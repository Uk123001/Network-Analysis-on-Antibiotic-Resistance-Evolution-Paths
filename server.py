"""
server.py: Flask backend for the optrA Resistance Analysis UI
Bridges model1.py / model2.py with a REST API consumed by the web frontend.
Directory layout expected:
    project/
    ├── server.py       ← this file
    ├── model1.py
    ├── model2.py
    ├── Data_4.xlsx
    ├── Data_6.xlsx     
    ├── index.html
    ├── style.css
    └── app.js
Endpoints:
  POST /api/analyze        : analyse a raw DNA sequence
  GET  /api/test-sequence  : return a random TEST sequence from Data_4.xlsx
  GET  /api/health         : liveness check
"""

# Imported modules
import os
import sys
import json
import random
import logging
import warnings
import threading
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Import core classes from model2 (self-contained, no circular deps) ──────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model2 import (OptrAFeatureExtractor, DeBruijnGraphBuilder,
                    NodeFeatureComputer, GCNEncoder,
                    EdgeFeatureBuilder, MLPEdgeScorer)

from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ============================================================================
# GLOBAL STATE  (populated once at startup, read-only during requests)
# ============================================================================

_state = {
    'lr_model':      None,   # LogisticRegression  (51 features → resistance)
    'lr_scaler':     None,   # StandardScaler for LR
    'edge_scorer':   None,   # MLPEdgeScorer        (locus finding)
    'feat_computer': None,   # NodeFeatureComputer
    'extractor':     None,   # OptrAFeatureExtractor
    'test_seqs':     [],     # list of {header, sequence} dicts from Data_4 TEST split
    'ready':         False,
}
_startup_lock = threading.Lock()

DATA4_PATH = 'Data_4.xlsx'
DATA6_PATH = 'Data_6.xlsx'
MAX_LR_TRAIN    = 600    # sequences used for LR startup training
MAX_EDGE_TRAIN  = 60     # sequences used for edge-scorer startup training

# ============================================================================
# ANTIBIOTIC TYPE MAPPING
# ============================================================================

# optrA gene always confers Oxazolidinone + Phenicol resistance at baseline.
# Additional antibiotics are inferred from extracted feature signals.

_ALWAYS_RESISTANT = [
    {'name': 'Linezolid',       'class': 'Oxazolidinone', 'base_conf': 0.88},
    {'name': 'Tedizolid',       'class': 'Oxazolidinone', 'base_conf': 0.82},
    {'name': 'Chloramphenicol', 'class': 'Phenicol',      'base_conf': 0.79},
    {'name': 'Florfenicol',     'class': 'Phenicol',      'base_conf': 0.74},
]

def identify_antibiotic_types(feat_dict):
    """
    Returns list of {name, class, confidence} dicts sorted by confidence desc.
    Uses heuristic rules on the 51-feature dictionary.
    """
    types = []

    rei  = feat_dict.get('51_Resistance_Evolution_Index', 0)
    hgt  = feat_dict.get('50_HGT_Risk_Score', 0)
    s83f = feat_dict.get('15_S83F_Mutation_Count', 0)
    d87n = feat_dict.get('16_D87N_Mutation_Count', 0)
    e84k = feat_dict.get('17_E84K_Mutation_Count', 0)
    s80i = feat_dict.get('18_S80I_Mutation_Count', 0)
    is_t = feat_dict.get('27_Total_IS_Elements', 0)
    prom = feat_dict.get('21_Promoter_Strength', 0)

    # Always-present optrA resistances (boosted by evolution index)
    for entry in _ALWAYS_RESISTANT:
        conf = min(entry['base_conf'] + rei * 0.12, 0.99)
        types.append({'name': entry['name'], 'class': entry['class'],
                      'confidence': round(conf, 4)})

    # Fluoroquinolones (GyrA / ParC mutations)
    fq_signal = s83f + d87n
    if fq_signal > 0:
        conf = min(0.45 + fq_signal * 0.12, 0.88)
        types.append({'name': 'Ciprofloxacin',  'class': 'Fluoroquinolone', 'confidence': round(conf, 4)})
        types.append({'name': 'Enrofloxacin',   'class': 'Fluoroquinolone', 'confidence': round(conf * 0.9, 4)})

    # Oxazolidinone via ParC mutations
    parC_signal = e84k + s80i
    if parC_signal > 0:
        conf = min(0.50 + parC_signal * 0.10, 0.87)
        types.append({'name': 'Contezolid', 'class': 'Oxazolidinone', 'confidence': round(conf, 4)})

    # Mobile-element / HGT mediated broad-spectrum
    if is_t > 2 or hgt > 0.5:
        conf = min(0.38 + is_t * 0.04 + hgt * 0.25, 0.80)
        types.append({'name': 'Tetracycline',   'class': 'Tetracycline',    'confidence': round(conf, 4)})
        types.append({'name': 'Erythromycin',   'class': 'Macrolide',       'confidence': round(conf * 0.85, 4)})

    # Efflux pump driven (strong promoter + hydrophobic regions)
    hydro = feat_dict.get('29_Hydrophobic_Region_Count', 0)
    if prom > 0.6 and hydro > 0:
        conf = min(0.42 + prom * 0.20 + hydro * 0.03, 0.78)
        types.append({'name': 'Ampicillin',  'class': 'Beta-lactam', 'confidence': round(conf, 4)})

    # De-duplicate by name, keep highest confidence
    seen = {}
    for t in types:
        if t['name'] not in seen or t['confidence'] > seen[t['name']]['confidence']:
            seen[t['name']] = t

    result = sorted(seen.values(), key=lambda x: x['confidence'], reverse=True)
    return result[:8]

# ============================================================================
# STARTUP  –  trains LR + edge scorer
# ============================================================================

def _startup_train():
    """
    Called once in a background thread just before the first request.
    Loads Data_4.xlsx, re-extracts 51 features, trains LR + edge scorer.
    """
    with _startup_lock:
        if _state['ready']:
            return
        log.info("=== Startup training begins ===")

        extractor     = OptrAFeatureExtractor()
        feat_computer = NodeFeatureComputer()
        _state['extractor']     = extractor
        _state['feat_computer'] = feat_computer

        # ── Load sequences ────────────────────────────────────────────────────
        if not os.path.exists(DATA4_PATH):
            log.warning(f"{DATA4_PATH} not found – models will be untrained.")
            _state['ready'] = True
            return

        df = pd.read_excel(DATA4_PATH, sheet_name='Data_Split')
        df = df.dropna(subset=['Sequence', 'Label'])
        df['Sequence'] = df['Sequence'].astype(str)

        test_df = df[df['Type'].str.upper() == 'TEST'].reset_index(drop=True)
        _state['test_seqs'] = test_df[['Header', 'Sequence']].to_dict(orient='records')
        log.info(f"Test sequences available for UI: {len(_state['test_seqs'])}")

        # ── Re-extract 51 features for LR training ───────────────────────────
        train_df = df.sample(min(MAX_LR_TRAIN, len(df)), random_state=42)
        log.info(f"Extracting 51 features for {len(train_df)} sequences (LR training)...")

        feat_rows, labels = [], []
        for _, row in train_df.iterrows():
            try:
                fd = extractor.extract_all_features(row['Sequence'])
                feat_rows.append(list(fd.values()))
                labels.append(int(row['Label']))
            except Exception as e:
                log.debug(f"Feature extraction failed: {e}")

        if len(feat_rows) < 10:
            log.warning("Too few valid sequences for LR training.")
            _state['ready'] = True
            return

        X_lr = np.array(feat_rows)
        y_lr = np.array(labels)

        lr_scaler = StandardScaler()
        X_lr_sc   = lr_scaler.fit_transform(X_lr)
        lr_model  = LogisticRegression(max_iter=2000, random_state=42,
                                       class_weight='balanced', solver='lbfgs')
        lr_model.fit(X_lr_sc, y_lr)
        _state['lr_model']  = lr_model
        _state['lr_scaler'] = lr_scaler
        log.info(f"LR trained | acc={lr_model.score(X_lr_sc, y_lr):.4f}")

        # ── Train edge scorer on de Bruijn graphs ─────────────────────────────
        log.info(f"Building de Bruijn graphs for edge scorer training (up to {MAX_EDGE_TRAIN} seqs)...")
        edge_df = df.sample(min(MAX_EDGE_TRAIN, len(df)), random_state=7)

        all_efeats, all_elabels = [], []
        for _, row in edge_df.iterrows():
            seq = row['Sequence']
            lbl = int(row['Label'])
            if len(seq) < DeBruijnGraphBuilder.K:
                continue
            try:
                G, edge_list = DeBruijnGraphBuilder.build(seq)
                node_index   = list(G.nodes())
                if not node_index:
                    continue
                node_feats = feat_computer.compute(seq, G)
                in_dim     = feat_computer.n_features
                gcn        = GCNEncoder(in_dim=in_dim, hidden_dim=64, out_dim=32)
                H          = gcn.encode(node_index, node_feats, G)
                ef, _      = EdgeFeatureBuilder.build(H, node_index, edge_list)
                if len(ef) == 0:
                    continue
                all_efeats.append(ef)
                all_elabels.append(np.full(len(ef), lbl, dtype=int))
            except Exception as e:
                log.debug(f"Graph build failed for edge training: {e}")

        if all_efeats:
            X_e = np.vstack(all_efeats)
            y_e = np.concatenate(all_elabels)
            scorer = MLPEdgeScorer()
            scorer.fit(X_e, y_e)
            _state['edge_scorer'] = scorer
            log.info(f"Edge scorer trained on {len(X_e)} edges.")
        else:
            log.warning("Edge scorer could not be trained – no valid graphs.")

        _state['ready'] = True
        log.info("=== Startup training complete ===")

# ============================================================================
# ANALYSIS LOGIC
# ============================================================================

def analyse_sequence(sequence):
    """
    Core analysis function.
    Returns a dict ready to be JSON-serialised for the frontend.
    """
    seq = sequence.upper().strip()

    # ── 1. Extract 51 features ────────────────────────────────────────────────
    extractor = _state['extractor'] or OptrAFeatureExtractor()
    feat_dict = extractor.extract_all_features(seq)
    feat_vec  = np.array(list(feat_dict.values())).reshape(1, -1)

    # ── 2. LR classification + confidence ────────────────────────────────────
    if _state['lr_model'] and _state['lr_scaler']:
        X_sc       = _state['lr_scaler'].transform(feat_vec)
        prob       = float(_state['lr_model'].predict_proba(X_sc)[0, 1])
        is_resist  = prob >= 0.5
    else:
        # Fallback: use resistance evolution index as proxy
        prob      = float(min(feat_dict.get('51_Resistance_Evolution_Index', 0) * 2, 1.0))
        is_resist = prob >= 0.5

    # ── 3. Build de Bruijn graph ──────────────────────────────────────────────
    G, edge_list = DeBruijnGraphBuilder.build(seq)
    node_index   = list(G.nodes())

    if not node_index or len(edge_list) == 0:
        return {
            'is_resistant':     is_resist,
            'confidence':       round(prob * 100, 2),
            'loci':             [],
            'antibiotic_types': identify_antibiotic_types(feat_dict),
            'error':            'Sequence too short to build graph.',
        }

    # ── 4. GCN encode ────────────────────────────────────────────────────────
    feat_computer = _state['feat_computer'] or NodeFeatureComputer()
    node_feats    = feat_computer.compute(seq, G)
    in_dim        = feat_computer.n_features
    gcn           = GCNEncoder(in_dim=in_dim, hidden_dim=64, out_dim=32)
    H             = gcn.encode(node_index, node_feats, G)

    # ── 5. Score edges ────────────────────────────────────────────────────────
    edge_feats, valid_edges = EdgeFeatureBuilder.build(H, node_index, edge_list)

    scorer = _state['edge_scorer']
    if scorer and len(edge_feats) > 0:
        scores = scorer.score(edge_feats)
    else:
        # Fallback: use GC content as score proxy
        scores = edge_feats[:, -4] if len(edge_feats) > 0 else np.array([])

    # ── 6. Top-3 loci with ±5 node neighbourhood ─────────────────────────────
    TOP_N = 3
    NEIGH = 5
    top_indices = np.argsort(scores)[::-1][:TOP_N] if len(scores) >= TOP_N else \
                  np.argsort(scores)[::-1]

    # Ordered path nodes (sequence order, de-duplicated)
    path_nodes = _ordered_path_nodes(seq)

    loci = []
    for rank, ti in enumerate(top_indices):
        src, dst, pos, hexamer = valid_edges[ti]
        score = float(scores[ti])

        try:
            src_pos = path_nodes.index(src)
        except ValueError:
            src_pos = len(path_nodes) // 2

        lo        = max(0, src_pos - NEIGH)
        hi        = min(len(path_nodes), src_pos + NEIGH + 2)
        sub_nodes = path_nodes[lo:hi]

        # Build subgraph data for D3
        node_gc = {n: sum(1 for b in n if b in 'GC') / max(len(n), 1) for n in sub_nodes}
        nodes_d3 = [{'id': n, 'gc': round(node_gc[n], 3)} for n in sub_nodes]
        edges_d3 = []
        for i in range(len(sub_nodes) - 1):
            u, v = sub_nodes[i], sub_nodes[i + 1]
            e_score = _mean_edge_score(u, v, valid_edges, scores)
            edges_d3.append({
                'source':   u,
                'target':   v,
                'score':    round(e_score, 4),
                'is_locus': (u == src and v == dst),
                'hexamer':  seq[pos: pos + DeBruijnGraphBuilder.K]
                            if pos + DeBruijnGraphBuilder.K <= len(seq) else hexamer,
            })

        loci.append({
            'rank':     rank + 1,
            'hexamer':  hexamer,
            'position': int(pos),
            'score':    round(score, 4),
            'nodes':    nodes_d3,
            'edges':    edges_d3,
        })

    # ── 7. Antibiotic types ───────────────────────────────────────────────────
    ab_types = identify_antibiotic_types(feat_dict)

    # ── 8. Key feature summary for UI ────────────────────────────────────────
    feature_summary = {
        'gc_percent':          round(feat_dict.get('9_GC_Percent', 0), 2),
        'hgt_risk':            round(feat_dict.get('50_HGT_Risk_Score', 0), 4),
        'mutation_count':      int(sum(feat_dict.get(f, 0) for f in
                                      ['15_S83F_Mutation_Count', '16_D87N_Mutation_Count',
                                       '17_E84K_Mutation_Count', '18_S80I_Mutation_Count'])),
        'is_elements':         int(feat_dict.get('27_Total_IS_Elements', 0)),
        'promoter_strength':   round(feat_dict.get('21_Promoter_Strength', 0), 4),
        'evolution_index':     round(feat_dict.get('51_Resistance_Evolution_Index', 0), 4),
    }

    return {
        'is_resistant':     bool(is_resist),
        'confidence':       round(prob * 100, 2),
        'loci':             loci,
        'antibiotic_types': ab_types,
        'features':         feature_summary,
        'sequence_length':  len(seq),
    }

def _ordered_path_nodes(sequence):
    """Pentamers in order of first appearance (Eulerian path nodes)."""
    seq   = sequence.upper()
    k     = DeBruijnGraphBuilder.K
    seen, result = set(), []
    for i in range(len(seq) - k + 1):
        src = seq[i: i + k - 1]
        if src not in seen:
            result.append(src); seen.add(src)
    if len(seq) >= k:
        dst = seq[-(k - 1):]
        if dst not in seen:
            result.append(dst)
    return result

def _mean_edge_score(u, v, edge_list, scores):
    s = [scores[i] for i, (s_, d_, *_) in enumerate(edge_list)
         if s_ == u and d_ == v]
    return float(np.mean(s)) if s else 0.0

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.before_request
def ensure_ready():
    if not _state['ready']:
        _startup_train()


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'models_ready': _state['ready']})


@app.route('/api/test-sequence')
def test_sequence():
    seqs = _state.get('test_seqs', [])
    if not seqs:
        return jsonify({'error': 'No test sequences loaded.'}), 404
    choice = random.choice(seqs)
    return jsonify({'header': choice.get('Header', ''), 'sequence': choice['Sequence']})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    seq  = (data.get('sequence') or '').strip()

    if not seq:
        return jsonify({'error': 'No sequence provided.'}), 400
    if len(seq) < DeBruijnGraphBuilder.K:
        return jsonify({'error': f'Sequence must be at least {DeBruijnGraphBuilder.K} bases.'}), 400

    # Sanitise – keep only ATGCN
    seq_clean = ''.join(c for c in seq.upper() if c in 'ATGCN')
    if len(seq_clean) < DeBruijnGraphBuilder.K:
        return jsonify({'error': 'Sequence contains too few valid bases (A/T/G/C).'}), 400

    try:
        result = analyse_sequence(seq_clean)
        return jsonify(result)
    except Exception as e:
        log.exception("Analysis failed")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    log.info("Starting optrA Resistance Analysis Server …")
    _startup_train()   # train synchronously before serving
    app.run(host='0.0.0.0', port=5000, debug=False)