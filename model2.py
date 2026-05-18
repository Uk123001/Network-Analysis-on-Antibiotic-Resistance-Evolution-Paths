"""
Step 6: Model 2: GNN Resistance Locus Finder

Graph design: de Bruijn graph (k=6)
  • Nodes  = (k-1)-mers  = pentamers  (5-base substrings)
  • Edges  = k-mers      = hexamers   (6-base substrings, one directed edge per occurrence)
  → Any sequence of length L produces L-5 hexamer edges forming an Eulerian path.

Pipeline per sequence
  1. Build de Bruijn graph from sequence
  2. Assign node feature vectors (local 51-feature sub-scores for each pentamer's context)
  3. Run 2-layer GCN message passing (pure numpy – no external GNN library needed)
  4. Score every edge with a trained MLP:  score = MLP([h_u ‖ h_v ‖ edge_features])
  5. Identify top-3 highest-scoring edges (= resistance loci)
  6. Extract ±5-node neighbourhood around each locus
  7. Visualise the three subgraphs (Eulerian-path excerpt) with colour-coded scores

MLP training:
  • One graph per sequence, each edge produces a feature vector
  • Edges from Label=1 sequences → positive class; Label=0 → negative class (via Data_6 labels)
  • MLP (binary): trained on pooled edge-feature vectors across all sequences in Data_6

Inline feature extractor:
  Full OptrAFeatureExtractor is embedded below (no external file import required).

Input : Data_6.xlsx  (Resistant_Sequences sheet, output of model1.py)
Output: ./GNN_PLOTS/<header>_top3_loci.png  (one image per sequence)
"""

# Imported modules
import os
import re
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing  import StandardScaler

warnings.filterwarnings('ignore')

# ===========================================================================
# ██████████████████████  INLINED FEATURE EXTRACTOR  ████████████████████████
# ===========================================================================

class OptrAFeatureExtractor:
    """
    Extracts the 51 optrA-specific resistance features from a raw DNA sequence.
    Inlined verbatim from feature_extraction.py (Step 2) so model2.py is
    fully self-contained.
    """

    RESISTANCE_MUTATIONS = {
        'S83F': [('TCC', 'TTC'), ('TCT', 'TTT')],
        'D87N': [('GAC', 'AAC'), ('GAT', 'AAT')],
        'E84K': [('GAA', 'AAA'), ('GAG', 'AAG')],
        'S80I': [('TCA', 'ATA'), ('TCT', 'ATT')],
    }

    PROMOTER_SEQUENCES = {
        'TATAAT': 'Pribnow_box',
        'TTGACA': 'Consensus_-35',
        'TGACA':  '-35_variant',
    }

    IS_ELEMENTS = {
        'AAATAAA':   'IS1',
        'TTGTTT':    'IS10_left',
        'TGACA':     'IS5_inverted',
        'AATAATTGAA':'IS102',
    }

    def __init__(self):
        self.codon_table = self._create_codon_table()

    def _create_codon_table(self):
        return {
            'ATA':'I','ATC':'I','ATT':'I','ATG':'M',
            'ACA':'T','ACC':'T','ACG':'T','ACT':'T',
            'AAC':'N','AAT':'N','AAA':'K','AAG':'K',
            'AGC':'S','AGT':'S','AGA':'R','AGG':'R',
            'CTA':'L','CTC':'L','CTG':'L','CTT':'L',
            'CCA':'P','CCC':'P','CCG':'P','CCT':'P',
            'CAC':'H','CAT':'H','CAA':'Q','CAG':'Q',
            'CGA':'R','CGC':'R','CGG':'R','CGT':'R',
            'GTA':'V','GTC':'V','GTG':'V','GTT':'V',
            'GCA':'A','GCC':'A','GCG':'A','GCT':'A',
            'GAC':'D','GAT':'D','GAA':'E','GAG':'E',
            'GGA':'G','GGC':'G','GGG':'G','GGT':'G',
            'TCA':'S','TCC':'S','TCG':'S','TCT':'S',
            'TTC':'F','TTT':'F','TTA':'L','TTG':'L',
            'TAC':'Y','TAT':'Y','TAA':'*','TAG':'*',
            'TGC':'C','TGT':'C','TGA':'*','TGG':'W',
        }

    # ── main entry point ────────────────────────────────────────────────────

    def extract_all_features(self, sequence):
        seq = sequence.upper().strip()
        if not seq:
            return {f: 0.0 for f in self._feature_names()}
        f = {}

        # GROUP 1: Gene Structure & Integrity
        f['1_Gene_Length_bp']           = len(seq)
        f['2_Has_Start_Codon_ATG']      = 1.0 if 'ATG' in seq else 0.0
        f['3_Has_Stop_Codon']           = 1.0 if any(s in seq for s in ('TAA','TAG','TGA')) else 0.0
        f['4_ORF_Completeness']         = self._orf_completeness(seq)
        f['5_Frame_Integrity']          = self._frame_integrity(seq)
        f['6_Internal_Stop_Codons']     = float(self._count_internal_stops(seq))
        f['7_Sequence_Length_Normalized']= min(len(seq)/1200, 1.0)
        f['8_Is_Full_Length_Gene']      = 1.0 if 950 <= len(seq) <= 1300 else 0.0

        # GROUP 2: GC Composition
        gc = seq.count('G') + seq.count('C')
        n  = max(len(seq), 1)
        f['9_GC_Percent']               = gc / n * 100
        f['10_GC_Skew']                 = self._gc_skew(seq)
        f['11_AT_Skew']                 = self._at_skew(seq)
        f['12_GC_Codon_Pos3']           = self._gc_at_position(seq, 2)
        f['13_Purine_Percent']          = (seq.count('A')+seq.count('G')) / n * 100
        f['14_Pyrimidine_Percent']      = (seq.count('C')+seq.count('T')) / n * 100

        # GROUP 3: Known Resistance Mutations
        f['15_S83F_Mutation_Count']     = float(self._detect_mutation_pattern(seq,'S83F'))
        f['16_D87N_Mutation_Count']     = float(self._detect_mutation_pattern(seq,'D87N'))
        f['17_E84K_Mutation_Count']     = float(self._detect_mutation_pattern(seq,'E84K'))
        f['18_S80I_Mutation_Count']     = float(self._detect_mutation_pattern(seq,'S80I'))

        # GROUP 4: Promoter & Regulatory Elements
        f['19_Pribnow_Box_Count']       = float(seq.count('TATAAT'))
        f['20_Consensus_-35_Count']     = float(seq.count('TTGACA'))
        f['21_Promoter_Strength']       = self._promoter_strength(seq)
        f['22_Upstream_Regulatory_GC']  = self._upstream_regulatory_gc(seq)
        f['23_TATA_Box_Distance_to_Start'] = float(self._distance_to_start_codon(seq,'TATAAT'))

        # GROUP 5: IS Elements
        f['24_IS1_Count']               = float(seq.count('AAATAAA'))
        f['25_IS10_Count']              = float(seq.count('TTGTTT'))
        f['26_IS5_Count']               = float(seq.count('TGACA'))
        f['27_Total_IS_Elements']       = float(self._count_is_elements(seq))
        f['28_IS_Element_Density']      = f['27_Total_IS_Elements'] / max(n/100, 1)

        # GROUP 6: Efflux Pump Characteristics
        f['29_Hydrophobic_Region_Count']= float(self._count_hydrophobic_regions(seq))
        f['30_Transmembrane_Domain_Count']=float(self._predict_transmembrane_domains(seq))
        f['31_Protein_Length_aa']       = float(n // 3)
        f['32_Expected_MW_kDa']         = float((n // 3) * 0.110)
        f['33_Charge_Bias']             = self._calculate_charge_bias(seq)

        # GROUP 7: Dinucleotide & Codon Bias
        f['34_CpG_Count']               = float(seq.count('CG'))
        f['35_CpG_per_100bp']           = seq.count('CG') / n * 100
        f['36_CpG_ObsExp_Ratio']        = self._cpg_obsexp(seq)
        f['37_Codon_Adaptation_Index']  = self._codon_adaptation_index(seq)
        f['38_Codon_Bias_Strength']     = self._codon_bias_strength(seq)
        f['39_Unique_Codons_Used']      = float(self._count_unique_codons(seq))

        # GROUP 8: Sequence Complexity
        f['40_Entropy_Trinucleotide']   = self._entropy(self._get_kmers(seq,3))
        f['41_Entropy_Tetranucleotide'] = self._entropy(self._get_kmers(seq,4))
        f['42_4mer_Complexity']         = self._kmer_complexity(seq,4)
        f['43_Repeat_Fraction']         = self._repeat_fraction(seq)
        f['44_Longest_Homopolymer']     = float(self._longest_homopolymer(seq))

        # GROUP 9: Mutation Signals & Adaptation
        f['45_Synonymous_Site_Density']    = self._synonymous_site_density(seq)
        f['46_NonSynonymous_Site_Density'] = self._nonsynonymous_site_density(seq)
        f['47_dN_dS_Proxy']                = self._dn_ds_proxy(seq)
        f['48_Transition_Transversion_Proxy'] = self._ts_tv_proxy(seq)
        f['49_Mutability_Index']           = self._mutability_index(seq)

        # GROUP 10: Functional & Evolution Metrics
        f['50_HGT_Risk_Score']             = self._hgt_risk_score(seq)
        f['51_Resistance_Evolution_Index'] = self._resistance_evolution_index(seq)

        return f

    def _feature_names(self):
        return [
            '1_Gene_Length_bp','2_Has_Start_Codon_ATG','3_Has_Stop_Codon',
            '4_ORF_Completeness','5_Frame_Integrity','6_Internal_Stop_Codons',
            '7_Sequence_Length_Normalized','8_Is_Full_Length_Gene',
            '9_GC_Percent','10_GC_Skew','11_AT_Skew','12_GC_Codon_Pos3',
            '13_Purine_Percent','14_Pyrimidine_Percent',
            '15_S83F_Mutation_Count','16_D87N_Mutation_Count',
            '17_E84K_Mutation_Count','18_S80I_Mutation_Count',
            '19_Pribnow_Box_Count','20_Consensus_-35_Count',
            '21_Promoter_Strength','22_Upstream_Regulatory_GC',
            '23_TATA_Box_Distance_to_Start',
            '24_IS1_Count','25_IS10_Count','26_IS5_Count',
            '27_Total_IS_Elements','28_IS_Element_Density',
            '29_Hydrophobic_Region_Count','30_Transmembrane_Domain_Count',
            '31_Protein_Length_aa','32_Expected_MW_kDa','33_Charge_Bias',
            '34_CpG_Count','35_CpG_per_100bp','36_CpG_ObsExp_Ratio',
            '37_Codon_Adaptation_Index','38_Codon_Bias_Strength',
            '39_Unique_Codons_Used',
            '40_Entropy_Trinucleotide','41_Entropy_Tetranucleotide',
            '42_4mer_Complexity','43_Repeat_Fraction','44_Longest_Homopolymer',
            '45_Synonymous_Site_Density','46_NonSynonymous_Site_Density',
            '47_dN_dS_Proxy','48_Transition_Transversion_Proxy',
            '49_Mutability_Index',
            '50_HGT_Risk_Score','51_Resistance_Evolution_Index',
        ]

    # ── helper methods (identical to feature_extraction.py) ─────────────────

    def _orf_completeness(self, seq):
        has_start = 1.0 if 'ATG' in seq else 0.0
        has_stop  = 1.0 if any(s in seq for s in ('TAA','TAG','TGA')) else 0.0
        return (has_start + has_stop + min(len(seq)/1200,1.0)) / 3.0

    def _frame_integrity(self, seq):
        if len(seq) < 3: return 0.0
        codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
        valid  = sum(1 for c in codons if len(c)==3 and all(b in 'ATGC' for b in c))
        return valid / len(codons) if codons else 0.0

    def _count_internal_stops(self, seq):
        stops = {'TAA','TAG','TGA'}
        return sum(1 for i in range(0, len(seq)-5, 3) if seq[i:i+3] in stops)

    def _gc_skew(self, seq):
        g, c = seq.count('G'), seq.count('C')
        return (g-c)/(g+c) if (g+c) > 0 else 0.0

    def _at_skew(self, seq):
        a, t = seq.count('A'), seq.count('T')
        return (a-t)/(a+t) if (a+t) > 0 else 0.0

    def _gc_at_position(self, seq, pos):
        total = count = 0
        for i in range(pos, len(seq)-2, 3):
            total += 1
            if seq[i] in 'GC': count += 1
        return (count/total*100) if total > 0 else 0.0

    def _detect_mutation_pattern(self, seq, mutation):
        count = 0
        for _, mutant in self.RESISTANCE_MUTATIONS.get(mutation, []):
            count += seq.count(mutant)
        return count

    def _promoter_strength(self, seq):
        score = 0.0
        if 'TATAAT' in seq: score += 0.5
        if 'TTGACA' in seq: score += 0.3
        if 'TGACA'  in seq: score += 0.2
        return min(score, 1.0)

    def _upstream_regulatory_gc(self, seq):
        up = seq[:300]
        gc = up.count('G') + up.count('C')
        return (gc/len(up)*100) if up else 0.0

    def _distance_to_start_codon(self, seq, motif):
        s, m = seq.find('ATG'), seq.find(motif)
        return abs(s-m) if s >= 0 and m >= 0 else len(seq)

    def _count_hydrophobic_regions(self, seq):
        hydro = set('ILVFM')
        protein, count, consec = self._translate(seq), 0, 0
        for aa in protein:
            if aa in hydro:
                consec += 1
                if consec >= 10: count += 1; consec = 0
            else:
                consec = 0
        return count

    def _predict_transmembrane_domains(self, seq):
        kd = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,
              'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,
              'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,
              'W':-0.9,'Y':-1.3}
        protein, w, count = self._translate(seq), 9, 0
        for i in range(len(protein)-w):
            if sum(kd.get(aa,0) for aa in protein[i:i+w])/w > 1.5:
                count += 1
        return count

    def _calculate_charge_bias(self, seq):
        p = self._translate(seq)
        pos = p.count('K') + p.count('R')
        neg = p.count('D') + p.count('E')
        return (pos-neg)/len(p) if p else 0.0

    def _translate(self, seq):
        return ''.join(self.codon_table.get(seq[i:i+3],'X')
                       for i in range(0, len(seq)-2, 3))

    def _cpg_obsexp(self, seq):
        cpg, c, g = seq.count('CG'), seq.count('C'), seq.count('G')
        exp = (c*g)/max(len(seq),1)
        return (cpg/exp) if exp > 0 else 0.0

    def _codon_adaptation_index(self, seq):
        codons = [seq[i:i+3] for i in range(0,len(seq)-2,3)]
        if not codons: return 0.0
        rare = sum(1 for c in codons if c in {'CGA','CGG','AGA','AGG','TTA'})
        return 1.0 - (rare/len(codons))

    def _codon_bias_strength(self, seq):
        codons = [seq[i:i+3] for i in range(0,len(seq)-2,3)]
        if not codons: return 0.0
        top = Counter(codons).most_common(1)[0][1]
        return top/len(codons)

    def _count_unique_codons(self, seq):
        return len({seq[i:i+3] for i in range(0,len(seq)-2,3)})

    def _get_kmers(self, seq, k):
        return [seq[i:i+k] for i in range(len(seq)-k+1)]

    def _entropy(self, kmers):
        if not kmers: return 0.0
        total = len(kmers)
        return -sum((c/total)*np.log2(c/total)
                    for c in Counter(kmers).values() if c > 0)

    def _kmer_complexity(self, seq, k):
        kmers = self._get_kmers(seq,k)
        return len(set(kmers)) / (4**k) if kmers else 0.0

    def _repeat_fraction(self, seq):
        kmers = self._get_kmers(seq,4)
        if not kmers: return 0.0
        cnt = Counter(kmers)
        return sum(1 for c in cnt.values() if c > 1) / len(cnt)

    def _longest_homopolymer(self, seq):
        if not seq: return 0
        mx = cr = 1
        for i in range(1, len(seq)):
            cr = cr+1 if seq[i] == seq[i-1] else 1
            mx = max(mx, cr)
        return mx

    def _synonymous_site_density(self, seq):
        codons = [seq[i:i+3] for i in range(0,len(seq)-2,3)]
        return len(codons)/max(len(seq)/100,1)

    def _nonsynonymous_site_density(self, seq):
        return self._synonymous_site_density(seq) * 0.8

    def _dn_ds_proxy(self, seq):
        syn = self._synonymous_site_density(seq)
        return (self._nonsynonymous_site_density(seq)/syn) if syn > 0 else 0.0

    def _ts_tv_proxy(self, seq):
        ts = seq.count('GA')+seq.count('AG')+seq.count('CT')+seq.count('TC')
        tv = seq.count('AC')+seq.count('CA')+seq.count('GT')+seq.count('TG')
        return (ts/tv) if tv > 0 else 0.0

    def _mutability_index(self, seq):
        return (seq.count('CG') + seq.count('TA') + seq.count('AT')) / max(len(seq),1)

    def _hgt_risk_score(self, seq):
        gc  = (seq.count('G')+seq.count('C')) / max(len(seq),1)
        is_ = self._count_is_elements(seq)
        ps  = self._promoter_strength(seq)
        return min(is_*0.3 + abs(gc-0.5)*0.4 + ps*0.3, 1.0)

    def _count_is_elements(self, seq):
        return sum(seq.count(e) for e in self.IS_ELEMENTS)

    def _resistance_evolution_index(self, seq):
        mut = sum(self._detect_mutation_pattern(seq,m)
                  for m in ('S83F','D87N','E84K','S80I')) / 4.0
        return mut*0.5 + self._hgt_risk_score(seq)*0.5

# ===========================================================================
# ████████████████████████  DE BRUIJN GRAPH BUILDER  ████████████████████████
# ===========================================================================

class DeBruijnGraphBuilder:
    """
    Build a directed de Bruijn graph (k=6) from a DNA sequence.

    Nodes: pentamers  (k-1 = 5 base substrings)
    Edges: hexamers   (k   = 6 base substrings, one occurrence = one directed edge)

    An Eulerian path through this graph reconstructs the original sequence.
    """

    K = 6   # hexamer edges

    @staticmethod
    def build(sequence):
        """
        Returns:
            G          – networkx DiGraph
            edge_list  – list of (src_node, dst_node, position_in_seq, hexamer)
        """
        seq       = sequence.upper().strip()
        G         = nx.DiGraph()
        edge_list = []

        for pos in range(len(seq) - DeBruijnGraphBuilder.K + 1):
            hexamer = seq[pos: pos + DeBruijnGraphBuilder.K]
            src     = hexamer[:-1]   # pentamer prefix
            dst     = hexamer[1:]    # pentamer suffix

            if not G.has_node(src):
                G.add_node(src, occurrences=0, positions=[])
            if not G.has_node(dst):
                G.add_node(dst, occurrences=0, positions=[])

            G.nodes[src]['occurrences'] += 1
            G.nodes[src]['positions'].append(pos)

            # Allow parallel edges via key=pos
            G.add_edge(src, dst, hexamer=hexamer, position=pos, score=0.0)
            edge_list.append((src, dst, pos, hexamer))

        return G, edge_list

# ===========================================================================
# ████████████████████████  NODE FEATURE COMPUTER  ██████████████████████████
# ===========================================================================

class NodeFeatureComputer:
    """
    For each pentamer node in the graph, compute a 51-element feature vector
    by extracting full sequence features from a local context window centred
    on the node's first occurrence in the sequence.
    """

    WINDOW = 60    # bp context around each node occurrence

    def __init__(self):
        self.extractor = OptrAFeatureExtractor()
        self._feat_keys = list(self.extractor.extract_all_features('A'*100).keys())
        self.n_features = len(self._feat_keys)

    def compute(self, sequence, G):
        """
        Returns node_features: dict  node_label -> np.ndarray(51,)
        """
        seq          = sequence.upper()
        node_feats   = {}
        zero_vec     = np.zeros(self.n_features)

        for node in G.nodes():
            positions = G.nodes[node].get('positions', [])
            if not positions:
                node_feats[node] = zero_vec.copy()
                continue

            # Use first occurrence; average over up to 3 occurrences for stability
            vecs = []
            for p in positions[:3]:
                start = max(0, p - self.WINDOW // 2)
                end   = min(len(seq), p + self.WINDOW // 2 + 5)
                window_seq = seq[start:end]
                if len(window_seq) < 12:
                    continue
                fdict = self.extractor.extract_all_features(window_seq)
                vecs.append(np.array([fdict[k] for k in self._feat_keys], dtype=float))

            node_feats[node] = np.mean(vecs, axis=0) if vecs else zero_vec.copy()

        return node_feats

# ===========================================================================
# ████████████████████████  GCN MESSAGE PASSING  ████████████████████████████
# ===========================================================================

class GCNLayer:
    """
    A single Graph Convolutional Network layer (Kipf & Welling 2017).
    H_out = ReLU( A_hat @ H @ W )

    A_hat = D^{-1/2} (A + I) D^{-1/2}   (symmetric normalised adjacency)

    Implemented in pure numpy — no PyTorch or PyG dependency.
    """

    def __init__(self, in_dim, out_dim, random_state=42):
        rng         = np.random.RandomState(random_state)
        # Xavier initialisation
        limit       = np.sqrt(6.0 / (in_dim + out_dim))
        self.W      = rng.uniform(-limit, limit, (in_dim, out_dim))
        self.in_dim  = in_dim
        self.out_dim = out_dim

    def forward(self, A_hat, H):
        """A_hat: (N,N)  H: (N, in_dim)  → (N, out_dim)"""
        return np.maximum(0, A_hat @ H @ self.W)   # ReLU

class GCNEncoder:
    """
    2-layer GCN encoder.
    Produces enriched node embeddings that incorporate 2-hop neighbourhood
    context — the key advantage over raw per-node features.
    """

    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        self.layer1 = GCNLayer(in_dim,     hidden_dim, random_state=42)
        self.layer2 = GCNLayer(hidden_dim, out_dim,    random_state=43)

    def encode(self, node_index, node_features, G):
        """
        node_index  : list of node labels in fixed order
        node_features: dict  label -> np.ndarray(in_dim)
        G           : networkx DiGraph

        Returns: H (N, out_dim) in the same node_index order
        """
        N   = len(node_index)
        idx = {n: i for i, n in enumerate(node_index)}
        dim = next(iter(node_features.values())).shape[0]

        # ── Build feature matrix ────────────────────────────────────────────
        H0  = np.zeros((N, dim))
        for node, i in idx.items():
            H0[i] = node_features.get(node, np.zeros(dim))

        # ── Build adjacency (undirected for GCN) + self-loops ───────────────
        A   = np.zeros((N, N))
        for u, v in G.edges():
            if u in idx and v in idx:
                A[idx[u], idx[v]] = 1.0
                A[idx[v], idx[u]] = 1.0   # treat as undirected
        A  += np.eye(N)                    # self-loop (A + I)

        # ── Symmetric normalisation D^{-1/2} A D^{-1/2} ────────────────────
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-6)))
        A_hat      = D_inv_sqrt @ A @ D_inv_sqrt

        # ── Forward pass ────────────────────────────────────────────────────
        H1 = self.layer1.forward(A_hat, H0)
        H2 = self.layer2.forward(A_hat, H1)

        return H2   # shape (N, out_dim)

# ===========================================================================
# ████████████████████████  EDGE FEATURE BUILDER  ███████████████████████████
# ===========================================================================

class EdgeFeatureBuilder:
    """
    For each edge (u → v) in the graph, build an edge feature vector:
        [h_u  ‖  h_v  ‖  edge_signal]

    edge_signal: a 4-element vector derived directly from the hexamer:
        [GC_frac, has_mutation_motif, is_promoter_adjacent, hexamer_entropy]
    """

    MUTATION_HEXAMERS = {
        'TTC', 'TTT',   # S83F
        'AAC', 'AAT',   # D87N
        'AAA', 'AAG',   # E84K
        'ATA', 'ATT',   # S80I
    }

    PROMOTER_HEXAMERS = {'TATAAT', 'TTGACA'}

    @staticmethod
    def build(H_matrix, node_index, edge_list):
        """
        H_matrix  : (N, out_dim) GCN embeddings
        node_index: list of node labels
        edge_list : list of (src, dst, pos, hexamer)

        Returns:
            edge_features – np.ndarray  (E, 2*out_dim + 4)
            valid_edges   – filtered edge_list matching rows
        """
        idx   = {n: i for i, n in enumerate(node_index)}
        feats = []
        valid = []

        for src, dst, pos, hexamer in edge_list:
            if src not in idx or dst not in idx:
                continue
            h_u = H_matrix[idx[src]]
            h_v = H_matrix[idx[dst]]

            # Hexamer-level signals
            gc   = sum(1 for b in hexamer if b in 'GC') / 6.0
            mut  = 1.0 if any(m in hexamer for m in EdgeFeatureBuilder.MUTATION_HEXAMERS) else 0.0
            prom = 1.0 if hexamer in EdgeFeatureBuilder.PROMOTER_HEXAMERS else 0.0
            ent  = EdgeFeatureBuilder._hex_entropy(hexamer)

            edge_vec = np.concatenate([h_u, h_v, [gc, mut, prom, ent]])
            feats.append(edge_vec)
            valid.append((src, dst, pos, hexamer))

        return np.array(feats, dtype=float), valid

    @staticmethod
    def _hex_entropy(hexamer):
        cnt   = Counter(hexamer)
        total = len(hexamer)
        return -sum((c/total)*np.log2(c/total) for c in cnt.values() if c > 0)

# ===========================================================================
# ████████████████████████  MLP EDGE SCORER  ████████████████████████████████
# ===========================================================================

class MLPEdgeScorer:
    """
    Binary MLP trained on pooled edge features from all sequences.
    Label = sequence-level Label (resistant=1, non-resistant=0).

    After training, scores individual edges: higher score = more likely
    this edge (hexamer locus) contributes to resistance.

    Architecture: (128, 64, 32) – mirrors the Step-2 classifier.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        self._fitted = False

    def fit(self, all_edge_features, all_edge_labels):
        """
        all_edge_features : (E_total, D)
        all_edge_labels   : (E_total,)  – sequence label broadcast to all edges
        """
        print(f"  [MLP] Training on {len(all_edge_features)} edges  "
              f"(resistant={int(sum(all_edge_labels==1))}, "
              f"non-resistant={int(sum(all_edge_labels==0))})")

        X_sc = self.scaler.fit_transform(all_edge_features)
        # Guard: need both classes to train
        if len(np.unique(all_edge_labels)) < 2:
            print("  [MLP] ⚠️  Only one class found – using raw GC-content as score proxy.")
            self._fitted = False
            return

        self.model.fit(X_sc, all_edge_labels)
        self._fitted = True
        acc = self.model.score(X_sc, all_edge_labels)
        print(f"  [MLP] Train accuracy: {acc:.4f}")

    def score(self, edge_features):
        """Returns probability of resistance for each edge."""
        if not self._fitted or len(edge_features) == 0:
            # Fallback: use GC fraction (index = 2*out_dim element)
            return edge_features[:, -4] if edge_features.shape[1] > 4 else \
                   np.zeros(len(edge_features))
        X_sc = self.scaler.transform(edge_features)
        return self.model.predict_proba(X_sc)[:, 1]

# ===========================================================================
# ██████████████████████████  VISUALISER  ████████████████████████████████████
# ===========================================================================

class LocusVisualiser:
    """
    For each sequence visualises top-3 resistance loci as Eulerian-path
    subgraph excerpts (±5 nodes around each top-scoring edge).

    Layout:
      • 3 subplots side-by-side (one per locus)
      • Nodes: circles, coloured by GC fraction of the pentamer
      • Edges: arrows, width & colour proportional to resistance score
      • Title shows hexamer + position in original sequence + score
    """

    GNN_PLOTS_DIR = 'GNN_PLOTS'
    TOP_N         = 3
    NEIGHBOURHOOD = 5       # nodes on each side of the locus edge

    def __init__(self):
        os.makedirs(self.GNN_PLOTS_DIR, exist_ok=True)

    def plot(self, header, sequence, G, edge_list, edge_scores):
        """
        G           : full de Bruijn DiGraph
        edge_list   : list of (src, dst, pos, hexamer) – same order as edge_scores
        edge_scores : np.ndarray  (E,)  resistance probability per edge
        """
        if len(edge_scores) == 0:
            print(f"  [VIS] No edges to visualise for {header}")
            return

        # ── Top-3 loci ───────────────────────────────────────────────────────
        top_indices = np.argsort(edge_scores)[::-1][:self.TOP_N]

        fig, axes = plt.subplots(1, self.TOP_N,
                                 figsize=(6 * self.TOP_N, 6),
                                 facecolor='#F0F4F8')
        fig.suptitle(
            f'Top-3 Resistance Loci – {header[:60]}',
            fontsize=13, fontweight='bold', y=1.02
        )

        cmap_node = plt.cm.YlOrRd
        cmap_edge = plt.cm.RdYlGn_r   # red = high resistance

        for ax_idx, top_i in enumerate(top_indices):
            ax  = axes[ax_idx] if self.TOP_N > 1 else axes
            src, dst, pos, hexamer = edge_list[top_i]
            score = edge_scores[top_i]

            # Build ordered node list from Eulerian path (sequence order)
            path_nodes = self._sequence_order_nodes(sequence)

            # Find position of src in path
            try:
                src_pos_in_path = path_nodes.index(src)
            except ValueError:
                src_pos_in_path = len(path_nodes) // 2

            # Extract ±NEIGHBOURHOOD nodes
            lo  = max(0, src_pos_in_path - self.NEIGHBOURHOOD)
            hi  = min(len(path_nodes), src_pos_in_path + self.NEIGHBOURHOOD + 2)
            sub_nodes = path_nodes[lo:hi]

            # Build subgraph
            SG = nx.DiGraph()
            SG.add_nodes_from(sub_nodes)
            for i in range(len(sub_nodes) - 1):
                u, v = sub_nodes[i], sub_nodes[i+1]
                # Get score for this edge (mean of all occurrences)
                e_score = self._edge_score_for_pair(u, v, edge_list, edge_scores)
                SG.add_edge(u, v, score=e_score)

            # Layout: straight line (Eulerian path is a sequential walk)
            pos_layout = {n: (i, 0) for i, n in enumerate(sub_nodes)}

            # Node colours by GC fraction
            node_gc = {n: sum(1 for b in n if b in 'GC') / max(len(n), 1)
                       for n in sub_nodes}
            node_colors = [cmap_node(node_gc[n]) for n in sub_nodes]

            # Edge colours and widths
            edge_colors = []
            edge_widths = []
            for u, v, data in SG.edges(data=True):
                s = data.get('score', 0.0)
                edge_colors.append(cmap_edge(s))
                edge_widths.append(1.5 + s * 5)

            # Draw
            ax.set_facecolor('#FAFAFA')
            nx.draw_networkx_nodes(SG, pos_layout, ax=ax,
                                   node_color=node_colors,
                                   node_size=800,
                                   edgecolors='#333333',
                                   linewidths=1.2)
            nx.draw_networkx_labels(SG, pos_layout, ax=ax,
                                    font_size=7,
                                    font_color='#111111')
            nx.draw_networkx_edges(SG, pos_layout, ax=ax,
                                   edge_color=edge_colors,
                                   width=edge_widths,
                                   arrows=True,
                                   arrowsize=20,
                                   connectionstyle='arc3,rad=0.15')

            # Highlight locus edge
            if SG.has_edge(src, dst):
                nx.draw_networkx_edges(
                    SG, pos_layout, ax=ax,
                    edgelist=[(src, dst)],
                    edge_color='red',
                    width=4.0,
                    arrows=True,
                    arrowsize=24,
                    connectionstyle='arc3,rad=0.15',
                    style='dashed',
                )

            ax.set_title(
                f'Locus #{ax_idx+1}\n'
                f'Hexamer: {hexamer}  |  Pos: {pos}\n'
                f'Resistance Score: {score:.4f}',
                fontsize=9, pad=8
            )
            ax.axis('off')

        # Shared colourbars
        sm_node = plt.cm.ScalarMappable(cmap=cmap_node,
                                        norm=plt.Normalize(0, 1))
        sm_node.set_array([])
        sm_edge = plt.cm.ScalarMappable(cmap=cmap_edge,
                                        norm=plt.Normalize(0, 1))
        sm_edge.set_array([])

        cbar_node = fig.colorbar(sm_node, ax=axes, location='bottom',
                                 shrink=0.4, pad=0.08, label='Node GC fraction')
        cbar_edge = fig.colorbar(sm_edge, ax=axes, location='bottom',
                                 shrink=0.4, pad=0.15, label='Edge resistance score')

        plt.tight_layout()

        safe_header = re.sub(r'[^\w\-]', '_', header[:50])
        out_path = os.path.join(self.GNN_PLOTS_DIR, f'{safe_header}_top3_loci.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [VIS] Saved → {out_path}")
        return out_path

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sequence_order_nodes(sequence):
        """Return pentamers in the order they first appear in the sequence."""
        seq    = sequence.upper()
        k      = DeBruijnGraphBuilder.K
        seen   = []
        seen_s = set()
        for i in range(len(seq) - k + 1):
            hexamer = seq[i:i+k]
            src     = hexamer[:-1]
            if src not in seen_s:
                seen.append(src)
                seen_s.add(src)
        # Add final dst
        if len(seq) >= k:
            dst = seq[-k+1:]
            if dst not in seen_s:
                seen.append(dst)
        return seen

    @staticmethod
    def _edge_score_for_pair(u, v, edge_list, edge_scores):
        """Return mean score for all occurrences of edge (u, v)."""
        scores = [edge_scores[i]
                  for i, (s, d, _, _) in enumerate(edge_list)
                  if s == u and d == v]
        return float(np.mean(scores)) if scores else 0.0

# ===========================================================================
# ████████████████████████  MAIN PIPELINE  ██████████████████████████████████
# ===========================================================================

class Model2Pipeline:
    """
    Full GNN locus-finding pipeline.

    Phase A – Training (all sequences in Data_6):
      Build graphs → compute node features → GCN encode → build edge features
      → train MLP edge scorer

    Phase B – Inference (each sequence):
      Score all edges → pick top-3 → visualise ±5-node neighbourhood
    """

    def __init__(self,
                 input_file='Data_6.xlsx',
                 sheet='Resistant_Sequences'):
        self.input_file    = input_file
        self.sheet         = sheet
        self.feat_computer = NodeFeatureComputer()
        self.visualiser    = LocusVisualiser()

    # ── public entry ─────────────────────────────────────────────────────────

    def run(self):
        print("\n" + "="*80)
        print("MODEL 2 – GNN RESISTANCE LOCUS FINDER")
        print("de Bruijn Graph (k=6)  |  2-layer GCN  |  MLP Edge Scorer")
        print(f"Input : {self.input_file}  |  Output: ./GNN_PLOTS/")
        print("="*80)

        # [0] Load
        df = self._load_data()

        # [1] Build graphs + GCN embeddings for all sequences
        print(f"\n{'='*80}")
        print(f"[STEP 1] BUILDING DE BRUIJN GRAPHS + GCN ENCODING")
        print(f"{'='*80}")

        all_graphs    = []     # (header, sequence, G, edge_list, H, node_index)
        all_edge_feats = []
        all_edge_lbls  = []

        for _, row in df.iterrows():
            header   = str(row.get('Header', 'unknown'))
            sequence = str(row.get('Sequence', ''))
            label    = int(row.get('Label_Predicted',
                                   row.get('Label_Original', 1)))

            if len(sequence) < DeBruijnGraphBuilder.K:
                print(f"  ⚠️  Skipping {header}: sequence too short ({len(sequence)} bp)")
                continue

            print(f"  Processing: {header[:60]}  ({len(sequence)} bp)")

            # Build graph
            G, edge_list = DeBruijnGraphBuilder.build(sequence)
            node_index   = list(G.nodes())

            if len(node_index) == 0:
                continue

            # Node features (51-dim)
            node_feats = self.feat_computer.compute(sequence, G)
            in_dim     = self.feat_computer.n_features

            # GCN encode (→ 32-dim embeddings)
            gcn     = GCNEncoder(in_dim=in_dim, hidden_dim=64, out_dim=32)
            H       = gcn.encode(node_index, node_feats, G)

            # Build edge feature vectors
            edge_feats, valid_edges = EdgeFeatureBuilder.build(H, node_index, edge_list)

            if len(edge_feats) == 0:
                continue

            all_graphs.append((header, sequence, G, valid_edges, H, node_index))
            all_edge_feats.append(edge_feats)
            all_edge_lbls.append(np.full(len(edge_feats), label, dtype=int))

            print(f"    Nodes: {len(node_index):>5d}  Edges: {len(valid_edges):>6d}")

        if not all_graphs:
            print("❌ No valid sequences to process.")
            return

        # [2] Train MLP edge scorer
        print(f"\n{'='*80}")
        print(f"[STEP 2] TRAINING MLP EDGE SCORER")
        print(f"{'='*80}")

        X_all = np.vstack(all_edge_feats)
        y_all = np.concatenate(all_edge_lbls)

        scorer = MLPEdgeScorer()
        scorer.fit(X_all, y_all)

        # [3] Score edges per sequence + visualise
        print(f"\n{'='*80}")
        print(f"[STEP 3] SCORING EDGES + VISUALISING TOP-3 LOCI")
        print(f"{'='*80}")

        for (header, sequence, G, valid_edges, H, node_index), edge_feats in \
                zip(all_graphs, all_edge_feats):

            print(f"\n  Sequence: {header[:60]}")

            # Score
            scores = scorer.score(edge_feats)

            # Report top-3
            top_idx = np.argsort(scores)[::-1][:3]
            for rank, ti in enumerate(top_idx, 1):
                src, dst, pos, hexamer = valid_edges[ti]
                print(f"    Locus #{rank}: hexamer={hexamer}  pos={pos}  "
                      f"src={src} → dst={dst}  score={scores[ti]:.4f}")

            # Visualise
            self.visualiser.plot(header, sequence, G, valid_edges, scores)

        print(f"\n{'='*80}")
        print("✅ MODEL 2 COMPLETE")
        print(f"   Plots saved in ./GNN_PLOTS/")
        print(f"{'='*80}\n")

    # ── private ──────────────────────────────────────────────────────────────

    def _load_data(self):
        print(f"\n{'='*80}")
        print(f"[MODEL 2 | STEP 0] LOADING DATA")
        print(f"{'='*80}")
        print(f"File : {self.input_file}  |  Sheet: {self.sheet}")

        # model1.py writes a title on row 1 and a subtitle on row 2;
        # actual column headers are on row 4  →  header=3  (0-indexed)
        df = pd.read_excel(self.input_file, sheet_name=self.sheet, header=3)

        # Drop any fully-empty rows that may appear after the title block
        df.dropna(how='all', inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"✅ Loaded {len(df)} resistant sequences")
        print(f"   Columns: {list(df.columns)}")

        if 'Sequence' not in df.columns:
            raise ValueError(
                f"'Sequence' column not found in Data_6.xlsx "
                f"(found: {list(df.columns)}).  "
                f"Check that model1.py ran successfully and produced data rows."
            )
        return df

# =========================================================================
# ██████████████████████  TEST SEQUENCE INTERFACE  ████████████████████████
# =========================================================================

class SingleSequenceTester:
    """
    Convenience interface: given a raw DNA sequence, extract features,
    build its de Bruijn graph, score all edges (using a pre-fitted scorer),
    and visualise top-3 loci.

    Usage:
        tester = SingleSequenceTester(scorer, feat_computer, visualiser)
        tester.test(sequence, header="my_test_seq")
    """

    def __init__(self, scorer, feat_computer, visualiser):
        self.scorer        = scorer
        self.feat_computer = feat_computer
        self.visualiser    = visualiser

    def test(self, sequence, header='test_sequence'):
        print(f"\n{'='*80}")
        print(f"[SINGLE TEST] {header}  ({len(sequence)} bp)")
        print(f"{'='*80}")

        seq = sequence.upper().strip()
        if len(seq) < DeBruijnGraphBuilder.K:
            print(f"❌ Sequence too short (need ≥ {DeBruijnGraphBuilder.K} bp).")
            return

        G, edge_list = DeBruijnGraphBuilder.build(seq)
        node_index   = list(G.nodes())
        node_feats   = self.feat_computer.compute(seq, G)

        in_dim = self.feat_computer.n_features
        gcn    = GCNEncoder(in_dim=in_dim, hidden_dim=64, out_dim=32)
        H      = gcn.encode(node_index, node_feats, G)

        edge_feats, valid_edges = EdgeFeatureBuilder.build(H, node_index, edge_list)
        if len(edge_feats) == 0:
            print("❌ No valid edges extracted.")
            return

        scores = self.scorer.score(edge_feats)

        top_idx = np.argsort(scores)[::-1][:3]
        print("  Top-3 resistance loci:")
        for rank, ti in enumerate(top_idx, 1):
            src, dst, pos, hexamer = valid_edges[ti]
            print(f"    Locus #{rank}: hexamer={hexamer}  pos={pos}  "
                  f"{src} → {dst}  score={scores[ti]:.4f}")

        self.visualiser.plot(header, seq, G, valid_edges, scores)

# ====================== ENTRY POINT ======================

if __name__ == "__main__":
    pipeline = Model2Pipeline(
        input_file='Data_6.xlsx',
        sheet='Resistant_Sequences',
    )
    pipeline.run()

    # ── Optional: test a custom sequence without running the full pipeline ──
    # Uncomment and replace with a real optrA sequence to test interactively.
    #
    # TEST_SEQ = (
    #     "ATGAAAGTTATAATCCTGACAACAGGTTTTTTAGACATTGATCGTAAGCACGATGATGTTTATCGG"
    #     "GCAGTTGAAGAAATCAAAGACAAACTTGAAGCAATGGAAGTTGCTCGTCGTAAAGGTATTCGTTTA"
    # )
    # tester = SingleSequenceTester(
    #     scorer        = pipeline._build_scorer_for_test(),  # see below if needed
    #     feat_computer = pipeline.feat_computer,
    #     visualiser    = pipeline.visualiser,
    # )
    # tester.test(TEST_SEQ, header="custom_test_sequence")