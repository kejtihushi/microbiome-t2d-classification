############################################################
# SKRIPTI 2: NDËRTIMI I RRJETAVE DHE VEÇORITË PËR-SAMPLE
# Hapi 6: një rrjet i vetëm mbi train (për ML) +
#         dy rrjeta të ndara healthy/T2D (për krahasim)
# Hapi 7: nxjerrja e veçorive në nivel mostre
# Të gjitha ndërtohen VETËM mbi train (shmangie data leakage)
############################################################

import pandas as pd
import numpy as np
import networkx as nx
from scipy import linalg
from scipy.stats import spearmanr
import warnings

import os
os.makedirs("data", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
warnings.filterwarnings('ignore')

THRESHOLD = 0.3   # prag fiks i korrelacionit Spearman

# ---------- Leximi i të dhënave nga Skripti 1 ----------
X_train = pd.read_csv("data/dt_X_train.csv", index_col=0)
X_test  = pd.read_csv("data/dt_X_test.csv",  index_col=0)
y_train = pd.read_csv("data/dt_y_train.csv", index_col=0)['disease'].values
y_test  = pd.read_csv("data/dt_y_test.csv",  index_col=0)['disease'].values

print("="*60)
print("SKRIPTI 2: RRJETAT DHE VEÇORITË PËR-SAMPLE")
print("="*60)

n_features = X_train.shape[1]

# ============================================================
# FUNKSIONET
# ============================================================
def build_network(X_group, threshold):
    """Ndërton një rrjet të papeshuar nga korrelacioni Spearman."""
    corr, _ = spearmanr(X_group.values)
    corr = np.nan_to_num(corr, nan=0.0)  # taksonet konstante -> 0
    n = X_group.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    iu = np.triu_indices(n, k=1)
    mask = np.abs(corr[iu]) >= threshold
    edges = [(int(iu[0][k]), int(iu[1][k])) for k in range(len(mask)) if mask[k]]
    G.add_edges_from(edges)
    return G

def node_metrics(G, n):
    """Llogarit metrikat e nyjeve një herë për rrjet."""
    deg     = np.array([G.degree(i) for i in range(n)], dtype=float)
    close   = np.array(list(nx.closeness_centrality(G).values()))
    clust   = np.array(list(nx.clustering(G).values()))
    betw    = np.array(list(nx.betweenness_centrality(G).values()))
    dc      = np.array(list(nx.degree_centrality(G).values()))
    # Modulet (komunitetet)
    comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    module = np.zeros(n, dtype=int)
    for idx, c in enumerate(comms):
        for node in c:
            module[node] = idx
    # Eigenvektoret e Laplacianit (3 të parët jo-trivial)
    L = nx.laplacian_matrix(G).toarray().astype(float)
    _, eigvecs = np.linalg.eigh(L)
    eig1 = eigvecs[:, 1] if n > 1 else np.zeros(n)
    eig2 = eigvecs[:, 2] if n > 2 else np.zeros(n)
    eig3 = eigvecs[:, 3] if n > 3 else np.zeros(n)
    return {
        'deg': deg, 'close': close, 'clust': clust, 'betw': betw, 'dc': dc,
        'module': module, 'n_modules': len(comms),
        'eig1': eig1, 'eig2': eig2, 'eig3': eig3
    }

def per_sample_features(X_samples, m):
    """Nxjerr veçori në nivel mostre duke ponderuar metrikat me abundancën."""
    rows = []
    vals = X_samples.values
    for i in range(vals.shape[0]):
        ab = vals[i]
        total = np.sum(ab) + 1e-10
        # top-3 modulet sipas abundancës së mostrës
        mod_sums_all = np.array([np.sum(ab[m['module']==k]) for k in range(m['n_modules'])])
        top3 = np.argsort(mod_sums_all)[::-1][:3]
        mod1 = mod_sums_all[top3[0]] if len(top3)>0 else 0.0
        mod2 = mod_sums_all[top3[1]] if len(top3)>1 else 0.0
        mod3 = mod_sums_all[top3[2]] if len(top3)>2 else 0.0
        rows.append({
            'w_degree'     : np.sum(m['deg']   * ab) / total,
            'w_closeness'  : np.sum(m['close'] * ab) / total,
            'w_clustering' : np.sum(m['clust'] * ab) / total,
            'w_betweenness': np.sum(m['betw']  * ab) / total,
            'w_deg_centr'  : np.sum(m['dc']    * ab) / total,
            'module_1_ab'  : mod1,
            'module_2_ab'  : mod2,
            'module_3_ab'  : mod3,
            'w_eigenvec1'  : np.sum(m['eig1']  * ab) / total,
            'w_eigenvec2'  : np.sum(m['eig2']  * ab) / total,
            'w_eigenvec3'  : np.sum(m['eig3']  * ab) / total,
        })
    return pd.DataFrame(rows, index=X_samples.index)

def global_metrics(G, n):
    """Veçori globale të rrjetit (vetëm për krahasim biologjik, JO për ML)."""
    deg = [d for _, d in G.degree() if d > 0]
    feats = {}
    feats['n_active_nodes'] = len(deg)
    feats['n_edges'] = G.number_of_edges()
    feats['avg_degree'] = float(np.mean(deg)) if deg else 0.0
    feats['density'] = nx.density(G)
    feats['avg_clustering'] = nx.average_clustering(G)
    comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    feats['modularity'] = nx.algorithms.community.modularity(G, comms)
    # spektrale mbi komponentin më të madh të lidhur
    lcc = max(nx.connected_components(G), key=len)
    Gl = G.subgraph(lcc).copy()
    L = nx.laplacian_matrix(Gl).toarray().astype(float)
    ev = np.sort(np.real(linalg.eigvals(L)))
    feats['algebraic_connectivity'] = float(ev[1]) if len(ev) > 1 else 0.0
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eva = np.real(linalg.eigvals(A))
    feats['spectral_radius'] = float(np.max(np.abs(eva)))
    return feats

# ============================================================
# A) NJË RRJET I VETËM MBI TRAIN -> veçori për-sample për ML
# ============================================================
print("\n--- A) Rrjeti i vetëm mbi train (për ML) ---")
G_all = build_network(X_train, THRESHOLD)
print(f"Rrjeti i vetëm: {G_all.number_of_nodes()} nyje, {G_all.number_of_edges()} lidhje")

m_all = node_metrics(G_all, n_features)
feat_train = per_sample_features(X_train, m_all)
feat_test  = per_sample_features(X_test,  m_all)

print(f"Veçori për-sample: {feat_train.shape[1]}")
print(f"Vlera unike (w_degree, train): {feat_train['w_degree'].nunique()}/{len(feat_train)}")

feat_train.to_csv("data/dt_feat_train.csv")
feat_test.to_csv("data/dt_feat_test.csv")
print("U ruajtën: dt_feat_train.csv, dt_feat_test.csv")

# ============================================================
# B) DY RRJETA TË NDARA healthy/T2D -> vetëm për krahasim biologjik
# ============================================================
print("\n--- B) Dy rrjeta të ndara (për krahasim biologjik) ---")
X_healthy = X_train[y_train == 1]
X_t2d     = X_train[y_train == 0]

G_h = build_network(X_healthy, THRESHOLD)
G_t = build_network(X_t2d, THRESHOLD)

glob_h = global_metrics(G_h, n_features)
glob_t = global_metrics(G_t, n_features)

cmp = pd.DataFrame({'Healthy': glob_h, 'T2D': glob_t})
print(cmp.round(4).to_string())
cmp.round(6).to_csv("results/tables/dt_krahasim_global.csv")
print("\nU ruajt: dt_krahasim_global.csv")
print("Skripti 2 përfundoi me sukses.")
