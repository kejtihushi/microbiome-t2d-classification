############################################################
# Gjenerimi i figurave dhe tabelave për Kapitullin IV
# Dataset: QinJ_2012
############################################################

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Konfigurimi i stilit
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'

# Leximi i te dhenave
abundance = pd.read_csv("data/qin2012_abundance.csv", index_col=0)
metadata  = pd.read_csv("data/qin2012_metadata.csv",  index_col=0)

# Folderi ku ruhen figurat
OUT = "results/figures/"
import os

import os
os.makedirs("results/figures", exist_ok=True)
os.makedirs(OUT, exist_ok=True)

############################################################
# FIGURA 4.1 - Shpërndarja e klasave
############################################################
fig, ax = plt.subplots(figsize=(7, 5))
counts = metadata['disease'].value_counts()
colors = ['#2E86AB', '#E63946']
bars = ax.bar(counts.index, counts.values, color=colors, width=0.55, edgecolor='black', linewidth=1)
for bar, val in zip(bars, counts.values):
    pct = val/counts.sum()*100
    ax.text(bar.get_x()+bar.get_width()/2, val+3, f'{val}\n({pct:.1f}%)', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Numri i mostrave', fontsize=12)
ax.set_xlabel('Klasa', fontsize=12)
ax.set_title('Shpërndarja e klasave healthy dhe T2D', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(counts.values)+30)
plt.tight_layout()
plt.savefig(OUT+'fig1_shperndarja_klasave.png', bbox_inches='tight')
plt.close()

############################################################
# FIGURA 4.2 - Histogram i abundancave
############################################################
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
nonzero = abundance.values[abundance.values > 0]
axes[0].hist(nonzero, bins=60, color='#2E86AB', edgecolor='black', linewidth=0.3)
axes[0].set_xlabel('Vlera e abundancës (%)', fontsize=11)
axes[0].set_ylabel('Frekuenca (shkallë log)', fontsize=11)
axes[0].set_title('Histogram i vlerave jo-zero', fontsize=12, fontweight='bold')
axes[0].set_yscale('log')
axes[1].hist(np.log10(nonzero+1e-6), bins=60, color='#A23B72', edgecolor='black', linewidth=0.3)
axes[1].set_xlabel('log10(abundancë)', fontsize=11)
axes[1].set_ylabel('Frekuenca', fontsize=11)
axes[1].set_title('Shpërndarja në shkallë logaritmike', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT+'fig2_histogram_abundances.png', bbox_inches='tight')
plt.close()

############################################################
# FIGURA 4.3 - Sparsity
############################################################
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
prevalence = (abundance > 0).sum(axis=0) / len(abundance) * 100
axes[0].hist(prevalence, bins=50, color='#06A77D', edgecolor='black', linewidth=0.3)
axes[0].set_xlabel('Prevalenca e veçorisë (%)', fontsize=11)
axes[0].set_ylabel('Numri i veçorive', fontsize=11)
axes[0].set_title('Shpërndarja e prevalencës së veçorive', fontsize=12, fontweight='bold')
total = abundance.shape[0]*abundance.shape[1]
zeros = int((abundance==0).sum().sum())
nonzeros = total - zeros
axes[1].pie([zeros, nonzeros],
            labels=[f'Zero\n{zeros/total*100:.1f}%', f'Jo-zero\n{nonzeros/total*100:.1f}%'],
            colors=['#D5D5D5','#06A77D'], startangle=90,
            wedgeprops={'edgecolor':'black','linewidth':1})
axes[1].set_title('Sparsity i matricës së abundancës', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT+'fig3_sparsity.png', bbox_inches='tight')
plt.close()

############################################################
# FIGURA 4.4 - Heatmap i korrelacionit (30 speciet më të abundanta)
############################################################
top30 = abundance.mean(axis=0).sort_values(ascending=False).head(30).index
sub = abundance[top30]
corr_sub, _ = spearmanr(sub.values)
names = [c.split('s__')[-1].replace('_',' ')[:24] for c in top30]
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_sub, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=names, yticklabels=names,
            square=True, cbar_kws={'label':'Korrelacioni Spearman','shrink':0.8}, ax=ax,
            linewidths=0.3, linecolor='white')
ax.set_title('Heatmap i korrelacionit Spearman\n(30 speciet më të abundanta)', fontsize=12, fontweight='bold')
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(OUT+'fig4_heatmap_korrelacioni.png', bbox_inches='tight')
plt.close()

############################################################
# Ndërtimi i rrjetave për Figurat 4.5 dhe 4.6
############################################################
meta_target = metadata[['disease']].copy()
df = abundance.join(meta_target, how='inner')
X = df.drop(columns=['disease']); y = df['disease']
le = LabelEncoder(); y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
X_train_norm = X_train/100.0
train_df = X_train_norm.copy(); train_df['disease'] = y_train
X_healthy = train_df[train_df['disease']==1].drop(columns=['disease'])
X_t2d     = train_df[train_df['disease']==0].drop(columns=['disease'])

THRESHOLD = 0.3
def build_network(Xg, t):
    corr,_ = spearmanr(Xg.values); n=Xg.shape[1]
    G=nx.Graph(); G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1,n):
            if abs(corr[i,j])>=t: G.add_edge(i,j)
    return G

G_h = build_network(X_healthy, THRESHOLD)
G_t = build_network(X_t2d, THRESHOLD)
G_h_vis = G_h.subgraph([n for n in G_h.nodes() if G_h.degree(n)>0]).copy()
G_t_vis = G_t.subgraph([n for n in G_t.nodes() if G_t.degree(n)>0]).copy()

############################################################
# FIGURA 4.5 - Vizualizim krahasues i rrjetave
############################################################
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
pos_h = nx.spring_layout(G_h_vis, k=0.15, iterations=30, seed=42)
deg_h = dict(G_h_vis.degree())
nx.draw_networkx_nodes(G_h_vis, pos_h, node_size=[deg_h[n]*3 for n in G_h_vis.nodes()],
                       node_color='#2E86AB', alpha=0.7, ax=axes[0])
nx.draw_networkx_edges(G_h_vis, pos_h, alpha=0.1, width=0.3, ax=axes[0])
axes[0].set_title(f'Rrjeti HEALTHY\n{G_h_vis.number_of_nodes()} nyje aktive, {G_h_vis.number_of_edges()} lidhje', fontsize=12, fontweight='bold')
axes[0].axis('off')
pos_t = nx.spring_layout(G_t_vis, k=0.15, iterations=30, seed=42)
deg_t = dict(G_t_vis.degree())
nx.draw_networkx_nodes(G_t_vis, pos_t, node_size=[deg_t[n]*3 for n in G_t_vis.nodes()],
                       node_color='#E63946', alpha=0.7, ax=axes[1])
nx.draw_networkx_edges(G_t_vis, pos_t, alpha=0.1, width=0.3, ax=axes[1])
axes[1].set_title(f'Rrjeti T2D\n{G_t_vis.number_of_nodes()} nyje aktive, {G_t_vis.number_of_edges()} lidhje', fontsize=12, fontweight='bold')
axes[1].axis('off')
plt.tight_layout()
plt.savefig(OUT+'fig5_rrjetat_krahasim.png', bbox_inches='tight')
plt.close()

############################################################
# FIGURA 4.6 - Shpërndarja e degree
############################################################
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
deg_list_h = [d for _,d in G_h.degree() if d>0]
deg_list_t = [d for _,d in G_t.degree() if d>0]
axes[0].hist(deg_list_h, bins=40, color='#2E86AB', alpha=0.7, label='Healthy', edgecolor='black', linewidth=0.3)
axes[0].hist(deg_list_t, bins=40, color='#E63946', alpha=0.6, label='T2D', edgecolor='black', linewidth=0.3)
axes[0].set_xlabel('Degree (shkalla e nyjes)', fontsize=11)
axes[0].set_ylabel('Numri i nyjeve', fontsize=11)
axes[0].set_title('Shpërndarja e degree në të dy rrjetat', fontsize=12, fontweight='bold')
axes[0].legend()
metrics = ['Avg Degree', 'Density x100', 'Clustering']
h_vals = [np.mean(deg_list_h), nx.density(G_h)*100, nx.average_clustering(G_h)]
t_vals = [np.mean(deg_list_t), nx.density(G_t)*100, nx.average_clustering(G_t)]
xpos = np.arange(len(metrics)); width=0.35
axes[1].bar(xpos-width/2, h_vals, width, label='Healthy', color='#2E86AB', edgecolor='black', linewidth=0.5)
axes[1].bar(xpos+width/2, t_vals, width, label='T2D', color='#E63946', edgecolor='black', linewidth=0.5)
axes[1].set_xticks(xpos); axes[1].set_xticklabels(metrics, fontsize=10)
axes[1].set_ylabel('Vlera', fontsize=11)
axes[1].set_title('Krahasim i metrikave topologjike', fontsize=12, fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig(OUT+'fig6_degree_distribution.png', bbox_inches='tight')
plt.close()

############################################################
# TË DHËNAT PËR TABELAT (printohen në ekran)
############################################################
print("="*55)
print("TABELA 4.2 - KONTROLLI PARAPRAK")
print("="*55)
print(f"Mostra ne abundance: {abundance.shape[0]}")
print(f"Mostra ne metadata: {metadata.shape[0]}")
print(f"Perputhja e ID-ve: {'po' if all(abundance.index==metadata.index) else 'jo'}")
print(f"Missing ne disease: {metadata['disease'].isna().sum()}")
print(f"NaN ne abundance: {abundance.isna().sum().sum()}")
print(f"Vlera negative: {(abundance<0).sum().sum()}")
print(f"Niveli taksonomik: species level")
print(f"Shkalla e abundances: 0-{abundance.max().max():.2f}")

print("\n" + "="*55)
print("TABELA 4.3 - METADATA ME VLERA MUNGESË")
print("="*55)
missing = metadata.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, val in missing.items():
    print(f"{col}: {val} ({val/len(metadata)*100:.1f}%)")

print("\nTë gjitha figurat u ruajtën në folderin 'results/figures/'")
