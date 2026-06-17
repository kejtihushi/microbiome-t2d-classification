############################################################
# SKRIPTI 4: GRAFIKËT E REZULTATEVE
# - ROC curves për tre skenarët
# - Confusion matrices për modelet më të mira
# - Feature importance (Random Forest)
# - Krahasim i metrikave midis skenarëve
############################################################

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'
RANDOM_STATE = 42

import os
OUT = "results/figures/"
os.makedirs(OUT, exist_ok=True)

# ---------- Leximi ----------
X_train = pd.read_csv("data/dt_X_train.csv", index_col=0)
X_test  = pd.read_csv("data/dt_X_test.csv",  index_col=0)
y_train = pd.read_csv("data/dt_y_train.csv", index_col=0)['disease'].values
y_test  = pd.read_csv("data/dt_y_test.csv",  index_col=0)['disease'].values
feat_train = pd.read_csv("data/dt_feat_train.csv", index_col=0)
feat_test  = pd.read_csv("data/dt_feat_test.csv",  index_col=0)
results = pd.read_csv("results/tables/rezultatet_finale.csv")

print("="*60)
print("SKRIPTI 4: GRAFIKT E REZULTATEVE")
print("="*60)

# ---------- Përgatitja e tre skenarëve ----------
scenarios = {
    'S1': (X_train.values, X_test.values),
    'S2': (feat_train.values, feat_test.values),
    'S3': (np.hstack([X_train.values, feat_train.values]),
           np.hstack([X_test.values, feat_test.values])),
}
scenario_labels = {'S1':'Abundancë', 'S2':'Rrjet (per-sample)', 'S3':'Kombinim'}

# Parametrat me te mire te SVM-se per cdo skenar, ashtu sic i gjeti
# grid search-i te 05_modeling.py (kolona SVM_params e rezultateve).
# Keshtu figurat dalin identike me tabelen.
SVM_BEST = {
    'S1': {'C': 0.1, 'gamma': 'scale'},
    'S2': {'C': 1,   'gamma': 0.001},
    'S3': {'C': 0.1, 'gamma': 'scale'},
}

def make_models(sc):
    return {
        'Logistic Regression': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=RANDOM_STATE, max_iter=1000),
        'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, **SVM_BEST[sc]),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
    }
SCALE = {'Logistic Regression','SVM'}

# ============================================================
# FIGURA 1: ROC curves - një panel për skenar, të 4 modelet
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
colors = {'Logistic Regression':'#2E86AB','SVM':'#A23B72','Random Forest':'#06A77D','XGBoost':'#E63946'}

for ax, (sc,(Xtr,Xte)) in zip(axes, scenarios.items()):
    for mn, model in make_models(sc).items():
        if mn in SCALE:
            est = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        else:
            est = model
        est.fit(Xtr, y_train)
        yb = est.predict_proba(Xte)[:,1]
        fpr, tpr, _ = roc_curve(y_test, yb)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[mn], lw=2, label=f'{mn} (AUC={roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{sc} - {scenario_labels[sc]}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.suptitle('Kurbat ROC sipas skenarëve dhe modeleve', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT+'fig_roc_curves.png', bbox_inches='tight')
plt.close()
print("Figura ROC u ruajt")

# ============================================================
# FIGURA 2: Confusion matrices - modeli më i mirë për çdo skenar
# ============================================================
best_per_sc = {}
for sc in scenarios:
    sub = results[results['Skenari']==sc]
    best_row = sub.loc[sub['F1_healthy'].idxmax()]
    best_per_sc[sc] = best_row['Modeli']

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, sc in zip(axes, scenarios):
    Xtr, Xte = scenarios[sc]
    mn = best_per_sc[sc]
    model = make_models(sc)[mn]
    if mn in SCALE:
        est = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    else:
        est = model
    est.fit(Xtr, y_train)
    yp = est.predict(Xte)
    cm = confusion_matrix(y_test, yp)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['T2D','Healthy'], yticklabels=['T2D','Healthy'], ax=ax,
                annot_kws={'size':14,'weight':'bold'})
    ax.set_xlabel('Parashikuar', fontsize=11)
    ax.set_ylabel('Reale', fontsize=11)
    ax.set_title(f'{sc} - {mn}', fontsize=12, fontweight='bold')
plt.suptitle('Confusion Matrix për modelin më të mirë të çdo skenari', fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(OUT+'fig_confusion_matrices.png', bbox_inches='tight')
plt.close()
print("Figura Confusion Matrices u ruajt")

# ============================================================
# FIGURA 3: Feature importance (Random Forest, skenari S3)
# ============================================================
Xtr_s3 = np.hstack([X_train.values, feat_train.values])
feat_names = list(X_train.columns) + list(feat_train.columns)
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(Xtr_s3, y_train)
importances = rf.feature_importances_
idx_sorted = np.argsort(importances)[::-1][:20]  # top 20

fig, ax = plt.subplots(figsize=(10, 8))
top_names = []
for i in idx_sorted:
    nm = feat_names[i]
    if nm in feat_train.columns:
        top_names.append(nm + ' (rrjet)')
    else:
        short = nm.split('s__')[-1].replace('_',' ')[:35]
        top_names.append(short)
colors_fi = ['#E63946' if feat_names[i] in feat_train.columns else '#2E86AB' for i in idx_sorted]
ax.barh(range(len(idx_sorted)), importances[idx_sorted][::-1], color=colors_fi[::-1], edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(idx_sorted)))
ax.set_yticklabels(top_names[::-1], fontsize=9)
ax.set_xlabel('Rëndësia (Gini importance)', fontsize=11)
ax.set_title('Top 20 veçoritë më të rëndësishme (Random Forest, S3)', fontsize=12, fontweight='bold')
from matplotlib.patches import Patch

import os
os.makedirs("results/figures", exist_ok=True)
legend_el = [Patch(facecolor='#E63946', label='Veçori rrjeti'), Patch(facecolor='#2E86AB', label='Veçori abundance')]
ax.legend(handles=legend_el, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(OUT+'fig_feature_importance.png', bbox_inches='tight')
plt.close()
print("Figura Feature Importance u ruajt")

# Sa nga top 20 janë veçori rrjeti
n_net_top = sum(1 for i in idx_sorted if feat_names[i] in feat_train.columns)
print(f"  Nga top 20: {n_net_top} veçori rrjeti, {20-n_net_top} veçori abundance")

# ============================================================
# FIGURA 4: Krahasim i metrikave midis skenarëve (F1 dhe AUC)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
models_order = ['Logistic Regression','SVM','Random Forest','XGBoost']
sc_order = ['S1','S2','S3']
sc_colors = {'S1':'#2E86AB','S2':'#A23B72','S3':'#06A77D'}

for ax, metric in zip(axes, ['F1_healthy','ROC_AUC']):
    x = np.arange(len(models_order)); width=0.25
    for i, sc in enumerate(sc_order):
        vals = [results[(results['Skenari']==sc)&(results['Modeli']==m)][metric].values[0] for m in models_order]
        ax.bar(x+(i-1)*width, vals, width, label=f'{sc} ({scenario_labels[sc]})', color=sc_colors[sc], edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([m.replace(' ','\n') for m in models_order], fontsize=9)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'Krahasim i {metric} midis skenarëve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig(OUT+'fig_krahasim_metrikave.png', bbox_inches='tight')
plt.close()
print("Figura Krahasim Metrikave u ruajt")

print("\nTë gjithë grafikët u ruajtën në folderin:", OUT)
print("Skripti 4 përfundoi me sukses.")
