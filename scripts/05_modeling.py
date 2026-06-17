############################################################
# SKRIPTI 3: MODELIMI DHE VLERËSIMI
# Hapi 8: tre skenaret (S1, S2, S3) x katër algoritme
# Vlërsim i brendshëm: 5-fold CV i stratifikuar
#   (rrjeti dhe veçoritë për-sample rillogariten brenda CDO fold,
#    vetëm mbi pjesën e trajnimit të fold-it -> pa data leakage)
# Vlerësim final: mbi test set
# SVM: grid search per C dhe gamma
############################################################

import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             balanced_accuracy_score)
from xgboost import XGBClassifier
import warnings

import os
os.makedirs("results/tables", exist_ok=True)
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
THRESHOLD = 0.3
N_SPLITS = 5

# ---------- Leximi ----------
X_train = pd.read_csv("data/dt_X_train.csv", index_col=0)
X_test  = pd.read_csv("data/dt_X_test.csv",  index_col=0)
y_train = pd.read_csv("data/dt_y_train.csv", index_col=0)['disease'].values
y_test  = pd.read_csv("data/dt_y_test.csv",  index_col=0)['disease'].values
feat_train = pd.read_csv("data/dt_feat_train.csv", index_col=0)
feat_test  = pd.read_csv("data/dt_feat_test.csv",  index_col=0)

n_features = X_train.shape[1]

print("="*60)
print("SKRIPTI 3: MODELIMI DHE VLERESIMI")
print("="*60)

# ============================================================
# Funksionet për nxjerrjen e veçorive për-sample (njësoj si Skripti 2)
# ============================================================
def build_network(X_group, threshold):
    corr, _ = spearmanr(X_group.values)
    corr = np.nan_to_num(corr, nan=0.0)
    n = X_group.shape[1]
    G = nx.Graph(); G.add_nodes_from(range(n))
    iu = np.triu_indices(n, k=1)
    mask = np.abs(corr[iu]) >= threshold
    edges = [(int(iu[0][k]), int(iu[1][k])) for k in range(len(mask)) if mask[k]]
    G.add_edges_from(edges)
    return G

def node_metrics(G, n):
    deg   = np.array([G.degree(i) for i in range(n)], dtype=float)
    close = np.array(list(nx.closeness_centrality(G).values()))
    clust = np.array(list(nx.clustering(G).values()))
    betw  = np.array(list(nx.betweenness_centrality(G).values()))
    dc    = np.array(list(nx.degree_centrality(G).values()))
    comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    module = np.zeros(n, dtype=int)
    for idx, c in enumerate(comms):
        for node in c: module[node] = idx
    L = nx.laplacian_matrix(G).toarray().astype(float)
    _, eigvecs = np.linalg.eigh(L)
    eig1 = eigvecs[:,1] if n>1 else np.zeros(n)
    eig2 = eigvecs[:,2] if n>2 else np.zeros(n)
    eig3 = eigvecs[:,3] if n>3 else np.zeros(n)
    return {'deg':deg,'close':close,'clust':clust,'betw':betw,'dc':dc,
            'module':module,'n_modules':len(comms),'eig1':eig1,'eig2':eig2,'eig3':eig3}

def per_sample_features(X_samples, m):
    rows = []
    vals = X_samples.values
    for i in range(vals.shape[0]):
        ab = vals[i]; total = np.sum(ab)+1e-10
        mod_all = np.array([np.sum(ab[m['module']==k]) for k in range(m['n_modules'])])
        top3 = np.argsort(mod_all)[::-1][:3]
        mod1 = mod_all[top3[0]] if len(top3)>0 else 0.0
        mod2 = mod_all[top3[1]] if len(top3)>1 else 0.0
        mod3 = mod_all[top3[2]] if len(top3)>2 else 0.0
        rows.append([
            np.sum(m['deg']*ab)/total, np.sum(m['close']*ab)/total,
            np.sum(m['clust']*ab)/total, np.sum(m['betw']*ab)/total,
            np.sum(m['dc']*ab)/total, mod1, mod2, mod3,
            np.sum(m['eig1']*ab)/total, np.sum(m['eig2']*ab)/total, np.sum(m['eig3']*ab)/total
        ])
    return np.array(rows)

# ============================================================
# MODELET
# ============================================================
def get_models():
    return {
        'Logistic Regression': LogisticRegression(penalty='l1', solver='liblinear',
                                                   C=1.0, random_state=RANDOM_STATE, max_iter=1000),
        'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                 eval_metric='logloss', verbosity=0),
    }

SCALE_MODELS = {'Logistic Regression', 'SVM'}  # këto kërkojne standardizim

# ============================================================
# 5-fold CV ku rrjeti+veçoritë rillogariten brenda çdo fold
# ============================================================
def cv_method_B(scenario, X_tr_abund, y_tr, feat_tr_precomputed):
    """
    scenario: 'S1', 'S2', 'S3'
    Për S2/S3 rrjeti dhe veçoritë për-sample rillogariten brenda çdo fold
    vetëm mbi pjesën e trajnimit të fold-it (pa data leakage).
    Për S1 përdoret vetëm abundanca.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model_names = list(get_models().keys())
    fold_f1 = {mn: [] for mn in model_names}

    Xab = X_tr_abund.values
    for tr_idx, va_idx in skf.split(Xab, y_tr):
        Xab_tr, Xab_va = X_tr_abund.iloc[tr_idx], X_tr_abund.iloc[va_idx]
        y_f_tr, y_f_va = y_tr[tr_idx], y_tr[va_idx]

        # Ndërtimi i veçorive sipas skenarit, vetëm mbi fold-train
        if scenario == 'S1':
            Xtr_feat = Xab_tr.values
            Xva_feat = Xab_va.values
        else:
            # rrjeti dhe metrikat rillogariten VETËM mbi fold-train
            G = build_network(Xab_tr, THRESHOLD)
            m = node_metrics(G, n_features)
            net_tr = per_sample_features(Xab_tr, m)
            net_va = per_sample_features(Xab_va, m)
            if scenario == 'S2':
                Xtr_feat, Xva_feat = net_tr, net_va
            else:  # S3
                Xtr_feat = np.hstack([Xab_tr.values, net_tr])
                Xva_feat = np.hstack([Xab_va.values, net_va])

        # Trajnimi dhe vlerësimi i secilit model brenda fold-it.
        # Modelet që kërkojnë standardizim vendosen brenda një Pipeline së bashku
        # me StandardScaler, kështu standardizimi kryhet veçmas brenda çdo fold-i
        # (mbi pjesën e trajnimit të fold-it), pa rrjedhje informacioni.
        for mn, model in get_models().items():
            if mn in SCALE_MODELS:
                est = Pipeline([('scaler', StandardScaler()), ('clf', model)])
            else:
                est = model
            est.fit(Xtr_feat, y_f_tr)
            yp = est.predict(Xva_feat)
            fold_f1[mn].append(f1_score(y_f_va, yp, zero_division=0))

    return {mn: (np.mean(v), np.std(v)) for mn, v in fold_f1.items()}

# ============================================================
# Vlerësimi final mbi test set (veçoritë janë llogaritur në Skriptin 2
# mbi gjithë train-in -> pa data leakage)
# ============================================================
def final_eval(scenario):
    if scenario == 'S1':
        Xtr, Xte = X_train.values, X_test.values
    elif scenario == 'S2':
        Xtr, Xte = feat_train.values, feat_test.values
    else:
        Xtr = np.hstack([X_train.values, feat_train.values])
        Xte = np.hstack([X_test.values,  feat_test.values])

    out = []
    for mn, model in get_models().items():
        # SVM: StandardScaler brenda Pipeline -> scaler-i fit vetëm mbi pjesën e
        # trajnimit të çdo fold-i të GridSearchCV (pa data leakage gjatë grid search)
        if mn == 'SVM':
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
            ])
            grid = GridSearchCV(
                pipe,
                {'svc__C':[0.1,1,10], 'svc__gamma':['scale','auto',0.01,0.001]},
                cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
                scoring='f1', n_jobs=-1)
            grid.fit(Xtr, y_train)        # Xtr i paskaluar; Pipeline standardizon brenda CV
            best = grid.best_estimator_
            best_params = grid.best_params_
            yp = best.predict(Xte)
            yb = best.predict_proba(Xte)[:,1]
        else:
            # LR (dhe çdo model tjetër që kërkon standardizim) vendoset brenda një
            # Pipeline me StandardScaler; scaler-i fit vetëm mbi train, pastaj test-i
            # transformohet me të njëjtin scaler (pa rrjedhje informacioni).
            if mn in SCALE_MODELS:
                est = Pipeline([('scaler', StandardScaler()), ('clf', model)])
            else:
                est = model
            best = est.fit(Xtr, y_train)
            best_params = None
            yp = best.predict(Xte)
            yb = best.predict_proba(Xte)[:,1]

        tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
        # Kodimi: healthy = klasa pozitive (1), T2D = klasa negative (0).
        # Raportohen metrikat për të dyja klasat, plus Macro F1 dhe Balanced Accuracy.
        out.append({
            'Skenari': scenario, 'Modeli': mn,
            'Accuracy': round(accuracy_score(y_test,yp),4),
            'Balanced_Accuracy': round(balanced_accuracy_score(y_test,yp),4),
            'Macro_F1': round(f1_score(y_test,yp,average='macro',zero_division=0),4),
            'Precision_T2D': round(precision_score(y_test,yp,pos_label=0,zero_division=0),4),
            'Recall_T2D': round(recall_score(y_test,yp,pos_label=0,zero_division=0),4),
            'F1_T2D': round(f1_score(y_test,yp,pos_label=0,zero_division=0),4),
            'Precision_healthy': round(precision_score(y_test,yp,pos_label=1,zero_division=0),4),
            'Recall_healthy': round(recall_score(y_test,yp,pos_label=1,zero_division=0),4),
            'F1_healthy': round(f1_score(y_test,yp,pos_label=1,zero_division=0),4),
            'ROC_AUC': round(roc_auc_score(y_test,yb),4),
            'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn),
            'SVM_params': str(best_params) if best_params else ''
        })
    return out

# ============================================================
# EKZEKUTIMI
# ============================================================
scenarios = {'S1':'Abundancë', 'S2':'Rrjet (per-sample)', 'S3':'Kombinim'}

print("\n>>> 5-FOLD CV (rillogaritje brenda çdo fold) <<<")
cv_results = {}
for sc in scenarios:
    print(f"\n--- {sc} ({scenarios[sc]}) ---")
    res = cv_method_B(sc, X_train, y_train, feat_train)
    cv_results[sc] = res
    for mn,(mean,std) in res.items():
        print(f"  {mn:22s} CV F1 = {mean:.3f} (+- {std:.3f})")

print("\n>>> VLERËSIMI FINAL MBI TEST SET <<<")
all_final = []
for sc in scenarios:
    print(f"\n--- {sc} ({scenarios[sc]}) ---")
    res = final_eval(sc)
    all_final.extend(res)
    for r in res:
        extra = f" [{r['SVM_params']}]" if r['SVM_params'] else ""
        print(f"  {r['Modeli']:22s} Acc={r['Accuracy']:.3f} BalAcc={r['Balanced_Accuracy']:.3f} "
              f"MacroF1={r['Macro_F1']:.3f} F1_T2D={r['F1_T2D']:.3f} F1_H={r['F1_healthy']:.3f} "
              f"AUC={r['ROC_AUC']:.3f}{extra}")
        print(f"  {'':22s} CM: TP={r['TP']} TN={r['TN']} FP={r['FP']} FN={r['FN']}")

# ---------- Ruajtja e rezultateve ----------
df_final = pd.DataFrame(all_final)
# shtojmë kolonat e CV F1
df_final['CV_F1_mean'] = df_final.apply(lambda r: round(cv_results[r['Skenari']][r['Modeli']][0],4), axis=1)
df_final['CV_F1_std']  = df_final.apply(lambda r: round(cv_results[r['Skenari']][r['Modeli']][1],4), axis=1)
cols = ['Skenari','Modeli','CV_F1_mean','CV_F1_std','Accuracy','Balanced_Accuracy','Macro_F1',
        'Precision_T2D','Recall_T2D','F1_T2D','Precision_healthy','Recall_healthy','F1_healthy',
        'ROC_AUC','TP','TN','FP','FN','SVM_params']
df_final = df_final[cols]
df_final.to_csv("results/tables/rezultatet_finale.csv", index=False)

print("\n" + "="*60)
print("TABELA PERMBLEDHESE")
print("="*60)
print(df_final.drop(columns=['SVM_params']).to_string(index=False))
print("\nU ruajt: rezultatet_finale.csv")
print("Skripti 3 përfundoi me sukses.")
