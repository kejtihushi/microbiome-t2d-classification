############################################################
# Gjenerimi i figurës dhe tabelës për pragun e korrelacionit
# (Kapitulli IV, mbështetja empirike e pragut 0.3)
# Dataset: QinJ_2012
#
# Prodhon:
#   - figura/fig_shperndarja_korrelacioneve.png
#   - dt_percentilet_korrelacioni.csv
#
# Logjika është e njejtë me Skriptin 2: korrelacioni Spearman
# llogaritet vetëm mbi train set (dt_X_train.csv).
############################################################

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings

import os
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
warnings.filterwarnings('ignore')

# ---------- Konfigurimi i stilit (i njejtë me figurat e tjera) ----------
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'

OUT = "results/figures/"
os.makedirs(OUT, exist_ok=True)

THRESHOLD = 0.3  # prag fiks i korrelacionit Spearman

# ---------- Leximi i të dhënave të trajnimit ----------
X_train = pd.read_csv("data/dt_X_train.csv", index_col=0)
print(f"Train shape: {X_train.shape}")

# ---------- Llogaritja e korrelacioneve absolute ----------
corr, _ = spearmanr(X_train.values)
corr = np.nan_to_num(corr, nan=0.0)  # taksonet konstante -> 0
n = corr.shape[0]
iu = np.triu_indices(n, k=1)          # vetëm gjysma e sipërme, pa diagonalen
absvals = np.abs(corr[iu])
total = absvals.size
print(f"Çifte korrelacioni total: {total}")

# ---------- Tabela e percentileve ----------
percentilet = [90, 95, 97.5, 99, 99.5]
rreshtat = []
for p in percentilet:
    rreshtat.append({'Percentili': p, 'Vlera_absolute': round(np.percentile(absvals, p), 3)})

# pozicioni i saktë i pragut 0.3 në shpërndarje
pct_i_pragut = (absvals <= THRESHOLD).mean() * 100
rreshtat.append({'Percentili': round(pct_i_pragut, 1),
                 'Vlera_absolute': THRESHOLD})

tabela = pd.DataFrame(rreshtat).sort_values('Percentili').reset_index(drop=True)
tabela.to_csv("results/tables/dt_percentilet_korrelacioni.csv", index=False)
print("\nTabela e percentileve:")
print(tabela.to_string(index=False))

kept = int((absvals >= THRESHOLD).sum())
print(f"\nPragu 0.3 -> percentili {pct_i_pragut:.2f}")
print(f"Lidhje të mbajtura: {kept} nga {total} ({kept/total*100:.2f}%)")

# ---------- Figura: histogram me vijën e pragut ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(absvals, bins=60, color='#2E86AB', edgecolor='black', linewidth=0.3)
ax.set_yscale('log')  # shkallë log, sepse shumica e vlerave janë afer zeros
ax.axvline(THRESHOLD, color='#E63946', linestyle='--', linewidth=2,
           label=f'Pragu = 0.3 (percentili {pct_i_pragut:.1f})')
ax.axvspan(THRESHOLD, absvals.max(), color='#E63946', alpha=0.08)
ax.set_xlabel('Vlera absolute e korrelacionit Spearman', fontsize=12)
ax.set_ylabel('Frekuenca (shkallë log)', fontsize=12)
ax.set_title('Shpërndarja e korrelacioneve absolute dhe vendosja e pragut 0.3',
             fontsize=13, fontweight='bold')
ax.text(0.5, 8e2,
        f'Lidhje të mbajtura: {kept} nga {total}\n({kept/total*100:.2f}% më të forta)',
        fontsize=10.5, color='#E63946', fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plt.savefig(OUT + 'fig_shperndarja_korrelacioneve.png', bbox_inches='tight')
plt.close()
print("\nU ruajt: " + OUT + "fig_shperndarja_korrelacioneve.png")
print("U ruajt: dt_percentilet_korrelacioni.csv")
