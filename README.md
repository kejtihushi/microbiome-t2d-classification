# Klasifikimi i gjendjeve shëndetësore nga të dhënat e mikrobiomës përmes analizës së rrjetave komplekse dhe metodave të machine learning

*Classification of health conditions from microbiome data through complex network analysis and machine learning methods*

## Përshkrimi
Ky repository përmban kodin e plotë të analizës së temës së diplomës, ku gjendjet shëndetësore (healthy kundrejt T2D) klasifikohen nga të dhënat e mikrobiomës së zorrës, duke integruar veçori topologjike dhe spektrale të rrjetave mikrobike me Machine Learning.

## Dataset-i
- Burimi: Qin et al. (2012), *Nature*, dataset-i QinJ_2012.
- Marrë nëpërmjet paketës curatedMetagenomicData (Bioconductor).
- 363 mostra: 193 healthy dhe 170 T2D, 651 veçori në nivel species.

## Marrja e të dhënave
Të dhënat nuk ngarkohen në repository. Ato gjenerohen duke ekzekutuar skriptin `scripts/01_download_dataset.R`, i cili i shkarkon nga curatedMetagenomicData dhe i eksporton në folderin `data/` si `qin2012_abundance.csv` dhe `qin2012_metadata.csv`. Detaje te `data/README_data.md`.

## Versionet kryesore të paketave
- Python 3: pandas, numpy, scipy, scikit-learn, networkx, xgboost, matplotlib, seaborn
- R: curatedMetagenomicData (Bioconductor)

Versionet jepen te `requirements.txt`.

## Rendi i ekzekutimit të skripteve
Skriptet ekzekutohen nga rrënja e repository-t, sipas këtij rendi:

1. `scripts/01_download_dataset.R` — merr dataset-in dhe e eksporton në `data/`.
2. `scripts/02_data_check.py` — kontrolli paraprak i integritetit të të dhënave.
3. `scripts/03_preprocessing.py` — ndarja train/test (80/20 e stratifikuar) dhe normalizimi.
4. `scripts/04_network_features.py` — ndërtimi i rrjetave me Spearman dhe nxjerrja e veçorive në nivel mostre.
5. `scripts/05_modeling.py` — trajnimi dhe vlerësimi i modeleve për tre skenarët.
6. `scripts/06_results_figures.py` — figurat e rezultateve.
7. `scripts/07_descriptive_figures.py` — figurat përshkruese të të dhënave.
8. `scripts/08_correlation_threshold.py` — figura e pragut të korrelacionit dhe tabela e percentileve.

## Ku gjenden tabelat dhe figurat
- Tabelat e rezultateve: `results/tables/` (`rezultatet_finale.csv`, `dt_krahasim_global.csv`, `dt_percentilet_korrelacioni.csv`).
- Figurat: `results/figures/` (figurat përshkruese fig1–fig6 dhe ato të rezultateve: ROC curves, confusion matrices, feature importance, krahasimi i metrikave).

## Riprodhimi i analizës
1. Instalo kërkesat: `pip install -r requirements.txt`
2. Ekzekuto skriptin R për të gjeneruar dataset-in në `data/`.
3. Ekzekuto skriptet Python sipas rendit të mësipërm (2 deri 8), nga rrënja e repository-t.
4. Tabelat krijohen në `results/tables/` dhe figurat në `results/figures/`. Folderat krijohen automatikisht nga skriptet.

## Struktura e repository-t
```
microbiome-t2d-classification/
├── README.md
├── requirements.txt
├── data/              (vetëm README_data.md; të dhënat gjenerohen)
├── scripts/           (skriptet R dhe Python)
├── results/
│   ├── tables/        (tabelat e rezultateve)
│   └── figures/       (figurat)
└── docs/              (analysis_summary.pdf)
```

## Autori
Kejti Hushi. Punim diplome.
