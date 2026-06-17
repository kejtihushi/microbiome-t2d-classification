############################################################
# SKRIPTI 1: PERGATITJA E TE DHENAVE
# Hapat 1-5: lexim, kontroll paraprak, ndarje train/test,
#            normalizim (pjesetim me 100)
# Dataset: QinJ_2012
############################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

import os
os.makedirs("data", exist_ok=True)
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------- Hapi 1-2: Leximi i te dhenave ----------
abundance = pd.read_csv("data/qin2012_abundance.csv", index_col=0)
metadata  = pd.read_csv("data/qin2012_metadata.csv",  index_col=0)

print("="*60)
print("SKRIPTI 1: PERGATITJA E TE DHENAVE")
print("="*60)

# ---------- Hapi 3: Kontrolli paraprak ----------
print("\n--- Kontrolli paraprak ---")
print(f"Mostra ne abundance: {abundance.shape[0]}")
print(f"Vecori mikrobike (species): {abundance.shape[1]}")
print(f"Mostra ne metadata: {metadata.shape[0]}")
print(f"Perputhja e ID-ve: {'po' if all(abundance.index==metadata.index) else 'jo'}")
print(f"Missing ne disease: {metadata['disease'].isna().sum()}")
print(f"NaN ne abundance: {abundance.isna().sum().sum()}")
print(f"Vlera negative: {(abundance<0).sum().sum()}")
print(f"Shkalla e abundances: 0 - {abundance.max().max():.2f}")
print(f"Shperndarja e klasave:\n{metadata['disease'].value_counts().to_string()}")

# ---------- Pergatitja e X dhe y ----------
df = abundance.join(metadata[['disease']], how='inner')
X = df.drop(columns=['disease'])
y = df['disease']

# Etiketimi: healthy=1, T2D=0 (LabelEncoder: T2D=0, healthy=1 sipas rendit alfabetik)
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nEtiketat: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ---------- Hapi 4: Ndarja train/test (stratifikuar, para cdo transformimi) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)
print(f"\n--- Ndarja train/test (80/20 stratifikuar) ---")
print(f"Train: {X_train.shape[0]} mostra")
print(f"Test : {X_test.shape[0]} mostra")
print(f"Train healthy/T2D: {np.sum(y_train==1)}/{np.sum(y_train==0)}")
print(f"Test  healthy/T2D: {np.sum(y_test==1)}/{np.sum(y_test==0)}")

# ---------- Hapi 5: Normalizimi (pjesetim me 100 -> proporcione 0-1) ----------
# Operacion fiks, nuk meson parametra nga te dhenat, pa data leakage
X_train_norm = X_train / 100.0
X_test_norm  = X_test  / 100.0
print(f"\n--- Normalizimi (pjesetim me 100) ---")
print(f"Vlera max para: {X_train.max().max():.2f}, pas: {X_train_norm.max().max():.4f}")

# ---------- Ruajtja ----------
X_train_norm.to_csv("data/dt_X_train.csv")
X_test_norm.to_csv("data/dt_X_test.csv")
pd.Series(y_train, index=X_train.index, name='disease').to_csv("data/dt_y_train.csv")
pd.Series(y_test, index=X_test.index, name='disease').to_csv("data/dt_y_test.csv")

print("\nU ruajten: dt_X_train.csv, dt_X_test.csv, dt_y_train.csv, dt_y_test.csv")
print("Skripti 1 perfundoi me sukses.")
