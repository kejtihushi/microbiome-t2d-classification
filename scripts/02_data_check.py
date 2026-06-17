import pandas as pd

abundance = pd.read_csv("data/qin2012_abundance.csv", index_col=0)
metadata  = pd.read_csv("data/qin2012_metadata.csv",  index_col=0)

print("=== 1. NIVELI TAKSONOMIK ===")
cols = abundance.columns.tolist()
has_species = any('s__' in c for c in cols)
has_genus = any('g__' in c for c in cols)
print("Përmban nivel specie (s__):", has_species)
print("ërmban nivel gjinie (g__):", has_genus)
print("Shembull kolone:", cols[0])

print("\n=== 2. VLERAT E ABUNDANCES ===")
print("Vlera maksimale:", abundance.max().max())
print("Vlera minimale:", abundance.min().min())
print("Mesatarja:", abundance.mean().mean().round(4))

print("\n=== 3. VLERA NEGATIVE ===")
print("Vlera negative:", (abundance < 0).sum().sum())

print("\n=== 4. VLERA MUNGESE ===")
print("Missing në disease:", metadata['disease'].isna().sum())
print("NaN në abundancë:", abundance.isna().sum().sum())

print("\n=== 5. ID-TE PERPUTHEN ===")
print("Mostra abundancë:", abundance.shape[0])
print("Mostra metadata:", metadata.shape[0])
print("Të gjitha ID përputhen:", all(abundance.index == metadata.index))

print("\n=== 6. VARIABLI TARGET ===")
print(metadata['disease'].value_counts())
print("Vlera mungese në disease:", metadata['disease'].isna().sum())

print("\n=== 7. VEÇORITE MIKROBIKE vs METADATA ===")
print("Veçori mikrobike (species):", abundance.shape[1])
print("Variabla metadata:", metadata.shape[1])

print("\n=== 8. MOSTRA PA INFORMACION TË PLOTË ===")
print("Missing për çdo kolonë metadata:")
print(metadata.isnull().sum()[metadata.isnull().sum() > 0])


# ============================================
# REZULTATET E KONTROLLIT PARAPRAK
# Dataset: QinJ_2012, curatedMetagenomicData 3.14.0
# ============================================

# 1. NIVELI TAKSONOMIK: species level (s__)
#    Shembull: g__Dialister|s__Dialister_sp_CAG_357

# 2. VLERAT E ABUNDANCËS: përqindje relative (0-100)
#    Vlera max: 87.77 -> konfirmon shkallë 0-100
#    Vlera min: 0.0
#    Mesatarja: 0.1535

# 3. VLERA NEGATIVE: 0

# 4. VLERA MUNGESE:
#    Missing në disease: 0
#    NaN në abundancë: 0

# 5. ID-TË E MOSTRAVE: perputhen plotesisht
#    363 mostra në abundance = 363 mostra në metadata

# 6. VARIABLI TARGET: kolona "disease"
#    healthy: 193 mostra
#    T2D: 170 mostra
#    Vlera mungese: 0

# 7. VEÇORITE:
#    Veçori mikrobike: 651
#    Variabla metadata: 24

# 8. MOSTRA PA INFORMACION TË PLOTË
#    Kolona target "disease": 0 missing -> e sigurt për klasifikim
#    Kolonat me missing të larta (treatment 218, c_peptide 318)
#    nuk perdoren si target, vetëm për përshkrim nëse nevojitet
