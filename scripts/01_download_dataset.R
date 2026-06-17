############################################################
# Shkarkimi dhe përgatitja e dataset-it QinJ_2012
# Dataset: Gut microbiome - Type 2 Diabetes
# Burimi: curatedMetagenomicData (Bioconductor)
# R version: 4.4.3
############################################################

# ----------------------------------------------------------
# 1. Instalimi i paketave të nevojshme
# ----------------------------------------------------------

# Instalimi i BiocManager (nëse mungon)
install.packages("BiocManager")

# Instalimi i paketave Bioconductor
BiocManager::install(c(
  "curatedMetagenomicData",
  "SummarizedExperiment",
  "Biostrings",
  "DECIPHER",
  "TreeSummarizedExperiment",
  "mia"
))

# ----------------------------------------------------------
# 2. Ngarkimi i librarive
# ----------------------------------------------------------

library(curatedMetagenomicData)
library(SummarizedExperiment)

# ----------------------------------------------------------
# 3. Shkarkimi i dataset-it QinJ_2012
# ----------------------------------------------------------

qin_data <- curatedMetagenomicData(
  "QinJ_2012.relative_abundance",
  dryrun = FALSE
)

# ----------------------------------------------------------
# 4. Nxjerrja e objektit kryesor
# ----------------------------------------------------------

tse <- qin_data[[1]]

# ----------------------------------------------------------
# 5. Krijimi i matricës së abundancave
# ----------------------------------------------------------

# Transpozim:
# rreshtat = mostra
# kolonat = specie/gjini bakteriale

abundance_matrix <- as.data.frame(
  t(assay(tse))
)

# ----------------------------------------------------------
# 6. Nxjerrja e metadata
# ----------------------------------------------------------

metadata <- as.data.frame(
  colData(tse)
)

# ----------------------------------------------------------
# 7. Eksportimi në CSV
# ----------------------------------------------------------
dir.create("data", showWarnings = FALSE)
# ----------------------------------------------------------

write.csv(
  abundance_matrix,
  "data/qin2012_abundance.csv"
)

write.csv(
  metadata,
  "data/qin2012_metadata.csv"
)

# ----------------------------------------------------------
# 8. Informacion për dataset-in
# ----------------------------------------------------------

cat("Dataset-i u eksportua me sukses!\n")

cat("Numri i mostrave:",
    nrow(abundance_matrix), "\n")

cat("Numri i veçorive mikrobike:",
    ncol(abundance_matrix), "\n")

cat("Numri i kolonave metadata:",
    ncol(metadata), "\n")