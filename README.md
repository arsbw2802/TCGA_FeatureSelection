# Assessing Algorithmic Stability in Feature Selection of Pan-Cancer Gene Expression Data Project
Group members: Aarushi Biswas, Krisha Shetty, Kristin Keith 

This project evaluates the trade-off between predictive performance and selection stability for three feature selection methods applied to a TCGA pan-cancer RNA-Seq dataset:
* Variance filtering (unsupervised)
* Mutual Information (supervised)
* L1-regularized Logistic Regression / LASSO (embedded)

We assess:
* Feature stability using Jaccard similarity
* Predictive performance using a Random Forest classifier
* A baseline RF model on the full dataset
* A shuffled-label control to validate against data leakage

## Our Dataset
* **Source**: [UCI Pan-Cancer Gene Expression RNA-Seq Dataset (TCGA HiSeq)](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq)
* **Samples**: 801 tumor samples
* **Genes**: ~20,000 HGNC-mapped genes
* **Classes**:
  * BRCA – Breast Invasive Carcinoma
  * KIRC – Kidney Renal Clear Cell Carcinoma
  * COAD – Colon Adenocarcinoma
  * LUAD – Lung Adenocarcinoma
  * PRAD – Prostate Adenocarcinoma

 ### Accessing the Cleaned Data
Since our dataset is too large to include on GitHub, we've uploaded `RNAseq_with_HGNC_symbols1.csv` and `labels.csv` to [this OneDrive](https://gtvault-my.sharepoint.com/:f:/g/personal/kkeith9_gatech_edu/IgDnuPFr2umcSLkOo4-8lJKUAeRHgjN2kEnNlPKLkqmm7dc?e=y782V1) for easy access and local download.  
### Input Files 
```bash
RNAseq_with_HGNC_symbols1.csv   # expression matrix (samples × genes)
labels.csv                     # tumor class labels
```
## Methods Overview

1. **Feature Selection**
   
Performed inside 5-fold stratified cross-validation:
| Method             | Type         | Output                             |
| ------------------ | ------------ | ---------------------------------- |
| Variance Filtering | Unsupervised | Top-k most variable genes per fold |
| Mutual Information | Supervised   | Top-k most class-informative genes |
| L1 (LASSO)         | Embedded     | Non-zero coefficient genes         |

Each method outputs:
* Fold-specific selected gene sets
* Pairwise Jaccard similarity across folds

2. **Stability Evaluation (Jaccard Index)**
* Calculate Jaccard similarity for all 10 fold-pair combinations
* Report mean Jaccard score as overall stability

3. **Random Forest Classification**
* Model: `RandomForestClassifier`
* Trees: 500
* CV: Same 5-fold stratified splits
* Outputs per fold:
  * `y_true`
  * `y_pred`
* Evaluated for all three feature selection methods as well as the full, unfiltered dataset
* Performed shuffled-label leakage test with same RF framework on randomized tumor labels
  * Confirmed no data leakage or label memorization

## Core Findings
* All feature selection methods preserve near-ceiling classification accuracy (~0.997)
* Feature space reduced from ~20,000 → ~200 genes
* MI and LASSO slightly outperform variance in RF accuracy
* Shuffled-label control confirms no leakage
* Baseline RF shows TCGA cancer types are highly separable

