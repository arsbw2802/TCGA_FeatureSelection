import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from itertools import combinations

X = pd.read_csv("RNAseq_with_HGNC_symbols1.csv", index_col=0)

labels_df = pd.read_csv("labels.csv", index_col=0)
y = labels_df['Class'].values

print("Class distribution:")
print(pd.Series(y).value_counts(), "\n")

k_genes = 2000  # change to 500 / 5000 later to compare stability
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

selected_gene_sets = []  

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    X_train = X.iloc[train_idx]
    y_train = y[train_idx]
    gene_variances = X_train.var(axis=0)
    top_genes = (gene_variances
        .sort_values(ascending=False)
        .head(k_genes)
        .index)

    selected_gene_sets.append(set(top_genes))
    print(f"Fold {fold_idx}: selected {len(top_genes)} genes")

def jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0

pairwise_jaccards = []
for (i, genes_i), (j, genes_j) in combinations(enumerate(selected_gene_sets), 2):
    jac = jaccard(genes_i, genes_j)
    pairwise_jaccards.append(jac)
    print(f"Jaccard(Fold {i+1}, Fold {j+1}) = {jac:.3f}")

mean_jaccard = np.mean(pairwise_jaccards)
print(f"\nMean pairwise Jaccard (variance filter, k={k_genes}): {mean_jaccard:.3f}")

