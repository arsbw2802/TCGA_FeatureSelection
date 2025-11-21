
import argparse
import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import jaccard_score

def infer_label_column(df, user_label_col=None):
    if user_label_col and user_label_col in df.columns:
        return user_label_col
    candidates = ["label","class","target","y","phenotype","group","condition","status"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_data(data_csv, label_column=None, labels_csv=None):
    df = pd.read_csv(data_csv)
    if "sample_id" not in df.columns:
        raise ValueError("Expected a 'sample_id' column in the data CSV.")
    # Extract features
    feature_cols = [c for c in df.columns if c not in ("sample_id",)]
    # If label is in the same CSV
    if labels_csv is None:
        inferred = infer_label_column(df, label_column)
        if inferred is None:
            raise ValueError(
                "No label column found in data CSV. "
                "Pass --label_column if it exists (e.g., --label_column label), "
                "or provide a separate labels file via --labels_csv with columns 'sample_id,label'."
            )
        y = df[inferred].values
        feature_cols = [c for c in feature_cols if c != inferred]
        X = df[feature_cols].values
        sample_ids = df["sample_id"].astype(str).values
        return X, y, feature_cols, sample_ids
    else:
        # labels provided separately
        lab = pd.read_csv(labels_csv)
        # Expect columns sample_id and Class
        if "sample_id" not in lab.columns or "Class" not in lab.columns:
            raise ValueError("Labels CSV must have columns: 'sample_id' and 'Class'.")
        # Merge on sample_id (inner to ensure alignment)
        merged = pd.merge(df, lab[["sample_id","Class"]], on="sample_id", how="inner", validate="one_to_one")
        y = merged["Class"].values
        feature_cols = [c for c in df.columns if c not in ("sample_id",)]
        X = merged[feature_cols].values
        sample_ids = merged["sample_id"].astype(str).values
        return X, y, feature_cols, sample_ids

def jaccard_index(set_a, set_b):
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return inter / union if union > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="L1-regularized Logistic Regression feature selection with 5-fold CV and Jaccard similarity.")
    parser.add_argument("--data_csv", required=True, help="Path to the RNA-seq CSV with 'sample_id' and gene columns. If it also contains labels, specify --label_column or a detectable name (label/class/target/...).")
    parser.add_argument("--label_column", default=None, help="Name of the label column in data_csv (e.g., 'label' or 'class'). Optional if detectable automatically.")
    parser.add_argument("--labels_csv", default=None, help="Optional path to a separate labels CSV with columns 'sample_id,label'.")
    parser.add_argument("--C", type=float, default=0.1, help="Inverse of regularization strength for L1 LogisticRegression (smaller -> more sparsity). Default 0.1")
    parser.add_argument("--max_iter", type=int, default=5000, help="Max iterations for LogisticRegression(saga).")
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    X, y, feature_names, sample_ids = load_data(args.data_csv, label_column=args.label_column, labels_csv=args.labels_csv)

    # Sanity checks
    if X.ndim != 2:
        raise ValueError("X must be 2D [n_samples, n_features].")
    if len(np.unique(y)) < 2:
        raise ValueError("y must have at least 2 classes for classification.")

    # Build pipeline: scale then L1 logistic regression
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="saga",
            C=args.C,
            max_iter=args.max_iter,
            n_jobs=-1,
            multi_class="ovr"  # robust for binary or multiclass with l1
        ))
    ])

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    fold_results = []
    feature_sets = []
    fold_idx = 0
    for train_idx, test_idx in skf.split(X, y):
        fold_idx += 1
        X_train, y_train = X[train_idx], y[train_idx]

        start = time.time()
        pipe.fit(X_train, y_train)
        elapsed = time.time() - start

        clf = pipe.named_steps["clf"]
        coefs = clf.coef_  # shape (n_classes or 1, n_features)
        # Select features with any non-zero coefficient across classes
        nonzero_mask = (np.abs(coefs) > 0).any(axis=0)
        selected_features = [fname for fname, keep in zip(feature_names, nonzero_mask) if keep]

        # Save per-fold selected features
        fold_out = os.path.join(args.output_dir, f"selected_genes_fold_{fold_idx}.csv")
        pd.Series(selected_features, name=f"fold_{fold_idx}_selected_genes").to_csv(fold_out, index=False)

        fold_results.append({
            "fold": fold_idx,
            "n_selected": int(nonzero_mask.sum()),
            "fit_time_sec": round(elapsed, 4)
        })
        feature_sets.append(set(selected_features))

    # Summary CSV
    summary_df = pd.DataFrame(fold_results)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Jaccard matrix across the 5 folds
    k = len(feature_sets)
    jaccard_mat = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            jaccard_mat[i, j] = jaccard_index(feature_sets[i], feature_sets[j])
    jaccard_df = pd.DataFrame(jaccard_mat, index=[f"Fold_{i+1}" for i in range(k)], columns=[f"Fold_{i+1}" for i in range(k)])
    jaccard_path = os.path.join(args.output_dir, "jaccard_matrix.csv")
    jaccard_df.to_csv(jaccard_path)

    # Also save average off-diagonal Jaccard
    off_diag = [jaccard_mat[i, j] for i in range(k) for j in range(k) if i != j]
    avg_jaccard = float(np.mean(off_diag)) if off_diag else 1.0
    meta = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "unique_classes": [str(c) for c in np.unique(y)],
        "C": args.C,
        "max_iter": args.max_iter,
        "random_state": args.random_state,
        "average_pairwise_jaccard": avg_jaccard
    }
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Wrote: {summary_path}, {jaccard_path}, selected_genes_fold_*.csv, and run_metadata.json")

if __name__ == "__main__":
    main()
