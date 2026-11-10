"""Shared helper functions for the Machine Learning midterm.

Both notebooks (breast_cancer.ipynb and wine.ipynb) import this module.
Every function receives the dataset-specific settings as arguments, so the
two datasets never share data, fitted models or results.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, adjusted_rand_score, silhouette_score,
)

# --------------------------------------------------------------------------
# Global experiment settings (identical for both datasets)
# --------------------------------------------------------------------------
SEEDS = [42, 52, 62]
TEST_SEED = 42          # seed used once to reserve the fixed test set
EVAL_SEED = 42          # predeclared seed for the final test evaluation
TEST_SIZE = 0.20
VAL_SIZE = 0.125        # 0.125 * 80% = 10% of the full dataset
TREE_DEPTHS = [2, 3, 5, 10, None]
KNN_NEIGHBORS = [1, 3, 5, 7, 11]
REPRESENTATIONS = ["unscaled", "scaled"]

FIG_DIR = Path("figures")
RES_DIR = Path("results")
PARTS_DIR = RES_DIR / "parts"


def make_output_dirs():
    """Create output folders if they do not exist yet."""
    for d in (FIG_DIR, RES_DIR, PARTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    """Save a figure into figures/ and return the path."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------
def load_dataset(name):
    """Return X (DataFrame), y (Series) and target names for a dataset."""
    loaders = {"breast_cancer": load_breast_cancer, "wine": load_wine}
    bunch = loaders[name]()
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = pd.Series(bunch.target, name="target")
    return X, y, list(bunch.target_names)


def class_mapping_table(y, target_names):
    """Label -> name mapping together with class counts."""
    counts = y.value_counts().sort_index()
    return pd.DataFrame({
        "label": counts.index,
        "target_name": [target_names[i] for i in counts.index],
        "n_samples": counts.values,
        "percent": (100 * counts.values / len(y)).round(2),
    })


# --------------------------------------------------------------------------
# 2. Quality checks
# --------------------------------------------------------------------------
def missing_values_table(X):
    return X.isna().sum().rename("missing_count").to_frame()


def dtype_summary(X):
    dtypes = X.dtypes.astype(str).rename("dtype").to_frame()
    n_num = X.select_dtypes(include="number").shape[1]
    n_cat = X.shape[1] - n_num
    return dtypes, n_num, n_cat


def outlier_table(X, z_thresh=3.0):
    """Count potential outliers per feature with IQR (1.5 rule) and |z| > 3."""
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_count = ((X < lower) | (X > upper)).sum()
    z = np.abs(stats.zscore(X, ddof=0))
    z_count = (pd.DataFrame(z, columns=X.columns) > z_thresh).sum()
    out = pd.DataFrame({
        "IQR_lower": lower, "IQR_upper": upper,
        "IQR_outliers": iqr_count, "z>3_outliers": z_count,
    })
    out["IQR_outlier_%"] = (100 * out["IQR_outliers"] / len(X)).round(2)
    return out.sort_values("IQR_outliers", ascending=False)


# --------------------------------------------------------------------------
# 3. EDA
# --------------------------------------------------------------------------
def descriptive_stats(X):
    return pd.DataFrame({
        "mean": X.mean(), "median": X.median(), "min": X.min(),
        "max": X.max(), "std": X.std(), "Q1": X.quantile(0.25),
        "Q3": X.quantile(0.75),
    })


def plot_class_distribution(y, target_names, dataset_title, fig_name):
    counts = y.value_counts().sort_index()
    labels = [f"{i}: {target_names[i]}" for i in counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color="#4C72B0")
    ax.bar_label(bars)
    ax.set_title(f"{dataset_title} - class distribution (full dataset)")
    ax.set_xlabel("Class (label: name)")
    ax.set_ylabel("Number of samples")
    save_fig(fig, fig_name)
    plt.show()


def plot_correlation_heatmap(X, dataset_title, fig_name):
    corr = X.corr(method="pearson")
    n = len(corr)
    size = max(8, 0.38 * n)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    if n <= 15:  # annotate small matrices only, to stay readable
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center",
                        va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title(f"{dataset_title} - Pearson feature correlation")
    save_fig(fig, fig_name)
    plt.show()
    return corr


def top_correlated_pairs(corr, k=3):
    """Strongest unique pairs by |r|, excluding diagonal and mirrored pairs."""
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().rename("r").reset_index()
    pairs.columns = ["feature_1", "feature_2", "r"]
    pairs["abs_r"] = pairs["r"].abs()
    return pairs.sort_values("abs_r", ascending=False).head(k).reset_index(drop=True)


def plot_boxplots(X, dataset_title, fig_prefix, per_fig=10, ncols=5):
    """One small boxplot per feature (own y-scale), grouped into figures."""
    cols = list(X.columns)
    groups = [cols[i:i + per_fig] for i in range(0, len(cols), per_fig)]
    for g, group in enumerate(groups, start=1):
        nrows = int(np.ceil(len(group) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, group):
            ax.boxplot(X[col], vert=True, widths=0.6)
            ax.set_title(col, fontsize=9)
            ax.set_xticks([])
        for ax in axes[len(group):]:
            ax.axis("off")
        fig.suptitle(f"{dataset_title} - boxplots (figure {g}/{len(groups)}, full dataset)")
        fig.tight_layout()
        save_fig(fig, f"{fig_prefix}_{g}")
        plt.show()


# --------------------------------------------------------------------------
# 4. Splitting
# --------------------------------------------------------------------------
def make_splits(X, y):
    """Fixed 20% test set, then a 70/10 train/validation split per seed.

    Returns (X_test, y_test, splits) where splits[seed] holds
    X_train, X_val, y_train, y_val with original row indices preserved.
    """
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=TEST_SEED)
    splits = {}
    for seed in SEEDS:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_rest, y_rest, test_size=VAL_SIZE, stratify=y_rest,
            random_state=seed)
        splits[seed] = {"X_train": X_tr, "X_val": X_va,
                        "y_train": y_tr, "y_val": y_va}
    return X_test, y_test, splits


def split_summary(y, y_test, splits, target_names):
    """Nine-row table: three subsets for each of the three seeds."""
    rows = []
    n_total = len(y)
    for seed in SEEDS:
        subsets = {"train": splits[seed]["y_train"],
                   "validation": splits[seed]["y_val"], "test": y_test}
        for subset, ys in subsets.items():
            row = {"seed": seed, "subset": subset, "n": len(ys),
                   "percent": round(100 * len(ys) / n_total, 2)}
            for lbl in sorted(y.unique()):
                row[f"class_{lbl}_{target_names[lbl]}"] = int((ys == lbl).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def verify_splits(X, y_test, splits):
    """Check overlaps, totals and identical test indices; return a table."""
    test_idx = set(y_test.index)
    rows = []
    for seed in SEEDS:
        tr = set(splits[seed]["y_train"].index)
        va = set(splits[seed]["y_val"].index)
        rows.append({
            "seed": seed,
            "train∩val": len(tr & va),
            "train∩test": len(tr & test_idx),
            "val∩test": len(va & test_idx),
            "sum_of_counts": len(tr) + len(va) + len(test_idx),
            "dataset_size": len(X),
            "covers_all_rows": (tr | va | test_idx) == set(X.index),
            "test_indices_same_as_seed42": test_idx == set(y_test.index),
        })
    out = pd.DataFrame(rows)
    assert (out[["train∩val", "train∩test", "val∩test"]] == 0).all().all()
    assert (out["sum_of_counts"] == out["dataset_size"]).all()
    assert out["covers_all_rows"].all()
    return out


def split_indices_table(y_test, splits):
    """Long table of original row indices for every seed and subset."""
    rows = []
    for seed in SEEDS:
        for subset, ys in (("train", splits[seed]["y_train"]),
                           ("validation", splits[seed]["y_val"]),
                           ("test", y_test)):
            rows += [{"seed": seed, "subset": subset, "row_index": int(i)}
                     for i in ys.index]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5-6. Preprocessing and representations
# --------------------------------------------------------------------------
def build_representations(X_test, splits, needs_imputation):
    """Fit imputer (only if needed) and a NEW StandardScaler per seed on the
    training subset only. Returns:
      reps[seed][rep] = {"X_train", "X_val"}
      prep[seed] = {"imputer", "scaler"}
      test_reps = {"unscaled": X_test_u, "scaled": X_test_s}  (seed 42 preprocessing)
    """
    reps, prep = {}, {}
    for seed in SEEDS:
        X_tr = splits[seed]["X_train"]
        X_va = splits[seed]["X_val"]
        imputer = None
        if needs_imputation:
            imputer = SimpleImputer(strategy="median").fit(X_tr)
            X_tr = pd.DataFrame(imputer.transform(X_tr), index=X_tr.index, columns=X_tr.columns)
            X_va = pd.DataFrame(imputer.transform(X_va), index=X_va.index, columns=X_va.columns)
        scaler = StandardScaler().fit(X_tr)            # training data only
        X_tr_s = pd.DataFrame(scaler.transform(X_tr), index=X_tr.index, columns=X_tr.columns)
        X_va_s = pd.DataFrame(scaler.transform(X_va), index=X_va.index, columns=X_va.columns)
        reps[seed] = {"unscaled": {"X_train": X_tr, "X_val": X_va},
                      "scaled": {"X_train": X_tr_s, "X_val": X_va_s}}
        prep[seed] = {"imputer": imputer, "scaler": scaler}

    # Fixed test set: transformed only with seed-42 preprocessing
    X_te = X_test
    if needs_imputation:
        X_te = pd.DataFrame(prep[EVAL_SEED]["imputer"].transform(X_te),
                            index=X_te.index, columns=X_te.columns)
    X_te_s = pd.DataFrame(prep[EVAL_SEED]["scaler"].transform(X_te),
                          index=X_te.index, columns=X_te.columns)
    test_reps = {"unscaled": X_te, "scaled": X_te_s}
    return reps, prep, test_reps


# --------------------------------------------------------------------------
# 7-8. Classification experiments and metrics
# --------------------------------------------------------------------------
def classifier_grid():
    """List of (algorithm, param_label, factory(seed)) - 11 settings."""
    grid = []
    for d in TREE_DEPTHS:
        grid.append(("decision_tree", f"depth{d}",
                     lambda seed, d=d: DecisionTreeClassifier(max_depth=d, random_state=seed)))
    for k in KNN_NEIGHBORS:
        grid.append(("knn", f"k{k}",
                     lambda seed, k=k: KNeighborsClassifier(n_neighbors=k, weights="uniform",
                                                            metric="euclidean")))
    grid.append(("gaussian_nb", "default", lambda seed: GaussianNB()))
    return grid


def undefined_metric_classes(y_true, y_pred, labels):
    """Classes whose precision (no predictions) or recall (no true samples)
    is undefined BEFORE zero_division replacement."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    no_pred = [int(c) for c in labels if (y_pred == c).sum() == 0]
    no_true = [int(c) for c in labels if (y_true == c).sum() == 0]
    return no_pred, no_true


def roc_auc(model, X, y_true, dataset, positive_label):
    """ROC-AUC with the assignment conventions.

    breast_cancer: evaluation target 1 for original class 0 (malignant),
                   score = predicted probability of original class 0.
    wine:          macro one-vs-rest on class probabilities.
    """
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if dataset == "breast_cancer":
        y_eval = (np.asarray(y_true) == positive_label).astype(int)
        score = proba[:, classes.index(positive_label)]
        return roc_auc_score(y_eval, score)
    # Wine: probability columns follow model.classes_ -> pass labels explicitly
    assert classes == sorted(classes), "probability column order mismatch"
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                         labels=classes)


def classification_metrics(model, X, y_true, dataset, positive_label):
    y_pred = model.predict(X)
    labels = list(model.classes_)
    no_pred, no_true = undefined_metric_classes(y_true, y_pred, labels)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro",
                                           labels=labels, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro",
                                     labels=labels, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels,
                             zero_division=0),
        "roc_auc": roc_auc(model, X, y_true, dataset, positive_label),
        "precision_undefined_classes": str(no_pred) if no_pred else "",
        "recall_undefined_classes": str(no_true) if no_true else "",
    }


def run_classifiers(dataset, reps, splits, positive_label):
    """Run all 66 classifier fits of one dataset.

    Returns (runs DataFrame, fitted models of seed 42 keyed by config_id).
    """
    rows, models_s42 = [], {}
    for seed in SEEDS:
        y_tr = splits[seed]["y_train"]
        y_va = splits[seed]["y_val"]
        for rep in REPRESENTATIONS:
            X_tr = reps[seed][rep]["X_train"]
            X_va = reps[seed][rep]["X_val"]
            for algo, param, factory in classifier_grid():
                config_id = f"{dataset}_{algo}_{rep}_{param}"
                run_id = f"{config_id}_s{seed}"
                model = factory(seed).fit(X_tr, y_tr)      # training data only
                tr_pred = model.predict(X_tr)
                val = classification_metrics(model, X_va, y_va, dataset, positive_label)
                rows.append({
                    "dataset": dataset, "run_id": run_id, "config_id": config_id,
                    "seed": seed, "algorithm": algo, "representation": rep,
                    "parameter": param,
                    "train_accuracy": accuracy_score(y_tr, tr_pred),
                    "train_macro_f1": f1_score(y_tr, tr_pred, average="macro", zero_division=0),
                    **{f"val_{k}": v for k, v in val.items()},
                })
                if seed == EVAL_SEED:
                    models_s42[config_id] = model
    runs = pd.DataFrame(rows)
    assert len(runs) == 66, len(runs)
    return runs, models_s42


METRIC_COLS = ["train_accuracy", "train_macro_f1", "val_accuracy",
               "val_macro_precision", "val_macro_recall", "val_macro_f1",
               "val_roc_auc"]


def summarise_runs(runs):
    """22-row summary: mean and sample std (ddof=1) over the three seeds."""
    keys = ["dataset", "config_id", "algorithm", "representation", "parameter"]
    g = runs.groupby(keys, sort=False)[METRIC_COLS]
    mean = g.mean().add_suffix("_mean")
    std = g.std(ddof=1).add_suffix("_std")
    summary = pd.concat([mean, std], axis=1).reset_index()
    summary["n_seeds"] = g.size().values
    ordered = keys + [c for m in METRIC_COLS for c in (f"{m}_mean", f"{m}_std")] + ["n_seeds"]
    summary = summary[ordered]
    assert len(summary) == 22, len(summary)
    return summary


def select_configuration(summary):
    """Highest mean val macro-F1; ties -> mean val accuracy, then lower
    macro-F1 std, then alphabetical config_id."""
    ranked = summary.sort_values(
        ["val_macro_f1_mean", "val_accuracy_mean", "val_macro_f1_std", "config_id"],
        ascending=[False, False, True, True]).reset_index(drop=True)
    return ranked.iloc[0], ranked


def plot_param_curve(summary, algorithm, param_values, param_prefix, xlabel,
                     dataset_title, fig_name):
    """Mean train/val macro-F1 vs parameter, both representations, seed std
    as error bars."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(param_values))
    styles = {"unscaled": "-", "scaled": "--"}
    markers = {"unscaled": "o", "scaled": "x"}  # trees are scale-invariant: lines may overlap
    colors = {"train": "#4C72B0", "val": "#DD8452"}
    for rep in REPRESENTATIONS:
        sub = summary[(summary.algorithm == algorithm) & (summary.representation == rep)]
        sub = sub.set_index("parameter").loc[[f"{param_prefix}{v}" for v in param_values]]
        for split, col in (("train", "train_macro_f1"), ("val", "val_macro_f1")):
            ax.errorbar(x, sub[f"{col}_mean"], yerr=sub[f"{col}_std"],
                        linestyle=styles[rep], marker=markers[rep], markersize=8, capsize=4,
                        color=colors[split],
                        label=f"{split} - {rep}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in param_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro-F1 (mean ± std over 3 seeds)")
    ax.set_title(f"{dataset_title} - {algorithm}: training vs validation macro-F1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, fig_name)
    plt.show()


def scaling_effect_table(summary):
    """Mean val macro-F1 scaled minus unscaled for every setting."""
    piv = summary.pivot_table(index=["algorithm", "parameter"],
                              columns="representation",
                              values="val_macro_f1_mean")
    piv["scaled_minus_unscaled"] = piv["scaled"] - piv["unscaled"]
    return piv.sort_values("scaled_minus_unscaled", ascending=False)


# --------------------------------------------------------------------------
# 9. Final test evaluation
# --------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, labels, target_names, dataset_title, fig_name):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=[f"{l}: {target_names[l]}" for l in labels])
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{dataset_title} - test confusion matrix (seed 42 model)")
    save_fig(fig, fig_name)
    plt.show()
    return pd.DataFrame(cm, index=[f"Actual {l}: {target_names[l]}" for l in labels],
                        columns=[f"Predicted {l}: {target_names[l]}" for l in labels])


def manual_one_vs_rest(cm, labels, cls):
    """Precision, recall, F1 for one class from a confusion matrix (counts)."""
    i = labels.index(cls)
    tp = int(cm[i, i])
    fp = int(cm[:, i].sum() - tp)
    fn = int(cm[i, :].sum() - tp)
    tn = int(cm.sum() - tp - fp - fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"class": cls, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall, "f1": f1}


def plot_roc_binary(model, X, y_true, positive_label, target_names, dataset_title, fig_name):
    classes = list(model.classes_)
    score = model.predict_proba(X)[:, classes.index(positive_label)]
    y_eval = (np.asarray(y_true) == positive_label).astype(int)
    fpr, tpr, _ = roc_curve(y_eval, score)
    auc = roc_auc_score(y_eval, score)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"{target_names[positive_label]} vs rest (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{dataset_title} - test ROC curve (positive = {target_names[positive_label]})")
    ax.legend(loc="lower right")
    save_fig(fig, fig_name)
    plt.show()
    return auc


def plot_roc_ovr(model, X, y_true, target_names, dataset_title, fig_name):
    classes = list(model.classes_)
    proba = model.predict_proba(X)
    y_arr = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    aucs = {}
    for j, c in enumerate(classes):
        y_bin = (y_arr == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proba[:, j])
        aucs[c] = roc_auc_score(y_bin, proba[:, j])
        ax.plot(fpr, tpr, label=f"class {c}: {target_names[c]} vs rest (AUC = {aucs[c]:.3f})")
    macro = roc_auc_score(y_arr, proba, multi_class="ovr", average="macro", labels=classes)
    ax.plot([0, 1], [0, 1], "k:", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{dataset_title} - test one-vs-rest ROC (macro AUC = {macro:.3f})")
    ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, fig_name)
    plt.show()
    return aucs, macro


def misclassified_table(model, X, y_true, target_names):
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    df = pd.DataFrame({"true": np.asarray(y_true), "predicted": y_pred}, index=X.index)
    for j, c in enumerate(model.classes_):
        df[f"P(class {c}: {target_names[c]})"] = proba[:, j].round(3)
    return df[df.true != df.predicted]


# --------------------------------------------------------------------------
# 10. K-Means
# --------------------------------------------------------------------------
def safe_silhouette(X, clusters):
    n_clusters = len(np.unique(clusters))
    if n_clusters < 2 or n_clusters >= len(X):
        return np.nan
    return silhouette_score(X, clusters)


def run_kmeans(dataset, reps, splits, n_clusters):
    """12 K-Means fits in total across both notebooks: 6 per dataset."""
    rows, models_s42 = [], {}
    for seed in SEEDS:
        y_va = splits[seed]["y_val"]
        for rep in REPRESENTATIONS:
            X_tr = reps[seed][rep]["X_train"]
            X_va = reps[seed][rep]["X_val"]
            km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X_tr)  # no y
            val_clusters = km.predict(X_va)
            rows.append({
                "dataset": dataset,
                "run_id": f"{dataset}_kmeans_{rep}_c{n_clusters}_s{seed}",
                "seed": seed, "representation": rep, "n_clusters": n_clusters,
                "train_inertia": km.inertia_,
                "val_ari": adjusted_rand_score(y_va, val_clusters),
                "val_silhouette": safe_silhouette(X_va, val_clusters),
                "val_n_distinct_clusters": len(np.unique(val_clusters)),
            })
            if seed == EVAL_SEED:
                models_s42[rep] = km
    runs = pd.DataFrame(rows)
    assert len(runs) == 6
    return runs, models_s42


def summarise_kmeans(runs):
    rows = []
    for rep in REPRESENTATIONS:
        sub = runs[runs.representation == rep]
        row = {"dataset": sub.dataset.iloc[0], "representation": rep}
        for m in ("train_inertia", "val_ari", "val_silhouette"):
            vals = sub[m].dropna()               # N/A is never replaced with 0
            row[f"{m}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{m}_std"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
            row[f"{m}_valid_seeds"] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def kmeans_test_eval(dataset, km_models, test_reps, y_test, n_clusters):
    rows, contingency, test_clusters = [], {}, {}
    for rep in REPRESENTATIONS:
        Xte = test_reps[rep]
        clusters = km_models[rep].predict(Xte)
        test_clusters[rep] = clusters
        counts = pd.Series(clusters).value_counts().sort_index()
        sil = safe_silhouette(Xte, clusters)
        row = {"dataset": dataset, "representation": rep, "eval_seed": EVAL_SEED,
               "n_clusters": n_clusters, "n_test": len(Xte),
               "test_ari": adjusted_rand_score(y_test, clusters),
               "test_silhouette": sil}
        for c in range(n_clusters):
            row[f"cluster_{c}_count"] = int(counts.get(c, 0))
        rows.append(row)
        contingency[rep] = pd.crosstab(
            pd.Series(np.asarray(y_test), name="true class"),
            pd.Series(clusters, name="cluster ID"))
    return pd.DataFrame(rows), contingency, test_clusters


def choose_plot_features(X_train, y_train):
    """Pick the two original features with the highest ANOVA F-statistic on
    the seed-42 TRAINING data (used for visualisation only)."""
    f, _ = f_classif(X_train, y_train)
    order = np.argsort(f)[::-1][:2]
    return [X_train.columns[i] for i in order], pd.Series(f, index=X_train.columns)


def plot_class_vs_cluster(X_rep, y_test, clusters, features, target_names, rep,
                          dataset_title, fig_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    f1, f2 = features
    y_arr = np.asarray(y_test)
    for c in sorted(np.unique(y_arr)):
        m = y_arr == c
        axes[0].scatter(X_rep.loc[m, f1], X_rep.loc[m, f2], s=25, alpha=0.8,
                        label=f"{c}: {target_names[c]}")
    for k in sorted(np.unique(clusters)):
        m = clusters == k
        axes[1].scatter(X_rep.loc[m, f1], X_rep.loc[m, f2], s=25, alpha=0.8,
                        marker="s", label=f"cluster {k}")
    axes[0].set_title("True classes")
    axes[1].set_title("Predicted K-Means cluster IDs")
    for ax in axes:
        ax.set_xlabel(f1 + (" (scaled)" if rep == "scaled" else ""))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f2 + (" (scaled)" if rep == "scaled" else ""))
    fig.suptitle(f"{dataset_title} - test set, {rep} representation, seed-42 K-Means "
                 f"(2-feature projection of an all-feature model)")
    fig.tight_layout()
    save_fig(fig, fig_name)
    plt.show()


# --------------------------------------------------------------------------
# 12. Export
# --------------------------------------------------------------------------
CSV_EXPECTED_ROWS = {
    "classifier_runs": 132, "classifier_summary": 44, "classifier_test": 2,
    "kmeans_runs": 12, "kmeans_summary": 4, "kmeans_test": 4,
}


def export_results(dataset, tables):
    """Save this dataset's tables to results/parts/, then rebuild the six
    combined CSV files from every dataset part that exists."""
    for name, df in tables.items():
        df.to_csv(PARTS_DIR / f"{dataset}__{name}.csv", index=False)
    status = []
    for name, expected in CSV_EXPECTED_ROWS.items():
        parts = sorted(PARTS_DIR.glob(f"*__{name}.csv"))
        combined = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        combined.to_csv(RES_DIR / f"{name}.csv", index=False)
        status.append({"file": f"{name}.csv", "rows": len(combined),
                       "expected_rows_both_datasets": expected,
                       "datasets_included": ", ".join(sorted(combined.dataset.unique())),
                       "complete": len(combined) == expected})
    return pd.DataFrame(status)


def package_versions():
    import sys, platform, sklearn, scipy, matplotlib
    return pd.DataFrame({
        "package": ["python", "platform", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib"],
        "version": [sys.version.split()[0], platform.platform(), np.__version__, pd.__version__,
                    sklearn.__version__, scipy.__version__, matplotlib.__version__],
    })
