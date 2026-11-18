from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# Konfigurasi
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "results" / "figures" / "classification"
METRIC_DIR = ROOT / "results" / "metrics" / "classification"
MODEL_DIR = ROOT / "results" / "models" / "classification"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
QUALITY_THRESHOLD = 6

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET = "target"

PLOT_COLORS = {
    "blue": "#2563eb",
    "red": "#dc2626",
    "green": "#16a34a",
    "gray": "#64748b",
    "orange": "#ea580c",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# Data
def load_wine() -> pd.DataFrame:
    frames = []
    for filename, wine_type in [
        ("winequality-red.csv", "red"),
        ("winequality-white.csv", "white"),
    ]:
        part = pd.read_csv(RAW_DIR / filename, sep=";")
        part["wine_type"] = wine_type
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df["quality"] >= QUALITY_THRESHOLD).astype(int)

    print(f"Data mentah          : {n_raw} baris")
    print(f"Setelah deduplikasi  : {len(df)} baris ({n_raw - len(df)} duplikat dibuang)")
    print(f"Kelas baik (>= {QUALITY_THRESHOLD})  : {int(df[TARGET].sum())} "
          f"({df[TARGET].mean():.1%})")
    print(f"Kelas buruk (< {QUALITY_THRESHOLD})   : {int((1 - df[TARGET]).sum())} "
          f"({1 - df[TARGET].mean():.1%})")
    return df


# Evaluasi
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    print(f"\n--- {name} ---")
    for key in ["accuracy", "balanced_accuracy", "precision", "recall",
                "f1", "f1_macro", "roc_auc", "pr_auc"]:
        print(f"  {key:18s}: {metrics[key]:.4f}")
    print(f"  confusion matrix  : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    return metrics


# Visualisasi
def plot_class_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    quality_counts = df["quality"].value_counts().sort_index()
    axes[0].bar(quality_counts.index.astype(int), quality_counts.values,
                color=PLOT_COLORS["blue"], edgecolor="white")
    axes[0].set_title("Distribusi skor quality (setelah deduplikasi)")
    axes[0].set_xlabel("quality")
    axes[0].set_ylabel("Jumlah baris")
    for x, y in zip(quality_counts.index.astype(int), quality_counts.values):
        axes[0].text(x, y + 25, str(y), ha="center", fontsize=8)

    binary_counts = df[TARGET].value_counts().sort_index()
    bars = axes[1].bar(
        ["buruk (0)", "baik (1)"],
        [binary_counts.get(0, 0), binary_counts.get(1, 0)],
        color=[PLOT_COLORS["red"], PLOT_COLORS["green"]],
        edgecolor="white",
    )
    axes[1].set_title(f"Target biner: quality >= {QUALITY_THRESHOLD} = baik")
    axes[1].set_ylabel("Jumlah baris")
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, height + 25,
                     f"{int(height)}", ha="center", fontsize=9)

    axes[0].grid(False)
    axes[1].grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dt_class_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_depth_curve(best_model, X_train, y_train, X_test, y_test,
                     cv, best_depth: int) -> dict:
    depths = list(range(1, 21))
    train_scores, test_scores, cv_scores, cv_stds = [], [], [], []

    for depth in depths:
        model = clone(best_model).set_params(max_depth=depth)
        model.fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))
        scores = cross_val_score(
            clone(best_model).set_params(max_depth=depth),
            X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1,
        )
        cv_scores.append(scores.mean())
        cv_stds.append(scores.std())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(depths, train_scores, "o-", color=PLOT_COLORS["blue"],
            label="Akurasi train")
    ax.plot(depths, cv_scores, "s-", color=PLOT_COLORS["green"],
            label="Akurasi CV (5-fold)")
    ax.fill_between(depths,
                    np.array(cv_scores) - np.array(cv_stds),
                    np.array(cv_scores) + np.array(cv_stds),
                    color=PLOT_COLORS["green"], alpha=0.15)
    ax.plot(depths, test_scores, "^-", color=PLOT_COLORS["red"],
            label="Akurasi test")
    ax.axvline(best_depth, color=PLOT_COLORS["gray"], linestyle="--", linewidth=1,
               label=f"max_depth terpilih = {best_depth}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Akurasi")
    ax.set_title("Kurva overfitting: akurasi train, CV, dan test terhadap max_depth")
    ax.legend(loc="lower right")
    ax.set_xticks(depths)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dt_maxdepth_curve.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "depths": depths,
        "train_accuracy": [float(v) for v in train_scores],
        "cv_accuracy": [float(v) for v in cv_scores],
        "cv_std": [float(v) for v in cv_stds],
        "test_accuracy": [float(v) for v in test_scores],
    }


def plot_confusion_matrix(y_test, y_pred) -> None:
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["buruk (0)", "baik (1)"],
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion matrix — model terbaik (data test)")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dt_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_test, y_proba) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    axes[0].plot(fpr, tpr, color=PLOT_COLORS["blue"], linewidth=2,
                 label=f"ROC-AUC = {roc_auc_score(y_test, y_proba):.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=PLOT_COLORS["gray"],
                 linewidth=1, label="Acak")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("Kurva ROC")
    axes[0].legend(loc="lower right")

    baseline = y_test.mean()
    axes[1].plot(recall, precision, color=PLOT_COLORS["green"], linewidth=2,
                 label=f"PR-AUC = {average_precision_score(y_test, y_proba):.4f}")
    axes[1].axhline(baseline, linestyle="--", color=PLOT_COLORS["gray"],
                    linewidth=1, label=f"Baseline = {baseline:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Kurva Precision-Recall")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "dt_roc_pr.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(best_model, X_test, y_test) -> dict:
    impurity = pd.Series(best_model.feature_importances_, index=FEATURES)

    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=10,
        random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=-1,
    )
    perm_series = pd.Series(perm.importances_mean, index=FEATURES)
    perm_std = pd.Series(perm.importances_std, index=FEATURES)

    order_imp = impurity.sort_values(ascending=True)
    order_perm = perm_series.sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    axes[0].barh(order_imp.index, order_imp.values, color=PLOT_COLORS["blue"])
    axes[0].set_title("Feature importance (impurity / Gini)")
    axes[0].set_xlabel("Kontribusi penurunan impurity")
    axes[0].set_xlim(0, max(order_imp.max() * 1.15, 0.05))

    axes[1].barh(order_perm.index, order_perm.values,
                 xerr=perm_std.reindex(order_perm.index).values,
                 color=PLOT_COLORS["green"], capsize=3)
    axes[1].set_title("Permutation importance (f1_macro, data test)")
    axes[1].set_xlabel("Penurunan skor saat fitur diacak")
    axes[1].axvline(0, color=PLOT_COLORS["gray"], linewidth=0.8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "dt_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "impurity": {k: float(v) for k, v in impurity.items()},
        "permutation_mean": {k: float(v) for k, v in perm_series.items()},
        "permutation_std": {k: float(v) for k, v in perm_std.items()},
    }


def plot_learning_curve(best_model, X_train, y_train, cv) -> dict:
    sizes, train_scores, cv_scores = learning_curve(
        best_model, X_train, y_train, cv=cv, scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1, random_state=RANDOM_STATE,
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(sizes, train_scores.mean(axis=1), "o-", color=PLOT_COLORS["blue"],
            label="Skor train")
    ax.plot(sizes, cv_scores.mean(axis=1), "s-", color=PLOT_COLORS["green"],
            label="Skor CV")
    ax.fill_between(sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1),
                    color=PLOT_COLORS["blue"], alpha=0.12)
    ax.fill_between(sizes,
                    cv_scores.mean(axis=1) - cv_scores.std(axis=1),
                    cv_scores.mean(axis=1) + cv_scores.std(axis=1),
                    color=PLOT_COLORS["green"], alpha=0.12)
    ax.set_xlabel("Jumlah sampel training")
    ax.set_ylabel("f1_macro")
    ax.set_title("Learning curve model terbaik")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dt_learning_curve.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "train_sizes": [int(v) for v in sizes],
        "train_f1_macro": [float(v) for v in train_scores.mean(axis=1)],
        "cv_f1_macro": [float(v) for v in cv_scores.mean(axis=1)],
    }


def plot_tree_figure(best_model) -> None:
    """Dua versi: depth 2 dan depth 3."""
    for max_depth, filename, figsize, fontsize in [
        (2, "dt_tree_depth2.png", (16, 8), 11),
        (3, "dt_tree_depth3.png", (26, 12), 9),
    ]:
        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(
            best_model,
            max_depth=max_depth,
            feature_names=FEATURES,
            class_names=["buruk", "baik"],
            filled=True,
            rounded=True,
            proportion=True,
            impurity=True,
            fontsize=fontsize,
            ax=ax,
        )
    
        for text in ax.texts:
            label = text.get_text().strip()
            if label == "True":
                text.set_text("ya")
            elif label == "False":
                text.set_text("tidak")

        ax.set_title(
            f"Decision Tree terbaik (max_depth {max_depth})",
            pad=14,
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        plt.close(fig)


# Main
def main() -> None:
    start = time.time()
    for directory in (FIG_DIR, METRIC_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    df = load_wine()
    plot_class_distribution(df)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"\nTrain: {len(X_train)} baris | Test: {len(X_test)} baris")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # baseline
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)

    default_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    default_tree.fit(X_train, y_train)

    # tuning
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": list(range(2, 13)),
        "min_samples_leaf": [1, 5, 20, 50],
        "ccp_alpha": [0.0, 0.0005, 0.001],
        "class_weight": [None, "balanced"],
    }
    n_combinations = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"\nGridSearchCV: {n_combinations} kombinasi x {N_FOLDS} fold ...")

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        return_train_score=True,
        refit=True,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_depth = best_model.get_depth()
    print(f"Selesai dalam {time.time() - start:.1f} detik")
    print(f"Parameter terbaik: {grid.best_params_}")
    print(f"CV f1_macro terbaik: {grid.best_score_:.4f}")
    print(f"max_depth pohon terbaik: {best_depth}, jumlah leaf: {best_model.get_n_leaves()}")

    # evaluasi
    results = {
        "dummy": evaluate_model("DummyClassifier (mayoritas)", dummy, X_test, y_test),
        "default_tree": evaluate_model("DecisionTree default (tanpa tuning)",
                                       default_tree, X_test, y_test),
        "tuned_tree": evaluate_model("DecisionTree hasil tuning", best_model, X_test, y_test),
    }

    cv_best = cross_val_score(best_model, X_train, y_train, cv=cv,
                              scoring="accuracy", n_jobs=-1)
    cv_best_f1 = cross_val_score(best_model, X_train, y_train, cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
    print(f"\nCV akurasi model terbaik: {cv_best.mean():.4f} +/- {cv_best.std():.4f}")
    print(f"CV f1_macro model terbaik: {cv_best_f1.mean():.4f} +/- {cv_best_f1.std():.4f}")

    print("\nClassification report (test):")
    report_text = classification_report(
        y_test, best_model.predict(X_test),
        target_names=["buruk (0)", "baik (1)"], digits=4,
    )
    print(report_text)

    # figur
    print("\nMembuat visualisasi ...")
    depth_curve = plot_depth_curve(best_model, X_train, y_train, X_test, y_test,
                                   cv, best_depth)
    plot_confusion_matrix(y_test, best_model.predict(X_test))
    plot_roc_pr(y_test, best_model.predict_proba(X_test)[:, 1])
    importance = plot_feature_importance(best_model, X_test, y_test)
    learning = plot_learning_curve(best_model, X_train, y_train, cv)
    plot_tree_figure(best_model)

    rules_depth3 = export_text(best_model, feature_names=FEATURES,
                               max_depth=3, decimals=2)
    rules_full = export_text(best_model, feature_names=FEATURES, decimals=2)
    (METRIC_DIR / "dt_rules_depth3.txt").write_text(rules_depth3, encoding="utf-8")
    (METRIC_DIR / "dt_rules_full.txt").write_text(rules_full, encoding="utf-8")

    # simpan
    joblib.dump(best_model, MODEL_DIR / "decision_tree_best.joblib")

    grid_top = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score").head(10)
    grid_top[[
        "rank_test_score", "mean_test_score", "std_test_score",
        "mean_train_score", "param_criterion", "param_max_depth",
        "param_min_samples_leaf", "param_ccp_alpha", "param_class_weight",
    ]].to_csv(METRIC_DIR / "dt_grid_top10.csv", index=False)

    payload = {
        "dataset": {
            "n_rows_raw": 6497,
            "n_rows_dedup": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features": FEATURES,
            "target_rule": f"quality >= {QUALITY_THRESHOLD}",
            "positive_rate": float(y.mean()),
        },
        "best_params": grid.best_params_,
        "best_cv_f1_macro": float(grid.best_score_),
        "best_depth": int(best_depth),
        "best_n_leaves": int(best_model.get_n_leaves()),
        "cv_accuracy": {"mean": float(cv_best.mean()), "std": float(cv_best.std())},
        "cv_f1_macro": {"mean": float(cv_best_f1.mean()), "std": float(cv_best_f1.std())},
        "test_metrics": results,
        "permutation_importance": importance,
        "depth_curve": depth_curve,
        "learning_curve": learning,
        "classification_report": report_text,
        "runtime_seconds": float(time.time() - start),
    }
    with open(METRIC_DIR / "dt_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\nArtefak tersimpan:")
    for path in [
        FIG_DIR / "dt_class_distribution.png",
        FIG_DIR / "dt_tree_depth2.png",
        FIG_DIR / "dt_tree_depth3.png",
        FIG_DIR / "dt_maxdepth_curve.png",
        FIG_DIR / "dt_confusion_matrix.png",
        FIG_DIR / "dt_roc_pr.png",
        FIG_DIR / "dt_feature_importance.png",
        FIG_DIR / "dt_learning_curve.png",
        METRIC_DIR / "dt_metrics.json",
        METRIC_DIR / "dt_grid_top10.csv",
        METRIC_DIR / "dt_rules_depth3.txt",
        METRIC_DIR / "dt_rules_full.txt",
        MODEL_DIR / "decision_tree_best.joblib",
    ]:
        print(f"  - {path.relative_to(ROOT)}")
    print(f"\nTotal waktu: {time.time() - start:.1f} detik")


if __name__ == "__main__":
    main()
