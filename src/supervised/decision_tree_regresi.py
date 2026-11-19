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
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

# Konfigurasi
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "results" / "figures" / "regression"
METRIC_DIR = ROOT / "results" / "metrics" / "regression"
MODEL_DIR = ROOT / "results" / "models" / "regression"

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
TARGET = "quality"

PLOT_COLORS = {
    "blue": "#2563eb",
    "red": "#dc2626",
    "green": "#16a34a",
    "gray": "#64748b",
    "orange": "#ea580c",
    "purple": "#7c3aed",
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

    print(f"Data mentah          : {n_raw} baris")
    print(f"Setelah deduplikasi  : {len(df)} baris "
          f"({n_raw - len(df)} duplikat dibuang)")
    print(f"Target               : quality {df[TARGET].min()}-{df[TARGET].max()} "
          f"| mean={df[TARGET].mean():.4f} | std={df[TARGET].std():.4f}")
    print(f"Distribusi quality   : "
          + ", ".join(f"{k}={v}" for k, v in df[TARGET].value_counts().sort_index().items()))
    return df


# Evaluasi
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    residual = y_test.to_numpy(dtype=float) - y_pred
    mse = mean_squared_error(y_test, y_pred)

    y_pred_round = np.clip(np.round(y_pred), 3, 9)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mse),
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "medae": float(median_absolute_error(y_test, y_pred)),
        "mape": float(np.mean(np.abs(residual) / y_test.to_numpy(dtype=float)) * 100),
        "max_error": float(np.max(np.abs(residual))),
        "within_1": float(np.mean(np.abs(residual) <= 1.0)),
        "mae_rounded": float(mean_absolute_error(y_test, y_pred_round)),
        "rmse_rounded": float(root_mean_squared_error(y_test, y_pred_round)),
        "r2_rounded": float(r2_score(y_test, y_pred_round)),
        "accuracy_threshold": float(accuracy_score(y_test >= QUALITY_THRESHOLD,
                                                   y_pred >= QUALITY_THRESHOLD)),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
    }

    print(f"\n--- {name} ---")
    for key in ["mae", "rmse", "r2", "medae", "mape", "max_error",
                "within_1", "mae_rounded", "accuracy_threshold"]:
        if key == "within_1":
            print(f"  {key:18s}: {metrics[key]:.4f} ({metrics[key]:.1%})")
        elif key in ("mape",):
            print(f"  {key:18s}: {metrics[key]:.4f}%")
        else:
            print(f"  {key:18s}: {metrics[key]:.4f}")
    return metrics


# Visualisasi
def plot_quality_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    counts = df[TARGET].value_counts().sort_index()
    axes[0].bar(counts.index.astype(int), counts.values,
                color=PLOT_COLORS["blue"], edgecolor="white")
    axes[0].set_title("Distribusi skor quality (setelah deduplikasi)")
    axes[0].set_xlabel("quality")
    axes[0].set_ylabel("Jumlah baris")
    for x, y in zip(counts.index.astype(int), counts.values):
        axes[0].text(x, y + 25, str(y), ha="center", fontsize=8)

    grouped = [df.loc[df[TARGET] == level, "alcohol"].to_numpy()
               for level in counts.index.astype(int)]
    box = axes[1].boxplot(grouped, tick_labels=counts.index.astype(int),
                          patch_artist=True, medianprops={"color": "#111827"})
    for patch in box["boxes"]:
        patch.set_facecolor("#bfdbfe")
        patch.set_edgecolor("#1e3a8a")
    axes[1].set_title("Sebaran alcohol per skor quality")
    axes[1].set_xlabel("quality")
    axes[1].set_ylabel("alcohol")

    axes[0].grid(False)
    axes[1].grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dtr_quality_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(y_test, y_pred, metrics: dict) -> None:
    y_true = y_test.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    jitter_y = y_true + np.random.default_rng(RANDOM_STATE).uniform(-0.18, 0.18, len(y_true))

    ax.scatter(y_pred, jitter_y, s=14, alpha=0.35, color=PLOT_COLORS["blue"],
               edgecolor="none", label="Satu baris test")
    lims = [2.5, 9.5]
    ax.plot(lims, lims, "--", color=PLOT_COLORS["gray"], linewidth=1.2,
            label="Prediksi sempurna (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Prediksi quality")
    ax.set_ylabel("Quality aktual")
    ax.set_title("Prediksi vs aktual — Decision Tree Regressor")
    ax.text(
        0.03, 0.97,
        f"MAE = {metrics['mae']:.4f}\n"
        f"RMSE = {metrics['rmse']:.4f}\n"
        f"R² = {metrics['r2']:.4f}\n"
        f"prediksi ±1 poin = {metrics['within_1']:.1%}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cbd5e1"),
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dtr_actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_test, y_pred) -> None:
    y_true = y_test.to_numpy(dtype=float)
    residual = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].scatter(y_pred, residual, s=14, alpha=0.35, color=PLOT_COLORS["orange"],
                    edgecolor="none")
    axes[0].axhline(0, color=PLOT_COLORS["gray"], linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Prediksi quality")
    axes[0].set_ylabel("Residual (aktual - prediksi)")
    axes[0].set_title("Residual vs prediksi")

    axes[1].hist(residual, bins=np.arange(-4.25, 4.25, 0.5), color=PLOT_COLORS["green"],
                 edgecolor="white")
    axes[1].axvline(0, color=PLOT_COLORS["gray"], linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Jumlah baris")
    axes[1].set_title(f"Distribusi residual (mean={residual.mean():.3f}, "
                      f"std={residual.std():.3f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dtr_residuals.png", bbox_inches="tight")
    plt.close(fig)


def plot_prediction_distribution(y_test, y_pred) -> None:
    y_true = y_test.to_numpy(dtype=float)
    levels = np.arange(3, 10)
    actual = np.array([np.sum(y_true == level) for level in levels])
    predicted = np.array([np.sum(np.clip(np.round(y_pred), 3, 9) == level) for level in levels])
    unique_pred = np.unique(np.round(y_pred, 4))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.38
    ax.bar(levels - width / 2, actual, width, label="Aktual",
           color=PLOT_COLORS["blue"], edgecolor="white")
    ax.bar(levels + width / 2, predicted, width, label="Prediksi (dibulatkan)",
           color=PLOT_COLORS["orange"], edgecolor="white")
    ax.set_xticks(levels)
    ax.set_ylim(0, max(actual.max(), predicted.max()) * 1.25)
    ax.set_xlabel("quality")
    ax.set_ylabel("Jumlah baris")
    ax.set_title("Distribusi aktual vs prediksi per skor quality")
    ax.legend(loc="upper right")
    ax.text(
        0.02, 0.99,
        f"Tree hanya menghasilkan {len(unique_pred)} nilai prediksi\n"
        f"berbeda (rata-rata leaf), karena itu\n"
        f"prediksi tampak bertingkat.",
        transform=ax.transAxes, va="top", fontsize=8.5, color="#334155",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dtr_prediction_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_depth_curve(best_model, X_train, y_train, X_test, y_test,
                     cv, best_depth: int) -> dict:
    depths = list(range(1, 21))
    train_rmse, test_rmse, cv_rmse, cv_stds = [], [], [], []

    for depth in depths:
        model = clone(best_model).set_params(max_depth=depth)
        model.fit(X_train, y_train)
        train_rmse.append(root_mean_squared_error(y_train, model.predict(X_train)))
        test_rmse.append(root_mean_squared_error(y_test, model.predict(X_test)))
        scores = cross_val_score(
            clone(best_model).set_params(max_depth=depth),
            X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        cv_rmse.append(-scores.mean())
        cv_stds.append(scores.std())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(depths, train_rmse, "o-", color=PLOT_COLORS["blue"], label="RMSE train")
    ax.plot(depths, cv_rmse, "s-", color=PLOT_COLORS["green"], label="RMSE CV (5-fold)")
    ax.fill_between(depths,
                    np.array(cv_rmse) - np.array(cv_stds),
                    np.array(cv_rmse) + np.array(cv_stds),
                    color=PLOT_COLORS["green"], alpha=0.15)
    ax.plot(depths, test_rmse, "^-", color=PLOT_COLORS["red"], label="RMSE test")
    ax.axvline(best_depth, color=PLOT_COLORS["gray"], linestyle="--", linewidth=1,
               label=f"max_depth terpilih = {best_depth}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("RMSE")
    ax.set_title("Pengaruh max_depth pohon terhadap error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dtr_depth_curve.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "depths": depths,
        "train_rmse": [float(v) for v in train_rmse],
        "cv_rmse": [float(v) for v in cv_rmse],
        "cv_std": [float(v) for v in cv_stds],
        "test_rmse": [float(v) for v in test_rmse],
    }


def plot_feature_importance(best_model, X_test, y_test) -> dict:
    impurity = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    perm_series = pd.Series(perm.importances_mean, index=FEATURES).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].barh(impurity.index, impurity.values, color=PLOT_COLORS["blue"])
    axes[0].set_title("Impurity importance (MSE reduction)")
    axes[0].set_xlabel("Kontribusi")
    for i, value in enumerate(impurity.values):
        axes[0].text(value + 0.002, i, f"{value:.3f}", va="center", fontsize=8)

    axes[1].barh(perm_series.index, perm_series.values, color=PLOT_COLORS["orange"])
    axes[1].set_title("Permutation importance (RMSE, 10x)")
    axes[1].set_xlabel("Kenaikan RMSE saat fitur diacak")
    for i, value in enumerate(perm_series.values):
        axes[1].text(value + 0.001, i, f"{value:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "dtr_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "impurity": {k: float(v) for k, v in impurity.sort_values(ascending=False).items()},
        "permutation": {k: float(v) for k, v in perm_series.sort_values(ascending=False).items()},
    }


def plot_learning_curve(best_model, X_train, y_train, cv) -> dict:
    sizes, train_scores, cv_scores = learning_curve(
        best_model, X_train, y_train, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    train_rmse = -train_scores.mean(axis=1)
    cv_rmse = -cv_scores.mean(axis=1)
    cv_std = cv_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, train_rmse, "o-", color=PLOT_COLORS["blue"], label="RMSE train")
    ax.plot(sizes, cv_rmse, "s-", color=PLOT_COLORS["green"], label="RMSE CV (5-fold)")
    ax.fill_between(sizes, cv_rmse - cv_std, cv_rmse + cv_std,
                    color=PLOT_COLORS["green"], alpha=0.15)
    ax.set_xlabel("Jumlah data training")
    ax.set_ylabel("RMSE")
    ax.set_title("Learning curve — Decision Tree Regressor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dtr_learning_curve.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "train_sizes": [int(s) for s in sizes],
        "train_rmse": [float(v) for v in train_rmse],
        "cv_rmse": [float(v) for v in cv_rmse],
        "cv_std": [float(v) for v in cv_std],
    }


def plot_tree_figure(best_model) -> None:
    for depth, filename in [(2, "dtr_tree_depth2.png"), (3, "dtr_tree_depth3.png")]:
        fig, ax = plt.subplots(figsize=(16 if depth == 2 else 22, 8 if depth == 2 else 10))
        plot_tree(
            best_model,
            max_depth=depth,
            feature_names=FEATURES,
            filled=True,
            rounded=True,
            impurity=True,
            proportion=False,
            fontsize=9 if depth == 2 else 7,
            ax=ax,
        )
        ax.set_title(
            f"Struktur Decision Tree Regressor (max_depth {depth})",
            fontsize=13,
        )
        fig.savefig(FIG_DIR / filename, bbox_inches="tight")
        plt.close(fig)


# Main
def main() -> None:
    start = time.time()
    for directory in (FIG_DIR, METRIC_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    df = load_wine()
    plot_quality_distribution(df)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    print(f"\nTrain: {len(X_train)} baris | Test: {len(X_test)} baris")

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # baseline
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)

    default_tree = DecisionTreeRegressor(random_state=RANDOM_STATE)
    default_tree.fit(X_train, y_train)

    # tuning
    param_grid = {
        "criterion": ["squared_error", "absolute_error"],
        "max_depth": list(range(2, 15)),
        "min_samples_leaf": [1, 5, 20, 50],
        "ccp_alpha": [0.0, 0.0005, 0.002, 0.005],
    }
    n_combos = (
        len(param_grid["criterion"])
        * len(param_grid["max_depth"])
        * len(param_grid["min_samples_leaf"])
        * len(param_grid["ccp_alpha"])
    )
    print(f"\nGridSearchCV: {n_combos} kombinasi x {N_FOLDS} fold "
          f"(scoring = neg_root_mean_squared_error)")

    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_params = {k: (v if not isinstance(v, np.generic) else v.item())
                   for k, v in grid.best_params_.items()}
    print(f"Parameter terbaik: {best_params}")
    print(f"CV RMSE terbaik : {-grid.best_score_:.4f}")

    grid_mae = GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid_mae.fit(X_train, y_train)
    best_model_mae = grid_mae.best_estimator_
    best_params_mae = {k: (v if not isinstance(v, np.generic) else v.item())
                       for k, v in grid_mae.best_params_.items()}
    print(f"\nVarian MAE -> {best_params_mae}")
    print(f"CV MAE terbaik : {-grid_mae.best_score_:.4f}")

    # evaluasi
    model_results = {
        "dummy_mean": evaluate_model("Dummy (rata-rata)", dummy, X_test, y_test),
        "default_tree": evaluate_model("Decision Tree default", default_tree, X_test, y_test),
        "tuned_tree": evaluate_model("Decision Tree tuned (RMSE)", best_model, X_test, y_test),
        "tuned_tree_mae": evaluate_model("Decision Tree tuned (MAE)",
                                         best_model_mae, X_test, y_test),
    }

    cv_mae = -cross_val_score(best_model, X_train, y_train, cv=cv,
                              scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_rmse = -cross_val_score(best_model, X_train, y_train, cv=cv,
                               scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_r2 = cross_val_score(best_model, X_train, y_train, cv=cv,
                            scoring="r2", n_jobs=-1)
    print(f"\nCV model terbaik (5-fold):")
    print(f"  MAE  : {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
    print(f"  RMSE : {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"  R²   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    cv_mae_variant = -cross_val_score(best_model_mae, X_train, y_train, cv=cv,
                                      scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_rmse_variant = -cross_val_score(best_model_mae, X_train, y_train, cv=cv,
                                       scoring="neg_root_mean_squared_error", n_jobs=-1)

    # visualisasi
    y_pred = best_model.predict(X_test)
    metrics_test = model_results["tuned_tree"]

    depth_info = plot_depth_curve(best_model, X_train, y_train, X_test, y_test,
                                  cv, best_params["max_depth"])
    plot_actual_vs_predicted(y_test, y_pred, metrics_test)
    plot_residuals(y_test, y_pred)
    plot_prediction_distribution(y_test, y_pred)
    importance = plot_feature_importance(best_model, X_test, y_test)
    learning_info = plot_learning_curve(best_model, X_train, y_train, cv)
    plot_tree_figure(best_model)

    # aturan
    rules_depth3 = export_text(best_model, feature_names=FEATURES, max_depth=3, decimals=4)
    rules_full = export_text(best_model, feature_names=FEATURES, decimals=4)
    (METRIC_DIR / "dtr_rules_depth3.txt").write_text(rules_depth3, encoding="utf-8")
    (METRIC_DIR / "dtr_rules_full.txt").write_text(rules_full, encoding="utf-8")

    tree = best_model.tree_
    n_leaves = int(np.sum(tree.children_left == -1))
    actual_depth = int(best_model.get_depth())

    # simpan
    joblib.dump(best_model, MODEL_DIR / "decision_tree_regresi_best.joblib")

    cv_table = pd.DataFrame(grid.cv_results_)
    cv_table = cv_table.sort_values("rank_test_score").head(10)
    cv_table[
        ["rank_test_score", "mean_test_score", "std_test_score", "params",
         "mean_fit_time"]
    ].to_csv(METRIC_DIR / "dtr_grid_top10.csv", index=False)

    cv_table_mae = pd.DataFrame(grid_mae.cv_results_)
    cv_table_mae = cv_table_mae.sort_values("rank_test_score").head(10)
    cv_table_mae[
        ["rank_test_score", "mean_test_score", "std_test_score", "params",
         "mean_fit_time"]
    ].to_csv(METRIC_DIR / "dtr_grid_top10_mae.csv", index=False)

    payload = {
        "dataset": {
            "n_raw": int(len(pd.concat([
                pd.read_csv(RAW_DIR / f, sep=";") for f in
                ("winequality-red.csv", "winequality-white.csv")
            ]))),
            "n_unique": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "quality_mean": float(y.mean()),
            "quality_std": float(y.std()),
            "quality_min": int(y.min()),
            "quality_max": int(y.max()),
            "quality_counts": {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
        },
        "best_params": best_params,
        "best_params_mae_variant": best_params_mae,
        "cv_best_rmse": float(-grid.best_score_),
        "cv_best_mae_variant": float(-grid_mae.best_score_),
        "cv_tuned": {
            "mae_mean": float(cv_mae.mean()), "mae_std": float(cv_mae.std()),
            "rmse_mean": float(cv_rmse.mean()), "rmse_std": float(cv_rmse.std()),
            "r2_mean": float(cv_r2.mean()), "r2_std": float(cv_r2.std()),
        },
        "cv_tuned_mae_variant": {
            "mae_mean": float(cv_mae_variant.mean()),
            "mae_std": float(cv_mae_variant.std()),
            "rmse_mean": float(cv_rmse_variant.mean()),
            "rmse_std": float(cv_rmse_variant.std()),
        },
        "tree": {
            "depth": actual_depth,
            "leaves": n_leaves,
            "nodes": int(tree.node_count),
        },
        "metrics": model_results,
        "permutation_importance": importance["permutation"],
        "impurity_importance": importance["impurity"],
        "depth_curve": depth_info,
        "learning_curve": learning_info,
        "rules_depth3": rules_depth3,
        "runtime_seconds": float(time.time() - start),
    }
    with open(METRIC_DIR / "dtr_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # ringkas
    print("\n=== RINGKASAN MODEL TERBAIK (tuning RMSE) ===")
    print(f"  Kedalaman : {actual_depth}")
    print(f"  Leaf      : {n_leaves}")
    print(f"  MAE test  : {metrics_test['mae']:.4f}")
    print(f"  RMSE test : {metrics_test['rmse']:.4f}")
    print(f"  R² test   : {metrics_test['r2']:.4f}")
    print(f"  MAE bulat : {metrics_test['mae_rounded']:.4f}")
    print(f"  Akurasi ambang quality>={QUALITY_THRESHOLD}: "
          f"{metrics_test['accuracy_threshold']:.4f}")
    mae_variant = model_results["tuned_tree_mae"]
    print("\n=== VARIAN TUNING MAE ===")
    print(f"  MAE test  : {mae_variant['mae']:.4f}")
    print(f"  RMSE test : {mae_variant['rmse']:.4f}")
    print(f"  R² test   : {mae_variant['r2']:.4f}")

    print("\nArtefak:")
    for path in [
        FIG_DIR / "dtr_quality_distribution.png",
        FIG_DIR / "dtr_actual_vs_predicted.png",
        FIG_DIR / "dtr_residuals.png",
        FIG_DIR / "dtr_prediction_distribution.png",
        FIG_DIR / "dtr_depth_curve.png",
        FIG_DIR / "dtr_feature_importance.png",
        FIG_DIR / "dtr_learning_curve.png",
        FIG_DIR / "dtr_tree_depth2.png",
        FIG_DIR / "dtr_tree_depth3.png",
        METRIC_DIR / "dtr_metrics.json",
        METRIC_DIR / "dtr_grid_top10.csv",
        METRIC_DIR / "dtr_rules_full.txt",
        MODEL_DIR / "decision_tree_regresi_best.joblib",
    ]:
        print(f"  {path.relative_to(ROOT)}")

    print(f"\nSelesai dalam {time.time() - start:.1f} detik.")


if __name__ == "__main__":
    main()
