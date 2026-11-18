"""
Simulasi manual Decision Tree Regressor (depth 2) pada 8 baris Wine Quality.

Tujuan
------
Menunjukkan perhitungan variance / MSE, weighted MSE, dan variance reduction
secara manual, lalu memverifikasi hasilnya dengan
DecisionTreeRegressor(max_depth=2) pada 8 baris yang sama.

Aturan yang dipakai (disamakan dengan scikit-learn, criterion="squared_error"):
- Kandidat ambang = titik tengah dua nilai unik bertetangga.
- Aturan split: x <= ambang -> kiri, x > ambang -> kanan.
- Split terbaik = weighted MSE terkecil (= variance reduction terbesar).
- Prediksi leaf = rata-rata nilai target di leaf tersebut.
- Tie-break: MSE terkecil, lalu fitur urutan pertama, lalu ambang terkecil.
- Berhenti jika node homogen, tidak ada kandidat, atau kedalaman maksimum.

Jalankan:
    python3 src/supervised/simulasi_manual_dtr.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
DATA_OUT = RESULTS_DIR / "data"
METRICS_OUT = RESULTS_DIR / "metrics"

FEATURES = ["alcohol", "volatile acidity"]
TARGET = "quality"
MAX_DEPTH = 2
BASE_SEED = 42
QUALITY_LEVELS = [4, 5, 6, 7]


# 1. Data
def load_wine() -> pd.DataFrame:
    frames = []
    for filename, wine_type in [
        ("winequality-red.csv", "red"),
        ("winequality-white.csv", "white"),
    ]:
        df = pd.read_csv(RAW_DIR / filename, sep=";")
        df["wine_type"] = wine_type
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def pick_sample(df: pd.DataFrame, seed: int, n_per_level: int = 2) -> pd.DataFrame:
    """Ambil sampel merata: 2 baris untuk setiap level quality 4, 5, 6, 7."""
    parts = [
        df[df[TARGET] == level].sample(n=n_per_level, random_state=seed)
        for level in QUALITY_LEVELS
    ]
    sample = pd.concat(parts).sort_values(FEATURES).reset_index(drop=True)
    return sample


def has_unique_best(x: np.ndarray, y: np.ndarray) -> bool:
    """Split terbaik harus unik (tidak seri), agar tie-break sklearn tidak ambigu."""
    cands = sorted(all_candidates(x, y), key=lambda r: r["mse_weighted"])
    if len(cands) < 2:
        return False
    return cands[0]["mse_weighted"] < cands[1]["mse_weighted"] - 1e-9


def has_mixed_leaf(tree: dict) -> bool:
    """Minimal satu leaf berisi lebih dari satu nilai berbeda, supaya
    prediksi rata-rata (mean) benar-benar terpakai."""
    if tree["left"] is None:
        return tree["n"] >= 2 and tree["mse"] > 0
    return has_mixed_leaf(tree["left"]) or has_mixed_leaf(tree["right"])


def is_instructive(sample: pd.DataFrame) -> bool:
    """Pohon depth-2 informatif kalau root dan kedua anaknya masih bisa di-split,
    setiap split terbaiknya tidak seri, dan ada leaf yang nilainya bervariasi."""
    x = sample[FEATURES].to_numpy()
    y = sample[TARGET].to_numpy(dtype=float)
    root = best_split(x, y)
    if root is None or not has_unique_best(x, y):
        return False
    left, right = apply_split(sample, root)
    if not (len(left) >= 2 and len(right) >= 2):
        return False
    if left[TARGET].nunique() < 2 or right[TARGET].nunique() < 2:
        return False
    x_left = left[FEATURES].to_numpy()
    x_right = right[FEATURES].to_numpy()
    if not (
        has_unique_best(x_left, left[TARGET].to_numpy(dtype=float))
        and has_unique_best(x_right, right[TARGET].to_numpy(dtype=float))
    ):
        return False
    return has_mixed_leaf(build_tree(sample))


# 2. Perhitungan manual
def mse(y: np.ndarray) -> float:
    """Variance (MSE) sebuah node: rata-rata kuadrat simpangan terhadap mean."""
    if len(y) == 0:
        return 0.0
    return float(np.mean((y - y.mean()) ** 2))


def candidate_thresholds(values: np.ndarray) -> list[float]:
    uniq = np.unique(values)
    if len(uniq) < 2:
        return []
    return [float((a + b) / 2.0) for a, b in zip(uniq[:-1], uniq[1:])]


def evaluate_split(x: np.ndarray, y: np.ndarray, feature_idx: int, threshold: float) -> dict:
    mask = x[:, feature_idx] <= threshold
    y_left, y_right = y[mask], y[~mask]
    n = len(y)
    m_parent = mse(y)
    m_left, m_right = mse(y_left), mse(y_right)
    weighted = (len(y_left) / n) * m_left + (len(y_right) / n) * m_right
    return {
        "feature": FEATURES[feature_idx],
        "feature_idx": feature_idx,
        "threshold": threshold,
        "n_left": int(mask.sum()),
        "n_right": int((~mask).sum()),
        "mean_left": float(y_left.mean()),
        "mean_right": float(y_right.mean()),
        "mse_parent": m_parent,
        "mse_left": m_left,
        "mse_right": m_right,
        "mse_weighted": weighted,
        "gain": m_parent - weighted,
    }


def all_candidates(x: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for idx in range(x.shape[1]):
        for threshold in candidate_thresholds(x[:, idx]):
            rows.append(evaluate_split(x, y, idx, threshold))
    return rows


def best_split(x: np.ndarray, y: np.ndarray) -> dict | None:
    candidates = all_candidates(x, y)
    if not candidates:
        return None
    # MSE terkecil; tie-break fitur terdahulu, lalu ambang terkecil
    return sorted(
        candidates,
        key=lambda r: (round(r["mse_weighted"], 12), r["feature_idx"], r["threshold"]),
    )[0]


def apply_split(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df[split["feature"]] <= split["threshold"]
    return df[mask].copy(), df[~mask].copy()


# 3. Bangun pohon depth-2 manual
def build_tree(df: pd.DataFrame, depth: int = 0, max_depth: int = MAX_DEPTH) -> dict:
    y = df[TARGET].to_numpy(dtype=float)
    node = {
        "depth": depth,
        "n": len(df),
        "mean": float(y.mean()),
        "mse": mse(y),
        "feature": None,
        "threshold": None,
        "left": None,
        "right": None,
    }
    if depth >= max_depth or len(np.unique(y)) == 1:
        return node

    split = best_split(df[FEATURES].to_numpy(), y)
    if split is None:
        return node

    node["feature"] = split["feature"]
    node["threshold"] = split["threshold"]
    node["gain"] = split["gain"]
    left_df, right_df = apply_split(df, split)
    node["left"] = build_tree(left_df, depth + 1, max_depth)
    node["right"] = build_tree(right_df, depth + 1, max_depth)
    return node


def rules(tree: dict, path: str = "") -> list[str]:
    if tree["left"] is None:
        condition = path if path else "seluruh data"
        return [f"JIKA {condition} MAKA prediksi = {tree['mean']:.4f} "
                f"(n={tree['n']}, MSE={tree['mse']:.4f})"]
    cond = f"{tree['feature']} <= {tree['threshold']:.4f}"
    left_path = f"{path} DAN ({cond})" if path else cond
    right_path = f"{path} DAN BUKAN ({cond})" if path else f"BUKAN ({cond})"
    return rules(tree["left"], left_path) + rules(tree["right"], right_path)


# 4. Verifikasi sklearn
def sklearn_tree(sample: pd.DataFrame) -> tuple[DecisionTreeRegressor, str]:
    model = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=MAX_DEPTH,
        min_samples_leaf=1,
        random_state=BASE_SEED,
    )
    model.fit(sample[FEATURES], sample[TARGET])
    text = export_text(model, feature_names=FEATURES, decimals=4)
    return model, text


def node_depths(t) -> dict[int, int]:
    depths = {0: 0}
    stack = [0]
    while stack:
        node = stack.pop()
        for child in (t.children_left[node], t.children_right[node]):
            if child != -1:
                depths[int(child)] = depths[node] + 1
                stack.append(int(child))
    return depths


def describe_sklearn_tree(model: DecisionTreeRegressor) -> list[dict]:
    t = model.tree_
    depths = node_depths(t)
    rows = []
    for node_id in range(t.node_count):
        left = int(t.children_left[node_id])
        right = int(t.children_right[node_id])
        rows.append(
            {
                "node": node_id,
                "depth": depths[node_id],
                "n": int(t.n_node_samples[node_id]),
                "mse": float(t.impurity[node_id]),
                "feature": None if t.feature[node_id] < 0 else FEATURES[t.feature[node_id]],
                "threshold": None if t.threshold[node_id] < 0 else float(t.threshold[node_id]),
                "left_child": None if left < 0 else left,
                "right_child": None if right < 0 else right,
                "prediction": float(t.value[node_id].ravel()[0]),
            }
        )
    return rows


def compare_trees(tree: dict, model: DecisionTreeRegressor) -> list[dict]:
    """Bandingkan setiap split manual dengan node sklearn yang sepadan."""
    t = model.tree_
    rows = []
    stack = [(tree, 0)]
    while stack:
        node, node_id = stack.pop()
        if node["left"] is None:
            continue
        sk_feature = FEATURES[t.feature[node_id]]
        sk_threshold = float(t.threshold[node_id])
        rows.append(
            {
                "manual_feature": node["feature"],
                "manual_threshold": node["threshold"],
                "sklearn_feature": sk_feature,
                "sklearn_threshold": sk_threshold,
                "match": (
                    node["feature"] == sk_feature
                    and abs(node["threshold"] - sk_threshold) < 1e-6
                ),
            }
        )
        stack.append((node["left"], int(t.children_left[node_id])))
        stack.append((node["right"], int(t.children_right[node_id])))
    return rows


# 5. Visualisasi & format tabel
def plot_manual_tree(tree: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes: list[dict] = []

    def flatten(node: dict) -> None:
        nodes.append(node)
        if node["left"] is not None:
            flatten(node["left"])
            flatten(node["right"])

    flatten(tree)
    positions = [
        (0.50, 0.87),   # root
        (0.28, 0.56),   # anak kiri
        (0.12, 0.24),   # cucu kiri-kiri
        (0.40, 0.24),   # cucu kiri-kanan
        (0.72, 0.56),   # anak kanan
        (0.60, 0.24),   # cucu kanan-kiri
        (0.88, 0.24),   # cucu kanan-kanan
    ]

    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for idx, node in enumerate(nodes):
        x, y = positions[idx]
        if node["left"] is None:
            color = "#fde68a"
            text = (f"PREDIKSI = {node['mean']:.3f}\n"
                    f"MSE={node['mse']:.3f}\nn={node['n']}")
        else:
            color = "#dbeafe" if idx == 0 else "#e0e7ff"
            text = (f"{node['feature']} ≤ {node['threshold']:.4f}\n"
                    f"MSE={node['mse']:.3f} · mean={node['mean']:.3f} · n={node['n']}")
        ax.text(
            x, y, text, ha="center", va="center", fontsize=9.5, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=color,
                      edgecolor="#334155", linewidth=1.1),
        )

    for idx, node in enumerate(nodes):
        if node["left"] is None:
            continue
        px, py = positions[idx]
        lidx = idx + 1
        ridx = lidx + count_nodes(node["left"])

        for child_idx, side in [(lidx, "kiri"), (ridx, "kanan")]:
            cx, cy = positions[child_idx]
            ax.plot([px, cx], [py - 0.055, cy + 0.055], color="#64748b", linewidth=1.2)
            label = "ya" if side == "kiri" else "tidak"
            ax.text(
                (px + cx) / 2, (py + cy) / 2, label, fontsize=8.5, color="#475569",
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
            )

    ax.set_title(
        "Pohon Decision Tree Regressor hasil perhitungan manual "
        "(max_depth = 2, data 8 baris)",
        fontsize=13, pad=12,
    )
    ax.text(
        0.5, 0.02,
        "Setiap leaf memprediksi rata-rata quality di leaf tersebut; "
        "MSE = rata-rata kuadrat simpangan terhadap rata-rata.",
        ha="center", fontsize=9, color="#475569",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def count_nodes(node: dict) -> int:
    if node["left"] is None:
        return 1
    return 1 + count_nodes(node["left"]) + count_nodes(node["right"])


def md_table(rows: list[dict], columns: list[tuple[str, str]], decimals: dict | None = None) -> str:
    decimals = decimals or {}
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                digits = decimals.get(key, 4)
                cells.append(f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.mkdir(parents=True, exist_ok=True)

    df = load_wine()
    print(f"Data unik: {len(df)} baris | quality {df[TARGET].min()}-{df[TARGET].max()} | "
          f"rata-rata={df[TARGET].mean():.4f}")

    sample = None
    used_seed = None
    for seed in range(BASE_SEED, BASE_SEED + 5000):
        candidate = pick_sample(df, seed)
        if is_instructive(candidate):
            sample = candidate
            used_seed = seed
            break
    if sample is None:
        raise RuntimeError("Tidak menemukan sampel yang informatif; periksa data.")

    sample_display = sample.copy()
    sample_display[TARGET] = sample_display[TARGET].astype(int)
    sample_display = sample_display[["alcohol", "volatile acidity", TARGET]]
    sample_display.to_csv(DATA_OUT / "simulasi_dtr_8baris.csv", index=False)

    print(f"\nSeed sampel: {used_seed} (dicari agar kedua anak root masih bisa di-split)")
    print("\n=== TABEL 1. Delapan baris data ===")
    print(sample_display.to_string(index=False))

    y = sample[TARGET].to_numpy(dtype=float)
    x = sample[FEATURES].to_numpy()
    parent_mse = mse(y)
    print(f"\nMean parent = {y.mean():.4f}")
    print(f"MSE parent  = (1/8) * Σ(yi - 5.5)^2 = {parent_mse:.4f}")

    print("\n=== TABEL 2. Semua kandidat split di root ===")
    cands = all_candidates(x, y)
    cands_sorted = sorted(
        cands, key=lambda r: (round(r["mse_weighted"], 12), r["feature_idx"], r["threshold"])
    )
    for i, row in enumerate(cands_sorted, 1):
        row["rank"] = i
        row["left_text"] = f"{row['n_left']} (mean {row['mean_left']:.3f})"
        row["right_text"] = f"{row['n_right']} (mean {row['mean_right']:.3f})"
    print(md_table(
        cands_sorted,
        [
            ("rank", "Peringkat"),
            ("feature", "Fitur"),
            ("threshold", "Ambang"),
            ("left_text", "Kiri n (mean)"),
            ("right_text", "Kanan n (mean)"),
            ("mse_left", "MSE kiri"),
            ("mse_right", "MSE kanan"),
            ("mse_weighted", "MSE terbobot"),
            ("gain", "Variance reduction"),
        ],
        decimals={"threshold": 4, "mse_left": 4, "mse_right": 4,
                  "mse_weighted": 4, "gain": 4},
    ))

    root = best_split(x, y)
    print(f"\nRoot terpilih: {root['feature']} <= {root['threshold']:.4f} "
          f"(MSE terbobot = {root['mse_weighted']:.4f}, gain = {root['gain']:.4f})")

    left_df, right_df = apply_split(sample, root)
    print(f"  Kiri : n={len(left_df)}, mean={left_df[TARGET].mean():.4f}")
    print(f"  Kanan: n={len(right_df)}, mean={right_df[TARGET].mean():.4f}")

    for name, part in [("KIRI", left_df), ("KANAN", right_df)]:
        print(f"\n=== TABEL 3. Kandidat split level 2 - anak {name} (n={len(part)}, "
              f"MSE={mse(part[TARGET].to_numpy(dtype=float)):.4f}) ===")
        sub_cands = all_candidates(part[FEATURES].to_numpy(),
                                   part[TARGET].to_numpy(dtype=float))
        sub_sorted = sorted(
            sub_cands,
            key=lambda r: (round(r["mse_weighted"], 12), r["feature_idx"], r["threshold"]),
        )
        for i, row in enumerate(sub_sorted, 1):
            row["rank"] = i
            row["left_text"] = f"{row['n_left']} (mean {row['mean_left']:.3f})"
            row["right_text"] = f"{row['n_right']} (mean {row['mean_right']:.3f})"
        print(md_table(
            sub_sorted,
            [
                ("rank", "Peringkat"),
                ("feature", "Fitur"),
                ("threshold", "Ambang"),
                ("left_text", "Kiri n (mean)"),
                ("right_text", "Kanan n (mean)"),
                ("mse_left", "MSE kiri"),
                ("mse_right", "MSE kanan"),
                ("mse_weighted", "MSE terbobot"),
                ("gain", "Variance reduction"),
            ],
            decimals={"threshold": 4, "mse_left": 4, "mse_right": 4,
                      "mse_weighted": 4, "gain": 4},
        ))
        best_sub = best_split(part[FEATURES].to_numpy(),
                              part[TARGET].to_numpy(dtype=float))
        print(f"  -> Split terpilih: {best_sub['feature']} <= {best_sub['threshold']:.4f} "
              f"(MSE terbobot = {best_sub['mse_weighted']:.4f})")

    tree = build_tree(sample)
    print("\n=== ATURAN HASIL PERHITUNGAN MANUAL ===")
    for rule in rules(tree):
        print(" ", rule)

    fig_path = RESULTS_DIR / "figures" / "dtr_manual_tree.png"
    plot_manual_tree(tree, fig_path)

    model, text = sklearn_tree(sample)
    print("\n=== VERIFIKASI: export_text sklearn ===")
    print(text)

    sk_rows = describe_sklearn_tree(model)
    print("=== STRUKTUR INTERNAL SKLEARN ===")
    print(md_table(
        sk_rows,
        [
            ("node", "Node"),
            ("depth", "Depth"),
            ("n", "n"),
            ("mse", "MSE"),
            ("feature", "Fitur"),
            ("threshold", "Ambang"),
            ("left_child", "Anak kiri"),
            ("right_child", "Anak kanan"),
            ("prediction", "Prediksi (mean)"),
        ],
        decimals={"mse": 4, "threshold": 4, "prediction": 4},
    ))

    comparison = compare_trees(tree, model)
    print("\n=== PERBANDINGAN SEMUA SPLIT: MANUAL vs SKLEARN ===")
    print(md_table(
        comparison,
        [
            ("manual_feature", "Fitur manual"),
            ("manual_threshold", "Ambang manual"),
            ("sklearn_feature", "Fitur sklearn"),
            ("sklearn_threshold", "Ambang sklearn"),
            ("match", "Cocok"),
        ],
        decimals={"manual_threshold": 4, "sklearn_threshold": 4},
    ))

    sk_root_feature = FEATURES[model.tree_.feature[0]]
    sk_root_threshold = float(model.tree_.threshold[0])
    cocok = all(row["match"] for row in comparison)

    # bandingkan prediksi leaf manual vs sklearn
    pred_manual = manual_predict(tree, sample)
    pred_sklearn = model.predict(sample[FEATURES])
    max_diff = float(np.max(np.abs(pred_manual - pred_sklearn)))

    print(f"\nRoot manual : {root['feature']} <= {root['threshold']:.6f}")
    print(f"Root sklearn: {sk_root_feature} <= {sk_root_threshold:.6f}")
    print(f"Hasil semua split: {'COCOK' if cocok else 'TIDAK COCOK'} "
          f"({sum(r['match'] for r in comparison)}/{len(comparison)} split sama)")
    print(f"Selisih prediksi leaf manual vs sklearn: maksimum {max_diff:.2e}")

    payload = {
        "seed": used_seed,
        "rows": sample_display.to_dict(orient="records"),
        "parent_mean": float(y.mean()),
        "parent_mse": parent_mse,
        "root": root,
        "tree_manual": tree,
        "rules_manual": rules(tree),
        "sklearn_text": text,
        "sklearn_nodes": sk_rows,
        "comparison": comparison,
        "match": bool(cocok and max_diff < 1e-9),
        "max_leaf_prediction_diff": max_diff,
        "prediksi_manual": pred_manual.tolist(),
        "prediksi_sklearn": pred_sklearn.tolist(),
    }
    with open(METRICS_OUT / "simulasi_manual_dtr.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nDisimpan: {DATA_OUT / 'simulasi_dtr_8baris.csv'}")
    print(f"Disimpan: {METRICS_OUT / 'simulasi_manual_dtr.json'}")
    print(f"Disimpan: {fig_path}")


def manual_predict(tree: dict, df: pd.DataFrame) -> np.ndarray:
    preds = []
    for _, row in df.iterrows():
        node = tree
        while node["left"] is not None:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        preds.append(node["mean"])
    return np.asarray(preds)


if __name__ == "__main__":
    main()
