"""
Simulasi manual Decision Tree (depth 2) pada 8 baris data Wine Quality.

Tujuan
------
Menunjukkan perhitungan Gini, weighted Gini, dan information gain secara manual,
lalu memverifikasi hasilnya dengan DecisionTreeClassifier(max_depth=2) pada
8 baris yang sama.

Aturan yang dipakai (disamakan dengan scikit-learn):
- Kandidat ambang = titik tengah dua nilai unik bertetangga.
- Aturan split: x <= ambang -> kiri, x > ambang -> kanan.
- Split terbaik = information gain terbesar.
- Tie-break: gain terbesar, lalu fitur urutan pertama (alcohol sebelum
  volatile acidity), lalu ambang terkecil.
- Berhenti jika node murni, tidak ada kandidat, atau kedalaman maksimum tercapai.

Jalankan:
    python3 src/supervised/simulasi_manual_dt.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
DATA_OUT = RESULTS_DIR / "data"
METRICS_OUT = RESULTS_DIR / "metrics"

FEATURES = ["alcohol", "volatile acidity"]
TARGET = "target"
MAX_DEPTH = 2
BASE_SEED = 42


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
    df[TARGET] = (df["quality"] >= 6).astype(int)
    return df


def pick_sample(df: pd.DataFrame, seed: int, n_per_class: int = 4) -> pd.DataFrame:
    """Ambil sampel terstratifikasi: 4 kelas baik + 4 kelas buruk."""
    parts = [
        df[df[TARGET] == cls].sample(n=n_per_class, random_state=seed)
        for cls in (0, 1)
    ]
    sample = pd.concat(parts).sort_values(FEATURES).reset_index(drop=True)
    return sample


def is_instructive(sample: pd.DataFrame) -> bool:
    """Pohon depth-2 baru informatif kalau kedua anak root masih impure."""
    x = sample[FEATURES].to_numpy()
    y = sample[TARGET].to_numpy()
    root = best_split(x, y)
    if root is None:
        return False
    left, right = apply_split(sample, root)
    return (
        len(left) >= 2
        and len(right) >= 2
        and left[TARGET].nunique() > 1
        and right[TARGET].nunique() > 1
        and best_split(left[FEATURES].to_numpy(), left[TARGET].to_numpy()) is not None
        and best_split(right[FEATURES].to_numpy(), right[TARGET].to_numpy()) is not None
    )


# 2. Perhitungan manual
def gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p1 = y.mean()
    p0 = 1.0 - p1
    return float(1.0 - p0**2 - p1**2)


def candidate_thresholds(values: np.ndarray) -> list[float]:
    uniq = np.unique(values)
    if len(uniq) < 2:
        return []
    return [float((a + b) / 2.0) for a, b in zip(uniq[:-1], uniq[1:])]


def evaluate_split(x: np.ndarray, y: np.ndarray, feature_idx: int, threshold: float) -> dict:
    mask = x[:, feature_idx] <= threshold
    y_left, y_right = y[mask], y[~mask]
    n = len(y)
    g_parent = gini(y)
    g_left, g_right = gini(y_left), gini(y_right)
    weighted = (len(y_left) / n) * g_left + (len(y_right) / n) * g_right
    return {
        "feature": FEATURES[feature_idx],
        "feature_idx": feature_idx,
        "threshold": threshold,
        "n_left": int(mask.sum()),
        "n_right": int((~mask).sum()),
        "good_left": int(y_left.sum()),
        "good_right": int(y_right.sum()),
        "gini_parent": g_parent,
        "gini_left": g_left,
        "gini_right": g_right,
        "gini_weighted": weighted,
        "gain": g_parent - weighted,
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
    # gain terbesar; tie-break fitur terdahulu, lalu ambang terkecil
    return sorted(
        candidates,
        key=lambda r: (-round(r["gain"], 12), r["feature_idx"], r["threshold"]),
    )[0]


def apply_split(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df[split["feature"]] <= split["threshold"]
    return df[mask].copy(), df[~mask].copy()


# 3. Bangun pohon depth-2 manual
def build_tree(df: pd.DataFrame, depth: int = 0, max_depth: int = MAX_DEPTH) -> dict:
    y = df[TARGET].to_numpy()
    node = {
        "depth": depth,
        "n": len(df),
        "good": int(y.sum()),
        "bad": int(len(y) - y.sum()),
        "gini": gini(y),
        "prediction": int(round(y.mean())),
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
        return [f"JIKA {condition} MAKA prediksi={'baik' if tree['prediction'] else 'buruk'} "
                f"(n={tree['n']}, baik={tree['good']}, buruk={tree['bad']})"]
    cond = f"{tree['feature']} <= {tree['threshold']:.4f}"
    left_path = f"{path} DAN ({cond})" if path else cond
    right_path = f"{path} DAN BUKAN ({cond})" if path else f"BUKAN ({cond})"
    return rules(tree["left"], left_path) + rules(tree["right"], right_path)


# 4. Verifikasi sklearn
def sklearn_tree(sample: pd.DataFrame) -> tuple[DecisionTreeClassifier, str]:
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=MAX_DEPTH,
        min_samples_leaf=1,
        random_state=BASE_SEED,
    )
    model.fit(sample[FEATURES], sample[TARGET])
    text = export_text(model, feature_names=FEATURES, decimals=4)
    return model, text


def describe_sklearn_tree(model: DecisionTreeClassifier) -> list[dict]:
    t = model.tree_
    depths = node_depths(t)
    rows = []
    for node_id in range(t.node_count):
        left = int(t.children_left[node_id])
        right = int(t.children_right[node_id])
        n = int(t.n_node_samples[node_id])
        vals = np.asarray(t.value[node_id][0], dtype=float)
        # sklearn >= 1.9 menyimpan proporsi; versi lama menyimpan jumlah sampel
        counts = vals * n if abs(vals.sum() - 1.0) < 1e-9 else vals
        rows.append(
            {
                "node": node_id,
                "depth": depths[node_id],
                "n": n,
                "gini": float(t.impurity[node_id]),
                "feature": None if t.feature[node_id] < 0 else FEATURES[t.feature[node_id]],
                "threshold": None if t.threshold[node_id] < 0 else float(t.threshold[node_id]),
                "left_child": None if left < 0 else left,
                "right_child": None if right < 0 else right,
                "prediction": int(np.argmax(counts)),
                "counts": [int(round(c)) for c in counts],
            }
        )
    return rows


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


def compare_trees(tree: dict, model: DecisionTreeClassifier) -> list[dict]:
    """Bandingkan setiap internal node manual dengan node sklearn yang sepadan."""
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
        # anak kiri/kanan sklearn mengikuti urutan yang sama dengan manual
        stack.append((node["left"], int(t.children_left[node_id])))
        stack.append((node["right"], int(t.children_right[node_id])))
    return rows


# 5. Visualisasi & format tabel
def plot_manual_tree(tree: dict, out_path: Path) -> None:
    """Gambar pohon hasil perhitungan manual (depth 2) untuk lampiran laporan."""
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
    # urutan flatten = preorder: root, anak kiri + subtree, anak kanan + subtree
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
            color = "#bbf7d0" if node["prediction"] == 1 else "#fecaca"
            label = "BAIK (1)" if node["prediction"] == 1 else "BURUK (0)"
            text = f"{label}\nGini={node['gini']:.3f}\nn={node['n']} [B={node['bad']}, G={node['good']}]"
        else:
            color = "#dbeafe" if idx == 0 else "#e0e7ff"
            text = (f"{node['feature']} ≤ {node['threshold']:.4f}\n"
                    f"Gini={node['gini']:.3f} · n={node['n']} [B={node['bad']}, G={node['good']}]")
        ax.text(
            x, y, text, ha="center", va="center", fontsize=9.5, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=color, edgecolor="#334155", linewidth=1.1),
        )

    # garis penghubung + label cabang
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
        "Pohon Decision Tree hasil perhitungan manual (max_depth = 2, data 8 baris)",
        fontsize=13, pad=12,
    )
    ax.text(
        0.5, 0.02,
        "B = jumlah baris berlabel buruk (quality < 6), G = jumlah baris berlabel baik (quality ≥ 6).",
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
    print(f"Data unik: {len(df)} baris | baik={int(df[TARGET].sum())} buruk={int((1 - df[TARGET]).sum())}")

    # cari seed yang menghasilkan pohon depth-2 dengan dua anak root yang masih impure
    sample = None
    used_seed = None
    for seed in range(BASE_SEED, BASE_SEED + 500):
        candidate = pick_sample(df, seed)
        if is_instructive(candidate):
            sample = candidate
            used_seed = seed
            break
    if sample is None:
        raise RuntimeError("Tidak menemukan sampel yang informatif; periksa data.")

    sample_display = sample.copy()
    sample_display["quality"] = sample_display["quality"].astype(int)
    sample_display["label"] = sample_display[TARGET].map({0: "buruk (0)", 1: "baik (1)"})
    sample_display = sample_display[["alcohol", "volatile acidity", "quality", "label"]]
    sample_display.to_csv(DATA_OUT / "simulasi_dt_8baris.csv", index=False)

    print(f"\nSeed sampel: {used_seed} (dicari agar kedua anak root masih impure)")
    print("\n=== TABEL 1. Delapan baris data ===")
    print(sample_display.to_string(index=False))

    y = sample[TARGET].to_numpy()
    x = sample[FEATURES].to_numpy()
    parent_gini = gini(y)
    print(f"\nGini parent = 1 - (4/8)^2 - (4/8)^2 = {parent_gini:.4f}")

    print("\n=== TABEL 2. Semua kandidat split di root ===")
    cands = all_candidates(x, y)
    cands_sorted = sorted(cands, key=lambda r: (-round(r["gain"], 12), r["feature_idx"], r["threshold"]))
    for i, row in enumerate(cands_sorted, 1):
        row["rank"] = i
        row["left_dist"] = f"{row['n_left']} ({row['good_left']}/{row['n_left'] - row['good_left']})"
        row["right_dist"] = f"{row['n_right']} ({row['good_right']}/{row['n_right'] - row['good_right']})"
    print(md_table(
        cands_sorted,
        [
            ("rank", "Peringkat"),
            ("feature", "Fitur"),
            ("threshold", "Ambang"),
            ("left_dist", "Kiri n (baik/buruk)"),
            ("right_dist", "Kanan n (baik/buruk)"),
            ("gini_left", "Gini kiri"),
            ("gini_right", "Gini kanan"),
            ("gini_weighted", "Gini terbobot"),
            ("gain", "Information gain"),
        ],
        decimals={"threshold": 4, "gini_left": 4, "gini_right": 4,
                  "gini_weighted": 4, "gain": 4},
    ))

    root = best_split(x, y)
    print(f"\nRoot terpilih: {root['feature']} <= {root['threshold']:.4f} "
          f"(IG = {root['gain']:.4f})")

    left_df, right_df = apply_split(sample, root)
    print(f"  Kiri : n={len(left_df)}, baik={int(left_df[TARGET].sum())}, buruk={int((1-left_df[TARGET]).sum())}")
    print(f"  Kanan: n={len(right_df)}, baik={int(right_df[TARGET].sum())}, buruk={int((1-right_df[TARGET]).sum())}")

    for name, part in [("KIRI", left_df), ("KANAN", right_df)]:
        print(f"\n=== TABEL 3. Kandidat split level 2 - anak {name} (n={len(part)}, "
              f"Gini={gini(part[TARGET].to_numpy()):.4f}) ===")
        sub_cands = all_candidates(part[FEATURES].to_numpy(), part[TARGET].to_numpy())
        sub_sorted = sorted(sub_cands, key=lambda r: (-round(r["gain"], 12), r["feature_idx"], r["threshold"]))
        for i, row in enumerate(sub_sorted, 1):
            row["rank"] = i
            row["left_dist"] = f"{row['n_left']} ({row['good_left']}/{row['n_left'] - row['good_left']})"
            row["right_dist"] = f"{row['n_right']} ({row['good_right']}/{row['n_right'] - row['good_right']})"
        print(md_table(
            sub_sorted,
            [
                ("rank", "Peringkat"),
                ("feature", "Fitur"),
                ("threshold", "Ambang"),
                ("left_dist", "Kiri n (baik/buruk)"),
                ("right_dist", "Kanan n (baik/buruk)"),
                ("gini_left", "Gini kiri"),
                ("gini_right", "Gini kanan"),
                ("gini_weighted", "Gini terbobot"),
                ("gain", "Information gain"),
            ],
            decimals={"threshold": 4, "gini_left": 4, "gini_right": 4,
                      "gini_weighted": 4, "gain": 4},
        ))
        best_sub = best_split(part[FEATURES].to_numpy(), part[TARGET].to_numpy())
        print(f"  -> Split terpilih: {best_sub['feature']} <= {best_sub['threshold']:.4f} "
              f"(IG = {best_sub['gain']:.4f})")

    tree = build_tree(sample)
    print("\n=== ATURAN HASIL PERHITUNGAN MANUAL ===")
    for rule in rules(tree):
        print(" ", rule)

    fig_path = RESULTS_DIR / "figures" / "dt_manual_tree.png"
    plot_manual_tree(tree, fig_path)

    model, text = sklearn_tree(sample)
    print("\n=== VERIFIKASI: export_text sklearn ===")
    print(text)

    sk_rows = describe_sklearn_tree(model)
    print("=== STRUKTUR INTERNAL SKLEARN ===")
    for row in sk_rows:
        row["counts_text"] = f"[{row['counts'][0]}, {row['counts'][1]}]"
    print(md_table(
        sk_rows,
        [
            ("node", "Node"),
            ("depth", "Depth"),
            ("n", "n"),
            ("gini", "Gini"),
            ("feature", "Fitur"),
            ("threshold", "Ambang"),
            ("left_child", "Anak kiri"),
            ("right_child", "Anak kanan"),
            ("prediction", "Prediksi"),
            ("counts_text", "Value [buruk, baik]"),
        ],
        decimals={"gini": 4, "threshold": 4},
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

    # bandingkan root
    sk_root_feature = FEATURES[model.tree_.feature[0]]
    sk_root_threshold = float(model.tree_.threshold[0])
    cocok = all(row["match"] for row in comparison)
    print(f"\nRoot manual : {root['feature']} <= {root['threshold']:.6f}")
    print(f"Root sklearn: {sk_root_feature} <= {sk_root_threshold:.6f}")
    print(f"Hasil semua split: {'COCOK' if cocok else 'TIDAK COCOK'} "
          f"({sum(r['match'] for r in comparison)}/{len(comparison)} split sama)")

    payload = {
        "seed": used_seed,
        "rows": sample_display.to_dict(orient="records"),
        "parent_gini": parent_gini,
        "root": root,
        "tree_manual": tree,
        "rules_manual": rules(tree),
        "sklearn_text": text,
        "sklearn_nodes": sk_rows,
        "comparison": comparison,
        "match": bool(cocok),
    }
    with open(METRICS_OUT / "simulasi_manual_dt.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nDisimpan: {DATA_OUT / 'simulasi_dt_8baris.csv'}")
    print(f"Disimpan: {METRICS_OUT / 'simulasi_manual_dt.json'}")


if __name__ == "__main__":
    main()
