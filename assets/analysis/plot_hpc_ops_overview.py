"""Generate analysis charts for the hpc-ops repository.

Outputs four 1x2 / 2x2 combined PNGs into the same directory.
Numbers come from README.md (Performance table) and CMakeLists/source layout.
"""

from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

OUT_DIR = Path(__file__).parent
plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# Figure 1: Performance overview — peak speedup + operator coverage
# ---------------------------------------------------------------------------
PERF_ROWS = [
    # (operator, peak_speedup, baseline_label)
    ("Sampler",                    8.50, "vs vLLM/FlashInfer"),
    ("BF16xFP32 GEMM",             3.22, "vs cuBLAS FP32/TF32"),
    ("Sparse Attn FP8",            3.16, "vs MIT-BSA / FA3"),
    ("Dynamic Decode Attn",        2.88, "vs static split-k"),
    ("Attn BF16 (decode)",         2.22, "vs FlashInfer/FA"),
    ("Attn FP8 (decode)",          2.00, "vs FlashInfer/FA3"),
    ("Group GEMM FP8 (decode)",    1.88, "vs DeepGEMM"),
    ("AllReduce+RMSNorm",          1.76, "vs NCCL/FlashInfer"),
    ("Fused MoE (TP)",             1.60, "vs vLLM/SGLang"),
    ("Fused MoE (EP)",             1.50, "vs vLLM/SGLang"),
    ("Attn BF16 (prefill)",        1.33, "vs FlashInfer/FA"),
    ("Attn FP8 (prefill)",         1.12, "vs FlashInfer/FA3"),
    ("Group GEMM FP8 (prefill)",   1.10, "vs DeepGEMM"),
]

OP_FAMILIES = [
    ("Attention",      6, "Prefill / Decode / Sparse / Paged-KV (BF16+FP8)"),
    ("GEMM",           1, "BF16xFP32 router GEMM"),
    ("Group GEMM",     2, "Per-tensor / blockwise FP8"),
    ("Fused MoE",      3, "cp.async / blockwise / per-tensor"),
    ("Communication",  2, "AllReduce+RMSNorm HT / LL"),
    ("Sampler",        2, "Full / temperature fast-path"),
    ("Norm/RoPE/Act",  3, "RMSNorm / RoPE+KV-store / Quant"),
    ("Stem (pre-prefill)", 4, "Prep / OAM / TPD"),
]


def fig_perf_overview():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 2]})

    # Left: peak speedup bars
    ax = axes[0]
    names = [r[0] for r in PERF_ROWS]
    speed = [r[1] for r in PERF_ROWS]
    order = np.argsort(speed)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(speed)))
    bars = ax.barh([names[i] for i in order], [speed[i] for i in order],
                   color=[colors[i] for i in order])
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="baseline = 1x")
    for bar, idx in zip(bars, order):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{speed[idx]:.2f}x", va="center", fontsize=9)
    ax.set_xlabel("Peak speedup vs reference baseline")
    ax.set_title("(a) HPC-Ops peak speedup per operator (H20 / SM90)")
    ax.set_xlim(0, max(speed) * 1.18)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # Right: operator family coverage donut + count bars
    ax2 = axes[1]
    labels = [f"{n}\n({c})" for n, c, _ in OP_FAMILIES]
    sizes = [c for _, c, _ in OP_FAMILIES]
    wedges, texts = ax2.pie(sizes, labels=labels, startangle=90, counterclock=False,
                            colors=plt.cm.tab20.colors[: len(OP_FAMILIES)],
                            wedgeprops={"width": 0.42, "edgecolor": "white"})
    for t in texts:
        t.set_fontsize(9)
    ax2.text(0, 0.05, f"{sum(sizes)}", ha="center", va="center", fontsize=22, fontweight="bold")
    ax2.text(0, -0.18, "kernel families", ha="center", va="center", fontsize=10)
    ax2.set_title("(b) Operator family coverage")

    fig.suptitle("HPC-Ops performance & coverage overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "hpc_ops_perf_overview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: Code distribution + optimization technique footprint (2x2)
# ---------------------------------------------------------------------------
MODULE_LOC = [
    ("attention",     34, 11457),
    ("group_gemm",    14, 3518),
    ("fuse_moe",       9, 3069),
    ("stem",          10, 2059),
    ("sampler",        5, 1729),
    ("communicator",  15, 1369),
    ("allreduce",      5, 1360),
    ("activation",     3, 1090),
    ("rope",           3, 1083),
    ("utils",          7, 1044),
    ("gemm",           3,  727),
    ("normalization",  3,  320),
]

TECHNIQUES = [
    # (name, # of operators that use it)
    ("CUTLASS / CuTe",         6),
    ("Warp-Specialization",    4),
    ("cp.async",               5),
    ("TMA",                    7),
    ("PDL",                    5),
    ("CUDA Multicast",         1),
    ("Lamport P2P",            1),
    ("FP8 e4m3",               7),
    ("BF16",                  10),
    ("Paged KV cache",         3),
    ("CUDA Graph",             6),
    ("Split-K + dynamic sched",2),
]

PRECISION = [
    ("BF16",     5),
    ("FP8 (per-tensor)", 4),
    ("FP8 (blockwise)",  3),
    ("BF16 x FP32",      1),
    ("FP32 (sampler)",   1),
]


def fig_code_and_tech():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Module LoC
    ax = axes[0, 0]
    names = [m[0] for m in MODULE_LOC]
    locs  = [m[2] for m in MODULE_LOC]
    files = [m[1] for m in MODULE_LOC]
    order = np.argsort(locs)[::-1]
    bars = ax.bar([names[i] for i in order], [locs[i] for i in order],
                  color=plt.cm.plasma(np.linspace(0.15, 0.9, len(names))))
    for bar, idx in zip(bars, order):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{locs[idx]}\n({files[idx]} files)", ha="center", fontsize=8)
    ax.set_ylabel("Lines of CUDA / C++ code")
    ax.set_title("(a) Source code per module (src/*)")
    ax.set_xticklabels([names[i] for i in order], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(locs) * 1.18)

    # (b) Optimization techniques
    ax = axes[0, 1]
    techs = [t[0] for t in TECHNIQUES]
    counts = [t[1] for t in TECHNIQUES]
    order = np.argsort(counts)
    ax.barh([techs[i] for i in order], [counts[i] for i in order],
            color=plt.cm.cividis(np.linspace(0.1, 0.9, len(techs))))
    ax.set_xlabel("# operators using this technique")
    ax.set_title("(b) Optimization technique footprint")
    for i, idx in enumerate(order):
        ax.text(counts[idx] + 0.1, i, f"{counts[idx]}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(counts) * 1.2)

    # (c) Precision pie
    ax = axes[1, 0]
    p_names = [p[0] for p in PRECISION]
    p_counts = [p[1] for p in PRECISION]
    ax.pie(p_counts, labels=[f"{n}\n({c})" for n, c in PRECISION],
           autopct=lambda pct: f"{pct:.0f}%", startangle=90,
           colors=plt.cm.Set2.colors[: len(PRECISION)],
           wedgeprops={"edgecolor": "white"})
    ax.set_title("(c) Precision modes supported across kernels")

    # (d) Total LoC summary stacked
    ax = axes[1, 1]
    cat_loc = {
        "Attention family": 11457,
        "MoE / Group GEMM": 3518 + 3069,
        "Stem (pre-prefill)": 2059,
        "Sampler": 1729,
        "Communication / AllReduce": 1369 + 1360,
        "Norm / RoPE / Act": 1090 + 1083 + 320,
        "GEMM / Utils": 727 + 1044,
    }
    total = sum(cat_loc.values())
    starts = 0
    colors = plt.cm.tab10.colors
    for i, (k, v) in enumerate(cat_loc.items()):
        ax.barh([0], [v], left=starts, color=colors[i % 10],
                label=f"{k}: {v} ({v/total*100:.0f}%)")
        starts += v
    ax.set_xlim(0, total)
    ax.set_xlabel("Cumulative lines of CUDA/C++")
    ax.set_yticks([])
    ax.set_title(f"(d) Total {total} LoC across kernel families")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)

    fig.suptitle("HPC-Ops codebase composition & technique footprint",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "hpc_ops_code_and_tech.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: Architecture map + dispatch flow (1x2)
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # ---- Left: stacked architecture diagram ----
    ax = axes[0]
    ax.set_xlim(0, 12); ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, label, color, fc=None):
        ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor="black", facecolor=color, lw=1.0))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)

    box(0.5, 8.5, 11, 1.0, "Python API  (hpc/__init__.py via torch.ops.load_library)", "#F4D35E")
    box(0.5, 7.0, 11, 1.2, "torch::Library entry (src/*/entry.cc) — register fake/meta ops", "#FFC857")

    # mid layer: kernels family
    families = [
        (0.5, 5.0, 2.4, "Attention\n(prefill / decode\nsparse / paged)"),
        (3.1, 5.0, 2.0, "GEMM\nBF16xFP32"),
        (5.3, 5.0, 2.4, "Group GEMM\nper-tensor / blockwise"),
        (7.9, 5.0, 1.8, "Fuse MoE\ncp.async path"),
        (9.9, 5.0, 1.6, "Stem\n(pre-prefill)"),
    ]
    for x, y, w, lbl in families:
        box(x, y, w, 1.5, lbl, "#7DDF64")
    box(0.5, 3.3, 5.4, 1.4, "Sampler  (2-kernel fused, temp fast-path)", "#48BB78")
    box(6.1, 3.3, 5.4, 1.4, "AllReduce + RMSNorm  (HT multicast / LL Lamport P2P)", "#48BB78")
    box(0.5, 1.6, 11.0, 1.4,
        "Norm / RoPE+KV-store / Activation-Quant / Communicator / Multicast Handle",
        "#A8DADC")
    # Foundation (white text on dark navy)
    ax.add_patch(plt.Rectangle((0.5, 0.1), 11.0, 1.2, edgecolor="black",
                              facecolor="#1D3557", lw=1.0))
    ax.text(6.0, 0.7,
            "CUTLASS 4.4.2  +  CUDA 12.8  +  CuTe / cp.async / TMA / PDL / Multicast  on  SM90 (H20)",
            ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.set_title("(a) HPC-Ops layered architecture", fontsize=12, fontweight="bold")

    # ---- Right: decode-attention dispatch flow ----
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    def node(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    linewidth=1.0, edgecolor="black", facecolor=color))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#222"))
        if label:
            ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, label, fontsize=8, color="#444")

    node(3.5, 9.0, 3.0, 0.8, "Decode request batch\n(varying KV lengths)", "#F4D35E")
    node(0.5, 7.0, 3.5, 1.2, "Static split-k\n(legacy, one CTA per head)", "#E5989B")
    node(6.0, 7.0, 3.5, 1.2, "Dynamic task assigner\nassign_attention_decode_task", "#7DDF64")
    arrow(4.5, 9.0, 2.3, 8.2)
    arrow(5.5, 9.0, 7.7, 8.2)

    node(0.5, 5.0, 3.5, 1.2, "Per-CTA fixed range\n→ tail latency on long req", "#E5989B")
    node(6.0, 5.0, 3.5, 1.2, "Greedy bin-pack tiles\nuniform 64-token tiles", "#A8DADC")
    arrow(2.3, 7.0, 2.3, 6.2)
    arrow(7.7, 7.0, 7.7, 6.2)

    node(2.5, 3.0, 5.0, 1.2, "attention_decode_fp8 / bf16\n(SM90 warp-spec, FP8 e4m3, paged KV)", "#48BB78")
    arrow(2.3, 5.0, 4.0, 4.2)
    arrow(7.7, 5.0, 6.0, 4.2)

    ax.add_patch(FancyBboxPatch((2.5, 1.0), 5.0, 1.2, boxstyle="round,pad=0.05",
                               linewidth=1.0, edgecolor="black", facecolor="#1D3557"))
    ax.text(5.0, 1.6, "splitk_combine kernel  ->  output", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
    arrow(5.0, 3.0, 5.0, 2.2)
    ax.set_title("(b) Dynamic decode attention dispatch flow", fontsize=12, fontweight="bold")

    fig.suptitle("HPC-Ops architecture & dispatch flow", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "hpc_ops_architecture.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: Speedup grouped by operator domain
# ---------------------------------------------------------------------------
GROUPED = {
    "Attention": [
        ("Sparse FP8",          3.16),
        ("Dynamic Decode",      2.88),
        ("BF16 decode",         2.22),
        ("FP8 decode",          2.00),
        ("BF16 prefill",        1.33),
        ("FP8 prefill",         1.12),
    ],
    "MoE / GEMM": [
        ("BF16xFP32 GEMM",      3.22),
        ("Group GEMM dec.",     1.88),
        ("Fused MoE TP",        1.60),
        ("Fused MoE EP",        1.50),
        ("Group GEMM pref.",    1.10),
    ],
    "System / post": [
        ("Sampler",             8.50),
        ("AllReduce+RMSNorm",   1.76),
    ],
}


def fig_grouped_speedup():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                              gridspec_kw={"width_ratios": [3, 3, 2]})
    color_map = {"Attention": plt.cm.Blues, "MoE / GEMM": plt.cm.Greens,
                 "System / post": plt.cm.Reds}
    for ax, (group, rows) in zip(axes, GROUPED.items()):
        names = [r[0] for r in rows]
        sp = [r[1] for r in rows]
        cmap = color_map[group]
        bars = ax.bar(names, sp, color=cmap(np.linspace(0.4, 0.9, len(rows))))
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
        for b, v in zip(bars, sp):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                    f"{v:.2f}x", ha="center", fontsize=9)
        ax.set_title(group, fontsize=12, fontweight="bold")
        ax.set_ylabel("Peak speedup vs baseline")
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(sp) * 1.15 + 0.2)

    fig.suptitle("Peak speedup grouped by operator domain", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "hpc_ops_grouped_speedup.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def write_csv():
    csv_path = OUT_DIR / "hpc_ops_perf_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["operator", "peak_speedup", "baseline"])
        for op, sp, base in PERF_ROWS:
            w.writerow([op, sp, base])
    return csv_path


if __name__ == "__main__":
    paths = []
    paths.append(write_csv())
    paths.append(fig_perf_overview())
    paths.append(fig_code_and_tech())
    paths.append(fig_architecture())
    paths.append(fig_grouped_speedup())
    for p in paths:
        print(p)
