"""
Generate all paper figures for BiTempQA
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
EVAL_DIR = DATA_DIR / "eval_results"
FIG_DIR = BASE / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open(DATA_DIR / "validated" / "bitpqa_test_zh.json", encoding="utf-8") as f:
    dataset = json.load(f)
qa_pairs = dataset.get("qa_pairs", dataset if isinstance(dataset, list) else [])

with open(EVAL_DIR / "temporal_breakdown.json", encoding="utf-8") as f:
    temporal = json.load(f)

with open(EVAL_DIR / "deep_diagnostics.json", encoding="utf-8") as f:
    diagnostics = json.load(f)

with open(EVAL_DIR / "statistical_analysis.json", encoding="utf-8") as f:
    stats = json.load(f)

SYSTEMS = ["FAISS Vector Store", "BM25", "Simple KG", "Naive RAG", "ChromaDB"]
SYS_SHORT = {"FAISS Vector Store": "FAISS", "BM25": "BM25", "Simple KG": "Simple KG", "Naive RAG": "Naive RAG", "ChromaDB": "ChromaDB", "Mem0": "Mem0", "Graphiti": "Graphiti"}
COLORS = {"FAISS": "#2563eb", "Naive RAG": "#7c3aed", "BM25": "#059669", "Simple KG": "#d97706", "ChromaDB": "#dc2626", "Mem0": "#6366f1", "Graphiti": "#a855f7"}

# ═══════════════════════════════════════════════════════════
# FIGURE 3: Heatmap — System × Question Type Accuracy
# ═══════════════════════════════════════════════════════════
print("Generating Figure 3: System × Question Type Heatmap...")

qt_data = {
    "Point-in-Time": {"FAISS": 77.8, "BM25": 85.7, "Naive RAG": 77.8, "Simple KG": 63.5, "ChromaDB": 31.7, "Mem0": 14.5, "Graphiti": 11.3},
    "Period Query": {"FAISS": 83.3, "BM25": 80.6, "Naive RAG": 86.1, "Simple KG": 88.9, "ChromaDB": 50.0, "Mem0": 8.6, "Graphiti": 5.7},
    "Temporal Order": {"FAISS": 89.3, "BM25": 89.3, "Naive RAG": 92.9, "Simple KG": 85.7, "ChromaDB": 64.3, "Mem0": 42.9, "Graphiti": 32.1},
    "Multi-hop": {"FAISS": 76.7, "BM25": 83.3, "Naive RAG": 73.3, "Simple KG": 76.7, "ChromaDB": 56.7, "Mem0": 6.9, "Graphiti": 6.9},
    "First Recorded": {"FAISS": 64.0, "BM25": 64.0, "Naive RAG": 64.0, "Simple KG": 56.0, "ChromaDB": 40.0, "Mem0": 12.5, "Graphiti": 8.3},
    "Change Detection": {"FAISS": 61.3, "BM25": 48.4, "Naive RAG": 54.8, "Simple KG": 38.7, "ChromaDB": 19.4, "Mem0": 6.9, "Graphiti": 10.3},
    "Complex Temporal": {"FAISS": 87.1, "BM25": 80.6, "Naive RAG": 83.9, "Simple KG": 71.0, "ChromaDB": 29.0, "Mem0": 10.7, "Graphiti": 10.7},
    "Counterfactual": {"FAISS": 53.8, "BM25": 41.0, "Naive RAG": 53.8, "Simple KG": 46.2, "ChromaDB": 33.3, "Mem0": 12.8, "Graphiti": 20.5},
}

sys_order = ["FAISS", "Naive RAG", "BM25", "Simple KG", "ChromaDB", "Mem0", "Graphiti"]
qt_order = list(qt_data.keys())

matrix = np.array([[qt_data[qt][s] for s in sys_order] for qt in qt_order])

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.heatmap(matrix, annot=True, fmt=".1f", cmap="RdYlGn", vmin=20, vmax=100,
            xticklabels=sys_order, yticklabels=qt_order, ax=ax,
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Accuracy (%)'})
ax.set_xlabel("Memory System")
ax.set_ylabel("Question Type")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_heatmap.pdf")
fig.savefig(FIG_DIR / "fig3_heatmap.png")
plt.close()
print("  -> fig3_heatmap.pdf/png")

# ═══════════════════════════════════════════════════════════
# FIGURE 4: Difficulty Degradation Curves
# ═══════════════════════════════════════════════════════════
print("Generating Figure 4: Difficulty Degradation...")

diff_data = {
    "FAISS": [77.6, 74.2, 74.7],
    "Naive RAG": [78.4, 72.2, 70.5],
    "BM25": [81.9, 71.1, 65.3],
    "Simple KG": [67.2, 69.1, 65.3],
    "ChromaDB": [41.4, 42.3, 40.0],
    "Mem0": [21.1, 7.5, 10.0],
    "Graphiti": [15.8, 7.5, 13.3],
}

fig, ax = plt.subplots(figsize=(6, 3.5))
levels = [1, 2, 3]
for sys_name, accs in diff_data.items():
    style = '--' if sys_name in ('Mem0', 'Graphiti') else '-'
    ax.plot(levels, accs, 'o' + style, color=COLORS[sys_name], label=sys_name, linewidth=2, markersize=6)
ax.set_xlabel("Difficulty Level")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["L1\n(Easy)", "L2\n(Medium)", "L3\n(Hard)"])
ax.set_ylim(30, 95)
ax.legend(loc='lower left', framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig4_difficulty.pdf")
fig.savefig(FIG_DIR / "fig4_difficulty.png")
plt.close()
print("  -> fig4_difficulty.pdf/png")

# ═══════════════════════════════════════════════════════════
# FIGURE 5: System Complementarity (Stacked Bar)
# ═══════════════════════════════════════════════════════════
print("Generating Figure 5: System Complementarity...")

comp = diagnostics.get("complementarity", {})
pairwise = comp.get("pairwise", {})

# Bar chart: accuracy of each system + oracle
sys_accs = {
    "FAISS": 75.6, "Naive RAG": 74.0, "BM25": 73.4,
    "Simple KG": 67.2, "ChromaDB": 41.2, "Mem0": 13.5, "Graphiti": 12.5,
    "Oracle\n(Any System)": 85.4,
}

fig, ax = plt.subplots(figsize=(7, 3.5))
bar_colors = [COLORS.get(name.replace("\n", " "), '#333') for name in sys_accs.keys()]
bar_colors[-1] = '#555'  # Oracle bar
bars = ax.bar(sys_accs.keys(), sys_accs.values(), color=bar_colors)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
for bar, val in zip(bars, sys_accs.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%",
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(y=85.4, color='gray', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig5_complementarity.pdf")
fig.savefig(FIG_DIR / "fig5_complementarity.png")
plt.close()
print("  -> fig5_complementarity.pdf/png")

# ═══════════════════════════════════════════════════════════
# FIGURE 6: Temporal Sensitivity Bar Chart
# ═══════════════════════════════════════════════════════════
print("Generating Figure 6: Temporal Sensitivity...")

ts_data = {
    "Event Only\n(n=110)": {"FAISS": 79.1, "Naive RAG": 78.2, "BM25": 77.3, "Simple KG": 66.4, "ChromaDB": 44.5, "Mem0": 18.9, "Graphiti": 15.1},
    "Record Only\n(n=24)": {"FAISS": 66.7, "Naive RAG": 62.5, "BM25": 66.7, "Simple KG": 58.3, "ChromaDB": 50.0, "Mem0": 26.1, "Graphiti": 26.1},
    "Both Required\n(n=174)": {"FAISS": 74.7, "Naive RAG": 73.0, "BM25": 71.8, "Simple KG": 69.0, "ChromaDB": 37.9, "Mem0": 8.3, "Graphiti": 8.9},
    "Version Track\n(n=78)": {"FAISS": 79.5, "Naive RAG": 74.4, "BM25": 69.2, "Simple KG": 75.6, "ChromaDB": 37.2, "Mem0": 5.2, "Graphiti": 9.1},
    "Retraction\n(n=28)": {"FAISS": 67.9, "Naive RAG": 67.9, "BM25": 57.1, "Simple KG": 53.6, "ChromaDB": 39.3, "Mem0": 14.3, "Graphiti": 21.4},
}

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(ts_data))
width = 0.1
for i, sys_name in enumerate(sys_order):
    vals = [ts_data[cat][sys_name] for cat in ts_data]
    ax.bar(x + i * width, vals, width, label=sys_name, color=COLORS[sys_name])

ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x + width * 3)
ax.set_xticklabels(ts_data.keys())
ax.set_ylim(0, 100)
ax.legend(loc='upper right', ncol=4, fontsize=7)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig6_temporal.pdf")
fig.savefig(FIG_DIR / "fig6_temporal.png")
plt.close()
print("  -> fig6_temporal.pdf/png")

# ═══════════════════════════════════════════════════════════
# FIGURE 7: Confidence Calibration Reliability Diagram
# ═══════════════════════════════════════════════════════════
print("Generating Figure 7: Confidence Calibration...")

r2 = {}
with open(EVAL_DIR / "round2_analyses.json", encoding="utf-8") as f:
    r2 = json.load(f)

calib = r2.get("calibration", {})
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

for ax_i, sys_name in enumerate(["FAISS", "Naive_RAG", "BM25"]):
    ax = axes[ax_i]
    data = calib.get(sys_name, {})
    bins = data.get("bins", [])
    if not bins:
        ax.text(0.5, 0.5, f"No data for {sys_name}", ha='center', va='center', transform=ax.transAxes)
        continue

    confs = []
    accs = []
    for bd in bins:
        bin_range = bd.get("bin", "0.0-0.1")
        confs.append(float(bin_range.split("-")[0]) + 0.05)
        accs.append(bd["accuracy"] * 100)

    ax.bar(range(len(confs)), accs, alpha=0.7, color=COLORS.get(sys_name, '#333'))
    ax.plot(range(len(confs)), [c * 100 for c in confs], 'r--o', label='Perfect calibration')
    ax.set_title(SYS_SHORT.get(sys_name, sys_name))
    ax.set_xlabel("Confidence Bin")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle("Confidence Calibration", fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig7_calibration.pdf")
fig.savefig(FIG_DIR / "fig7_calibration.png")
plt.close()
print("  -> fig7_calibration.pdf/png")

print("\nAll data figures generated!")
print(f"Output: {FIG_DIR}")
for f in sorted(FIG_DIR.glob("fig*")):
    print(f"  {f.name}")
