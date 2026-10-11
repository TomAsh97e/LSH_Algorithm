import csv
from pathlib import Path

import matplotlib.pyplot as plt


project_dir = Path(__file__).resolve().parents[1]
csv_path = project_dir / "benchmark" / "tuning_results.csv"
output_path = project_dir / "benchmark" / "tuning_tradeoff.png"

with csv_path.open(newline="", encoding="utf-8") as file:
    results = list(csv.DictReader(file))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for tables in sorted({int(row["L"]) for row in results}):
    group = [row for row in results if int(row["L"]) == tables]
    lsh_times = [float(row["lsh_us"]) for row in group]
    candidates = [float(row["candidates"]) for row in group]
    recalls = [float(row["recall"]) for row in group]

    axes[0].plot(lsh_times, recalls, marker="o", label=f"L={tables}")
    axes[1].plot(candidates, recalls, marker="o", label=f"L={tables}")

    for row, lsh_time, candidate_count, recall in zip(group, lsh_times, candidates, recalls):
        axes[0].annotate(f'b={row["b"]}', (lsh_time, recall), xytext=(5, 5), textcoords="offset points")
        axes[1].annotate(f'b={row["b"]}', (candidate_count, recall), xytext=(5, 5), textcoords="offset points")

axes[0].set(title="Quality vs search time", xlabel="Average LSH time [us]", ylabel="Recall@10")
axes[1].set(title="Quality vs candidate count", xlabel="Average candidate count", ylabel="Recall@10")

for axis in axes:
    axis.grid(alpha=0.3)
    axis.legend()

fig.suptitle("LSH parameter tuning (N=10,000)")
fig.tight_layout()
fig.savefig(output_path, dpi=160)
print(f"Saved {output_path}")
