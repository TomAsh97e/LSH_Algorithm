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
plt.close(fig)

test_csv_path = project_dir / "benchmark" / "test_results.csv"
test_output_path = project_dir / "benchmark" / "test_performance.png"

with test_csv_path.open(newline="", encoding="utf-8") as file:
    test_results = list(csv.DictReader(file))

document_counts = sorted({int(row["N"]) for row in test_results})
configs = [(8, 8), (16, 8), (16, 6)]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

exact_times = []
for document_count in document_counts:
    group = [row for row in test_results if int(row["N"]) == document_count]
    exact_times.append(sum(float(row["exact_us"]) for row in group) / len(group))

axes[0].plot(document_counts, exact_times, marker="o", linewidth=2, label="Exact Search")

for tables, bits in configs:
    lsh_times = []
    recalls = []

    for document_count in document_counts:
        group = [row for row in test_results if int(row["N"]) == document_count
                 and int(row["L"]) == tables and int(row["b"]) == bits]
        lsh_times.append(sum(float(row["lsh_us"]) for row in group) / len(group))
        recalls.append(sum(float(row["recall"]) for row in group) / len(group))

    label = f"LSH L={tables}, b={bits}"
    axes[0].plot(document_counts, lsh_times, marker="o", label=label)
    axes[1].plot(document_counts, recalls, marker="o", label=label)

axes[0].set(title="Average query time", xlabel="Document count", ylabel="Time [us]")
axes[0].set_yscale("log")
axes[1].set(title="Search quality", xlabel="Document count", ylabel="Recall@10", ylim=(0, 1))

for axis in axes:
    axis.set_xscale("log")
    axis.set_xticks(document_counts, labels=[str(count) for count in document_counts])
    axis.grid(alpha=0.3)
    axis.legend()

fig.suptitle("Final test results (3 seeds, 200 queries)")
fig.tight_layout()
fig.savefig(test_output_path, dpi=160)
print(f"Saved {test_output_path}")
