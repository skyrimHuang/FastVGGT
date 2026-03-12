import csv
from pathlib import Path

root = Path("tests/tests_result/hybrid_registration_7scenes")
src = root / "hybrid_registration_summary.csv"
trend = root / "hybrid_registration_summary_trend_v2.csv"
paper = root / "hybrid_registration_paper_table.csv"

with src.open("r", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

with trend.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

paper_rows = []
for row in rows:
    recall_percent = 100.0 * float(row["recall"])
    if row["method"] == "B":
        mean_icp_iterations = "N/A"
    else:
        mean_icp_iterations = f"{float(row['mean_icp_iterations']):.2f}"

    paper_rows.append(
        {
            "scene": row["scene"],
            "method": row["method"],
            "num_pairs": row["num_pairs"],
            "recall_percent": f"{recall_percent:.2f}",
            "mean_rotation_error_deg": f"{float(row['mean_rotation_error_deg']):.4f}",
            "mean_translation_error_m": f"{float(row['mean_translation_error_m']):.4f}",
            "mean_chamfer_distance": f"{float(row['mean_chamfer_distance']):.6f}",
            "mean_icp_iterations": mean_icp_iterations,
            "mean_runtime_ms": f"{float(row['mean_runtime_ms']):.2f}",
        }
    )

with paper.open("w", newline="", encoding="utf-8") as file:
    headers = list(paper_rows[0].keys())
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    writer.writerows(paper_rows)

print(trend)
print(paper)
