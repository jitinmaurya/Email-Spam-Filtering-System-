# scripts/patch_metrics_dataset.py
import json, sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: python scripts/patch_metrics_dataset.py <dataset_name> <metrics.json> [more.json ...]")
    sys.exit(1)

dataset = sys.argv[1]
for p in sys.argv[2:]:
    path = Path(p)
    if not path.exists():
        print(f"[skip] not found: {path}")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    if "dataset" not in data or not data["dataset"]:
        data["dataset"] = dataset
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[ok] set dataset={dataset} -> {path}")
    else:
        print(f"[keep] already has dataset={data['dataset']} -> {path}")
