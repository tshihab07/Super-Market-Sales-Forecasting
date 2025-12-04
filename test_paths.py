# run once
from pathlib import Path

project_root = Path(__file__).parent 
target_path = project_root / "artifacts" / "feature-selection" / "target_encoder.pkl"
ohe_path = project_root / "artifacts" / "feature-selection" / "ohe.pkl"

print("Project root:", project_root.resolve())
print("Target path:  ", target_path.resolve())
print("Exists?       ", target_path.exists())
print("Target path:  ", ohe_path.resolve())
print("Exists?       ", ohe_path.exists())