import os

folders = [
    "configs",
    "datasets/images",
    "datasets/labels",
    "models",
    "modules",
    "trainers",
    "scripts",
    "tests",
    "notebooks"
]

root = r"C:\OneDrive\Study\AI\STN_Final_Term"  # chỉnh lại đường dẫn gốc của bạn

for folder in folders:
    full_path = os.path.join(root, folder)
    try:
        os.makedirs(full_path, exist_ok=True)
        print(f"✅ Created: {full_path}")
    except Exception as e:
        print(f"❌ Failed: {full_path} – {e}")
