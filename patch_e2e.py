import re
import yaml
from pathlib import Path

with open('test/test_full_chain_e2e.py', 'r', encoding='utf-8') as f:
    text = f.read()

fix_code = """def _prepare_batch(batch_size: int = SMOKE_BATCH_SIZE) -> dict:
    import yaml
    with open("configs/train.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train_files = cfg.get("paths", {}).get("train_files")
    if train_files is None:
        train_dir = Path(cfg["paths"]["train_dir"])
        train_files = [str(file) for file in train_dir.glob("*.xlsx") if not file.name.startswith("~$")]
    df = load_case_files([train_files[0]])"""

text = re.sub(r'def _prepare_batch\(batch_size: int = SMOKE_BATCH_SIZE\) -> dict:[\s\S]*?df = load_case_files\(\[train_files\[0\]\]\)', fix_code, text)

# Just in case the second search fails because it's not present, let's also do a hard replace:
if "load_config()" in text and "train_files" in text:
    pass # Already replaced

with open('test/test_full_chain_e2e.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("test_full_chain_e2e.py patched")
