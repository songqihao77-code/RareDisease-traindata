import re

with open('src/evaluation/evaluator.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('"readout": {"type": "hyperedge"},', '"readout": {"type": "hyperedge", **model_cfg.get("readout", {}), "hidden_dim": hidden_dim},')

with open('src/evaluation/evaluator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("evaluator.py patched successfully")
