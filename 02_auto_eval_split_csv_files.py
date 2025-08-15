#!/usr/bin/env python3
import re
from pathlib import Path

# --- configure paths ---
in_path   = Path("plots/results_phate.csv")
out_emb   = Path("plots/EMB_DRIFT.csv")   # embedding drift changed
out_cka   = Path("plots/CKA_SIM.csv")         # cka similarity changed
# -----------------------

pattern = re.compile(
    r'[^F"]*EMBEDDING DRIFT:\s*0\.(\d+)\n[^"]*CKA SIMILARITY:\s*0\.(\d+)\n',
    flags=re.MULTILINE
)

text = in_path.read_text(encoding="utf-8")
text = in_path.read_text(encoding="utf-8").replace(",", ";")

# Replace entire match with group 1
text_emb = pattern.sub(lambda m: f"0,{m.group(1)}", text)

# Replace entire match with group 2
text_cka = pattern.sub(lambda m: f"0,{m.group(2)}", text)

out_emb.write_text(text_emb, encoding="utf-8")
out_cka.write_text(text_cka, encoding="utf-8")

print(f"Done.\n  Saved: {out_emb}\n  Saved: {out_cka}")