#!/usr/bin/env python3
"""Verify all authoritative artifacts before installing; never deserialize them."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
FILES=['MODEL_MANIFEST.json','base_model_ensemble.joblib','residual_preprocessor.joblib','residual_nn.keras']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
 expected=json.loads((ROOT/'config/accepted_model_hashes.json').read_text());dest=ROOT/'models/final_frozen'
 for n in FILES:
  src=a.source/n
  if not src.is_file() or sha(src)!=expected[n]:raise SystemExit(f'Missing or incorrect authoritative artifact: {src}. Nothing copied.')
  if (dest/n).exists() and sha(dest/n)!=expected[n]:raise SystemExit(f'Destination contains a different artifact: {dest/n}. Nothing copied.')
 dest.mkdir(parents=True,exist_ok=True)
 for n in FILES:
  if (a.source/n).resolve()!=(dest/n).resolve():shutil.copy2(a.source/n,dest/n)
 print('All four authoritative artifacts verified and installed. Run inference identity in the established model environment.')
if __name__=='__main__':main()
