import json,runpy
from pathlib import Path
root=Path(__file__).resolve().parent

def load_history():
 # Compact history lacks questions; parsed history candidates are already frozen.
 return json.loads((root/'history_frozen.json').read_text())

def load_lab():
 rows=[]
 for p in sorted(root.glob('lab_*.json')): rows.extend(json.loads(p.read_text()))
 return rows

if __name__=='__main__':
 import os
 os.chdir(root.parent)
 for script in ['audit.py','xor_lab_audit.py','additive_audit.py']:
  runpy.run_path(str(root/script),init_globals={'load_lab':load_lab,'load_history':load_history},run_name='__main__')
