"""Run unit tests and verify that saved matrices match a reloaded model."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from sgns import SGNS, load_vectors
from train import save_json

root = Path(__file__).resolve().parent
out = root/'outputs'
run = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                     cwd=root,text=True,capture_output=True)
(out/'tests.txt').write_text(run.stdout+run.stderr,encoding='utf-8')
print(run.stdout+run.stderr)
if run.returncode:
    raise SystemExit(run.returncode)
config = json.loads((out/'config.json').read_text())
words, vectors = load_vectors(out)
model = SGNS(len(words),config['dim'])
model.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True))
np.testing.assert_array_equal(vectors,model.center.weight.detach().numpy())
assert np.isfinite(vectors).all()
assert len(set(words)) == len(words)
assert vectors.shape == (len(words),config['dim'])
assert (np.linalg.norm(vectors,axis=1)>0).all()
save_json(out/'verification.json',{'unit_tests':7,'passed':True,'reload_exact_match':True,
          'vectors_shape':list(vectors.shape),'finite_and_nonzero_vectors':True})
print('Saved model / matrix exact match:',vectors.shape)
