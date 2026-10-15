"""Server-only check: set RUN_FROZEN_MODEL_IDENTITY=1 to explicitly enable inference."""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import tempfile
import unittest
from configuration import PIPELINE_ROOT
from qc import identity_test

@unittest.skipUnless(os.environ.get('RUN_FROZEN_MODEL_IDENTITY')=='1','Server inference identity test is opt-in; use the mandatory CLI identity check on the server.')
class PredictionIdentityTest(unittest.TestCase):
    def test_known_twenty_predictions(self):
        from model_loader import FrozenPredictor
        with tempfile.TemporaryDirectory(dir=PIPELINE_ROOT/'results/validation') as folder:
            reports=identity_test(Path(folder),FrozenPredictor())
            self.assertEqual(len(reports),18)
            self.assertTrue(all(r['passed'] for r in reports))

if __name__=='__main__':unittest.main()
