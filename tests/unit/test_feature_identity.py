import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import tempfile
import unittest
from configuration import PIPELINE_ROOT
from qc import identity_test

class FeatureIdentityTest(unittest.TestCase):
    def test_known_twenty_sequence_pairs(self):
        with tempfile.TemporaryDirectory(dir=PIPELINE_ROOT/'results/validation') as folder:
            reports=identity_test(Path(folder))
            self.assertEqual(len(reports),13)
            self.assertTrue(all(r['passed'] for r in reports))

if __name__=='__main__':unittest.main()
