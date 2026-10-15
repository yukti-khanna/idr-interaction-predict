"""No trained model is loaded by these guard tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import tempfile
import unittest
from unittest.mock import patch
from configuration import PIPELINE_ROOT
from model_loader import FrozenPredictor

class ArtifactGuardTest(unittest.TestCase):
    def test_wrong_manifest_stops_before_loading_any_models(self):
        with tempfile.TemporaryDirectory(dir=PIPELINE_ROOT/'results/validation') as folder:
            root=Path(folder);(root/'MODEL_MANIFEST.json').write_text('{}')
            with patch('model_loader.MODEL_DIR',root):
                with self.assertRaisesRegex(RuntimeError,'manifest identity mismatch'):
                    FrozenPredictor()

if __name__=='__main__':unittest.main()
