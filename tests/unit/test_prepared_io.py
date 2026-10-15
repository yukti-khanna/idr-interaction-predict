"""Prepared-run reuse checks; no model loading or feature calculation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from configuration import PIPELINE_ROOT, PUBLIC_FEATURES, sha256
from prepared_io import load_prepared_run, PREPARED_FILES
from predict_pairs import parser, validate_mode

class PreparedRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=PIPELINE_ROOT/'results/validation')
        self.root=Path(self.tmp.name)/'prepared'
        shutil.copytree(PIPELINE_ROOT/'tests/fixtures/prepared_run',self.root)
    def tearDown(self):self.tmp.cleanup()
    def test_legacy_run_reuses_saved_features_without_recalculation(self):
        with patch('feature_engineering.calculate_features',side_effect=AssertionError('must not calculate input features')):
            ready,requested,rejected,manifest,audit=load_prepared_run(self.root)
        expected=pd.read_csv(self.root/'feature_matrix.csv',float_precision='round_trip')
        pd.testing.assert_frame_equal(ready[PUBLIC_FEATURES],expected)
        self.assertEqual(len(requested),len(ready)+len(rejected))
        self.assertFalse(audit['recalculated_input_features'])
        self.assertEqual(audit['integrity_check'],'legacy_run_cross_checks_no_original_output_hashes')
    def test_feature_misalignment_is_rejected(self):
        p=self.root/'feature_row_ids.csv';d=pd.read_csv(p);d=d.iloc[::-1];d.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'alignment mismatch'):load_prepared_run(self.root)
    def test_changed_feature_value_is_rejected(self):
        p=self.root/'prepared_pairs.csv';d=pd.read_csv(p);d.loc[0,'mean_lambda_RGG']+=1;d.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'feature matrix disagrees'):load_prepared_run(self.root)
    def test_new_output_hashes_detect_changes(self):
        p=self.root/'run_manifest.json';m=json.loads(p.read_text())
        m['prepared_output_hashes']={name:sha256(self.root/name) for name in PREPARED_FILES};p.write_text(json.dumps(m))
        _,_,_,_,audit=load_prepared_run(self.root)
        self.assertEqual(audit['integrity_check'],'stored_output_hashes_and_cross_checks')
        with (self.root/'prepared_pairs.csv').open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):load_prepared_run(self.root)
    def test_prediction_mode_cannot_mix_input_options(self):
        ap=parser();args=ap.parse_args(['--prepared-run',str(self.root)])
        validate_mode(ap,args)
        args.features_only=True
        with patch.object(ap,'error',side_effect=ValueError('mixed options')):
            with self.assertRaisesRegex(ValueError,'mixed options'):validate_mode(ap,args)

if __name__=='__main__':unittest.main()
