import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import pandas as pd
from custom_model import read_sequences
class CustomInputTests(unittest.TestCase):
    def test_sequence_overlap_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.csv'
            pd.DataFrame([{'pair_id':'a','rAHD_sequence':'DDDDD','RGG_sequence':'RRRRR','split':'train'}, {'pair_id':'b','rAHD_sequence':'DDDDD','RGG_sequence':'GGGGG','split':'validation'}]).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Sequences overlap'):read_sequences(p,True)
            _,overlap=read_sequences(p,True,'pair')
            self.assertEqual(overlap[0]['shared_sequences'],1)
    def test_duplicate_sequence_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.csv'
            pd.DataFrame([{'pair_id':i,'rAHD_sequence':'DDDDD','RGG_sequence':'RRRRR'} for i in ['a','b']]).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Duplicate sequence pairs'):read_sequences(p)
