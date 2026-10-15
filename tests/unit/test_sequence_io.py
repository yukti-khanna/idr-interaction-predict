import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import tempfile
import unittest
from argparse import Namespace
import pandas as pd
from sequence_io import resolve_inputs, canonical_id
from configuration import PIPELINE_ROOT

class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=PIPELINE_ROOT/'results/validation')
        self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def args(self,**kw):
        d=dict(pairs=None,ahd_fasta=None,rgg_fasta=None,ahd_list=None,rgg_list=None,ahd_reference=[],rgg_reference=[])
        d.update(kw);return Namespace(**d)
    def pairs(self,rows):
        p=self.root/'pairs.csv';pd.DataFrame(rows).to_csv(p,index=False);return p
    def test_inline_normalization_and_rejections(self):
        good=dict(pair_id='a',AHD_name='ahd',RGG_name='rgg',AHD_sequence=' de d ed ',RGG_sequence='rgrgr')
        p=self.pairs([good,{**good,'pair_id':'b'}, {**good,'pair_id':'bad','AHD_name':'bad','AHD_sequence':'AXX'}, {**good,'pair_id':'empty','AHD_name':'empty','AHD_sequence':''}])
        df,_,_=resolve_inputs(self.args(pairs=p))
        self.assertEqual(df.status.tolist(),['valid','valid','rejected','rejected'])
        self.assertEqual(df.AHD_sequence.iloc[0],'DEDED')
        self.assertTrue(df.duplicate_sequence_pair.iloc[0])
        self.assertTrue(df.invalid_residue.iloc[2]);self.assertTrue(df.short_for_patch_features.iloc[2])
        self.assertTrue(df.missing_sequence.iloc[3])
    def test_duplicate_ids_reject_all_instances(self):
        r=dict(pair_id='same',AHD_name='a',RGG_name='r',AHD_sequence='DEDED',RGG_sequence='RGRGR')
        df,_,_=resolve_inputs(self.args(pairs=self.pairs([r,r])))
        self.assertTrue(df.duplicate_pair_id.all());self.assertTrue(df.status.eq('rejected').all())
    def test_ambiguous_registry_and_missing_name(self):
        a=self.root/'a.fasta';a.write_text('>a\nDEDED\n>a\nDDDDD\n')
        r=self.root/'r.fasta';r.write_text('>r\nRGRGR\n')
        rows=[dict(AHD_name='a',RGG_name='r'),dict(AHD_name='unknown',RGG_name='r')]
        df,_,_=resolve_inputs(self.args(pairs=self.pairs(rows),ahd_reference=[a],rgg_reference=[r]))
        self.assertTrue(df.status.eq('rejected').all())
        self.assertIn('ambiguous',df.failure_reason.iloc[0]);self.assertIn('missing',df.failure_reason.iloc[1])
    def test_all_modes_agree(self):
        e=PIPELINE_ROOT/'tests/examples'
        cases=[self.args(pairs=e/'pairs.tsv'),self.args(pairs=e/'pair_names.tsv',ahd_reference=[e/'ahds.fasta'],rgg_reference=[e/'rggs.fasta']),self.args(ahd_fasta=e/'ahds.fasta',rgg_fasta=e/'rggs.fasta'),self.args(ahd_list=e/'ahd_names.txt',rgg_list=e/'rgg_names.txt',ahd_reference=[e/'ahds.fasta'],rgg_reference=[e/'rggs.fasta'])]
        data=[resolve_inputs(a)[0] for a in cases]
        for frame in data:self.assertTrue(frame.status.eq('valid').all())
        cols=['pair_id','AHD_sequence','RGG_sequence']
        pd.testing.assert_frame_equal(data[0][cols],data[1][cols])
        pd.testing.assert_frame_equal(data[2][cols],data[3][cols])
    def test_separator_escape(self):
        self.assertEqual(canonical_id('ahd','rgg'),'rgg__ahd')
        self.assertNotEqual(canonical_id('x__y','z'),canonical_id('y','z__x'))

    def test_plain_pair_text_matches_table(self):
        e=PIPELINE_ROOT/'tests/examples'
        expected=pd.read_csv(e/'pair_names.tsv',sep='\t',dtype=str)
        text=self.root/'pairs.txt'
        text.write_text('# Pair names, RGG first\n\n'+'\n'.join(expected.pair_id)+'\n')
        common=dict(ahd_reference=[e/'ahds.fasta'],rgg_reference=[e/'rggs.fasta'])
        plain,_,_=resolve_inputs(self.args(pairs=text,**common))
        table,_,_=resolve_inputs(self.args(pairs=e/'pair_names.tsv',**common))
        pd.testing.assert_frame_equal(plain,table)

    def test_bad_text_pairs_are_preserved_as_rejections(self):
        a=self.root/'a.fasta';a.write_text('>a\nDEDED\n')
        r=self.root/'r.fasta';r.write_text('>r\nRGRGR\n')
        text=self.root/'pairs.txt'
        names=['r__a','no_separator','r__a__ambiguous','r__unknown','r__']
        text.write_text('\n'.join(names)+'\n')
        df,_,_=resolve_inputs(self.args(pairs=text,ahd_reference=[a],rgg_reference=[r]))
        self.assertEqual(df.pair_id.tolist(),names)
        self.assertEqual(df.status.tolist(),['valid']+['rejected']*4)
        self.assertIn('invalid_pair_name',df.failure_reason.iloc[1])
        self.assertIn('missing_sequence',df.failure_reason.iloc[3])

if __name__=='__main__':unittest.main()
