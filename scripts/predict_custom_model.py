#!/usr/bin/env python3
"""Predict with an independently trained custom model, never the frozen loader."""
from pathlib import Path
import argparse
import pandas as pd
from custom_model import read_sequences,load_custom,predict_components
from feature_engineering import calculate_features

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Output exists; choose a new path')
 df,_=read_sequences(a.data);bundle,manifest,residual=load_custom(a.model);features=calculate_features(df)
 result=pd.concat([df,features,predict_components(bundle,features,manifest,residual)],axis=1)
 a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False)
if __name__=='__main__':main()
