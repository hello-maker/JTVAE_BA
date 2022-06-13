import os
import sys
import torch
import tqdm
import pandas as pd

JTVAE_HOME_PATH = '/home/csy/work/ReBADD/JTVAE_BA'
if JTVAE_HOME_PATH not in sys.path:
    sys.path = [JTVAE_HOME_PATH] + sys.path
    
from BA_module.bascorer import DTA


if __name__=='__main__':
    ## 1. Load SMILES data
    filepath_input = 'zinc15_navitoclax-like.csv'
    df = pd.read_csv(filepath_input) # columns : [zinc_id, smiles, mwt, logp, length]
    print(f'Number of SMILES: {df.shape[0]}')
    
    ## 2. GPU check
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(use_cuda, device)
    
    ## 3. Init DTA
    scorer_bcl2  = DTA('Bcl-2', use_cuda, device)
    scorer_bclxl = DTA('Bcl-xl', use_cuda, device)
    scorer_bclw  = DTA('Bcl-w', use_cuda, device)
    
    ## 4. Prediction
    records = []
    for i in tqdm.trange(df.shape[0]):
        idx = df.loc[i,'zinc_id']
        smi = df.loc[i,'smiles']
        mwt = df.loc[i,'mwt']
        lgp = df.loc[i,'logp']
        lng = df.loc[i,'length']
        try:
            ba_bcl2  = scorer_bcl2(smi)
            ba_bclxl = scorer_bclxl(smi)
            ba_bclw  = scorer_bclw(smi)
            records.append((idx, smi, mwt, lgp, lng, ba_bcl2, ba_bclxl, ba_bclw))
        except:
            continue
            
    ## 5. Make a table
    df_res = pd.DataFrame.from_records(records)
    df_res = df_res.rename(columns={0:'zinc_id', 1:'smiles', 2:'mwt', 3:'logp', 4:'length', 5:'ba_bcl2', 6:'ba_bclxl', 7:'ba_bclw'})
    print(f'Number of SMILES whose scores are available: {df_res.shape[0]}')

    ## 6. Save the result table
    filepath_output = 'zinc15_navitoclax-like_dta.csv'
    df_res.to_csv(filepath_output, sep=',', index=False)
    print(f'The result is saved in {filepath_output}')