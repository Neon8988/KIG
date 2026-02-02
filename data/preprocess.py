import pandas as pd
import itertools
from tqdm import tqdm
from rdkit import Chem, rdBase
from data_preprocess import DataPreprocessor


dataset_name='kinase_organic'
ORIGINAL_DATASET = 'datasets/' + dataset_name + '.csv'
dec_min_len=8
scaf_min_len=8
PREPROCESSED_FILE = 'datasets/preprocessed_' + dataset_name + '_LEN_' + str(dec_min_len) + '_SCAF_' + str(scaf_min_len) + '.csv'
ATTACHMENT_POINT_TOKEN='*'
smarts='[*]!@-[*]'




preprocessor = DataPreprocessor(
            ORIGINAL_DATASET, 
            PREPROCESSED_FILE, 
            smarts,
            ATTACHMENT_POINT_TOKEN,
            dec_min_len,
            scaf_min_len,
            1
        )
smiles = preprocessor.read_smiles_from_file()
preprocessor.write_smiles_to_file([smiles])

data = pd.read_csv(PREPROCESSED_FILE, sep=';', names = ['scaffold', 'decorations', 'smiles'])
print('Preprocessed data shape:', data.shape)
data = data.drop_duplicates(subset=['scaffold', 'decorations', 'smiles'])
print('After removing duplicates, data shape:', data.shape)
data.to_csv(PREPROCESSED_FILE, sep=';', index=False, header=False)