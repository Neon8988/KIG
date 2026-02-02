
## Requirements

Install dependencies using conda:
```bash
conda env create -n KIG -f env.yml
conda activate KIG
```

## Project Structure

```
KIG/
├── env.yml                 # Conda environment specification
├── main.py                 # Main training script
├── train.py                # Training utilities
├── test.py                 # Testing and evaluation
├── kinase_properties.py    # Kinase-specific property calculations
├── data/                   # Data processing modules
│   ├── data_iter.py        # Data loaders and iterators
│   ├── data_preprocess.py  # Preprocessing utilities
│   └── preprocess.py       # SMILES processing functions
├── datasets/               # Training and preprocessed datasets
│   ├── kinase_organic.csv  # Original kinase dataset
│   └── preprocessed_*.csv  # Preprocessed datasets with different parameters
└── src/                    # Core model implementations
    ├── discriminator.py    # Discriminator network
    ├── filler.py           # Generator (Filler) network
    ├── mol_metrics.py      # Molecular property calculations
    ├── rollout.py          # Policy gradient rollout
    └── utils.py            # Utility functions
```

## Data Preprocessing

Preprocess your SMILES dataset with different decoration/scaffold length constraints:

```python
from data.data_preprocess import DataPreprocessor

preprocessor = DataPreprocessor(
    input_file='datasets/kinase_organic.csv',
    output_file='datasets/preprocessed_data.csv',
    smarts='[*]!@-[*]',  # Bond breaking pattern
    attachment_point='*',
    dec_min_len=3,       # Minimum decoration length
    scaf_min_len=5,      # Minimum scaffold length
    cuts=1               # Number of cuts per molecule
)
preprocessor.process()
```

## Training

Train the SpotGAN model:

```bash
python main.py --dataset kinase_organic \
               --dec_min_len 3 \
               --scaf_min_len 5 \
               --epochs 200 \
               --batch_size 64
```


## Testing
Test the trained model on kinase scaffolds:
```bash
python main.py --test --scaffold_name 'kinase_scaffold' \
    --test_property 'kinase_properties'
```

## Generation

Generate new kinase inhibitors:

```python
from src.filler import FillerSampler
from mol_metrics import Tokenizer

# Load trained model and generate molecules
tokenizer = Tokenizer()
sampler = FillerSampler(model, tokenizer, max_len=20, batch_size=64)
generated_molecules = sampler.sample_from_scaffolds(scaffolds)
```



