import gzip
import math
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import QED
from rdkit import DataStructs
from rdkit.Chem import PandasTools, Crippen
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from kinase_properties import batch_kinase_properties, batch_fsp3, batch_clogp, batch_tpsa
rdBase.DisableLog('rdApp.error')

# Kinase property optimization utilities
def kinase_multi_objective_reward(smiles_list, weights=None):
    """
    Multi-objective reward combining FSP3, clogP, and TPSA for kinase inhibitors
    
    Args:
        smiles_list: List of SMILES strings
        weights: Dict with keys 'fsp3', 'clogp', 'tpsa' (default: equal weighting)
    
    Returns:
        List of weighted combined scores
    """
    if weights is None:
        weights = {'fsp3': 1.0, 'clogp': 1.0, 'tpsa': 1.0}
    
    # Get individual property scores
    fsp3_scores = batch_fsp3(smiles_list)
    clogp_scores = batch_clogp(smiles_list)
    tpsa_scores = batch_tpsa(smiles_list)
    
    # Weighted combination
    total_weight = sum(weights.values())
    combined_scores = []
    
    for f, c, t in zip(fsp3_scores, clogp_scores, tpsa_scores):
        score = (weights['fsp3'] * f + weights['clogp'] * c + weights['tpsa'] * t) / total_weight
        combined_scores.append(score)
    
    return combined_scores

def kinase_diversity_penalty(smiles_list, diversity_weight=0.1):
    """
    Add diversity penalty to encourage exploration in kinase chemical space
    
    Args:
        smiles_list: List of SMILES strings
        diversity_weight: Weight for diversity penalty term
    
    Returns:
        List of diversity-penalized scores
    """
    base_scores = batch_kinase_properties(smiles_list)
    
    if len(smiles_list) <= 1:
        return base_scores
    
    # Calculate pairwise similarities
    diversity_penalty = batch_diversity(smiles_list)
    
    # Apply penalty (higher diversity -> lower penalty -> higher reward)
    penalized_scores = []
    for score in base_scores:
        penalized_score = score * (1.0 + diversity_weight * diversity_penalty)
        penalized_scores.append(min(1.0, penalized_score))  # Cap at 1.0
    
    return penalized_scores

# ===========================
class Tokenizer():

    def __init__(self):
        self.pad = '_'# Padding token (index 0)
        self.seq_start = '<'# Scaffold start token 
        self.seq_end = '>'# Scaffold end token

        self.mask_start = '{'# Decoration start token
        self.mask_end = '}'# Decoration end token
        self.mask = '*'# Attachment point marker
    
    # Define every token of the vocabulary 
    def build_vocab(self):
        chars=[]
        # atoms (carbon), replace Cl for Q and Br for W
        chars = chars + ['H', 'B', 'c', 'C', 'n', 'N', 'o', 'O', 'p', 'P', 's', 'S', 'F', 'Q', 'W', 'I']
        # hidrogens: H2 to Z, H3 to X
        chars = chars + ['[', ']', '+', 'Z', 'X']
        # bounding
        chars = chars + ['-', '=', '#', '.']
        # branches
        chars = chars + ['(', ')']
        # cycles
        chars = chars + ['1', '2', '3', '4', '5', '6', '7']
        # anit/clockwise
        chars = chars + ['@']
        # directional bonds
        chars = chars + ['/', '\\']
        #Important that pad gets value 0
        self.tokenlist = [self.pad, self.seq_start, self.seq_end, self.mask_start, self.mask_end, self.mask] + list(chars)
            
    @property
    def tokenlist(self):
        return self._tokenlist
    
    @tokenlist.setter
    def tokenlist(self, tokenlist):
        self._tokenlist = tokenlist
        # create the dictionaries      
        self.char_to_int = {c:i for i,c in enumerate(self._tokenlist)}
        self.int_to_char = {i:c for c,i in self.char_to_int.items()}
    
    # Encode a scaffold to a numerical list with seq_start and seq_end tokens
    def scaffold_encode(self, smi):
        encoded = []
        smi = smi.replace('Cl', 'Q')
        smi = smi.replace('Br', 'W')
        # hydrogens
        smi = smi.replace('H2', 'Z')
        smi = smi.replace('H3', 'X')

        return [self.char_to_int[self.seq_start]] + [self.char_to_int[s] for s in smi] + [self.char_to_int[self.seq_end]]
    
    # Encode a decoration to a numerical list with mask_start and mask_end tokens
    def decoration_encode(self, smi):
        encoded = []
        smi = smi.replace('Cl', 'Q')
        smi = smi.replace('Br', 'W')
        # hydrogens
        smi = smi.replace('H2', 'Z')
        smi = smi.replace('H3', 'X')

        return [self.char_to_int[self.mask_start]] + [self.char_to_int[s] for s in smi] + [self.char_to_int[self.mask_end]]
    
    # Decode the numerical list to a SMILES string
    def decode(self, ords):
        smi = ''.join([self.int_to_char[o] for o in ords]) 
        # hydrogens
        smi = smi.replace('Z', 'H2')
        smi = smi.replace('X', 'H3')
        # replace proxy atoms for double char atoms symbols
        smi = smi.replace('Q', 'Cl')
        smi = smi.replace('W', 'Br')
        
        return smi
    
    # Define the vocabulary size
    @property
    def n_tokens(self):
        return len(self.int_to_char)
    
# ===========================
# Select chemical properties
def reward_fn(properties, generated_smiles):
    """Main reward function for policy gradient training"""
    if properties == 'druglikeness':
        vals = batch_druglikeness(generated_smiles) 
    elif properties == 'solubility':
        vals = batch_solubility(generated_smiles)
    elif properties == 'synthesizability':
        vals = batch_SA(generated_smiles)   
    # elif properties == 'DRD2':
    #     vals = batch_DRD2(generated_smiles)
    elif properties == 'kinase_properties':
        # Use our new RDKit-based kinase property scoring
        vals = batch_kinase_properties(generated_smiles)
    elif properties == 'kinase_fsp3':
        # Individual FSP3 optimization
        vals = batch_fsp3(generated_smiles)
    elif properties == 'kinase_clogp':
        # Individual clogP optimization
        vals = batch_clogp(generated_smiles)
    elif properties == 'kinase_tpsa':
        # Individual TPSA optimization
        vals = batch_tpsa(generated_smiles)
    else:
        raise ValueError(f"Unknown property: {properties}")
        
    return vals

# ===========================
# Druglikeness
def batch_druglikeness(smiles):
    vals = []
    for sm in smiles:
        if len(sm) != 0:
            mol = Chem.MolFromSmiles(sm, sanitize=False)
            if mol is None:
                vals.append(0.0)
            else:
                try:
                    val = QED.default(mol)
                    vals.append(val)
                except ValueError:
                    vals.append(0.0)
        else:
            vals.append(0.0)
           
    return vals

# ===========================
# Diversity
def batch_diversity(smiles):
    scores = []
    df = pd.DataFrame({'smiles': smiles})
    PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
    fps = [GetMorganFingerprintAsBitVect(m, 4, nBits=2048) for m in df['mol'] if m is not None]
    for i in range(1, len(fps)):
        scores.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i], returnDistance=True))
    return np.mean(scores)

# ===========================
# Solubility
def batch_solubility(smiles):
    vals = []
    for sm in smiles: 
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            vals.append(0.0)
        else:
            low_logp = -2.12178879609
            high_logp = 6.0429063424
            logp = Crippen.MolLogP(mol)
            val = (logp - low_logp) / (high_logp - low_logp)
            val = np.clip(val, 0.1, 1.0)
            vals.append(val)
    return vals

# ===========================
# Read synthesizability model
def readSAModel(filename='SA_score.pkl.gz'):
    model_data = pickle.load(gzip.open(filename))
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    SA_model = outDict
    return SA_model
SA_model = readSAModel()

# ===========================
#synthesizability
def batch_SA(smiles):
    vals = []
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)
        if sm != '' and mol is not None and mol.GetNumAtoms() > 1:
            # fragment score
            fp = Chem.AllChem.GetMorganFingerprint(mol, 2)
            fps = fp.GetNonzeroElements()
            score1 = 0.
            nf = 0
            for bitId, v in fps.items():
                nf += v
                sfp = bitId
                score1 += SA_model.get(sfp, -4) * v
            score1 /= nf
            # features score
            nAtoms = mol.GetNumAtoms()
            nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            ri = mol.GetRingInfo()
            nSpiro = Chem.AllChem.CalcNumSpiroAtoms(mol)
            nBridgeheads = Chem.AllChem.CalcNumBridgeheadAtoms(mol)
            nMacrocycles = 0
            for x in ri.AtomRings():
                if len(x) > 8:
                    nMacrocycles += 1
            sizePenalty = nAtoms**1.005 - nAtoms
            stereoPenalty = math.log10(nChiralCenters + 1)
            spiroPenalty = math.log10(nSpiro + 1)
            bridgePenalty = math.log10(nBridgeheads + 1)
            macrocyclePenalty = 0.
            if nMacrocycles > 0:
                macrocyclePenalty = math.log10(2)
            score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
            score3 = 0.
            if nAtoms > len(fps):
                score3 = math.log(float(nAtoms) / len(fps)) * .5
            sascore = score1 + score2 + score3
            min = -4.0
            max = 2.5
            sascore = 11. - (sascore - min + 1) / (max - min) * 9.
            # smooth the 10-end
            if sascore > 8.:
                sascore = 8. + math.log(sascore + 1. - 9.)
            if sascore > 10.:
                sascore = 10.0
            elif sascore < 1.:
                sascore = 1.0
            val = (sascore - 5) / (1.5 - 5)
            val = np.clip(val, 0.1, 1.0)
            vals.append(val)
        else:
            vals.append(0.0)
    return vals

# ===========================
# Read DRD2 model
# try:
#     DRD2_model = pickle.load(open('DRD2_score.sav', 'rb'))
#     DRD2_model_available = True
# except (ValueError, EOFError, pickle.UnpicklingError) as e:
#     print(f"Warning: Could not load DRD2 model due to compatibility issue: {e}")
#     print("DRD2 scoring will return default values.")
#     DRD2_model = None
#     DRD2_model_available = False

# def batch_DRD2(smiles):
#     vals = []
#     if not DRD2_model_available:
#         # Return default values when model is not available
#         return [0.5] * len(smiles)  # Return neutral scores
    
#     for sm in smiles:
#         mol = Chem.MolFromSmiles(sm)
#         if len(sm) != 0 and mol:
#             try:
#                 morgan = [GetMorganFingerprintAsBitVect(mol, 2, 2048)]
#                 val = DRD2_model.predict_proba(np.array(morgan))[:, 1]
#                 val = val[0]
#                 vals.append(val)
#             except ValueError:
#                 vals.append(0.0)           
#         else:
#             vals.append(0.0)
           
#     return vals
























