# Kinase-specific molecular properties using consistent RDKit calculations
# All properties (FSP3, clogP, TPSA) calculated with RDKit for full consistency

from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors
import numpy as np

def calculate_fsp3(mol):
    """Calculate fraction of sp3 carbons - compatible with different RDKit versions"""
    if mol is None:
        return 0.0
    try:
        # Try newer RDKit method first
        return rdMolDescriptors.CalcFractionCsp3(mol)
    except AttributeError:
        # Fallback for older RDKit versions - manual calculation
        sp3_carbons = 0
        total_carbons = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                total_carbons += 1
                if atom.GetHybridization() == Chem.HybridizationType.SP3:
                    sp3_carbons += 1
        return sp3_carbons / total_carbons if total_carbons > 0 else 0.0

def get_kinase_properties(smiles):
    """
    Calculate molecular properties for kinase inhibitor optimization
    
    Args:
        smiles (str): SMILES string
        
    Returns:
        tuple: (fsp3, clogp, tpsa)
            - fsp3: Fraction of sp3 carbons (0-1)
            - clogp: Calculated logP using Crippen method
            - tpsa: Topological polar surface area (Å²)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, 0.0, 0.0
    
    try:
        # Calculate all properties using RDKit for consistency
        fsp3 = calculate_fsp3(mol)
        clogp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        
        return fsp3, clogp, tpsa
    except Exception as e:
        # Return zeros if calculation fails
        return 0.0, 0.0, 0.0

def batch_kinase_properties(smiles_list):
    """
    Calculate combined kinase-favorable properties score for a batch of SMILES
    
    Scoring based on optimal ranges for kinase inhibitors:
    - FSP3: 0.3-0.7 (higher sp3 content for better 3D shape diversity)
    - clogP: 1-4 (optimal lipophilicity for membrane permeability)
    - TPSA: 60-90 (drug-like polar surface area)
    
    Args:
        smiles_list (list): List of SMILES strings
        
    Returns:
        list: Combined scores (0-1) for each molecule
    """
    scores = []
    
    for smiles in smiles_list:
        if not smiles or len(smiles.strip()) == 0:
            scores.append(0.0)
            continue
            
        fsp3, clogp, tpsa = get_kinase_properties(smiles)
        
        # Skip if all properties are zero (calculation failed)
        if fsp3 == 0.0 and clogp == 0.0 and tpsa == 0.0:
            scores.append(0.0)
            continue
        
        # Score each property based on kinase-favorable ranges
        
        # FSP3 score: prefer 0.3-0.7, with higher values better
        if 0.3 <= fsp3 <= 0.7:
            fsp3_score = 1.0
        elif fsp3 < 0.3:
            fsp3_score = fsp3 / 0.3
        else:  # fsp3 > 0.7
            fsp3_score = max(0.0, 1.0 - (fsp3 - 0.7) / 0.3)  # Penalty for too high FSP3
        
        # clogP score: optimal range 1-4
        if 1.0 <= clogp <= 4.0:
            clogp_score = 1.0
        elif clogp < 1.0:
            clogp_score = max(0.0, clogp / 1.0)  # Linear decrease below 1
        else:  # clogp > 4
            clogp_score = max(0.0, 1.0 - (clogp - 4.0) / 6.0)  # Linear decrease above 4
        
        # TPSA score: optimal range 60-90
        if 60.0 <= tpsa <= 90.0:
            tpsa_score = 1.0
        elif tpsa < 60.0:
            tpsa_score = max(0.0, tpsa / 60.0)  # Linear increase up to 60
        else:  # tpsa > 90
            tpsa_score = max(0.0, 1.0 - (tpsa - 90.0) / 90.0)  # Linear decrease above 90
        
        # Combined score (equal weighting of all three properties)
        combined_score = (fsp3_score + clogp_score + tpsa_score) / 3.0
        scores.append(combined_score)
    
    return scores

def batch_fsp3(smiles_list):
    """Calculate normalized FSP3 scores for a batch of SMILES"""
    scores = []
    for smiles in smiles_list:
        if not smiles or len(smiles.strip()) == 0:
            scores.append(0.0)
            continue
        
        fsp3, _, _ = get_kinase_properties(smiles)
        
        # FSP3 score: optimal range 0.3-0.7
        if 0.3 <= fsp3 <= 0.7:
            fsp3_score = 1.0
        elif fsp3 < 0.3:
            fsp3_score = fsp3 / 0.3
        else:  # fsp3 > 0.7
            fsp3_score = max(0.0, 1.0 - (fsp3 - 0.7) / 0.3)  # Penalty for too high FSP3
        scores.append(max(0.0, min(1.0, fsp3_score)))
    
    return scores

def batch_clogp(smiles_list):
    """Calculate normalized clogP scores for a batch of SMILES"""
    scores = []
    for smiles in smiles_list:
        if not smiles or len(smiles.strip()) == 0:
            scores.append(0.0)
            continue
        
        _, clogp, _ = get_kinase_properties(smiles)
        
        # Normalize clogP to kinase-favorable range 1-4
        if 1.0 <= clogp <= 4.0:
            clogp_score = 1.0
        elif clogp < 1.0:
            clogp_score = max(0.0, clogp / 1.0)  # Linear decrease below 1
        else:  # clogp > 4
            clogp_score = max(0.0, 1.0 - (clogp - 4.0) / 6.0)  # Linear decrease above 4
        scores.append(clogp_score)
    
    return scores

def batch_tpsa(smiles_list):
    """Calculate normalized TPSA scores for a batch of SMILES"""
    scores = []
    for smiles in smiles_list:
        if not smiles or len(smiles.strip()) == 0:
            scores.append(0.0)
            continue
        
        _, _, tpsa = get_kinase_properties(smiles)
        
        # Normalize TPSA to kinase-favorable range 60-90
        if 60.0 <= tpsa <= 90.0:
            tpsa_score = 1.0
        elif tpsa < 60.0:
            tpsa_score = max(0.0, tpsa / 60.0)  # Linear increase up to 60
        else:  # tpsa > 90
            tpsa_score = max(0.0, 1.0 - (tpsa - 90.0) / 90.0)  # Linear decrease above 90
        scores.append(tpsa_score)
    
    return scores

