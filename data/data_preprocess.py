import warnings
import itertools
from tqdm import tqdm
from rdkit import Chem, rdBase
import pandas as pd
warnings.filterwarnings('ignore')
rdBase.DisableLog('rdApp.error')

# ===========================
class DataPreprocessor:
    
    def __init__(
        self, 
        input_file, 
        output_file, 
        smarts,
        attachment_point_token,
        dec_min_len=3,
        scaf_min_len=3,
        cuts=1
    ):
        """
        input_file: the original SMILES dataset
        output_file: the generated file that contains (scaffold, decorations, smiles) pairs
        cuts: the desired number of cuts (decorations) to perform
        smarts: split the bond that is not in a ring
        attachment_point_token: '*'
        dec_min_len: minimum number of atoms for decorations
        scaf_min_len: minimum number of atoms for scaffolds
        """
        self.input_file = input_file
        self.output_file = output_file
        self.cuts = cuts
        # Convert SMARTS string to RDKit molecule pattern
        self.smarts = Chem.MolFromSmarts(smarts)
        if self.smarts is None:
            raise ValueError(f"Invalid SMARTS pattern: {smarts}")
        self.attachment_point_token = attachment_point_token
        self.dec_min_len = dec_min_len
        self.scaf_min_len = scaf_min_len
        self.dec_min_len = dec_min_len
        
    # Define the cutting function according to self.smarts
    def get_matches(self, mol):
        """
        mol: a mol object
        return: a list of (atom1, atom2) pairs that can cut the mol between atom1 and atom2
        """
        matches = set()
        matches |= set(tuple(sorted(match)) for match in mol.GetSubstructMatches(self.smarts))

        return matches

    # Given a number of cuts, enumerate all possible combinations of scaffolds and decorations
    def enumerate_scaffold_decorations(self, mol):
        """
        mol: a mol object
        return: a list of possible (scaffold, decorations) tuples
        """
        matches = self.get_matches(mol)
        combined_mols = set()
        
        # Select the number of cuts pairs from the matched combinations
        for atom_pairs_to_cut in itertools.combinations(matches, self.cuts): 
            bonds_to_cut = list(sorted(mol.GetBondBetweenAtoms(aidx, oaidx).GetIdx() for aidx, oaidx in atom_pairs_to_cut)) # Get the bond IDs between the (atom1 and atom2) pairs
            attachment_point_idxs = [(i, i) for i in range(len(bonds_to_cut))]
            cut_mol = Chem.FragmentOnBonds(mol, bondIndices = bonds_to_cut, dummyLabels = attachment_point_idxs) # Obtain the substructures of the molecule with the attachment point  token
            
            # Set the representation of all atoms in the molecule with [atom:i]
            for atom in cut_mol.GetAtoms(): 
                if atom.GetSymbol() == self.attachment_point_token:
                    num = atom.GetIsotope()
                    atom.SetIsotope(0)
                    atom.SetProp('molAtomMapNumber', str(num))        
            cut_mol.UpdatePropertyCache()
            fragments = Chem.GetMolFrags(cut_mol, asMols=True, sanitizeFrags=True)
            
            # Save the generated scaffold and decorations
            scaffold = None
            decorations = []
            # Detect whether there is a scaffold and use the fragement with the same number of cuts as scaffold
            if self.cuts == 1:
                # Calculate the length of each frag
                len_frag0 = len([atom for atom in fragments[0].GetAtoms()])
                len_frag1 = len([atom for atom in fragments[1].GetAtoms()])
                # Generate both orderings to avoid systematic bias, checking both min scaffold and decoration lengths
                if len_frag0 >= self.scaf_min_len and len_frag1 >= self.dec_min_len:
                    # frag0 as scaffold, frag1 as decoration
                    combined_mols.add(tuple([fragments[0], tuple([fragments[1]])]))
                if len_frag1 >= self.scaf_min_len and len_frag0 >= self.dec_min_len:
                    # frag1 as scaffold, frag0 as decoration
                    combined_mols.add(tuple([fragments[1], tuple([fragments[0]])]))
            else:
                # Calculate the number of atoms for each frag
                len_frags = []
                for frag in fragments:
                    len_frags.append(frag.GetNumAtoms())
                    
                for frag in fragments:
                    num_attachment_points = len([atom for atom in frag.GetAtoms() if atom.GetSymbol() == self.attachment_point_token]) # Count the number of attachment point tokens of every slice
                    # Decide the scaffold and decorations
                    if num_attachment_points == self.cuts and not scaffold:
                        if frag.GetNumAtoms() == max(len_frags):
                            scaffold = frag
                    else:
                        decorations.append(frag)
                if scaffold:
                    combined_mols.add(tuple([scaffold, tuple(decorations)]))

        return list(combined_mols)

    def read_smiles_from_file(self):
        """
        return: a list of SMILES data
        """
        #smiles = []
        # with open(self.input_file, 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.strip()
        #         line = line.rstrip('\n')
        #         smiles.append(line)
        df=pd.read_csv(self.input_file)
        return df['SMILES'].tolist()

    def extract_decoration_properly(self, decoration_mol):
        """
        Properly extract decoration by removing attachment atom from molecular graph
        instead of string manipulation to avoid invalid SMILES
        """
        try:
            # Find attachment point atoms
            attachment_atoms = [atom.GetIdx() for atom in decoration_mol.GetAtoms() 
                              if atom.HasProp('molAtomMapNumber')]
            
            if not attachment_atoms:
                # No attachment point, return as-is
                return Chem.MolToSmiles(decoration_mol, canonical=False)
            
            # Create editable copy and remove attachment atom
            decoration_copy = Chem.RWMol(decoration_mol)
            decoration_copy.RemoveAtom(attachment_atoms[0])
            
            # Get fragments after removal
            frags = Chem.GetMolFrags(decoration_copy, asMols=True)
            
            if not frags:
                return ''  # Attachment was the only atom
            
            # Return largest fragment as SMILES
            largest_frag = max(frags, key=lambda x: x.GetNumAtoms())
            return Chem.MolToSmiles(largest_frag, canonical=False)
            
        except Exception as e:
            # Fallback to string replacement (may create invalid SMILES)
            decoration_smiles = Chem.MolToSmiles(decoration_mol, rootedAtAtom=0, canonical=False)
            return decoration_smiles.replace('[' + self.attachment_point_token + ':0]', '')

    # Write the (scaffold, decorations, mol) to output_file
    def write_smiles_to_file(self, smiles_generator):
        """
        smiles_generator: a generator that yields chunks of smiles data
        return: convert these SMILES data into (scaffold, decorations, smiles) pairs and save to output file
        """
        total_processed = 0
        total_valid = 0
        
        with open(self.output_file, 'w+') as fout:
            for smiles_chunk in smiles_generator:
                print(f"Processing chunk of {len(smiles_chunk)} molecules... (Total processed: {total_processed})")
                
                for smi in tqdm(smiles_chunk, desc="Processing molecules"):
                    total_processed += 1
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is None:
                            continue
                            
                        combined_mols = self.enumerate_scaffold_decorations(mol)  # Split the mol to (scaffold, decorations, smiles) pair
                        
                        for row in combined_mols:
                            scaffold = Chem.MolToSmiles(row[0], rootedAtAtom=0, canonical=False) #  Split the scaffold from the mol object                    
                            # Extract decorations using proper molecular graph method
                            decorations = []
                            for i in range(self.cuts):
                                # Use proper extraction method to avoid invalid SMILES
                                clean_decoration = self.extract_decoration_properly(row[1][i])
                                decorations.append(clean_decoration)
                            
                            # Find the order of decorations in scaffold and sort them
                            scaffold_positions = [scaffold.find('[' + self.attachment_point_token +':' + str(i) +']') for i in range(self.cuts)]  
                            sorted_indexs = sorted(range(self.cuts), key = lambda k: scaffold_positions[k]) # Order in which decorations appear in scaffold
                            decorations = [decorations[i] for i in sorted_indexs]  # Sort the decorations according to scaffold order
                            
                            # Now clean up the scaffold by replacing all attachment points
                            clean_scaffold = scaffold
                            for i in range(self.cuts):
                                clean_scaffold = clean_scaffold.replace('[' + self.attachment_point_token + ':' + str(i) + ']', self.attachment_point_token) # Replace the '[:*\d]' to '*'
                            
                            # Check if the combination is valid with better error handling
                            try:
                                # Skip if any decoration is empty (attachment was only atom)
                                if any(not dec.strip() for dec in decorations):
                                    continue
                                    
                                combined_smiles = clean_scaffold.replace(self.attachment_point_token, '{}').format(*decorations)
                                mol_reconstructed = Chem.MolFromSmiles(combined_smiles)
                                mol_original = Chem.MolFromSmiles(smi)
                                
                                # Check if reconstruction is valid and matches original
                                if (mol_reconstructed and mol_original and 
                                    Chem.MolToSmiles(mol_reconstructed) == Chem.MolToSmiles(mol_original)):
                                    
                                    if self.cuts > 1:
                                        decorations_str = ','.join(decorations)
                                    else:
                                        decorations_str = ''.join(decorations)
                                    #row_smiles = '{};{};{}'.format(clean_scaffold, decorations_str, combined_smiles) 
                                    row_smiles = '{};{};{}'.format(clean_scaffold, decorations_str, smi)
                                    fout.write('{}\n' .format(row_smiles))
                                    total_valid += 1
                                    
                            except (ValueError, IndexError, Exception):
                                # Skip invalid reconstructions (e.g., wrong number of decorations)
                                continue
                    except Exception:
                        # Skip problematic molecules
                        continue
                        
                print(f"Chunk complete. Valid pairs found: {total_valid} / {total_processed}")
        
        print(f"Preprocessing complete! Total valid pairs: {total_valid} from {total_processed} molecules")

                        
                        