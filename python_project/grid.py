"""
grid.py
------------------------
Feature Extraction Module: Implements the grid-based approach for SAS point generation and descriptors calculation.

Use:
    python grid.py file.pdb

Output:
    - pandas.DataFrame → DataFrame where each row represnts a SAS point and the columns include the calculated descriptors,
            the 'target' label and the PDB identifier.
"""

from html import parser

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from Bio.PDB import PDBParser
from rdkit import Chem 
import sys

def load_ligand_sdf(sdf_path):
    """
    Extracts the 3D atomic coordinates from an SDF ligand file.

    Args:
        sdf_path (str): Path to the ligand file in .sdf format.

    Returns:
        numpy.ndarray: An array of shape (N, 3) containing the coordinates of the N atoms in the ligand. 
        None: if the file cannot be read or contains no valid molecule.
    """
    try:
        mol = Chem.SDMolSupplier(sdf_path)[0]
        if mol is None:
            return None

        conf = mol.GetConformer()
        coords = []

        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])

        return np.array(coords)

    except Exception as e:
        print(f"Error reading ligand {sdf_path}: {e}")
        return None
    
# Reference dictionary
PROPERTIES = {
        'ALA': {'hydro': 1.8,  'aromatic': 0, 'polar': 0, 'charge': 0},
        'ARG': {'hydro': -4.5, 'aromatic': 0, 'polar': 1, 'charge': 1},
        'ASN': {'hydro': -3.5, 'aromatic': 0, 'polar': 1, 'charge': 0},
        'ASP': {'hydro': -3.5, 'aromatic': 0, 'polar': 1, 'charge': -1},
        'CYS': {'hydro': 2.5,  'aromatic': 0, 'polar': 1, 'charge': 0},
        'GLU': {'hydro': -3.5, 'aromatic': 0, 'polar': 1, 'charge': -1},
        'GLN': {'hydro': -3.5, 'aromatic': 0, 'polar': 1, 'charge': 0},
        'GLY': {'hydro': -0.4, 'aromatic': 0, 'polar': 0, 'charge': 0},
        'HIS': {'hydro': -3.2, 'aromatic': 0, 'polar': 1, 'charge': 0.5},
        'ILE': {'hydro': 4.5,  'aromatic': 0, 'polar': 0, 'charge': 0},
        'LEU': {'hydro': 3.8,  'aromatic': 0, 'polar': 0, 'charge': 0},
        'LYS': {'hydro': -3.9, 'aromatic': 0, 'polar': 1, 'charge': 1},
        'MET': {'hydro': 1.9,  'aromatic': 0, 'polar': 0, 'charge': 0},
        'PHE': {'hydro': 2.8,  'aromatic': 1, 'polar': 0, 'charge': 0},
        'PRO': {'hydro': -1.6, 'aromatic': 0, 'polar': 0, 'charge': 0},
        'SER': {'hydro': -0.8, 'aromatic': 0, 'polar': 1, 'charge': 0},
        'THR': {'hydro': -0.7, 'aromatic': 0, 'polar': 1, 'charge': 0},
        'TRP': {'hydro': -0.9, 'aromatic': 1, 'polar': 1, 'charge': 0},
        'TYR': {'hydro': -1.3, 'aromatic': 1, 'polar': 1, 'charge': 0},
        'VAL': {'hydro': 4.2,  'aromatic': 0, 'polar': 0, 'charge': 0}
    }


def process_sample(pdbfile, ligand_file): 
    """
    Generates a dataset that contains the structural and physicochemical descriptors calculated from a PDB file.

    It calculates the Solvent Accesible Surface (SAS) using a grid-based approach and characterizes each surface point by projecting the
    properties of the nearby atoms (hydrophobicity, charge, etc.). If a ligand is present near those points, it assigns a binary 'target' 
    label to them (1 for binding site, 0 otherwise), for the training of the model.
   
    Args:
        pdbfile (str): Path to the PDB file.
        ligand_file(str): Path to the .sdf ligand file.

    Returns:
        pandas.DataFrame: a DataFrame where each row represnts a SAS point and the columns include the calculated descriptors,
            the 'target' label and the PDB identifier.
        None: if a critical error occurred during the processing of the PDB file.

    """

    # Instance parser
    parser = PDBParser(QUIET=True)

    # Loading structure
    try:
        structure = parser.get_structure("file", pdbfile)
    except Exception as e: 
        print(f"Error loading {pdbfile}: {e}")
        return None 

    
    ligand_coords = load_ligand_sdf(ligand_file)

    if ligand_coords is None:
        ligand_coords = np.array([])

    # Obtaining coordinates of PDB Atomos
    protein_atoms = [a for a in structure.get_atoms() if a.get_parent().get_id()[0] == ' ']
    coords_atoms = np.array([a.get_coord() for a in protein_atoms])
    tree = KDTree(coords_atoms)
    list_atoms = protein_atoms

    # Creating Grid
    x_min = min(i[0] for i in coords_atoms)
    x_max = max(i[0] for i in coords_atoms)
    y_min = min(i[1] for i in coords_atoms)
    y_max = max(i[1] for i in coords_atoms)
    z_min = min(i[2] for i in coords_atoms)
    z_max = max(i[2] for i in coords_atoms)
    grid_points = np.mgrid[x_min:x_max:1, y_min:y_max:1, z_min:z_max:1].reshape(3, -1).T

    # Finding closer atoms
    distances, indices = tree.query(grid_points)

    # Finding the points in the surface 
    mask = (distances > 2.8) & (distances < 3.5)
    sas_points = grid_points[mask]

    # Asign Label to points 
    if len(ligand_coords) > 0:
        ligand_tree = KDTree(ligand_coords)
        distances_to_ligand = ligand_tree.query_ball_point(sas_points, r=4.0)
        target = [1 if len(neighbor) > 0 else 0 for neighbor in distances_to_ligand]
    else:
        target = [0] * len(sas_points)


    # Descriptors calculation for each point
    data_rows = []
    for i, punto in enumerate(sas_points):
        neighbor_6A = tree.query_ball_point(punto, 6.0)
        neighbor_10A = tree.query_ball_point(punto, 10.0)
        
        density_6A = len(neighbor_6A)

        # Features
        f_aromatic = 0
        f_hydro = 0
        f_bfactor = 0
        f_invalids = 0
        f_polar = 0    
        f_charge = 0

        for idx in neighbor_6A:
            atomo = list_atoms[idx]
            res_name = atomo.get_parent().get_resname() 
            dist = np.linalg.norm(punto - atomo.get_coord())
            peso = 1 / (dist + 0.5)       

            if res_name in PROPERTIES:
                props = PROPERTIES[res_name]
                f_hydro += props['hydro'] * peso
                f_aromatic += props['aromatic'] * peso
                f_bfactor += atomo.get_bfactor() * peso
                f_polar += props['polar'] * peso    
                f_charge += props['charge'] * peso

                # Feature Invalids
                atom_name = atomo.get_name().strip()
                if atom_name.startswith(('N', 'O')):
                    f_invalids += 1 * peso
    
        ratio_density = len(neighbor_6A) / (len(neighbor_10A) + 1)
        hydro_norm = f_hydro / (density_6A + 1)
        charge_norm = f_charge / (density_6A + 1)
        polar_norm = f_polar / (density_6A + 1)
        bfactor_norm = f_bfactor / (density_6A + 1)   
        hydro_polar_ratio = f_hydro / (f_polar + 1)
    
        unique_residues = len(set([
            list_atoms[idx].get_parent().get_resname()
            for idx in neighbor_6A
        ]))

        if density_6A > 0:
            bfactor_var = np.var([
                list_atoms[idx].get_bfactor()
                for idx in neighbor_6A
            ])
        else:
            bfactor_var = 0
    
        # Save descriptors
        row = {
            'protrusion': len(neighbor_10A),                
            'bfactor': bfactor_norm,
            'Invalids': f_invalids,
            'Aromatic': f_aromatic,
            'hydrophobic': hydro_norm,
            'polar': polar_norm,     
            'net_charge': charge_norm,
            'ratio_density': ratio_density,
            'bfactor_var': bfactor_var,
            'hydro_polar_ratio': hydro_polar_ratio,
            'unique_residues': unique_residues
        }

        data_rows.append(row)

    # Transform to DataFrame
    df = pd.DataFrame(data_rows)

    # Assign Labels (0 or 1)
    df['target'] = target  
    df['pdb_id'] = pdbfile

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_pdb = sys.argv[1]
        ligand_file = sys.argv[2]
        process_sample(file_pdb, ligand_file)
    else:
        print("Error: no file provided")