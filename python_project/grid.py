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

def cargar_ligando_sdf(sdf_path):
    """
    DOCSTRING
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
        print(f"Error leyendo ligando {sdf_path}: {e}")
        return None
    
def procesar_sample(pdbfile, ligand_file): 
    """
    Generates a dataset that contains the structural and physicochemical descriptors calculated from a PDB file.

    It calculates the Solvent Accesible Surface (SAS) using a grid-based approach and characterizes each surface point by projecting the
    properties of the nearby atoms (hydrophobicity, charge, etc.). If a ligand is present near those points, it assigns a binary 'target' 
    label to them (1 for binding site, 0 otherwise), for the training of the model.
   
    Args:
        pdbfile (str): Path to the PDB file.
    Returns:
        pandas.DataFrame: a DataFrame where each row represnts a SAS point and the columns include the calculated descriptors,
            the 'target' label and the PDB identifier.
        None: if a critical error occurred during the processing of the PDB file.

    """

    # 1. Instancias el parser
    parser = PDBParser(QUIET=True)

    # 2. Cargamos la estructura
    #"file": Es el ID que le asigno a la estructura dentro de Python
    #pdbfile: Es la ruta del archivo físico que quiero leer / Habría que cambiarlo para que lea muchos ficheros
    try:
        structure = parser.get_structure("file", pdbfile)
    except Exception as e: # Captura cualquier error de lectura - se podría hacer más específico
        print(f"Error cargando {pdbfile}: {e}")
        return None 

    
    ligand_coords = cargar_ligando_sdf(ligand_file)

    if ligand_coords is None:
        ligand_coords = np.array([])

    # 3. Obtener coordenadas de todos los átomos del PDB
    protein_atoms = [a for a in structure.get_atoms() if a.get_parent().get_id()[0] == ' ']
    coords_atoms = np.array([a.get_coord() for a in protein_atoms])
    tree = KDTree(coords_atoms)
    
    # IMPORTANTE: Esta es la lista que usaremos para recuperar la información química
    # Ahora el índice del KDTree coincidirá perfectamente con esta lista
    lista_atomos = protein_atoms

    # 4. Crear la rejilla (Grid)
    # Supongamos que x_range, y_range, z_range son tus límites
    x_min = min(i[0] for i in coords_atoms)
    x_max = max(i[0] for i in coords_atoms)
    y_min = min(i[1] for i in coords_atoms)
    y_max = max(i[1] for i in coords_atoms)
    z_min = min(i[2] for i in coords_atoms)
    z_max = max(i[2] for i in coords_atoms)
    grid_points = np.mgrid[x_min:x_max:1, y_min:y_max:1, z_min:z_max:1].reshape(3, -1).T

    # 5. Encontrar los átomos que están más cerca de cada punto (y guardar las distancias)
    # Consultar la distancia al átomo más cercano para TODOS los puntos a la vez
    distancias, indices = tree.query(grid_points)

    # 6. Encontrar los puntos que están en la superficie (2.8 A < distancia atomo-punto < 3.5 A)
    # Aplicar una "máscara" booleana para filtrar los puntos 
    mask = (distancias > 2.8) & (distancias < 3.5)

    # Estos son tus puntos finales de la superficie
    sas_points = grid_points[mask]

    #print(f"De {len(grid_points)} puntos iniciales, {len(sas_points)} son superficie.")

    # 7. ASIGNAR TARGETS (Comparando SAS_POINTS con LIGAND_COORDS)
    if len(ligand_coords) > 0:
        ligand_tree = KDTree(ligand_coords)
        distancias_al_ligando = ligand_tree.query_ball_point(sas_points, r=4.0)
        target = [1 if len(vecinos) > 0 else 0 for vecinos in distancias_al_ligando]
    else:
        target = [0] * len(sas_points)

    # 8. Diccionario de referencia 
    PROPIEDADES = {
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



    # 9. Procesar los puntos de la superficie:
    data_rows = []
    for i, punto in enumerate(sas_points):

        # Recuperar vecinos en diferentes radios
        vecinos_6A = tree.query_ball_point(punto, 6.0)
        vecinos_10A = tree.query_ball_point(punto, 10.0)
        #vecinos_3_5A = tree.query_ball_point(punto, 3.5)
        
        density_6A = len(vecinos_6A)

        # Inicializamos las features para este punto
        f_aromatic = 0
        f_hydro = 0
        f_bfactor = 0
        f_invalids = 0
        f_polar = 0    
        f_charge = 0

        # Para un punto concreto:
        for idx in vecinos_6A:
            atomo = lista_atomos[idx]
            res_name = atomo.get_parent().get_resname() # Esto da el aminoácido 
            dist = np.linalg.norm(punto - atomo.get_coord())
            peso = 1 / (dist + 0.5)       

            if res_name in PROPIEDADES:
                # Proyección con peso por distancia (opcional pero mejor)
                props = PROPIEDADES[res_name]
                f_hydro += props['hydro'] * peso
                f_aromatic += props['aromatic'] * peso
                f_bfactor += atomo.get_bfactor() * peso
                f_polar += props['polar'] * peso    
                f_charge += props['charge'] * peso

                # Cálculo de Invalids (N y O)
                atom_name = atomo.get_name().strip()
                if atom_name.startswith(('N', 'O')):
                    f_invalids += 1 * peso
    
        ratio_density = len(vecinos_6A) / (len(vecinos_10A) + 1)
        hydro_norm = f_hydro / (density_6A + 1)
        charge_norm = f_charge / (density_6A + 1)
        polar_norm = f_polar / (density_6A + 1)
        bfactor_norm = f_bfactor / (density_6A + 1)   
        hydro_polar_ratio = f_hydro / (f_polar + 1)
    
        unique_residues = len(set([
            lista_atomos[idx].get_parent().get_resname()
            for idx in vecinos_6A
        ]))

        if density_6A > 0:
            bfactor_var = np.var([
                lista_atomos[idx].get_bfactor()
                for idx in vecinos_6A
            ])
        else:
            bfactor_var = 0
    
        # Guardar todos los resultados en una fila
        fila = {
            'protrusion': len(vecinos_10A),                
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

        data_rows.append(fila)

    # 10. Convertir a DataFrame y guardar
    df = pd.DataFrame(data_rows)

    # 2. Añadimos la columna de etiquetas (0 o 1)
    df['target'] = target  
    df['pdb_id'] = pdbfile

    
    # 3. Guardamos el archivo final
    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
    #Coger el archivo que queremos predecir desde la command line
        file_pdb = sys.argv[1]
        ligand_file = sys.argv[2]
        print("Usar procesar_sample desde otro script")
        procesar_sample(file_pdb, ligand_file)
    else:
        print("Error: no file provided")