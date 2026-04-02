"""
inferencia2.py
------------------------
Predicts binding sites from a PDB file using a trained Random Forest model.

Use:
    python inferencia2.py mi_proteina.pdb 

Outputs:
    - binding_site_residues.txt     → list of amino acids in the binding site
    - visualization.pml             → script for PyMOL
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from Bio.PDB import PDBParser
import joblib
import sys
import os


# ─────────────────────────────────────────────
# 1. REPLICAR EXACTAMENTE EL CÁLCULO DE FEATURES DE grid.py
# ─────────────────────────────────────────────

PROPIEDADES = {
    'ALA': {'hydro': 1.8,  'aromatic': 0, 'polar': 0, 'charge': 0},
    'ARG': {'hydro': -4.5, 'aromatic': 0, 'polar': 1, 'charge': 1},
    'ASN': {'hydro': -3.5, 'aromatic': 0, 'polar': 1, 'charge': 0},
    'ASP': {'hydro': -3.5, 'aromatic': 0, 'polar': 1, 'charge': -1},
    'CYS': {'hydro': 2.5,  'aromatic': 0, 'polar': 1, 'charge': 0},
    'GLU': {'hydro': -3.5, 'aromatic': 0, 'polar': 1, 'charge': -1},
    'GLN': {'hydro': -3.5, 'aromatic': 0, 'polar': 1, 'charge': 0},
    'GLY': {'hydro': -0.4, 'aromatic': 0, 'polar': 0, 'charge': 0},
    'HIS': {'hydro': -3.2, 'aromatic': 1, 'polar': 1, 'charge': 0.5},
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

def calcular_features(pdbfile):
    """
    Generates a dataset that contains the structural and physicochemical descriptors calculated from a PDB file.

    It calculates the Solvent Accesible Surface (SAS) using a grid-based approach and characterizes each surface point by projecting the
    properties of the nearby atoms (hydrophobicity, charge, etc.). Unlike the training version, this function returns the structural 
    objects (tree, atoms) needed for posterior residue mapping and visualization.

    Args:
        pdbfile (str): Path to the PDB file.
    Returns:
        tuple: A tuple containing the following elements:
            - pandas.DataFrame ('df'): Dataset where each row is a SAS point with its descriptors (columns).
            - numpy.ndarray ('sas_points'): Grid coordinates of the generated SAS points.
            - list ('lista_atomos'): List of Bio.PDB.Atom objects used for the calculation. 
            - Bio.PDB.Structure.Structure ('structure'): The full protein structure object.
            - scipy.spatial.KDtree ('tree'): Spatial index of protein atoms for fast neighbor search. 
        None: If an error occurred during PDB parsing, returns (None, None, None, None, None).
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("proteina", pdbfile)
    except Exception as e:
        print(f"Error cargando {pdbfile}: {e}")
        return None, None, None, None, None

    # Átomos de la proteína (solo ATOM, no HETATM)
    protein_atoms = [a for a in structure.get_atoms()
                     if a.get_parent().get_id()[0] == ' ']
    coords_atoms  = np.array([a.get_coord() for a in protein_atoms])
    lista_atomos  = protein_atoms
    tree          = KDTree(coords_atoms)

    # Grid y puntos SAS — idéntico a grid.py
    x_min, x_max = coords_atoms[:,0].min(), coords_atoms[:,0].max()
    y_min, y_max = coords_atoms[:,1].min(), coords_atoms[:,1].max()
    z_min, z_max = coords_atoms[:,2].min(), coords_atoms[:,2].max()

    grid_points = np.mgrid[x_min:x_max:1,y_min:y_max:1,z_min:z_max:1].reshape(3, -1).T

    distancias, _ = tree.query(grid_points)
    mask       = (distancias > 2.8) & (distancias < 3.5)
    sas_points = grid_points[mask]
    print(f"Puntos SAS generados: {len(sas_points)}")

    # Features — idéntico a grid.py
    data_rows = []

    for i, punto in enumerate(sas_points):
        vecinos_6A   = tree.query_ball_point(punto, 6.0)
        vecinos_10A  = tree.query_ball_point(punto, 10.0)
        vecinos_3_5A = tree.query_ball_point(punto, 3.5)

        f_aromatic = 0
        f_hydro    = 0
        f_bfactor  = 0
        f_invalids = 0
        f_polar = 0    
        f_charge = 0

        for idx in vecinos_6A:
            atomo    = lista_atomos[idx]
            res_name = atomo.get_parent().get_resname()
            dist     = np.linalg.norm(punto - atomo.get_coord())
            peso     = 1 / (dist + 0.5)

            if res_name in PROPIEDADES:
                props       = PROPIEDADES[res_name]
                f_hydro    += props['hydro']    * peso
                f_aromatic += props['aromatic'] * peso
                f_bfactor  += atomo.get_bfactor() * peso

                atom_name = atomo.get_name().strip()
                if atom_name.startswith(('N', 'O')):
                    f_invalids += 1 * peso

        data_rows.append({
            'protrusion':    len(vecinos_10A),
            'atom0':         len(vecinos_3_5A),
            'bfactor':       f_bfactor / (len(vecinos_6A) + 1),
            'Invalids': f_invalids,
            'Aromatic':    f_aromatic,
            'hydrophobic':   f_hydro,
            'polar': f_polar,     
            'net_charge': f_charge
        })

    df = pd.DataFrame(data_rows)
    return df, sas_points, lista_atomos, structure, tree


# ─────────────────────────────────────────────
# 2. MAPEAR PUNTOS POSITIVOS → RESIDUOS
# ─────────────────────────────────────────────

def mapear_residuos(sas_points, predicciones, lista_atomos, tree, radio=4.0):
    """
    Identifies protein residues associated with the predicted binding site points.

    For each SAS point classified as 'binding' (1), this function finds all nearby protein atoms within a specified radius 
    and retrieves their parent residues. It uses a spatial index (KDTree) for efficient neighbor searching.

    Args:
        sas_points (numpy.ndarray): Grid coordinates (x,y,z) of the SAS points.
        predicciones: Binary classification results (0 or 1) from the model.
        lista_atomos (list): list of Bio.PDB.Atom objects corresponding to the KDTree.
        tree (scipy.spatial.KDtree): spatial index of protein atoms. 
        radio (float, optional): search radius in Angstroms. Defaults to 4.0.
    Returns:
        list of tuples ('residuos_sorted'): sorted list of unique residues identified, where each tuple contains 
            (chain_id, res_seq, res_name).
    """
    puntos_binding = sas_points[predicciones == 1]
    print(f"Puntos predichos como binding site: {len(puntos_binding)}")

    residuos_binding = set()
    indices_vecinos = tree.query_ball_point(puntos_binding, radio)

    for vecinos in indices_vecinos:
        for idx in vecinos:
            atomo   = lista_atomos[idx]
            residuo = atomo.get_parent()
            chain   = residuo.get_parent().get_id()
            res_id  = residuo.get_id()[1]        # número de secuencia
            res_name = residuo.get_resname()
            residuos_binding.add((chain, res_id, res_name))

    # Ordenar por cadena y número de residuo
    residuos_sorted = sorted(residuos_binding, key=lambda x: (x[0], x[1]))
    return residuos_sorted


# ─────────────────────────────────────────────
# 3. EXPORTAR LISTA DE RESIDUOS (.txt)
# ─────────────────────────────────────────────

def guardar_residuos_txt(residuos, pdbfile, output="binding_site_residues.txt"):
    """
    Exports the predicted binding site residues to a formatted text file.

    The created report includes the PDB filename, the total count of identified residues, and a table with the chain identifier, 
    sequence number, and amino acid name for each residue.

    Args:
        residuos (list of tuples): List of identified residues, where each tuple contains (chain_id, res_id, res_name).
        pdbfile (str): Path to the original PDB file.
        output (str, optional): Name or path of the file where results will be saved. Defaults to "binding_site_residues.txt".
    """

    pdb_name = os.path.basename(pdbfile)
    with open(output, 'w') as f:
        f.write(f"Binding site residues — {pdb_name}\n")
        f.write(f"Total residues: {len(residuos)}\n")
        f.write("=" * 40 + "\n")
        f.write(f"{'Chain':<8}{'ResNum':<10}{'ResName'}\n")
        f.write("-" * 40 + "\n")
        for chain, res_id, res_name in residuos:
            f.write(f"{chain:<8}{res_id:<10}{res_name}\n")
    print(f"Lista de residuos guardada en: {output}")


# ─────────────────────────────────────────────
# 4. EXPORTAR SCRIPT PYMOL (.pml)
# ─────────────────────────────────────────────
def guardar_pymol(residuos, pdbfile, output="visualization.pml"):
    """
    Generates a PyMOL (.pml) script for 3D visualization of the predicted binding site.

    The PyMOL script:
    1. Loads the protein structure and sets a neutral gray cartoon representation.
    2. Creates a specific selection of the predicted residues using chain and sequence IDs.
    3. Displays the binding site residues as red sticks and a semi-transparent surface.
    
    Args:
        residuos (list of tuples): List of identified residues, where each tuple contains (chain_id, res_id, res_name).
        pdbfile (str): Path to the original PDB file.
        output (str, optional): Name or path of the file where results will be saved. Defaults to "visualization.pml".
    """

    pdb_path = os.path.abspath(pdbfile)
    pdb_name = os.path.splitext(os.path.basename(pdbfile))[0]
 
    # Construir la selección por cadena
    selecciones_por_cadena = {}
    for chain, res_id, _ in residuos:
        selecciones_por_cadena.setdefault(chain, []).append(str(res_id))
 
    sel_parts = []
    for chain, ids in selecciones_por_cadena.items():
        ids_str = "+".join(ids)
        sel_parts.append(f"(chain {chain} and resi {ids_str})")
    sel_string = " or ".join(sel_parts)
 
    with open(output, 'w') as f:
        f.write(f"# PyMOL script — Binding site prediction\n")
        f.write(f"# Generated by predict_binding_site.py\n\n")
        f.write(f"load {pdb_path}, {pdb_name}\n\n")
        f.write(f"# Representación base\n")
        f.write(f"hide everything, {pdb_name}\n")
        f.write(f"show cartoon, {pdb_name}\n")
        f.write(f"color gray80, {pdb_name}\n\n")
        f.write(f"# Selección del binding site\n")
        f.write(f"select binding_site, {pdb_name} and ({sel_string})\n\n")
        f.write(f"# Colorear y mostrar el binding site\n")
        f.write(f"show sticks, binding_site\n")
        f.write(f"color red, binding_site\n")
        f.write(f"set stick_radius, 0.2\n\n")
        f.write(f"# Superficie del binding site\n")
        f.write(f"create bs_surface, binding_site\n")
        f.write(f"show surface, bs_surface\n")
        f.write(f"color tv_red, bs_surface\n")
        f.write(f"set transparency, 0.4, bs_surface\n\n")
        f.write(f"# Vista final\n")
        f.write(f"zoom binding_site\n")
        f.write(f"ray 1200, 900\n")
 
    print(f"Script PyMOL guardado en: {output}")
 

# ─────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def predecir_binding_site(pdbfile):

    """
    Main pipeline that transforms raw PDB structures into predicted binding sites.

    This function loads a pre-trained Random Forest model to evaluate SAS points based on their physicochemical descriptors
    and a confidence-based classification scheme. Finally, it maps the identified surface points back to their biological 
    residues, exporting the findings into a formatted text report and a PyMOL visualization script.

    Args:
        pdbfile (str): Path to the original PDB file to be analyzed.
    
    Returns: 
        tuple: a tuple containing:
        - list of tuples ('residuos')): List of identified residues, where each tuple contains (chain_id, res_id, res_name).
        - numpy.ndarray ('predicciones'): Binary classification results (0 or 1) for each SAS point.
        - numpy.ndarray ('probabilidades'): Probabilities scores (class 1) for each SAS point.
    """

    # Cargar modelo
    print("\nCargando modelo: modelo_rf_predictor.pkl")
    modelo = joblib.load("models/modelo_rf_predictor.pkl")

    # Calcular features
    print(f"\nCalculando features SAS para: {pdbfile}")
    resultado = calcular_features(pdbfile)
    if resultado[0] is None:
        print("Error: no se pudieron calcular las features.")
        return
    df_features, sas_points, lista_atomos, structure, tree = resultado

    # Asegurarse de que las columnas estén en el mismo orden que en training
    columnas_modelo = ['protrusion', 'atom0', 'bfactor', 'apRawInvalids',
                       'vsAromatic', 'hydrophobic']
    X_pred = df_features[columnas_modelo]

    # Predecir
    probabilidades = modelo.predict_proba(X_pred)[:, 1]

    # Definimos el umbral (threshold)
    umbral = 0.7
    print(f"\nEjecutando predicción con umbral de confianza {umbral}...")
    # Creamos las nuevas predicciones: 1 si prob >= 0.8, de lo contrario 0
    predicciones = (probabilidades >= umbral).astype(int)

    n_binding = int(predicciones.sum())
    print(f"Puntos totales evaluados: {len(predicciones)}")
    print(f"Puntos predichos como binding site (1): {n_binding}")
    print(f"Puntos predichos como no-binding  (0): {len(predicciones) - n_binding}")

    # Mapear a residuos
    print("\nMapeando puntos al binding site a residuos de la proteína...")
    residuos = mapear_residuos(sas_points, predicciones, lista_atomos, tree)
    print(f"Residuos únicos en el binding site: {len(residuos)}")

    # Guardar outputs
    pdb_base = os.path.splitext(os.path.basename(pdbfile))[0]
    guardar_residuos_txt(residuos, pdbfile,
                         output=f"{pdb_base}_binding_site_residues.txt")
    guardar_pymol(residuos, pdbfile,
                  output=f"{pdb_base}_visualization.pml")

    print("\n¡Predicción completada!")
    print(f"Archivos generados:")
    print(f"  - {pdb_base}_binding_site_residues.txt")
    print(f"  - {pdb_base}_visualization.pml")

    return residuos, predicciones, probabilidades


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python predict_binding_site.py <proteina.pdb> <modelo.pkl>")
        print("Ejemplo: python predict_binding_site.py 1OV9.pdb modelo_rf_predictor.pkl")
        sys.exit(1)

    pdbfile     = sys.argv[1]
    modelo_path = sys.argv[2]
    predecir_binding_site(pdbfile, modelo_path) 