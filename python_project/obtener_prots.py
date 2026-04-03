"""
obtener_prots.py
------------------------
Module for parsing the PDBbind dataset and selecting valid protein-ligand samples.

This script scans the PDBbind dataset directory, filters samples based on file existence and size, and generates a CSV file with paths 
to protein and ligand files. It supports random sampling to create smaller training/testing subsets.
"""

import os
import pandas as pd
import random

def encontrar_archivos_sample(sample_path):
    """
    Selects the PDB files from PDBbind to be used for the training of the model.

    Args:
        sample_path(str): Path to the individual PDB sample folder.
    
    Returns:
        tuple: A tuple containing the following elements:
        - protein (str): Path to the PDBbind _protein.pdb file. 
        - ligand (str): Path to the PDBbind _ligand.sdf file. 
        - pocket (str): Path to the PDBbind _pocket.pdb file. 
        None if the file is not found
    """
    protein = None
    ligand = None
    pocket = None

    for f in os.listdir(sample_path):
        f_lower = f.lower()

        if "protein.pdb" in f_lower:
            protein = os.path.join(sample_path, f)

        elif "ligand.sdf" in f_lower:
            ligand = os.path.join(sample_path, f)

        elif "pocket.pdb" in f_lower:
            pocket = os.path.join(sample_path, f)

    return protein, ligand, pocket


def obtener_prots_definitivo(
    root_dir,
    output_csv="samples.csv",
    min_protein_size_kb=50,
    require_pocket=False,
    sample_size=None,
    seed=42
):
    """
    Parses the PDBbind dataset and generates a curated CSV file.

    Args:
        root_dir (str): Path to the directory with the dataset folders
        output_csv (str): Name of the output CSV file to store the sample paths.
        min_protein_size_kb( int): Minimum size in KB for the protein PDB file to be considered valid.
        require_pocket (bool): If True, samples without a '_pocket.pdb' file will be discarded.
        sample_size (int, optional): Number of samples to randomly select for the final list.
        seed(int): Random seed for reproducibility of the sampling process.

    Returns:
        pandas.DataFrame ('samples.csv'): a DataFrame containing the columns: 'pdb_id', 'protein_path', 'ligand_path' and 'pocket_path'.
    """

    random.seed(seed)

    samples = []
    total_folders = 0

    for era in os.listdir(root_dir):
        ruta_era = os.path.join(root_dir, era)

        if not os.path.isdir(ruta_era):
            continue

        print(f"Exploring folder: {era}")

        for pdb_id in os.listdir(ruta_era):
            sample_path = os.path.join(ruta_era, pdb_id)

            if not os.path.isdir(sample_path):
                continue

            total_folders += 1

            protein, ligand, pocket = encontrar_archivos_sample(sample_path)

          
            # FILTERS
            # 1. deben existir protein y ligand
            if protein is None or ligand is None:
                continue

            # 2. tamaño mínimo proteína
            if os.path.getsize(protein) < min_protein_size_kb * 1024:
                continue

            # 3. opcional: requerir pocket
            if require_pocket and pocket is None:
                continue

            samples.append({
                "pdb_id": pdb_id,
                "protein_path": protein,
                "ligand_path": ligand,
                "pocket_path": pocket if pocket else ""
            })

    print(f"\nTotal folders analyzed: {total_folders}")
    print(f"Found valid samples: {len(samples)}")

    # OPTIONAL SAMPLING

    # ────────────────
    # SAMPLING OPCIONAL
    # ────────────────
    df = pd.DataFrame(samples)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        print(f"Submuestreo aplicado: {len(df)} samples")
    
    df.to_csv(output_csv, index=False)
    print(f"\nCSV saved in: {output_csv}")

    return df


if __name__ == "__main__":
    dataset_path = "P-L"  

    df = obtener_prots_definitivo(
        root_dir=dataset_path,
        output_csv="samples.csv",
        min_protein_size_kb=50,
        require_pocket=False,   # True si quieres usarlo más adelante
        sample_size=200,       
        seed=42
    )
