"""
obtaining_proteins.py
------------------------
Module for parsing the PDBbind dataset and selecting valid protein-ligand samples.

This script scans the PDBbind dataset directory, filters samples based on file existence and size, and generates a CSV file with paths 
to protein and ligand files. It supports random sampling to create smaller training/testing subsets.
"""

import os
import pandas as pd
import random

def find_sample_files(sample_path):
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
    protein_path = None
    ligand_path = None
    pocket_path = None

    for filename in os.listdir(sample_path):
        filename_lower = filename.lower()

        if "protein.pdb" in filename_lower:
            protein_path = os.path.join(sample_path, filename)

        elif "ligand.sdf" in filename_lower:
            ligand_path = os.path.join(sample_path, filename)

        elif "pocket.pdb" in filename_lower:
            pocket_path = os.path.join(sample_path, filename)

    return protein_path, ligand_path, pocket_path


def build_cleam_dataset(
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
        era_path = os.path.join(root_dir, era)

        #Skip non-directory entries
        if not os.path.isdir(era_path):
            continue

        print(f"Scanning directory: {era}")

        for pdb_id in os.listdir(era_path):
            sample_path = os.path.join(era_path, pdb_id)
            #Skip non-directory entries
            if not os.path.isdir(sample_path):
                continue

            total_folders += 1

            protein, ligand, pocket = find_sample_files(sample_path)

          
            # FILTERS
            # Protein and ligand must exist
            if protein is None or ligand is None:
                continue

            # Enforce minimum protein size
            if os.path.getsize(protein) < min_protein_size_kb * 1024:
                continue

            # Reqiuire pocket file if specified
            if require_pocket and pocket is None:
                continue

            samples.append({
                "pdb_id": pdb_id,
                "protein_path": protein,
                "ligand_path": ligand,
                "pocket_path": pocket if pocket else ""
            })

    print(f"\nTotal folders analyzed: {total_folders}")
    print(f"Valid samples found: {len(samples)}")

    # OPTIONAL SAMPLING
    df = pd.DataFrame(samples)

    #Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    #Subsample if requested
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        print(f"Subsampling applied: {len(df)} samples")
    
    #Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nCSV saved to: {output_csv}")

    return df

# Execution
if __name__ == "__main__":
    dataset_path = "P-L"  

    df = build_cleam_dataset(
        root_dir=dataset_path,
        output_csv="samples.csv",
        min_protein_size_kb=50,
        require_pocket=False,   # Set to True if you want to only include samples with pocket files
        sample_size=200,        # Set to None to include all valid samples, or specify a number for random sampling
        seed=42
    )
