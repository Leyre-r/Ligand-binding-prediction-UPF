"""
generate_training_data.py
------------------------
Module for generating the training dataset for Random Forest binding site prediction.

Processes a filtered proteins list via the 'grid' module to extract descriptors and exports a balanced CSV dataset. 
Part of the protein analysis pipeline for binding site detection.
"""

import pandas as pd
import grid

def generate_dataset(
    samples_csv,
    output_csv="dataset_test.csv",
    negative_to_positive_ratio=2,
    max_samples=None):

    """
    Parses a list of protein paths to calculate their descriptors and create a training/test set, balancing the proportion of negative SAS points. 

    Args:
        samples_csv (str): Path to the CSV file with the columns 'pdb_id', 'protein_path' and 'ligand_path'.
        output_csv (str): File where the calculated descriptors are saved.
        negative_to_positive_ratio (int): number of negative SAS points (target=0) to be included for each positive SAS point (target=1). By default is 2.
        max_samples (int, optional): maximum number of proteins from the list to be processed. If is None, process all of them.

        Returns:
            None: the function writes the result in 'output_csv'.
    """
    
    #Load samples
    df_samples = pd.read_csv(samples_csv)
    if max_samples:
        df_samples = df_samples.iloc[:max_samples]

    print(f"Processing {len(df_samples)} samples...\n")

    first_write = True
    total_rows = 0

    for i, row in df_samples.iterrows():

        pdb_id = row["pdb_id"]
        protein_path = row["protein_path"]
        ligand_path = row["ligand_path"]
        print(f"[{i+1}/{len(df_samples)}] Processing {pdb_id}")

        # PROCESS SAMPLE
        df = grid.process_sample(protein_path, ligand_path) # optionally add pocket_path

        if df is None or len(df) == 0:
            print("Empty result, skipping\n")
            continue

   
        # BALANCE DATASET
        df_positive = df[df["target"] == 1]
        df_negative = df[df["target"] == 0]

        #Skip samples without positive points
        if len(df_positive) == 0:
            print("No positive samples, skipping\n")
            continue

        # Sample negatives according to ratio
        df_negative_sampled = df_negative.sample(
            n=min(len(df_negative), negative_to_positive_ratio * len(df_positive)),
            random_state=42
        )
        #Combine balanced dataset
        df_balanced = pd.concat([df_positive, df_negative_sampled])

        # Write to CSV 
        if first_write:
            df_balanced.to_csv(output_csv, index=False)
            first_write = False
        else:
            df_balanced.to_csv(output_csv, mode="a", header=False, index=False)

        total_rows += len(df_balanced)

        print(f"added: {len(df_balanced)} filas | total: {total_rows}\n")

    print("Final dataset generated:", output_csv)
    print(f"Total rows: {total_rows}")

if __name__ == "__main__":
    generate_dataset("samples.csv", output_csv="dataset_test.csv", negative_to_positive_ratio=2, max_samples=500)
