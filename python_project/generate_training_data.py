"""
generate_training_data.py
------------------------
Module for generating the training dataset for Random Forest binding site prediction.

Processes a filtered proteins list via the 'grid' module to extract descriptors and exports a balanced CSV dataset. 
Part of the protein analysis pipeline for binding site detection.
"""

import pandas as pd
import grid

def generar_dataset(
    samples_csv,
    output_csv="dataset_test.csv",
    ratio_neg_pos=2,
    max_samples=None):

    """
    Parses a list of protein paths to calculate their descriptors and create a training/test set, balancing the proportion of negative SAS points. 

    Args:
        samples_csv (str): Path to the CSV file with the columns 'pdb_id', 'protein_path' and 'ligand_path'.
        output_csv (str): File where the calculated descriptors are saved.
        ratio_neg_pos (int): number of negative SAS points (target=0) to be included for each positive SAS point (target=1). By default is 2.
        max_samples (int, optional): maximum number of proteins from the list to be processed. If is None, process all of them.

        Returns:
            None: the function writes the result in 'output_csv'.
    """

    df_samples = pd.read_csv(samples_csv)

    if max_samples:
        df_samples = df_samples.iloc[:max_samples]

    print(f"Procesando {len(df_samples)} samples...\n")

    first_write = True
    total_rows = 0

    for i, row in df_samples.iterrows():

        pdb_id = row["pdb_id"]
        protein_path = row["protein_path"]
        ligand_path = row["ligand_path"]
        #pocket_path = row["pocket_path"]

        print(f"[{i+1}/{len(df_samples)}] {pdb_id}")

    
        # 1. PROCESSING SAMPLE
        df = grid.procesar_sample(protein_path, ligand_path) #pocket_path

        if df is None or len(df) == 0:
            print("⚠️ vacío, skip\n")
            continue

   
        # 2. BALANCING
        df_1 = df[df["target"] == 1]
        df_0 = df[df["target"] == 0]

        if len(df_1) == 0:
            print("⚠️ sin positivos, skip\n")
            continue

        df_0_sampled = df_0.sample(
            n=min(len(df_0), ratio_neg_pos * len(df_1)),
            random_state=42
        )

        df_bal = pd.concat([df_1, df_0_sampled])

        if first_write:
            df_bal.to_csv(output_csv, index=False)
            first_write = False
        else:
            df_bal.to_csv(output_csv, mode="a", header=False, index=False)

        total_rows += len(df_bal)

        print(f"added: {len(df_bal)} filas | total: {total_rows}\n")

    print("Final dataset generated:", output_csv)
    print(f"Total rows: {total_rows}")

if __name__ == "__main__":
    generar_dataset("samples.csv", output_csv="dataset_test.csv", ratio_neg_pos=2, max_samples=200)