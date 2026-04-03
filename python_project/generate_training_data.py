"""
generate_training_data.py
------------------------
Description. 

Use:

Output:
    dataset_test.csv  → 
"""
# import grid
# import pandas as pd
# import os

# lista_csvs = []
# carpeta = "files/"
# for archivo in os.listdir(carpeta):
#     if archivo.endswith(".pdb"):
#         # Aquí llamas a tu función que calcula features + target
#         ruta_completa = os.path.join(carpeta, archivo)
#         df_proteina = grid.procesar_pdb(ruta_completa)
#         if df_proteina is not None:
#             lista_csvs.append(df_proteina)

# # Unir todos en un solo archivo maestro
# if lista_csvs:
#     # pd.concat junta todos los DataFrames uno debajo del otro
#     dataset_final = pd.concat(lista_csvs, ignore_index=True) 
#     #pd.concat es lo que hace que el encabezado del csv solo salga 1 vez en el resultado final (pero funciona)    
# else:
#     print("No se generaron datos.")

# print(f"Dataset original: {len(dataset_final)} filas")

# # Aplicar el downsampling
# df_clase_0 = dataset_final[dataset_final['target'] == 0]
# df_clase_1 = dataset_final[dataset_final['target'] == 1]
# # balance 1:1 o 1:2 
# df_clase_0_reducido = df_clase_0.sample(n=2*len(df_clase_1), random_state=42)


# # Ahora sí, guardamos el CSV gigante con toda la información
# df_balanceado = pd.concat([df_clase_1, df_clase_0_reducido])
# print(f"Dataset balanceado: {len(df_balanceado)} filas")

# # PASO 4: Guardar el CSV final para el training
# df_balanceado.to_csv("dataset_training_optimizado.csv", index=False)
# print("Éxito! dataset_training_optimizado.csv ha sido creado")

import pandas as pd
import grid

def generar_dataset(
    samples_csv,
    output_csv="dataset_test.csv",
    ratio_neg_pos=2,
    max_samples=None
):

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

        # ─────────────────────────
        # 1. PROCESAR SAMPLE
        # ─────────────────────────
        df = grid.procesar_sample(protein_path, ligand_path) #pocket_path

        if df is None or len(df) == 0:
            print("⚠️ vacío, skip\n")
            continue

        # ─────────────────────────
        # 2. BALANCEO
        # ─────────────────────────
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

        # ─────────────────────────
        # 3. GUARDADO INCREMENTAL
        # ─────────────────────────
        if first_write:
            df_bal.to_csv(output_csv, index=False)
            first_write = False
        else:
            df_bal.to_csv(output_csv, mode="a", header=False, index=False)

        total_rows += len(df_bal)

        print(f"✔ añadido: {len(df_bal)} filas | total: {total_rows}\n")

    print("✅ Dataset final generado:", output_csv)
    print(f"Total filas: {total_rows}")

if __name__ == "__main__":
    generar_dataset("samples.csv", output_csv="dataset_test.csv", ratio_neg_pos=2, max_samples=200)