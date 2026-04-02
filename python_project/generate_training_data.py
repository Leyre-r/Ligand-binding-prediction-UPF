import grid
import pandas as pd
import os

lista_csvs = []
carpeta = "files/"
for archivo in os.listdir(carpeta):
    if archivo.endswith(".pdb"):
        # Aquí llamas a tu función que calcula features + target
        ruta_completa = os.path.join(carpeta, archivo)
        df_proteina = grid.procesar_pdb(ruta_completa)
        if df_proteina is not None:
            lista_csvs.append(df_proteina)

# Unir todos en un solo archivo maestro
if lista_csvs:
    # pd.concat junta todos los DataFrames uno debajo del otro
    dataset_final = pd.concat(lista_csvs, ignore_index=True) 
    #pd.concat es lo que hace que el encabezado del csv solo salga 1 vez en el resultado final (pero funciona)    
else:
    print("No se generaron datos.")

print(f"Dataset original: {len(dataset_final)} filas")

# Aplicar el downsampling
df_clase_0 = dataset_final[dataset_final['target'] == 0]
df_clase_1 = dataset_final[dataset_final['target'] == 1]
# balance 1:1 o 1:2 
df_clase_0_reducido = df_clase_0.sample(n=2*len(df_clase_1), random_state=42)


# Ahora sí, guardamos el CSV gigante con toda la información
df_balanceado = pd.concat([df_clase_1, df_clase_0_reducido])
print(f"Dataset balanceado: {len(df_balanceado)} filas")

# PASO 4: Guardar el CSV final para el training
df_balanceado.to_csv("dataset_training_optimizado.csv", index=False)
print("Éxito! dataset_training_optimizado.csv ha sido creado")