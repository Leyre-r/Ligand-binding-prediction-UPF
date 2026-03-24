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
        lista_csvs.append(df_proteina)
        
        if df_proteina is not None:
            lista_csvs.append(df_proteina)

# Unir todos en un solo archivo maestro
if lista_csvs:
    # pd.concat junta todos los DataFrames uno debajo del otro
    dataset_final = pd.concat(lista_csvs, ignore_index=True) 
    #pd.concat es lo que hace que el encabezado del csv solo salga 1 vez en el resultado final (pero funciona)
    
    # Ahora sí, guardamos el CSV gigante con toda la información
    dataset_final.to_csv("dataset_training_completo.csv", index=False) 
    print(f"¡Éxito! Dataset final creado con {len(dataset_final)} puntos.")
else:
    print("No se generaron datos.")