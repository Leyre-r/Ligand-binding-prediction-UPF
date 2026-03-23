import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from Bio.PDB import PDBParser

#para leer archivos de otra carpeta 
"""
import os

lista_csvs = []
for archivo in os.listdir("mis_pdbs/"):
    if archivo.endswith(".pdb"):
        # Aquí llamas a tu función que calcula features + target
        df_proteina = procesar_pdb_y_generar_features(archivo)
        lista_csvs.append(df_proteina)

# Unir todos en un solo archivo maestro
dataset_final = pd.concat(lista_csvs, ignore_index=True)
dataset_final.to_csv("train_data_completo.csv", index=False)
"""


# 1. Instancias el parser
parser = PDBParser(QUIET=True)

# 2. Cargamos la estructura
#"1OV9": Es el ID que le asigno a la estructura dentro de Python
#"1OV9.pdb": Es la ruta del archivo físico que quiero leer / Habría que cambiarlo para que lea muchos ficheros
try:
    structure = parser.get_structure("file", "files/1trn.pdb")
except FileNotFoundError:
    print("Error: El archivo PDB no se encuentra.") 

# 2. IDENTIFICAR COORDENADAS DEL LIGANDO (Tu nuevo bloque va aquí)
ligand_coords = []
for model in structure:
    for chain in model:
        for residue in chain:
            # Filtro de seguridad para HETATM
            if residue.get_id()[0].startswith('H_'):
                res_name = residue.get_resname()
                ignore = ['HOH', 'WAT', 'GOL', 'SO4', 'PO4', 'CL', 'MG', 'ZN', 'NA']
                
                # Solo ligandos con más de 5 átomos y que no estén en la lista negra
                if res_name not in ignore and len(residue) > 5:
                    for atom in residue:
                        ligand_coords.append(atom.get_coord())

ligand_coords = np.array(ligand_coords)

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

print(f"De {len(grid_points)} puntos iniciales, {len(sas_points)} son superficie.")


# 7. Busco qué átomos están a menos de 6A.
radio_busqueda = 6.0

# Esto devuelve una lista de listas: una lista de índices de átomos por cada punto
indices_vecinos = tree.query_ball_point(sas_points, radio_busqueda)

# 7.2. ASIGNAR TARGETS (Comparando SAS_POINTS con LIGAND_COORDS)
if len(ligand_coords) > 0:
    ligand_tree = KDTree(ligand_coords)
    distancias_al_ligando = ligand_tree.query_ball_point(sas_points, r=4.0)
    target = [1 if len(vecinos) > 0 else 0 for vecinos in distancias_al_ligando]
else:
    target = [0] * len(sas_points)

# 8. Diccionario de referencia (puedes ampliarlo con tablas de internet)
PROPIEDADES = {
    'ALA': {'hydro': 1.8,  'aromatic': 0, 'polar': 0},
    'ARG': {'hydro': -4.5, 'aromatic': 0, 'polar': 1},
    'ASN': {'hydro': -3.5, 'aromatic': 0, 'polar': 1},
    'ASP': {'hydro': -3.5, 'aromatic': 0, 'polar': 1},
    'CYS': {'hydro': 2.5,  'aromatic': 0, 'polar': 1},
    'GLU': {'hydro': -3.5, 'aromatic': 0, 'polar': 1},
    'GLN': {'hydro': -3.5, 'aromatic': 0, 'polar': 1},
    'GLY': {'hydro': -0.4, 'aromatic': 0, 'polar': 0},
    'HIS': {'hydro': -3.2, 'aromatic': 1, 'polar': 1},
    'ILE': {'hydro': 4.5,  'aromatic': 0, 'polar': 0},
    'LEU': {'hydro': 3.8,  'aromatic': 0, 'polar': 0},
    'LYS': {'hydro': -3.9, 'aromatic': 0, 'polar': 1},
    'MET': {'hydro': 1.9,  'aromatic': 0, 'polar': 0},
    'PHE': {'hydro': 2.8,  'aromatic': 1, 'polar': 0},
    'PRO': {'hydro': -1.6, 'aromatic': 0, 'polar': 0},
    'SER': {'hydro': -0.8, 'aromatic': 0, 'polar': 1},
    'THR': {'hydro': -0.7, 'aromatic': 0, 'polar': 1},
    'TRP': {'hydro': -0.9, 'aromatic': 1, 'polar': 1},
    'TYR': {'hydro': -1.3, 'aromatic': 1, 'polar': 1},
    'VAL': {'hydro': 4.2,  'aromatic': 0, 'polar': 0}
}



# 9. Procesar los puntos de la superficie:
data_rows = []
for i, punto in enumerate(sas_points):
    # Recuperar vecinos en diferentes radios
    vecinos_6A = indices_vecinos[i] # Ya lo tienes de antes
    vecinos_10A = tree.query_ball_point(punto, 10.0)
    vecinos_3_5A = tree.query_ball_point(punto, 3.5)
    
    # Inicializamos las features para este punto
    f_aromatic = 0
    f_hydro = 0
    f_bfactor = 0
    f_invalids = 0

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

            # Cálculo de Invalids (N y O)
            atom_name = atomo.get_name().strip()
            if atom_name.startswith(('N', 'O')):
                f_invalids += 1 * peso
    
    # Guardar todos los resultados en una fila
    fila = {
        'protrusion': len(vecinos_10A),      # Neighbor Count
        'atom0': len(vecinos_3_5A),           # Presencia de átomos
        'bfactor': f_bfactor / (len(vecinos_6A) + 1),
        'apRawInvalids': f_invalids,
        'vsAromatic': f_aromatic,
        'hydrophobic': f_hydro,
        # Aquí añadirías la columna "Target" si estás en fase de entrenamiento
    }

    data_rows.append(fila)

# 10. Convertir a DataFrame y guardar
df = pd.DataFrame(data_rows)

# 2. Añadimos la columna de etiquetas (0 o 1)
# IMPORTANTE: Esto solo funciona si 'target' tiene el mismo número de filas que 'df'
df['target'] = target  

# 3. Guardamos el archivo final
df.to_csv("datos_entrenamiento.csv", index=False)

# Un pequeño truco para verificar que todo ha ido bien:
n_positivos = sum(target)
n_negativos = len(target) - n_positivos
print(f"Procesamiento finalizado.")
print(f"Puntos totales: {len(target)}")
print(f"Puntos en el bolsillo (Target 1): {n_positivos}")
print(f"Puntos fuera (Target 0): {n_negativos}")


#NOTA: Al entrenar el modelo, recuerda usar un parámetro como class_weight='balanced' en tu Random Forest.