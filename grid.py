import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from Bio.PDB import PDBParser
#para leer archivos de otra carpeta - import os...

# 1. Instancias el parser
parser = PDBParser(QUIET=True)

# 2. Cargamos la estructura
#"1OV9": Es el ID que le asigno a la estructura dentro de Python
#"1OV9.pdb": Es la ruta del archivo físico que quiero leer / Habría que cambiarlo para que lea muchos ficheros
try:
    structure = parser.get_structure("1OV9", "1OV9.pdb")
except FileNotFoundError:
    print("Error: El archivo PDB no se encuentra.") 


# 3. Obtener coordenadas de todos los átomos del PDB
coords_atoms = np.array([atom.get_coord() for atom in structure.get_atoms()])
tree = KDTree(coords_atoms)

# 4. Crear la rejilla (Grid)
# Supongamos que x_range, y_range, z_range son tus límites
x_min = min(i[0] for i in coords_atoms)
x_max = max(i[0] for i in coords_atoms)
y_min = min(i[1] for i in coords_atoms)
y_max = max(i[1] for i in coords_atoms)
z_min = min(i[2] for i in coords_atoms)
z_max = max(i[2] for i in coords_atoms)
#print(x_min, x_max, y_min,y_max, z_min, z_max)
#.T → Para que cada fila sea un punto [x, y, z].
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
print(indices_vecinos)
# Crea una lista con todos los objetos 'Atom' en el mismo orden que el KDTree
lista_atomos = list(structure.get_atoms())

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


data_rows = []

# 9. Procesar los puntos de la superficie:
for i, punto in enumerate(sas_points):
    # Recuperar vecinos en diferentes radios
    vecinos_6A = indices_vecinos[i] # Ya lo tienes de antes
    vecinos_10A = tree.query_ball_point(punto, 10.0)
    vecinos_3_5A = tree.query_ball_point(punto, 3.5)
    
    print(f"--- Procesando punto {i} en coordenadas {punto} ---")
    
    # Inicializamos las features para este punto
    f_aromatic = 0
    f_hydro = 0
    f_bfactor = 0
    f_invalids = 0

    # Para un punto concreto:
    for idx in vecinos_6A:
        atomo = lista_atomos[idx]
        res_name = atomo.get_parent().get_resname() # Esto da el aminoácido 
        if res_name in ['HOH', 'WAT', 'H2O']: continue
        print(res_name)
        dist = np.linalg.norm(punto - atomo.get_coord())
        peso = 1 / (dist + 0.5)       

        if res_name in PROPIEDADES:
            # Proyección con peso por distancia (opcional pero mejor)
            props = PROPIEDADES[res_name]
            f_hydro += props['hydro'] * peso
            f_aromatic += props['aromatic'] * peso
            f_bfactor += atomo.get_bfactor() * peso

            # Cálculo de Invalids (N y O)
            if atomo.get_name().startswith(('N', 'O')):
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
df.to_csv("mi_dataset_proteina.csv", index=False)