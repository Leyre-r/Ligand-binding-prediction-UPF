#elegir las 20 proteinas para el analisis de ejemplos
import random 

file_completa = 'lista_500_final.txt'
file_training = 'list_training.txt'

lista_completa = []
with open(file_completa) as completa:
    for line in completa:
        line = line.strip('\n')
        line = line[14:]
        lista_completa.append(line)

lista_training = []
with open(file_training) as training:
    for line in training:
        line = line.strip('\n')
        line = line[14:]
        lista_training.append(line)

lista_posibles = []
nprote = 0

random.seed(42)

for prote in lista_completa:
    if prote not in lista_training:
        lista_posibles.append(prote)

size = 0
lista_analisis = []
while size < 21:
    protein = random.choice(lista_posibles)
    lista_analisis.append(protein)
    size = len(lista_analisis)

print(lista_analisis)

#comando para copiar los folders en Bash (desde la carpeta tests): 
# for folder in 5e7n 3s1h 2y7x 5jm4 4ehg 4c1g 4b12 3ti5 5jcj 3rf5 5fqb 5jm4 5vqu 4x6i 3qs4 4zam 4o2a 2ydj 2yc5 3qzt 4awi; do cp -r ../python_project/P-L/2011-2019/$folder .

#preparar carpetas en la carpeta tests para usar el script 'binding_site_evaluation'
#for folder in 5e7n 3s1h 2y7x 5jm4 4ehg 4c1g 4b12 3ti5 5jcj 3rf5 5fqb 5jm4 5vqu 4x6i 3qs4 4zam 4o2a 2ydj 2yc5 3qzt 4awi; do cp $folder/${folder}_protein.pdb pockets/; done
