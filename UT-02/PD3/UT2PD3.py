from UT2_PD3 import print_first_rows
from math import sqrt
from random import randrange
from csv import reader
from statistics import mean, stdev
import csv

columnas = ['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
 	'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

with open('wine.data') as dataset:
    csv_reader = csv.reader(dataset, delimiter=',')
    lineas = []
    line_count = 0
    for row in csv_reader:
        lineas.append(row)
        line_count += 1
    print(f'Processed {line_count} lines.')

print(lineas)

#Imprimir las primeras 10 lineas
print ("====================== PRIMERAS 10 LINEAS ======================")
for i in range(10):
		print(lineas[i])

mins = []
maxs = []
#Minimos y Máximos de cada columna
print ("====================== MAXIMOS Y MINIMOS ======================")
for x in range(len(columnas)):
    valuesInColumn = []
    for y in range(line_count):
        valuesInColumn.append(float(lineas[y][x + 1]))
        
    maxV = max(valuesInColumn)
    maxs.append(maxV)
    minV = min(valuesInColumn)
    mins.append(minV)
    print(columnas[x] + ": " + "Max " + str(maxV) + ", Min " + str(minV))

means = []
#Promedio de cada columna
print ("====================== PROMEDIOS ======================")
for x in range(len(columnas)):
    valuesInColumn = []
    for y in range(line_count):
        valuesInColumn.append(float(lineas[y][x + 1]))
        
    columnMean = mean(valuesInColumn)
    means.append(columnMean)
    print(columnas[x] + ": " + "Promedio " + str(round(columnMean,2)))

stds = []
#Desviacion standard
print ("====================== DESVIACIONES STANDARD ======================")
for x in range(len(columnas)):
    valuesInColumn = []
    for y in range(line_count):
        valuesInColumn.append(float(lineas[y][x + 1]))
        
    desviacion = stdev(valuesInColumn)
    stds.append(desviacion)
    print(columnas[x] + ": " + "Desviacion Standard " + str(round(desviacion,2)))

print ("====================== DESVIACIONES STANDARD ======================")
print(stds)
print ("====================== PROMEDIOS ======================")
print(means)
print ("====================== MINIMOS ======================")
print(mins)
print ("====================== MAXIMOS ======================")
print(maxs)
print ("====================== NORMALIZACION ======================")

normalizedDataset =  [ [ 0 for i in range(len(columnas) + 1) ] for j in range(line_count) ]
for x in range(len(columnas)):
    for y in range(line_count):
        value = float(lineas[y][x + 1])
        normalizedDataset[y][x + 1] = ((value + means[x])/stds[x])
        
standarizedDataset =  [ [ 0 for i in range(len(columnas) + 1) ] for j in range(line_count) ]
for x in range(len(columnas)):
    for y in range(line_count):
        value = float(lineas[y][x + 1])
        standarizedDataset[y][x + 1] = (value - mins[x])/(maxs[x] - mins[x])
    
print ("====================== DIVIDE DATASET ======================")
def train_test_split(dataset, split=0.60):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy

train, test = train_test_split(lineas)

print_first_rows(train)
print("")
print_first_rows(test)
print("")
