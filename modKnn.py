#from scipy.cluster.hierarchy import linkage, dendrogram
#from util.fosc import FOSC

from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import os
#import random


def intersection_knn(A, B):
    non_zero_indices = [i for i, (a, b) in enumerate(zip(A, B)) if a != 0 and b != 0]
    return non_zero_indices

pathFiles = "../dataset/pts2/"
listOfFiles    = os.listdir(pathFiles)

listOfMClSize = [2]#, 5, 8, 16, 20, 30]

#methodsLinkage = ["single", "average", "ward", "complete", "weighted"]

for fn in listOfFiles:
    if not fn.endswith(".csv"):continue
    
    varFileName = str(fn).rsplit(".", 1)[0]
    print ("\n\n\nPerforming experiments in dataset " + varFileName)
    
    matrix = np.genfromtxt(pathFiles+fn, dtype=float, delimiter=';', missing_values=np.nan)
    #print(distanceMat)
    distanceMatrix = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            point1 = matrix[i]
            point2 = matrix[j]
            distanceMatrix[i,j] = np.linalg.norm(point1 - point2)
            #print(distanceMatrix[i,j])
    #print(distanceMatrix)
    dMax = distanceMatrix.max()
    similarityMatrix = np.zeros_like(distanceMatrix)
    similarityMatrix = 1 - (distanceMatrix/dMax)
    #print(similarityMatrix)
    k = listOfMClSize[0]
    adjacencyMatrix = np.zeros_like(similarityMatrix)

    for i in range(similarityMatrix.shape[0]):
        indicesSorted = np.argsort(similarityMatrix[i])[::-1]  # Índices em ordem decrescente de similaridade
        kNearest = indicesSorted[1:k+1]  # Exclui o próprio ponto (índice i)
        adjacencyMatrix[i, kNearest] = 1
        adjacencyMatrix[kNearest, i] = 1
    
    #print(adjacencyMatrix)

    knnMatrix = adjacencyMatrix * similarityMatrix

    #print(knnMatrix)
    sigmaMatrix = np.zeros_like(knnMatrix)
    for i in range(knnMatrix.shape[0]):
        for j in range(knnMatrix.shape[0]):
            if i == j: continue
            intersection = intersection_knn(knnMatrix[i], knnMatrix[j])
            #print(f'Inteseção entre {i} e {j} é de {intersection}')
            if len(intersection) == 0: continue
            upper = 0
            for x in intersection:
                upper += (knnMatrix[i,x] * knnMatrix[j,x])
            lowerRight = 0
            lowerLeft = 0
            for x in knnMatrix[i]:
                lowerRight += x**2
            for x in knnMatrix[j]:
                lowerLeft += x**2
            lowerRight = np.sqrt(lowerRight)
            lowerLeft = np.sqrt(lowerLeft)
            sigmaMatrix[i,j] = upper/(lowerLeft * lowerRight)
    
    #print(sigmaMatrix)
    ts = np.sum(sigmaMatrix)
    modList = []
    acc = 5
    while True:
        strC5 = input(f'Elementos de C{acc} ')
        if strC5 == 'U': break
        LC5 = strC5.split()
        lista_inteiros = [int(elemento) for elemento in LC5]
        IS = 0
        DS = 0
        for i in lista_inteiros:
            for j in lista_inteiros:
                if i >= j: continue
                IS += sigmaMatrix[i,j]
        for i in lista_inteiros:
            for j in range(sigmaMatrix.shape[1]):
                if i >= j: continue
                DS += sigmaMatrix[i,j]
        modQ = (IS/ts) - (DS/ts)**2
        
        
        print(f"C{acc} = {modQ}")
        acc -= 1


    print(f"TS = {ts}")


    