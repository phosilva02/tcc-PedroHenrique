import igraph as ig
import matplotlib.pyplot as plt
import numpy as np


class Knn:


    def __init__(self, distanceMatrix, k=2):
        self.distanceMatrix = distanceMatrix
        self.dMax = self.distanceMatrix.max()
        self.similarityMatrix = self.createSimilarityMatrix()
        #print(self.similarityMatrix)

        self.k = k

        self.knnGraph = self.createknnGraph()

        self.TotalEdgesWeight = np.sum(self.knnGraph.es["weight"])





    def createSimilarityMatrix(self):
        similarityMatrix = np.zeros_like(self.distanceMatrix)
        similarityMatrix = 1 - (self.distanceMatrix/self.dMax)
        return similarityMatrix
    
    def createknnGraph(self):
        knnEdges = []
        for i in range(self.similarityMatrix.shape[0]):
            indicesSorted = np.argsort(self.similarityMatrix[i])[::-1]  # Índices em ordem decrescente de similaridade
            kNearest = indicesSorted[1:self.k+1]
            for j in kNearest:
                if i < j:
                    if (i,j) not in knnEdges:
                        knnEdges.append((i,j))
                else:
                    if (j,i) not in knnEdges:
                        knnEdges.append((j,i))
            #print(knnEdges)
        g = ig.Graph(self.similarityMatrix.shape[0], knnEdges)
        sigmaEdges = []
        sigmaWeight = []
        for edge in g.es:
            uNeighbors = g.neighbors(edge.source)
            vNeighbors = g.neighbors(edge.target)
            #print(uNeighbors)
            #print(vNeighbors)
            upper = 0
            lowerLeft = 0
            lowerRight = 0
            emptyIntersection = True
            for x in uNeighbors:
                lowerLeft += (self.similarityMatrix[edge.source, x] ** 2)
                if x in vNeighbors:
                    emptyIntersection = False
                    upper += (self.similarityMatrix[edge.source, x] * self.similarityMatrix[edge.target, x])
            if emptyIntersection: continue

            for x in vNeighbors:
                lowerRight += (self.similarityMatrix[edge.target, x] ** 2)
            simWeighet = upper/ (np.sqrt(lowerLeft) + np.sqrt(lowerRight))
            sigmaEdges.append((edge.source, edge.target))
            sigmaWeight.append(simWeighet)

        sig = ig.Graph(self.similarityMatrix.shape[0])
        for i in range(len(sigmaEdges)):
            sig.add_edge(sigmaEdges[i][0], sigmaEdges[i][1], weight=sigmaWeight[i])
        return sig
    def calculateModQ(self, objects):
        IS = 0.0
        DS = 0.0
        #alreadyEvaluated = []
        for object in objects:
            #if(objects == [591, 626]):
                #print(f"Avaliando vertice {object}")
            #print(f"Vertex id = {object}")
            #print(f"Vertex [{self.knnGraph.vs.indices}]")
            #print(objects)
            #print(f"Tam = [{len(self.knnGraph.vs)}]")
            neighbors = self.knnGraph.neighbors(object)
            #if(objects == [591, 626]):
                #print(f"Vizinhos de {object} \n\t{neighbors}")
            for neighbor in neighbors:
                if neighbor < object: continue
                edge = self.knnGraph.es.find(_source= object, _target=neighbor)
                if neighbor in objects:
                    IS += edge["weight"]
                DS += edge["weight"]
        return ((IS/self.TotalEdgesWeight) - (DS/self.TotalEdgesWeight)**2)

    def getSimilarityMatrix(self):
        return self.similarityMatrix
    def getKnnGraph(self):
        return self.knnGraph
    def getDistanceMatrix(self):
        return self.distanceMatrix
    def getDMax(self):
        return self.dMax
    def getTS(self):
        return self.TotalEdgesWeight