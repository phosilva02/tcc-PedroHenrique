from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.cluster import adjusted_rand_score
#from util.foscKnn import FOSC
from util.fosc import FOSC

from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import random

no_of_colors=200
palleteColors=["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(no_of_colors)]


def plotPartition(x,y, result, title, saveDescription=None):
    uniqueValues = np.unique(result)
    
    fig = plt.figure(figsize=(15, 10))
    
    dicColors = {}
    dicColors[0] = "#000000"
    
    for  i in range(len(uniqueValues)):
        if uniqueValues[i] != 0:
            dicColors[uniqueValues[i]]= palleteColors[i]   
    
    
    for k, v in dicColors.items():
        plt.scatter(x[result==k], y[result==k], color=v )
    
    
    plt.title(title, fontsize=15)
    
    if saveDescription != None:
        plt.savefig(saveDescription)
        plt.close(fig)
        return
    
    plt.show() #\




def plotDendrogram(Z, result, title, saveDescription=None):
    uniqueValues = np.unique(result)
            
    dicColors = {}
    dicColors[0] = "#000000"

    for i in range(len(uniqueValues)):
        if uniqueValues[i] != 0:
            dicColors[uniqueValues[i]]= palleteColors[i]    
    
    
    colorsLeaf={}
    for i in range(len(result)):
        colorsLeaf[i] = dicColors[result[i]]
    
    
    # notes:
    # * rows in Z correspond to "inverted U" links that connect clusters
    # * rows are ordered by increasing distance
    # * if the colors of the connected clusters match, use that color for link
    linkCols = {}
    for i, i12 in enumerate(Z[:,:2].astype(int)):
        c1, c2 = (linkCols[x] if x > len(Z) else colorsLeaf[x]
                  for x in i12)
                
        linkCols[i+1+len(Z)] = c1 if c1 == c2 else dicColors[0]
    
    fig = plt.figure(figsize=(15, 10))
    
    dn = dendrogram(Z=Z, color_threshold=None, leaf_font_size=10,
                     leaf_rotation=45, link_color_func=lambda x: linkCols[x])
    plt.title(title, fontsize=12)
    
    
    if saveDescription != None:
        plt.savefig(saveDescription)
        plt.close(fig)
        return
    
    plt.show() #\


testingSynthetic = True
testingModQ = True

if(testingSynthetic):
    pathFiles = "../../dataset/sintetico/"

else:
    pathFiles = "../../dataset/pts2/"

listOfFiles = os.listdir(pathFiles)

listOfMClSize = [2, 4]#, 5, 8, 16, 20, 30]
methodsLinkage = ["single", "average", "complete"]#, "ward", "weighted"]

for fn in listOfFiles:
    if not fn.endswith(".csv"):continue
    if(fn == "groundTruth.csv"):continue
    
    varFileName = str(fn).rsplit(".", 1)[0]
    print ("\n\n\nPerforming experiments in dataset " + varFileName)
    
    matrix = np.genfromtxt(pathFiles+fn, dtype=float, delimiter=';', missing_values=np.nan)

    mat = matrix

    # Running tests
    for m in listOfMClSize:
        print("--------------------------------------- MCLSIZE = %d ---------------------------------------" % m)
        
        for lm in methodsLinkage:
            
            titlePlot = varFileName + "\n(" + lm + " and mClSize=" + str(m) + ")"
            savePath = pathFiles + varFileName + "-" + lm + " mClSize-" + str(m)+ ".png"
            if(testingModQ):
                saveDendrogram = pathFiles + varFileName + "-dendrogram-" + lm + " mClSize-" + str(m)+ "usingKNN.png"
            else:
                saveDendrogram = pathFiles + varFileName + "-dendrogram-" + lm + " mClSize-" + str(m)+ "usingStability.png"

            
            print("Using linkage method %s" % lm)
            Z= linkage(mat, method=lm, metric="euclidean")

            #print(Z)
            
                        
            foscFramework = FOSC(Z, mClSize=m, filename=pathFiles+fn)
            infiniteStability = foscFramework.propagateTree()
            partition, lastObjects = foscFramework.findProminentClusters(1,infiniteStability)
            
            # Plot results
            #plotPartition(mat[:,0], mat[:,1], partition, titlePlot, savePath)     
            plotDendrogram(Z, partition, titlePlot, saveDendrogram)

            #print(f"partition shape = {mat.shape} ")

            #print(f"ARI = {adjusted_rand_score(groundTruth, partition)}")

            for cluster in foscFramework.getHierarchy():
                if cluster == None: continue
                print(f"Cluster {cluster.getLabel()} Value = {cluster.getStability()}")

            print("Clusters ids", np.unique(partition))
            print("Objetos significativos", lastObjects)

            if(testingSynthetic):
                groundTruth = np.genfromtxt("../../dataset/sintetico/groundTruth.csv", dtype=float, delimiter=';', missing_values=np.nan)
                print(f"ARI = {adjusted_rand_score(groundTruth, partition)}")



