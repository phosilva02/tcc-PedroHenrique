from util.dendrogram import DendrogramStructure
from util.clusterKNN import Cluster
from util.utilClasses import TreeSet

import numpy as np

class FOSC:
    WARNING_MESSAGE = "----------------------------------------------- WARNING -----------------------------------------------\n" \
                    "With your current settings, the cluster stability is not well-defined. It could receive\n" '''
                    infinite values for some data objects, either due to replicates in the data (not a set) or due to numerical\n
                    roundings (this does not affect the construction of the clustering hierarchy). For this reason,\n
                    the post-processing routine to extract a flat partition containing the most stable clusters may\n
                    produce unexpected results. It may be advisable to increase the value of mClSize.\n
                    -------------------------------------------------------------------------------------------------------"""
      '''


    def __init__(self, Z, mClSize=4,filename=None):
        
        self.ds = DendrogramStructure(Z)
        self.numObjects = self.ds.getNumObjects()
        
        nextClusterLabel = 2
        currentClusterLabels = np.ones(self.numObjects, dtype=np.int64)
        
        self.clusters = []
        self.clusters.append(None)
        
        # Creating the first cluster of the cluster tree
        self.clusters.append(Cluster(1, None, np.NaN, self.numObjects))

        
        affectedClusterLabels = TreeSet()
        
        for currentLevelWeight in self.ds.getSignificantLevels():
            #print("Labels at level %.5f: %s" %(currentLevelWeight, currentClusterLabels))
            affectedNodes = TreeSet()
            #print(affectedNodes)
            affectedNodes.addAll(self.ds.getAffectedNodesAtLevel(currentLevelWeight))
            #print(affectedNodes)
            for nodeId in affectedNodes:
                if currentClusterLabels[self.ds.getFirstObjectAtNode(nodeId)] != 0:
                    affectedClusterLabels.add(currentClusterLabels[self.ds.getFirstObjectAtNode(nodeId)])
            
            
            if affectedClusterLabels.isEmpty(): continue
            
            #print("Level %.5f. Affected labels %s" %(currentLevelWeight, affectedClusterLabels))
            
            while not affectedClusterLabels.isEmpty():
                examinedClusterLabel = affectedClusterLabels.last()
                affectedClusterLabels.remove(examinedClusterLabel)
                examinedNodes = TreeSet()
                
                # Get all the affected nodes that are members of the cluster currently being examined
                for nodeId in affectedNodes:
                    if currentClusterLabels[self.ds.getFirstObjectAtNode(nodeId)] == examinedClusterLabel:
                        examinedNodes.add(nodeId)
                
                #print("Level %.5f. Affected nodes in dendrogram for cluster %d: %s" %(currentLevelWeight, examinedClusterLabel, affectedNodes))
                # Check if the examinedNodes represent a cluster division or a cluster shrunk
                validChildren = TreeSet()
                virtualChildNodes = TreeSet()
                
                for nodeId in examinedNodes:
                    if self.ds.getNodeSize(nodeId) >= mClSize:
                        validChildren.add(nodeId)
                    else:
                        virtualChildNodes.add(nodeId)
                
                # If we have more than two valid child, we create new clusters, setup the
                # parent and ajust the parent's death level.
                # print("Level %.5f. Valid nodes for cluster %d: %s" %(currentLevelWeight, examinedClusterLabel, validChildren))
                
                if len(validChildren) >= 2:
                    for nodeId in validChildren:
                        newCluster = self._createNewCluster(self.ds.getObjectsAtNode(nodeId), currentClusterLabels, self.clusters[examinedClusterLabel], nextClusterLabel, currentLevelWeight)
                        self.clusters.append(newCluster)
                        nextClusterLabel += 1
                
                # We have to assign the noise label for all the objects in virtual child nodes list. We also have to update the respective cluster parent.
                for nodeId in virtualChildNodes:
                    if currentClusterLabels[self.ds.getFirstObjectAtNode(nodeId)] != 0:
                        self._createNewCluster(self.ds.getObjectsAtNode(nodeId), currentClusterLabels, self.clusters[examinedClusterLabel], 0, currentLevelWeight)
    
    
    '''createNewCluster
    Function to create a new cluster structure, or update the cluster when there is a shrunk of it
    (the children do not satisfy the mClSize parameter)
    '''
    def _createNewCluster(self, points, clusterLabels, parentCluster, clusterLabel, levelWeight):
        for point in points:
            clusterLabels[point] = clusterLabel
        
        parentCluster.detachPoints(len(points), levelWeight)
        
        if clusterLabel != 0:
            cluster = Cluster(clusterLabel, parentCluster, levelWeight, len(points))
            cluster.setObjects(points)
            return cluster
        else:
            parentCluster.addPointsToVirtualChildCluster(points)
            return None
    
    ''' propagateTree()
    Propagates constraint satisfaction, stability, bcubed index, and lowest child death level from each child
    cluster to each parent cluster in the tree.  This method must be called before calling
    findProminentClusters()
    return true if there are any clusters with infinite stability, false otherwise
    '''
    
    def propagateTree(self):
        clustersToExamine = TreeSet()
        addedToExaminationList = []
        infiniteStability = False
        
        
        for i in range(len(self.clusters)):
            addedToExaminationList.append(False)
        
        # Find all leaf clusters in the cluster tree
        for cluster in self.clusters:
            if cluster == None: continue
            
            if not cluster._hasChildren():
                clustersToExamine.add(cluster.getLabel())
                addedToExaminationList[cluster.getLabel()] = True
            
        while len(clustersToExamine) > 0:
            currentLabel = clustersToExamine.pop(-1)
            currentCluster = self.clusters[currentLabel]            
            currentCluster.propagate()
            
            if currentCluster.getStability() == np.Inf or currentCluster.getStability() == np.NaN:
                infiniteStability = True
            
            if currentCluster.getParent() != None:
                parent = currentCluster.getParent()
                
                if not addedToExaminationList[parent.getLabel()]:
                    clustersToExamine.add(parent.getLabel())
                    addedToExaminationList[parent.getLabel()] = True
            
        if infiniteStability:
            print(FOSC.WARNING_MESSAGE)
        
        return infiniteStability
    
    
    
    def findProminentClusters(self, rootTree, infiniteStability):
        partition = np.zeros(self.numObjects, dtype=np.int64)
        solution = self.clusters[rootTree].getPropagatedDescendants()
        significantObjects={}


        for cluster in solution:
            #print(f"Cluster {cluster.getLabel()} Value = {cluster.getStability()}")
            affectedNodes = self.ds.getAffectedNodesAtLevel(cluster.getDeathLevel())
            lastPoints=[]
            for idNode in affectedNodes:
                lastPoints+= self.ds.getObjectsAtNode(idNode)


            for point in cluster.getObjects():
                if point in lastPoints:
                    significantObjects[cluster.getLabel()]=point

                partition[point] = cluster.getLabel()
        
        return partition, significantObjects
    
    
    def getHierarchy(self):
        return self.clusters