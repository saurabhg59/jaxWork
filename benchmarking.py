#!/usr/bin/env python

import pandas as pd
import numpy as np
from math import *
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score
import sys, umap, re, hdbscan
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.cluster import estimate_bandwidth, DBSCAN, AffinityPropagation, KMeans, MeanShift, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfTransformer
from subprocess import check_call

def runKmeans(nc,matrix,trueLabels):
	km=KMeans(n_clusters=nc).fit(matrix)
	labels=km.labels_
	# print(trueLabels)
	# print(labels)
	ari = adjusted_rand_score(trueLabels,labels)
	fmi = fowlkes_mallows_score(trueLabels,labels)
	amis = adjusted_mutual_info_score(trueLabels,labels)
	hs = homogeneity_score(trueLabels,labels)
	cs = completeness_score(trueLabels,labels)
	vms = v_measure_score(trueLabels,labels)
	return ("kMeans\t"+str(ari)+"\t"+str(fmi)+"\t"+str(amis)+"\t"+str(hs)+"\t"+str(cs)+"\t"+str(vms))
	
def runAggClu(nc,link,matrix,trueLabels):
	ac=AgglomerativeClustering(n_clusters=nc,linkage=link).fit(matrix)
	labels=ac.labels_
	ari = adjusted_rand_score(trueLabels,labels)
	fmi = fowlkes_mallows_score(trueLabels,labels)
	amis = adjusted_mutual_info_score(trueLabels,labels)
	hs = homogeneity_score(trueLabels,labels)
	cs = completeness_score(trueLabels,labels)
	vms = v_measure_score(trueLabels,labels)
	return ("AggClust\t"+str(ari)+"\t"+str(fmi)+"\t"+str(amis)+"\t"+str(hs)+"\t"+str(cs)+"\t"+str(vms))
	
def runMeanShift(matrix,trueLabels,bw):
	if not bw==0:
		ms=MeanShift(bandwidth=bw,bin_seeding=True).fit(matrix)
	else:
		ms=MeanShift(bin_seeding=True).fit(matrix)
	labels=ms.labels_
	ari = adjusted_rand_score(trueLabels,labels)
	fmi = fowlkes_mallows_score(trueLabels,labels)
	amis = adjusted_mutual_info_score(trueLabels,labels)
	hs = homogeneity_score(trueLabels,labels)
	cs = completeness_score(trueLabels,labels)
	vms = v_measure_score(trueLabels,labels)
	return ("MeanShift\t"+str(ari)+"\t"+str(fmi)+"\t"+str(amis)+"\t"+str(hs)+"\t"+str(cs)+"\t"+str(vms))

def runAffProp(dp,matrix,trueLabels):
	ap = AffinityPropagation(damping=dp).fit(matrix)
	labels = ap.labels_
	ari = adjusted_rand_score(trueLabels,labels)
	fmi = fowlkes_mallows_score(trueLabels,labels)
	amis = adjusted_mutual_info_score(trueLabels,labels)
	hs = homogeneity_score(trueLabels,labels)
	cs = completeness_score(trueLabels,labels)
	vms = v_measure_score(trueLabels,labels)
	return ("AffProp\t"+str(ari)+"\t"+str(fmi)+"\t"+str(amis)+"\t"+str(hs)+"\t"+str(cs)+"\t"+str(vms))

def runDbscan(ep,matrix,trueLabels):
	#never used
	db=DBSCAN(eps=ep,min_samples=10).fit(matrix)
	labels=db.labels_
	ari = adjusted_rand_score(trueLabels,labels)
	fmi = fowlkes_mallows_score(trueLabels,labels)
	amis = adjusted_mutual_info_score(trueLabels,labels)
	hs = homogeneity_score(trueLabels,labels)
	cs = completeness_score(trueLabels,labels)
	vms = v_measure_score(trueLabels,labels)
	return ("dbscan\t"+str(ari)+"\t"+str(fmi)+"\t"+str(amis)+"\t"+str(hs)+"\t"+str(cs)+"\t"+str(vms))

def runHdbscan(matrix,trueLabels):
	labels = hdbscan.HDBSCAN(min_samples=10).fit_predict(matrix)
	ari = adjusted_rand_score(trueLabels,labels)
	fmi = fowlkes_mallows_score(trueLabels,labels)
	amis = adjusted_mutual_info_score(trueLabels,labels)
	hs = homogeneity_score(trueLabels,labels)
	cs = completeness_score(trueLabels,labels)
	vms = v_measure_score(trueLabels,labels)
	return ("hdbscan\t"+str(ari)+"\t"+str(fmi)+"\t"+str(amis)+"\t"+str(hs)+"\t"+str(cs)+"\t"+str(vms))

def scienceData(inFile):
	columnsToIgnore = 3
	data = pd.read_table(inFile,sep="\t")
	cellTypes=list(data)[columnsToIgnore:]

	fix = lambda x: x.split(".")[0] # Need to modify this part based on file being used

	for i in range(len(cellTypes)):
		cellTypes[i] = fix(cellTypes[i])

	le = LabelEncoder()
	le.fit(cellTypes)
	colorIndices = le.transform(cellTypes)
	print(cellTypes)
	data = np.transpose(data.as_matrix().tolist())
	peaks = data[:columnsToIgnore]
	matrix = np.asarray(data[columnsToIgnore:],dtype="float")
	return(matrix,colorIndices,peaks,cellTypes)


inFile=sys.argv[1]
nClusters = int(sys.argv[2])
outFile = inFile.split(".")[0] + ".metrics"
plotFile = inFile.split(".")[0] + "_plots.png"
(matrix,colorIndices,peaks,cellTypes)=scienceData(inFile)

###############################################################################################################
# UMAP Part
###############################################################################################################

values = "UMAP\nClust\tari\tfmi\tamis\ths\tcs\tvms\n"

reducer = umap.UMAP(n_components=2,metric='jaccard',n_neighbors=20,min_dist=0,random_state=42)
embedding = reducer.fit_transform(matrix)

values += runHdbscan(matrix=embedding,trueLabels=colorIndices)+"\n"
values += runMeanShift(bw=0,matrix=embedding,trueLabels=colorIndices)+"\n"
values += runAggClu(nc=nClusters,link='ward',matrix=embedding,trueLabels=colorIndices)+"\n"
values += runAffProp(dp=0.9,matrix=embedding,trueLabels=colorIndices)+"\n"
values += runKmeans(nc=nClusters,matrix=embedding,trueLabels=colorIndices)+"\n"

with open('umap.txt','w') as OUTPUT:
	for i in range(len(cellTypes)):
		OUTPUT.write(str(embedding[i,0])+"\t"+str(embedding[i,1])+"\t"+cellTypes[i]+"\n")

###############################################################################################################
# SVD+tSNE part (with tfidf)
###############################################################################################################

values += "tfIdf+SVD+tSNE\nClust\tari\tfmi\tamis\ths\tcs\tvms\n"

tfidf=TfidfTransformer().fit_transform(matrix)
svd=TruncatedSVD(n_components=50,random_state=100)
tsvd=svd.fit_transform(tfidf)
t=TSNE(random_state=100,n_iter=3000)
test=t.fit_transform(tsvd)

values += runHdbscan(matrix=test,trueLabels=colorIndices)+"\n"
values += runMeanShift(bw=0,matrix=test,trueLabels=colorIndices)+"\n"
values += runAggClu(nc=nClusters,link='ward',matrix=test,trueLabels=colorIndices)+"\n"
values += runAffProp(dp=0.9,matrix=test,trueLabels=colorIndices)+"\n"
values += runKmeans(nc=nClusters,matrix=test,trueLabels=colorIndices)+"\n"

with open('tfidfSVD.txt','w') as OUTPUT:
	for i in range(len(cellTypes)):
		OUTPUT.write(str(test[i,0])+"\t"+str(test[i,1])+"\t"+cellTypes[i]+"\n")

###############################################################################################################
# NMF
###############################################################################################################

values += "NMF\nClust\tari\tfmi\tamis\ths\tcs\tvms\n"

nmf=NMF(n_components=2,random_state=100).fit_transform(matrix)

values += runHdbscan(matrix=nmf,trueLabels=colorIndices)+"\n"
values += runMeanShift(bw=0,matrix=nmf,trueLabels=colorIndices)+"\n"
values += runAggClu(nc=nClusters,link='ward',matrix=nmf,trueLabels=colorIndices)+"\n"
values += runAffProp(dp=0.9,matrix=nmf,trueLabels=colorIndices)+"\n"
values += runKmeans(nc=nClusters,matrix=nmf,trueLabels=colorIndices)+"\n"

with open('nmf.txt','w') as OUTPUT:
	for i in range(len(cellTypes)):
		OUTPUT.write(str(nmf[i,0])+"\t"+str(nmf[i,1])+"\t"+cellTypes[i]+"\n")

###############################################################################################################
# SVD+tSNE part (without tfidf)
###############################################################################################################

values += "SVD+tSNE\nClust\tari\tfmi\tamis\ths\tcs\tvms\n"

svd=TruncatedSVD(n_components=50,random_state=100)
tsvd=svd.fit_transform(matrix)
t=TSNE(random_state=100,n_iter=3000)
test=t.fit_transform(tsvd)

values += runHdbscan(matrix=test,trueLabels=colorIndices)+"\n"
values += runMeanShift(bw=0,matrix=test,trueLabels=colorIndices)+"\n"
values += runAggClu(nc=nClusters,link='ward',matrix=test,trueLabels=colorIndices)+"\n"
values += runAffProp(dp=0.9,matrix=test,trueLabels=colorIndices)+"\n"
values += runKmeans(nc=nClusters,matrix=test,trueLabels=colorIndices)+"\n"

with open('justSVD.txt','w') as OUTPUT:
	for i in range(len(cellTypes)):
		OUTPUT.write(str(test[i,0])+"\t"+str(test[i,1])+"\t"+cellTypes[i]+"\n")

###############################################################################################################
# Writing metrics to output file
###############################################################################################################

with open(outFile,"w+") as OUTPUT:
	OUTPUT.write(values)

###############################################################################################################
# Making plots in R
###############################################################################################################

check_call(["./makePlots.R",plotFile])
check_call(["rm","umap.txt"])
check_call(["rm",'tfidfSVD.txt'])
check_call(["rm",'nmf.txt'])
check_call(["rm",'justSVD.txt'])