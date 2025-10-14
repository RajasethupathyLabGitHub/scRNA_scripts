import numpy as np
import pandas as p
import matplotlib.pyplot as plt
import scanpy as sc
import sklearn
import gseapy
import csv
import os
import phenograph
import scipy
import seaborn as sns
from collections import Counter
from pathlib import Path
import scanpy.external as sce

directory = Path("/Desktop/")
adatACC = sc.read(directory / "adata_acc.h5ad")

## pre-processing
sc.pp.calculate_qc_metrics(adatACC, inplace = True)

# Identify MT-genes
mito_genes = adatACC.var_names[adatACC.var_names.str.startswith('MT-')]
index_mito_genes = [adatACC.var_names.get_loc(j) for j in mito_genes]
mito_frac = np.asarray(np.sum(adatACC.X[:, index_mito_genes], axis = 1)/np.sum(adatACC.X, axis = 1)).squeeze() * 100
adatACC.obs['mito_frac'] = mito_frac

# Filter cells with low number genes
sc.pp.filter_genes(adatACC, min_cells=np.exp(3))

# Identify ribosomal genes
rb_genes = adatACC.var_names[adatACC.var_names.str.startswith(('RPS','RPL','GM'))]
index_rb_genes = [adatACC.var_names.get_loc(j) for j in rb_genes]
rb_frac = np.asarray(np.sum(adatACC.X[:, index_rb_genes], axis = 1)/np.sum(adatACC.X, axis = 1)).squeeze() * 100
ribo_mk = np.in1d(adatACC.var_names.values.astype(str), rb_genes)
adatACC = adatACC[:,~ribo_mk]

# Normalization
adatACC.layers['raw_data'] = adatACC.X.copy()
sc.pp.normalize_total(adatACC, inplace=True)
adatACC.layers['norm_counts'] = adatACC.X.copy()
adatACC.X = np.log2(adatACC.X + 1)

# HVG
sc.pp.highly_variable_genes(adatACC, layer='raw_data', n_top_genes=4000, flavor='seurat_v3')
adatACC.uns['id_hvg'] = np.where(adatACC.var['highly_variable'])[0]

# PCA
sc.tl.pca(adatACC, n_comps=100, use_highly_variable=None)
adatACC.uns['loadings'] = adatACC.varm['PCs'][adatACC.var['highly_variable'], :]
adatACC.obsm['X_pca'] = adatACC.obsm['X_pca'][:, 0:30]

adatACC.varm['PCs'] = adatACC.varm['PCs'][:, 0:30]
adatACC.uns['loadings'] = adatACC.uns['loadings'][:, 0:30]

sc.pp.neighbors(adatACC, n_neighbors=30, use_rep='X_pca', metric='euclidean', key_added='neighbors_30')
sc.tl.umap(adatACC, neighbors_key='neighbors_30', min_dist=0.1)

# Clustering using PhenoGraph
sc.external.tl.phenograph(adatACC,clustering_algo='leiden',k=30,
    jaccard=True,primary_metric='euclidean',resolution_parameter=0.1)

## Celltyping 
# visualization of marker genes 
markers = p.read_csv(directory / 'CellTypeMarkers.csv')
celltypes = list(markers.columns.values)

toplot_dic = {}
for c in celltypes:
    cc = np.unique(np.array(markers[c].values.tolist()))
    toplot_dic[c] = [x for x in cc if str(x) != 'nan']

sc.pl.matrixplot(adatACC, toplot_dic, groupby = 'pheno_leiden', standard_scale = 'var')

adatACC.layers['zs_norm_log'] = scipy.stats.zscore(adatACC.X)

for i in toplot_dic.keys():
    genes = toplot_dic[i]
    genes_up = []
    for g in genes:
        genes_up = np.append(genes_up,g.upper())
    
    top_genes = np.intersect1d(adatACC.var.index, genes_up)
    val = np.sum(adatACC[:,top_genes].layers['zs_norm_log'],axis=1)
    val = np.array(val).flatten()
    label = "%s_markers"%(i)
    adatACC.obs[label] = val/len(top_genes)
    sc.pl.scatter(adatACC,color=label,legend_loc='none',basis='umap',color_map='Spectral_r')

## Assign cell types
codes = p.to_numeric(adatACC.obs['pheno_leiden'], errors='coerce')

label_map = {
    1: 'Astrocytes', 7: 'Astrocytes',
    0: 'Microglia', 5: 'Microglia',
    6: 'Neurons',
    13: 'Oligodendrocytes',
    16: 'Fibroblast-like',
    3: 'Vascular endothelial', 17: 'Vascular endothelial', 18: 'Vascular endothelial', 15: 'Vascular endothelial',
    9: 'Vascular epithelial cells',
    11: 'Perycites',   # (same spelling as original)
    2: 'Vascular Smooth Muscle', 14: 'Vascular Smooth Muscle', 10: 'Vascular Smooth Muscle', 19: 'Vascular Smooth Muscle',
    4: 'Neuroblas-like neurons', 12: 'Neuroblas-like neurons', 20: 'Neuroblas-like neurons',
    8: 'Ependymal cells'
}

adatACC.obs['Cell Type'] = codes.map(label_map).fillna('na')

## Batch correct on latent space
sce.pp.harmony_integrate(adatACC, 'batch_id')

sc.pp.neighbors(adatACC, n_neighbors=30, use_rep='X_pca_harmony', metric='euclidean', 
                key_added='neighbors_30_corrected')
sc.tl.umap(adatACC, neighbors_key = 'neighbors_30_corrected', min_dist=0.1)



