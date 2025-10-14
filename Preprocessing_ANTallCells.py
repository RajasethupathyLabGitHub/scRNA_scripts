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
adataTH = sc.read(directory / "adata_thalamus.h5ad")

## pre-processing
sc.pp.calculate_qc_metrics(adataTH, inplace = True)

# Identify MT-genes
mito_genes = adataTH.var_names[adataTH.var_names.str.startswith('MT-')]
index_mito_genes = [adataTH.var_names.get_loc(j) for j in mito_genes]
mito_frac = np.asarray(np.sum(adataTH.X[:, index_mito_genes], axis = 1)/np.sum(adataTH.X, axis = 1)).squeeze() * 100
adataTH.obs['mito_frac'] = mito_frac

# Filter cells with low number genes
sc.pp.filter_genes(adataTH, min_cells=np.exp(3))

# Identify ribosomal genes
rb_genes = adataTH.var_names[adataTH.var_names.str.startswith(('RPS','RPL','GM'))]
index_rb_genes = [adataTH.var_names.get_loc(j) for j in rb_genes]
rb_frac = np.asarray(np.sum(adataTH.X[:, index_rb_genes], axis = 1)/np.sum(adataTH.X, axis = 1)).squeeze() * 100
ribo_mk = np.in1d(adataTH.var_names.values.astype(str), rb_genes)
adataTH = adataTH[:,~ribo_mk]

# Normalization
adataTH.layers['raw_data'] = adataTH.X.copy()
sc.pp.normalize_total(adataTH, inplace=True)
adataTH.layers['norm_counts'] = adataTH.X.copy()
adataTH.X = np.log2(adataTH.X + 1)

# HVG
sc.pp.highly_variable_genes(adataTH, layer='raw_data', n_top_genes=4000, flavor='seurat_v3')
adataTH.uns['id_hvg'] = np.where(adataTH.var['highly_variable'])[0]

# PCA
sc.tl.pca(adataTH, n_comps=100, use_highly_variable=None)
adataTH.uns['loadings'] = adataTH.varm['PCs'][adataTH.var['highly_variable'], :]
adataTH.obsm['X_pca'] = adataTH.obsm['X_pca'][:, 0:30]

adataTH.varm['PCs'] = adataTH.varm['PCs'][:, 0:30]
adataTH.uns['loadings'] = adataTH.uns['loadings'][:, 0:30]

sc.pp.neighbors(adataTH, n_neighbors=30, use_rep='X_pca', metric='euclidean', key_added='neighbors_30')
sc.tl.umap(adataTH, neighbors_key='neighbors_30', min_dist=0.1)

# Clustering using PhenoGraph
sc.external.tl.phenograph(adataTH,clustering_algo='leiden',k=30,
    jaccard=True,primary_metric='euclidean',resolution_parameter=0.1)

## Celltyping 
# visualization of marker genes 
markers = p.read_csv(directory / 'CellTypeMarkers.csv')
celltypes = list(markers.columns.values)

toplot_dic = {}
for c in celltypes:
    cc = np.unique(np.array(markers[c].values.tolist()))
    toplot_dic[c] = [x for x in cc if str(x) != 'nan']

sc.pl.matrixplot(adataTH, toplot_dic, groupby = 'pheno_leiden', standard_scale = 'var')

adataTH.layers['zs_norm_log'] = scipy.stats.zscore(adataTH.X)

for i in toplot_dic.keys():
    genes = toplot_dic[i]
    genes_up = []
    for g in genes:
        genes_up = np.append(genes_up,g.upper())
    
    top_genes = np.intersect1d(adataTH.var.index, genes_up)
    val = np.sum(adataTH[:,top_genes].layers['zs_norm_log'],axis=1)
    val = np.array(val).flatten()
    label = "%s_markers"%(i)
    adataTH.obs[label] = val/len(top_genes)
    sc.pl.scatter(adataTH,color=label,legend_loc='none',basis='umap',color_map='Spectral_r')

## Assign cell types
codes = p.to_numeric(adataTH.obs['pheno_leiden'], errors='coerce')

label_map = {
    1: 'Astrocytes', 8: 'Astrocytes',
    0: 'Microglia', 7: 'Microglia', 16: 'Microglia',
    6: 'Neurons',
    3: 'Oligodendrocytes', 11: 'Oligodendrocytes', 9: 'Oligodendrocytes', 13: 'Oligodendrocytes',
    12: 'Fibroblast-like',
    2: 'Vascular endothelial',
    5: 'Perycites',  # (typo preserved from original)
    4: 'Vascular Smooth Muscle', 14: 'Vascular Smooth Muscle', 15: 'Vascular Smooth Muscle',
    10: 'Neuroblast-like neurons',
    17: 'Ependymal cells'}

adataTH.obs['Cell Type'] = codes.map(label_map).fillna('na')

## Batch correct on latent space
sce.pp.harmony_integrate(adataTH, 'batch_id')

sc.pp.neighbors(adataTH, n_neighbors=30, use_rep='X_pca_harmony', metric='euclidean', 
                key_added='neighbors_30_corrected')
sc.tl.umap(adataTH, neighbors_key = 'neighbors_30_corrected', min_dist=0.1)


