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
import rpy2
import anndata2ri
import os
from sklearn.cluster import KMeans

directory = Path("/Desktop/scRNAseq_processing/")
adataN = sc.read(directory / "adata_thalamus_neurons.h5ad")


# Remove genes with 'RIK'
rk_genes = adataN.var_names[adataN.var_names.str.endswith('RIK')]
index_rk_genes = [adataN.var_names.get_loc(j) for j in rk_genes]
rk_g = np.in1d(adataN.var_names.values.astype(str), rk_genes)
adataN = adataN[:,~rk_g]

#Determine highly variable genes and recompute PCs
sc.pp.highly_variable_genes(adataN, layer = 'raw_data', n_top_genes = 2100, flavor = 'seurat_v3')
adataN.uns['id_hvg'] = np.where(adataN.var['highly_variable'])[0]
adataN.uns['loadings'] = adataN.varm['PCs'][adataN.var['highly_variable'], :]
adataN.obsm['X_pca'] = adataN.obsm['X_pca'][:, 0:25]

adataN.varm['PCs'] = adataN.varm['PCs'][:, 0:25]
adataN.uns['loadings'] = adataN.uns['loadings'][:, 0:25]

sc.pp.neighbors(adataN, n_neighbors=30, use_rep='X_pca', metric='euclidean', key_added='neighbors_30')
sc.tl.umap(adataN, neighbors_key = 'neighbors_30', min_dist=0.15)

#clustering
sc.external.tl.phenograph(adataN, clustering_algo='leiden', k=30, jaccard=True, primary_metric='euclidean', resolution_parameter = 0.1)

# label anterior/posterior thalamus marker genes
ant_markers = pd.read_csv(directory / 'antpost.csv')

# Build {set_name -> unique uppercase genes}, dropping NaNs/empties
neuron_dic = {col: sorted({str(g).strip().upper() for g in markers[col].dropna() if str(g).strip()})
    for col in markers.columns}

# Helper to sum across genes 
def _sum_rows(a):
    return (np.asarray(a.sum(1)).ravel() if sparse.issparse(a) else a.sum(axis=1).ravel())

# Map uppercase gene -> original gene name 
var_ = {g.upper(): g for g in adataN.var_names}

# 3) Score each gene set and plot
for label, genes_uc in neuron_dic.items():
    valid_orig = [var_[g] for g in genes_uc if g in var_]
    if not valid_orig:  # skip empty sets after intersection
        continue

    Xsub = adataN[:, valid_orig].X
    scores = _sum_rows(Xsub) / len(valid_orig)   # average of marker expression
    adataN.obs[label] = scores

    with plt.rc_context({'figure.figsize': (5, 5)}):
        sc.pl.umap(adataN, color=label, color_map='Spectral_r')

## Differential gene expression with MAST - between conditions
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
os.environ['R_USER'] = '/Library/Frameworks/R.framework/Resources'
anndata2ri.activate()
%load_ext rpy2.ipython

sce_v4 = ad.AnnData(X = adataN.X, 
                 obs = p.DataFrame({'batch_label': adataN.obs['batch_id'].astype('int')}, 
                                    index = adataN.obs.index),
                 var = p.DataFrame(index = adataN.var.index))
%%capture
%R library(scuttle)
%R library(scran)
%R library(MAST)
%R library(data.table)


%R -i sce_v4
%R counts(sce_v4) <- assay(sce_v4, "X"); 
print("Finished Setup")

%R id_cells_C2 <- which(colData(sce_v4)$batch_label ==0)
# identify cells belonging to batch 1:
%R id_cells_C3 <- which(colData(sce_v4)$batch_label == 1)

%R df1 <- t(data.frame(counts(sce_v4)[, id_cells_C2])) # transpose because in sce genes are rows
%R df2 <- t(data.frame(counts(sce_v4)[, id_cells_C3])) # transpose because in sce genes are rows

# Use external function:
%R source(directory / "run_MAST.r")

%R pairwise_de(df1, df2, paste0(directory / "MAST_EarlytrainingvsLatetraining.csv")

## Gseapy - Enrichr analysis
gseapy.get_library_name(organism = 'Mouse')
               
gene_list = p.read_csv(directory / 'MASTsorted_EarlytrainingvsLatetraining.csv')

gseapy_res = gseapy.enrichr(gene_list=gene_list[:300], #top DEGs
                            organism='Mouse',
                            gene_sets='GO_Biological_Process_2023', 
                            cutoff = 0.5)
gseapy_res.results.to_csv(directory / 'GO_EarlytrainingvsLatetraining.csv')

### heatmap DEGs
# Load DEG table
ANT_degs = pd.read_csv(directory / 'DEGtimpoints_TH.csv')

# Build dictionary of {group -> non-nan unique genes}
toplot_dic = {
    col: [g for g in pd.unique(ANT_degs[col].dropna()) if isinstance(g, str)]
    for col in ANT_degs.columns}

# Flatten all genes and build group labels
allgenes = ANT_degs.values.flatten(order='F')
allgenesgroup = np.repeat(ANT_degs.columns, ANT_degs.shape[0])


# Preserve original group order
uniquegenegroups, origindex = np.unique(allgenesgroup, return_index=True)
uniquegenegroups = uniquegenegroups[np.argsort(origindex)]

# Build usable gene list (uppercase + group suffix)
allgenesnew = [
    f"{g.upper()}--{grp}" for g, grp in zip(allgenes, allgenesgroup) if isinstance(g, str)]

# Keep only genes present in adataN
allgenesuse = [item for item in allgenesnew if item.split('--')[0] in adataN.var_names]

# Palette mapping
mypalette = dict(zip(uniquegenegroups, sc.pl.palettes.default_20))
allgenegroupsonly = [item.split('--')[1] for item in allgenesuse]
col_colors = [mypalette[g] for g in allgenegroupsonly]

# Build stats DataFrame
df_stats = pd.DataFrame(index=[f"Batch-{b}" for b in np.unique(adataN.obs['batch_id'])])

for item in allgenesuse:
    geneid = adataN.var_names.get_loc(item.split('--')[0])
    summarydata = [
        np.mean(adataN.X[adataN.obs['batch_id'] == ct, geneid])
        for ct in np.unique(adataN.obs['batch_id'])]
    df_stats[item] = summarydata

# Cluster within each gene group to order columns
columnsarranged = []
for group in uniquegenegroups:
    cols_in_group = [c for c in df_stats.columns if group in c]
    df_sub = df_stats[cols_in_group]
    n_clusters = max(int(np.ceil(df_sub.shape[1] / 1.5)), 2)
    res = KMeans(n_clusters=n_clusters, random_state=7).fit(df_sub.T)
    columnsarranged.extend(df_sub.columns[np.argsort(res.labels_)])

df_stats_arranged = df_stats[columnsarranged]

plt.rcParams["figure.figsize"] = (8,10)
g = sns.clustermap(df_stats_arranged, z_score =1,col_colors = col_colors, col_cluster = False,
                  xticklabels = False,row_cluster = False,cmap='RdBu_r',vmax=2.2)#standard_scale
for label in uniquegenegroups:
    g.ax_col_dendrogram.bar(0,0,color=mypalette[label],label=label)
g.ax_col_dendrogram.legend(loc='best',ncol=1,bbox_to_anchor=(1,0.,0.5,0))


## stacked histogram ridgeline plots of modules 

# Helper function
def calculate_expression_per_batch(adata, gene_modules):
    expression_data = {}

    for module_name, genes in gene_modules.items():
        # Ensure genes are in the data
        valid_genes = [gene for gene in genes if gene in adata.var_names]
        
        # Calculate average and variance expression per batch
        module_expr = adata[:, valid_genes].X.toarray().mean(axis=1)
        batch_df = p.DataFrame({
            'batch': adata.obs['batch_id'],
            'avg_expression': module_expr,
            'variance_expression': np.var(adata[:, valid_genes].X.toarray(), axis=1)
        })
        expression_data[module_name] = batch_df
    
    return expression_data

expression_data = calculate_expression_per_batch(adataN, gene_groups)

# Combine data into a single DataFrame for plotting
plot_data = []
for module_name, df in expression_data.items():
    df['module'] = module_name
    plot_data.append(df)

plot_df = p.concat(plot_data) 

# plotting 
g = sns.FacetGrid(plot_df, row='batch', hue='module',
                  aspect=15, height=1.5,  # Adjust aspect ratio 
                  sharex=True, sharey=True)  

# Add KDE plots for each batch with variance on the x-axis
g.map(sns.kdeplot, 'avg_expression',color='green',
      clip_on=False,bw_adjust=1,
      fill=True, alpha=0.3, linewidth=1.5)

# Add contour lines 
g.map(sns.kdeplot, 'avg_expression', 
       clip_on=False, bw_adjust=1,
      color="w", lw=2)

# Add horizontal lines for each plot
g.map(plt.axhline, y=0, lw=1, clip_on=False,color='black')

# Annotate each plot with the batch name
for i, ax in enumerate(g.axes.flat):
    batch_name = plot_df['batch'].unique()[i]
    ax.text(0.05, 0.8, batch_name,  transform=ax.transAxes,  
            fontweight='bold', fontsize=10, color=ax.lines[-1].get_color())

g.fig.subplots_adjust(hspace=0.1)  
g.set_titles("")
#g.set(yticks=[])
g.despine(bottom=True, left=True)
g.set(xlim=(-0.2, 2))
g.set(ylim=(0, 2.5))
g.fig.set_size_inches(3.5, 5.4)
g.fig.suptitle('Gene Module Expression per Batch',
               fontsize=14, fontweight='bold', y=1.02)


## Wasserstein distance between conditions 
import pertpy as pt

Distance = pt.tl.Distance(metric="wasserstein", obsm_key="X_pca")

# Helper to fetch embeddings for a condition
def get_X(adata, cond, obsm_key="X_pca", obs_key="condition"):
    mask = adata.obs[obs_key] == cond
    return adata.obsm[obsm_key][mask.values if hasattr(mask, "values") else mask]

# Helper to flatten what bootstrap returns into a p.Series
def to_series(res):
    if np.isscalar(res):
        return pd.Series({"stat": float(res)})
 
    attrs = ["stat", "distance", "pvalue", "ci", "ci_low", "ci_high"]
    flat = {}
    for a in attrs:
        if hasattr(res, a):
            v = getattr(res, a)
            if a == "ci" and isinstance(v, (list, tuple, np.ndarray)) and len(v) == 2:
                flat["ci_low"], flat["ci_high"] = float(v[0]), float(v[1])
            elif np.isscalar(v):
                flat[a] = float(v)
    return p.Series(flat) if flat else pd.Series({"result": res})

# Define contrasts 
contrasts = [
    ("T2", "R2HR"),
    ("T2", "R8HR"),
    ("T2", "R15HR"),
    ("T2", "R2LR"),
    ("T2", "R8LR"),
    ("T2", "R15LR")]

cols = {}
for a, b in contrasts:
    X = get_X(adataN, a, obsm_key="X_pca", obs_key="condition")
    Y = get_X(adataN, b, obsm_key="X_pca", obs_key="condition")
    res = Distance.bootstrap(X, Y) 
    cols[f"{a}_vs_{b}"] = to_series(res)
    
results_distances = p.DataFrame(cols).sort_index()






