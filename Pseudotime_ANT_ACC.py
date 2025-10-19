import palantir
import numpy as np
import pandas as p
import matplotlib.pyplot as plt
import scanpy as sc
import sklearn
import csv
import scipy
import seaborn as sns

#read pre processed data __ Here thalamus first 
directory = Path("/Desktop/scRNAseq_processing/")
palantir_obj = sc.read(directory / 'palantir_obj_ANT.h5ad')


##Run Palantir
sc.external.tl.palantir(palantir_obj, n_components=20, knn=30)

# Run diffusion maps
pca_projections = p.DataFrame(palantir_obj.obsm['X_pca'], index=palantir_obj.obs_names)
dm_res = palantir.utils.run_diffusion_maps(pca_projections, n_components=5)

ms_data = palantir.utils.determine_multiscale_space(dm_res)
tsne = palantir.utils.run_tsne(ms_data)
palantir.plot.plot_diffusion_components(tsne, dm_res)

#look at Correlated genes with diffusion components (Diff Comp 0 is to be ignored)
dati = palantir_obj.X
diffi_1 = palantir_obj.obsm['X_palantir_diff_comp'][:, 5]

corri_score = np.empty(dati.shape[1], dtype = np.float64)
for j in range(dati.shape[1]):
    corr_temp = np.corrcoef(dati[:, j], diffi_1)
    corri_score[j] = corr_temp[0, 1]

# Arrange the results in pandas data frame with associated gene names:
df_corri = p.DataFrame({'corr_score': corri_score}, index = palantir_obj.var.index)


# choose random start cell for pseudotime - here '231281549327068-0' was initially chosen in ANT
start_cell = '231281549327068-0'

#run pseudotime
pr_res = palantir.core.run_palantir(ms_data, start_cell,num_waypoints=500) 
palantir.plot.plot_palantir_results(pr_res, tsne)

#save results 
palantir_obj.obsm['X_tsne'] = tsne.values
palantir_obj.obs['pseudotime'] = pr_res.pseudotime


##start here if load processed data __ thalamus
#DEtermine density cells along trajectory per time point
sc.tl.embedding_density(palantir_obj, basis='tsne', groupby='batch_id')

with plt.rc_context({'figure.figsize': (5,4)}): 
    sc.pl.embedding_density(palantir_obj, basis='tsne',key = 'tsne_density_batch_id',bg_dotsize = 80,fg_dotsize = 200,vmin=0.4)

##Plot density of Fos cells
# FOS expression vector
gene_fos = palantir_obj.var_names.get_loc("FOS")
col = palantir_obj.X[:, gene_fos]
fos_expr = col.toarray().ravel() if sparse.issparse(col) else np.asarray(col).ravel()

# Map batch_id -> label (preserves your exact strings)
batch_map = {
    "0": "T2fos",
    "1": "T6fos",
    "2": "T 2d HRfos",
    "3": "T 2d LRfos",
    "4": "T 8d HRfos",
    "5": "T 8d LRfos",
    "6": "T 15d HRfos",
    "7": "T 15d LRfos",
    "8": "HCfos",
}

labels = palantir_obj.obs["batch_id"].astype(str).map(batch_map).fillna("na")
palantir_obj.obs["fos_batch"] = np.where(fos_expr > 0.2, labels, "na")

sc.tl.embedding_density(palantir_obj, basis='tsne', groupby='fos_batch')

with plt.rc_context({'figure.figsize': (5,4)}): 
    sc.pl.embedding_density(palantir_obj, basis='tsne',key = 'tsne_density_fos_batch',bg_dotsize = 80,fg_dotsize = 200,vmin=0.6)

    
#Average expression of TRs along pseudotime
expmef2c =palantir_obj[:, 'MEF2C'].X.mean(axis=1)
expcamta =palantir_obj[:, 'CAMTA1'].X.mean(axis=1)
exptcf4 =palantir_obj[:, 'TCF4'].X.mean(axis=1)
expmyt1l =palantir_obj[:, 'MYT1L'].X.mean(axis=1)

df = p.DataFrame({'MEF2C':expmef2c, 'CAMTA1':expcamta,
                  'TCF4':exptcf4, 'MYT1L':expmyt1l},index = palantir_obj.obs['pseudotime'])

if df.index.name == 'pseudotime':
    df.reset_index(inplace=True)

# Define the number of bins you want to use
num_bins = 10  # You can adjust this value based on how smooth you want the graph to be

# Create bin edges spanning from min to max pseudotime
bin_edges = np.linspace(df['pseudotime'].min(), df['pseudotime'].max(), num_bins + 1)

# Bin the pseudotime data
df['binned'] = p.cut(df['pseudotime'], bins=bin_edges, labels=[(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])

# Calculate the mean and SEM for each bin for each variable
grouped = df.groupby('binned')
means = grouped.mean()
sems = grouped.sem()

plt.figure(figsize=(5, 5))

# Plot each variable with error bars
for column in df.columns.drop(['pseudotime', 'binned']):  
    plt.errorbar(means.index.astype(float), means[column], yerr=sems[column], label=f'{column}', linestyle='-',
                markersize=8, capsize=8)  # Error bars for SEM

plt.xlabel('Pseudotime')
plt.xticks(np.linspace(df['pseudotime'].min(), df['pseudotime'].max(), num_bins + 1))  # Adjust x-ticks to align with bins
plt.xlim([df['pseudotime'].min(), df['pseudotime'].max()])


### loading preprocessed ACC pseudotime 
palantir_acc = sc.read_h5ad(directory / 'palantir_obj_ACC.h5ad')

#density per time points
sc.tl.embedding_density(palantir_acc, basis='tsne', groupby='batch_id')


# Fos density 
gene_fos = palantir_acc.var_names.get_loc("FOS")
col = palantir_acc.X[:, gene_fos]
fos_expr = col.toarray().ravel() if sparse.issparse(col) else np.asarray(col).ravel()

# Map batch_id -> label
batch_map = {
    "0": "T 2d HRfos",
    "1": "T 2d LRfos",
    "2": "T 8d HRfos",
    "3": "T 8d LRfos",
    "4": "T 15d HRfos",
    "5": "T 15d LRfos",
    "6": "T 20d HRfos",
    "7": "T 20d LRfos",
    "8": "HCfos",}

labels = palantir_acc.obs["batch_id"].astype(str).map(batch_map).fillna("HCfos")
palantir_acc.obs["fos_batch"] = np.where(fos_expr > 0.2, labels, "HCfos")

sc.tl.embedding_density(palantir_acc, basis='tsne', groupby='fos_batch')



## CellRank analysis
import scvi
import scvelo as scv
import cellrank as cr
from cellrank.estimators import GPCCA

#condition obs for ANT as example
context_dic = {
    '0': 'T', '1': 'T',
    '2': 'HR', '3': 'LR',
    '4': 'HR', '5': 'LR',
    '6': 'HR', '7': 'LR',
    '8': 'C'}

palantir_obj.obs['condition'] = (
    palantir_obj.obs['batch_id']
    .astype(str)
    .map(context_dic)
    .fillna('unknown'))

#recalculate neighbors for transition matrix
sc.pp.neighbors(palantir_obj,n_neighbors=30,use_rep='X_pca',metric='euclidean',key_added='neighbors')

#palantir transition matrix use pseudotime saved from palantir before
pk = cr.kernels.PseudotimeKernel(palantir_obj, time_key="pseudotime")
pk.compute_transition_matrix()
print(pk)


#Black dots denote sampled starting cells for random walks, yellow dots denote end cells. Terminated each random walk after a predefined number of steps. Random walks are colored according to how long they've been running for 
#orange/yellow. We can see that most random walks terminate in one of the two terminal states, as expected.

pk.plot_random_walks(
    n_sims=15,
    start_ixs={'batch_id': '0'},
    basis='X_tsne',
    color="batch_id",
    legend_loc="right",
    seed=1)

#CellRank estimators
g_fwd = cr.estimators.GPCCA(pk)
g_fwd.fit(cluster_key="batch_id", n_states=[2, 6])
g_fwd.plot_macrostates(which="all", discrete=True, legend_loc="right", basis="X_tsne",s=100)

g_fwd.predict_initial_states()
g_fwd.predict_terminal_states(allow_overlap=True)
g_fwd.compute_fate_probabilities()


#compute driver genes to each branch 
driver_clusters = ['T','HR','LR']

delta1_df = g_fwd.compute_lineage_drivers(lineages=['2'],cluster_key='condition',clusters= driver_clusters)
delta2_df = g_fwd.compute_lineage_drivers(lineages=['3'],cluster_key='condition',clusters= driver_clusters)
#can save gene list as csv

#Filter gene list by TFs
#load all known TFs file
tfactors = open(directory / 'TFs_AnimalTFDB.txt', 'r')
tfactor = tfactors.read().split('\n')

#helper 
def filter_genes_by_list(df, gene_list):
    # Convert gene_list string to a set for faster lookup
    gene_set = set(gene_list)
    filtered_df = df.loc[df.index.intersection(gene_set)]  
    return filtered_df

tf_filtered=filter_genes_by_list(delta2_df,tfactor)
#save list of TFs that drive branch commitement 
