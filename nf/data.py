import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt 
import scipy

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xenium_cluster import XeniumCluster

def prepare_synthetic_data(
    grid_size = 50,
    num_clusters = 5,
    data_dimension = 5,
    random_seed = 1,
    r = 1
):

    # Set grid dimensions and random seed
    np.random.seed(random_seed)

    # Step 1: Initialize an empty grid and randomly assign cluster "patches"
    ground_truth = np.zeros((grid_size, grid_size), dtype=int)
    for cluster_id in range(1, num_clusters):
        # Randomly choose a center for each cluster
        center_x, center_y = np.random.randint(0, grid_size, size=2)
        radius = np.random.randint(10, 30)  # Random radius for each cluster region

        # Assign cluster_id to a circular region around the chosen center
        for i in range(grid_size):
            for j in range(grid_size):
                if (i - center_x) ** 2 + (j - center_y) ** 2 < radius ** 2:
                    ground_truth[i, j] = cluster_id

    # Step 2: Add random noise within each patch
    noise_level = 0.5
    noisy_grid = ground_truth + noise_level * np.random.randn(grid_size, grid_size)

    # Step 3: Apply Gaussian smoothing to create spatial clustering
    sigma = 3  # Controls the amount of clustering smoothness
    smoothed_grid = scipy.ndimage.gaussian_filter(noisy_grid, sigma=sigma)

    # Step 4: Threshold to obtain integer values
    clustered_grid = np.round(smoothed_grid).astype(int)
    clustered_grid = np.clip(clustered_grid, 0, num_clusters)

    # Plot the clustered grid with flipped y axis
    fig, ax = plt.subplots(figsize=(6, 6)) 
    ax.imshow(clustered_grid, cmap="rainbow", interpolation="nearest", origin='lower')  # Flip y axis by setting origin to 'lower'
    ax.set_title("Ground Truth Clusters")
    plt.colorbar(ax.imshow(clustered_grid, cmap="rainbow", interpolation="nearest", origin='lower'), ax=ax, label="Cluster Level", ticks=range(num_clusters + 1))  # Flip y axis by setting origin to 'lower'
    os.makedirs("results/SYNTHETIC", exist_ok=True)
    plt.savefig("results/SYNTHETIC/ground_truth.png")

    # Create another copy of the image without titles or axes
    plt.figure(figsize=(6, 6))
    plt.imshow(clustered_grid, cmap="rainbow", interpolation="nearest", origin='lower')
    plt.axis('off')  # Remove axes
    os.makedirs("results/SYNTHETIC", exist_ok=True)
    plt.savefig("results/SYNTHETIC/ground_truth_no_axes.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()

    def find_indices_within_distance(grid, r=1):
        indices_within_distance = np.empty((grid.shape[0], grid.shape[1]), dtype=object)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                # Check all neighboring cells within a Manhattan distance of 1
                neighbors = []
                for x in range(max(0, i-r), min(grid.shape[0], i+r+1)):
                    for y in range(max(0, j-r), min(grid.shape[1], j+r+1)):
                        if abs(x - i) + abs(y - j) <= r:
                            neighbors.append((x, y))
                indices_within_distance[i, j] = neighbors
        return indices_within_distance

    # get the rook neighbors for each spot
    indices = find_indices_within_distance(clustered_grid, r=r)

    prior_weights = np.zeros((clustered_grid.shape[0] * clustered_grid.shape[1], num_clusters))

    # for each spot sample 
    for i in range(clustered_grid.shape[0]):
        for j in range(clustered_grid.shape[1]):
            for neighbor in indices[i, j]:
                prior_weights[i * clustered_grid.shape[1] + j, clustered_grid[neighbor]] += 1
    prior_weights = prior_weights / prior_weights.sum(axis=1, keepdims=True)

    # Initialize lists for means, covariances, and data points
    means = []
    covariances = []
    data = np.empty((clustered_grid.shape[0] * clustered_grid.shape[1], data_dimension))

    # Generate means and covariances for each cluster
    for i in range(num_clusters):
        # Randomly set the mean close to the origin to encourage overlap
        mean = np.random.uniform(-5, 5, data_dimension)
        # Generate a diagonal covariance matrix with random magnitudes
        covariance = np.diag(np.random.rand(data_dimension) * 2.5)
        
        means.append(mean)
        covariances.append(covariance)
        
    # Generate samples from the mixture.
    for i, weights in enumerate(prior_weights):
        data[i] = np.sum(weights[k] * np.random.multivariate_normal(means[k], covariances[k], 1) for k in range(num_clusters))

    # Create an anndata object
    adata = ad.AnnData(data)

    # Add row and col index
    adata.obs['spot_number'] = np.arange(clustered_grid.shape[0] * clustered_grid.shape[1])
    adata.obs['spot_number'] = adata.obs['spot_number'].astype('category')
    adata.obs['row'] = np.repeat(np.arange(clustered_grid.shape[0]), clustered_grid.shape[1])
    adata.obs['col'] = np.tile(np.arange(clustered_grid.shape[1]), clustered_grid.shape[0])
    clustering = XeniumCluster(data=adata.X, dataset_name="SYNTHETIC")
    clustering.xenium_spot_data = adata
    clustering.num_clusters = num_clusters
    clustering.data_dimension = data_dimension

    # Xenium_to_BayesSpace(clustering.xenium_spot_data, dataset_name="SYNTHETIC", spot_size=grid_size)

    return clustering.xenium_spot_data.X, clustering.xenium_spot_data.obs[['row', 'col']], clustering, prior_weights

def prepare_DLPFC_data(
    section_id=151670,
    num_pcs=5,
    log_normalize=True,
):
    section = ad.read_h5ad(f"../../xenium/data/DLPFC/{section_id}.h5ad")
    section.var["feature_name"] = section.var.index

    spatial_locations = section.obs[["array_row", "array_col"]]
    spatial_locations.columns = ["row", "col"]

    clustering = XeniumCluster(data=section.X, dataset_name="DLPFC")
    clustering.xenium_spot_data = section
    clustering.xenium_spot_data.X = clustering.xenium_spot_data.X.A
    if log_normalize:
        clustering.xenium_spot_data.X = np.log1p(clustering.xenium_spot_data.X)

    sc.tl.pca(clustering.xenium_spot_data, svd_solver='arpack', n_comps=num_pcs)
    data = clustering.xenium_spot_data.obsm["X_pca"]
    clustering.xenium_spot_data.obs.rename(columns={"array_row": "row", "array_col": "col"}, inplace=True)
    num_rows = max(clustering.xenium_spot_data.obs["row"]) - min(clustering.xenium_spot_data.obs["row"]) + 1
    clustering.xenium_spot_data.obs['spot_number'] = clustering.xenium_spot_data.obs["col"] * num_rows + clustering.xenium_spot_data.obs["row"]
    clustering.xenium_spot_data.obs['spot_number'] = clustering.xenium_spot_data.obs['spot_number'].astype('category')

    return data, spatial_locations, clustering

def prepare_Xenium_data(
        dataset="hBreast", 
        spots=True, 
        spot_size=100, 
        third_dim=False, 
        log_normalize=True,
        likelihood_mode="PCA",
        num_pcs=5,
        hvg_var_prop=0.5,
        min_expressions_per_spot=10
    ):

    data_filepath = f"../../xenium/data/spot_data/{dataset}/hBreast_SPOTSIZE={spot_size}um_z={third_dim}.h5ad"
    
    if spots:

        if os.path.exists(data_filepath):

            clustering = XeniumCluster(data=None, dataset_name="hBreast")
            clustering.set_spot_size(spot_size)
            print("Loading data.")
            clustering.xenium_spot_data = ad.read_h5ad(data_filepath)

        else:

            # Path to your .gz file
            file_path = f'data/{dataset}/transcripts.csv.gz'

            # Read the gzipped CSV file into a DataFrame
            df_transcripts = pd.read_csv(file_path, compression='gzip')
            df_transcripts["error_prob"] = 10 ** (-df_transcripts["qv"]/10)
            df_transcripts.head(), df_transcripts.shape

            # drop cells without ids
            df_transcripts = df_transcripts[df_transcripts["cell_id"] != -1]

            # drop blanks and controls
            df_transcripts = df_transcripts[~df_transcripts["feature_name"].str.startswith('BLANK_') & ~df_transcripts["feature_name"].str.startswith('NegControl')]

            clustering = XeniumCluster(data=df_transcripts, dataset_name="hBreast")
            clustering.set_spot_size(spot_size)

            if not os.path.exists(data_filepath):
                print("Generating and saving data")
                clustering.create_spot_data(third_dim=third_dim, save_data=True)
                clustering.xenium_spot_data.write_h5ad(data_filepath)

        print("Number of spots: ", clustering.xenium_spot_data.shape[0])
        clustering.xenium_spot_data = clustering.xenium_spot_data[clustering.xenium_spot_data.X.sum(axis=1) > min_expressions_per_spot]
        print("Number of spots after filtering: ", clustering.xenium_spot_data.shape[0])

        if log_normalize:
            clustering.normalize_counts(clustering.xenium_spot_data)

        if likelihood_mode == "PCA":
            sc.tl.pca(clustering.xenium_spot_data, svd_solver='arpack', n_comps=num_pcs)
            data = clustering.xenium_spot_data.obsm["X_pca"]
        elif likelihood_mode == "HVG":
            min_dispersion = torch.distributions.normal.Normal(0.0, 1.0).icdf(hvg_var_prop)
            clustering.filter_only_high_variable_genes(clustering.xenium_spot_data, flavor="seurat", min_mean=0.0125, max_mean=1000, min_disp=min_dispersion)
            clustering.xenium_spot_data = clustering.xenium_spot_data[:,clustering.xenium_spot_data.var.highly_variable==True]
            data = clustering.xenium_spot_data.X
        elif likelihood_mode == "ALL":
            data = clustering.xenium_spot_data.X

        spatial_locations = clustering.xenium_spot_data.obs[["row", "col"]]
    
    # prepare cells data
    else:

        cells = df_transcripts.groupby(['cell_id', 'feature_name']).size().reset_index(name='count')
        cells_pivot = cells.pivot_table(index='cell_id', 
                                        columns='feature_name', 
                                        values='count', 
                                        fill_value=0)
        
        location_means = df_transcripts.groupby('cell_id').agg({
            'x_location': 'mean',
            'y_location': 'mean',
            'z_location': 'mean'
        }).reset_index()

        cells_pivot = location_means.join(cells_pivot, on='cell_id')

        if log_normalize:
            # log normalization
            cells_pivot.iloc[:, 4:] = np.log1p(cells_pivot.iloc[:, 4:])

        if likelihood_mode == "PCA":
            pca = PCA(n_components=num_pcs)
            data = pca.fit_transform(cells_pivot.iloc[:, 4:])

        elif likelihood_mode == "HVG":
            genes = cells_pivot.iloc[:, 4:]
            gene_variances = genes.var(axis=0)
            gene_variances = gene_variances.sort_values(ascending=False)
            gene_var_proportions = (gene_variances / sum(gene_variances))
            relevant_genes = list(gene_var_proportions[(gene_var_proportions.cumsum() < hvg_var_prop)].index)
            cells_pivot.iloc[:, 4:] = cells_pivot.iloc[:, 4:][[relevant_genes]]
            data = cells_pivot.iloc[:, 4:]

        elif likelihood_mode == "ALL":
            data = cells_pivot.iloc[:, 4:]

        spatial_locations = cells_pivot[["x_location", "y_location"]]

    # the last one is to regain var/obs access from original data
    return data, spatial_locations, clustering 