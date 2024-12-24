import scanpy as sc
import scipy.cluster.hierarchy as sch
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import subprocess
import torch
import warnings

from matplotlib.colors import ListedColormap
from typing import List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mclustpy import mclustpy

class XeniumCluster:

    # TO DO
    # add option to only include high variable genes

    SPOT_SIZE = 100
    THIRD_DIM = False

    def __init__(self, data: pd.DataFrame, dataset_name: str, spot_size: int = 100) -> None:
        
        self.raw_xenium_data = data
        self.xenium_spot_data = None
        self.dataset_name = dataset_name
        self.SPOT_SIZE = spot_size
        self.spot_data_location = f"data/spot_data/{dataset_name}"

    def target_dir_setter(self, method, **kwargs):
        if self.dataset_name == "SYNTHETIC":
            self.target_dir = os.path.join(
                f"results/{self.dataset_name}/{method}/",
                f"DATA_DIM={self.data_dimension}",
                "/".join([f"{key}={value}" for key, value in kwargs.items()]),
                "clusters"
            )
        elif self.dataset_name == "hBreast":
            self.target_dir = os.path.join(
                f"results/{self.dataset_name}/{method}/",
                "/".join([f"{key}={value}" for key, value in kwargs.items()]),
                f"clusters/{self.SPOT_SIZE}"
            )
        elif self.dataset_name == "DLPFC":
            self.target_dir = os.path.join(
                f"results/{self.dataset_name}/{method}/",
                "/".join([f"{key}={value}" for key, value in kwargs.items()]),
                "clusters"
            )
        else:
            raise NotImplementedError("Dataset not supported.")

    def set_spot_size(self, new_spot_size):

        if not isinstance(new_spot_size, (int, float)): 
            raise TypeError("The spot size must be numeric.")
        if new_spot_size <= 0:
            raise ValueError("Spot size must be positive.")
        self.SPOT_SIZE = new_spot_size

    # update this to be a re-init procedure
    def set_data(self, data):

        self.raw_xenium_data = data

    def convert_pd_to_ad(self, data):

        obs_df = data.index.to_frame(index=False).astype("category")

        return sc.AnnData(X=data.values, obs=obs_df, var=pd.DataFrame(index=data.columns))
    
    def normalize_counts(self, data):

        data.layers['raw']=data.X
        # Why does the demo do this????
        # sc.pp.normalize_total(data, target_sum=1e4)
        sc.pp.log1p(data)

    def create_spot_data(self, third_dim=False, save_data=True):

        x_min, x_max = min(self.raw_xenium_data["x_location"]), max(self.raw_xenium_data["x_location"])
        y_min, y_max = min(self.raw_xenium_data["y_location"]), max(self.raw_xenium_data["y_location"])

        MIN_PAD = 1e-8

        x_values = np.arange(x_min - MIN_PAD, x_max + self.SPOT_SIZE, self.SPOT_SIZE)
        y_values = np.arange(y_min - MIN_PAD, y_max + self.SPOT_SIZE, self.SPOT_SIZE)
        

        self.xenium_spot_data = self.raw_xenium_data.copy()

        if third_dim:
            z_min, z_max = min(self.raw_xenium_data["z_location"]), max(self.raw_xenium_data["z_location"])
            z_values = np.arange(z_min - MIN_PAD, z_max + self.SPOT_SIZE, self.SPOT_SIZE)
            self.xenium_spot_data["col"] = np.searchsorted(x_values, self.xenium_spot_data["x_location"]) - 1
            self.xenium_spot_data["row"] = np.searchsorted(y_values, self.xenium_spot_data["y_location"]) - 1
            self.xenium_spot_data["z-index"] = np.searchsorted(z_values, self.xenium_spot_data["z_location"]) - 1
            self.xenium_spot_data["spot_number"] = (self.xenium_spot_data["col"] * len(y_values) * len(z_values)) + (self.xenium_spot_data["row"] * len(z_values)) + self.xenium_spot_data["z-index"]
        else:
            self.xenium_spot_data["col"] = np.searchsorted(x_values, self.xenium_spot_data["x_location"]) - 1
            self.xenium_spot_data["row"] = np.searchsorted(y_values, self.xenium_spot_data["y_location"]) - 1
            self.xenium_spot_data["spot_number"] = self.xenium_spot_data["col"] * len(y_values) + self.xenium_spot_data["row"]

        counts = self.xenium_spot_data.groupby(['spot_number', 'feature_name']).size().reset_index(name='count')

        counts_pivot = counts.pivot_table(index='spot_number', 
                                  columns='feature_name', 
                                  values='count', 
                                  fill_value=0)
        
        location_means = self.xenium_spot_data.groupby('spot_number').agg({
            'row': 'mean',
            'col': 'mean',
            'x_location': 'mean',
            'y_location': 'mean',
            'z_location': 'mean'
        }).reset_index()

        self.xenium_spot_data = location_means.join(counts_pivot, on='spot_number')

        if save_data:
            self.xenium_spot_data.to_csv(f"{self.spot_data_location}/{self.dataset_name}_SPOTSIZE={self.SPOT_SIZE}um_z={third_dim}.csv")

        self.xenium_spot_data.set_index(["spot_number", "x_location", "y_location", "z_location", "row", "col"], inplace=True)

        self.xenium_spot_data = self.convert_pd_to_ad(self.xenium_spot_data)

    def generate_neighborhood_graph(self, data: ad.AnnData, n_neighbors=15, n_pcs=20, plot_pcas=True):
        
        # generate the neigborhood graph based on pca
        sc.pp.pca(data, svd_solver='arpack')
        if plot_pcas:
            sc.pl.pca_variance_ratio(data, log=True)
        sc.pp.neighbors(data, n_neighbors=n_neighbors, n_pcs=n_pcs)

    def filter_only_high_variable_genes(self, data: ad.AnnData, min_mean: float=0.3, max_mean: float=7, min_disp: float=-0.5, flavor="seurat", plot_highly_variable_genes: bool=False, n_top_genes: int=None):
        sc.pp.highly_variable_genes(data, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp, n_top_genes=n_top_genes, flavor=flavor)
        if plot_highly_variable_genes:
            sc.pl.highly_variable_genes(data)

    def pca(self, data: ad.AnnData, num_pcs: int):
        sc.pp.pca(data, num_pcs)

    def Leiden(
            self,
            data: ad.AnnData,
            resolutions: List[float],
            embedding: str = "umap", 
            **kwargs
        ):

        for resolution in resolutions:
            key_added = f'leiden_{resolution}'
            
            if self.dataset_name == "DLPFC":

                non_na_mask = ~data.obs["Region"].isna()

                masked_data_clusters = sc.tl.leiden(data[non_na_mask], resolution=resolution, key_added=key_added, copy=True)

                data.obs[f'leiden_{resolution}'] = np.zeros_like(data.obs.index, dtype=int) - 1  # Initialize with -1 for non-NA regions
                data.obs.loc[non_na_mask, f'leiden_{resolution}'] = masked_data_clusters.obs[f'leiden_{resolution}']

            else:

                sc.tl.leiden(data, resolution=resolution, key_added=key_added)

            # # Calculate and plot embedding
            # get_embedding(data, embedding, **kwargs)

            # # plot embedding
            # _ = plot_embedding(data, key_added, embedding, **kwargs)

            self.target_dir_setter("Leiden", resolution=resolution)
            os.makedirs(self.target_dir, exist_ok=True)
            
            # Extracting row, col, and cluster values from the dataframe
            rows = torch.tensor(data.obs["row"].astype(int))
            cols = torch.tensor(data.obs["col"].astype(int))
            clusters = torch.tensor(data.obs[f'leiden_{resolution}'].astype(int))
            num_clusters = clusters.unique().size(0)

            num_rows = max(rows) + 1
            num_cols = max(cols) + 1

            cluster_grid = torch.zeros((num_rows, num_cols), dtype=torch.int)

            cluster_grid[rows, cols] = torch.tensor(clusters, dtype=torch.int) + 1

            if self.dataset_name == "SYNTHETIC":
                colormap = plt.cm.get_cmap('viridis', num_clusters)
            elif self.dataset_name == "DLPFC":
                colors = plt.cm.get_cmap('viridis', num_clusters)
                grey_color = [0.5, 0.5, 0.5, 1]  # Medium gray for unused cluster
                colormap_colors = np.vstack((grey_color, colors(np.linspace(0, 1, num_clusters-1))))
                colormap = ListedColormap(colormap_colors)
            else:
                colors = plt.cm.get_cmap('viridis', num_clusters + 1)
                colormap_colors = np.vstack(([[1, 1, 1, 1]], colors(np.linspace(0, 1, num_clusters))))
                colormap = ListedColormap(colormap_colors)

            plt.figure(figsize=(6, 6))
            if self.dataset_name == "DLPFC":

                rows, cols, clusters = rows.cpu(), cols.cpu(), clusters.cpu()

                # Create mapping between region names and integer codes
                region_to_int = {name: code + 1 for code, name in enumerate(data.obs["Region"].cat.categories)}
                int_to_region = {code + 1: name for code, name in enumerate(data.obs["Region"].cat.categories)}

                # Scatter plot
                plt.scatter(cols, rows, c=clusters+1, cmap=colormap, marker='h', s=12)#, edgecolors='white')

                # Calculate padding for the axis limits
                x_padding = (cols.max() - cols.min()) * 0.02  # 2% padding
                y_padding = (rows.max() - rows.min()) * 0.02        # 2% padding

                # Set axis limits with padding
                plt.xlim(cols.min() - x_padding, cols.max() + x_padding)
                plt.ylim(rows.min() - y_padding, rows.max() + y_padding)

                # Force square appearance by stretching the y-axis
                plt.gca().set_aspect((cols.max() - cols.min() + 2 * x_padding) / 
                                    (rows.max() - rows.min() + 2 * y_padding))  # Adjust for padded ranges
                plt.gca().invert_yaxis()  # Maintain spatial orientation
                # Add colorbar and title
                plt.colorbar(ticks=range(num_clusters), label="True Label").set_ticklabels(["NA"] + list(int_to_region.values()))
                plt.tight_layout()  # Minimize padding around the plot
            else:

                plt.imshow(cluster_grid.cpu(), cmap=colormap, interpolation='nearest', origin='lower')
                plt.colorbar(ticks=range(num_clusters + 1), label='Cluster Values')
            plt.title(f'Cluster Assignment with Leiden ($\gamma$ = {resolution})')

            plt.savefig(
                f"{self.target_dir}/clusters_RES={resolution}.png"
            )

        return {resolution: data.obs[f'leiden_{resolution}'].values.astype(int) for resolution in resolutions}

    def Louvain(
            self,
            data: ad.AnnData,
            resolutions: List[float],
            embedding: str = "umap", 
            **kwargs
        ):

        for resolution in resolutions:

            key_added = f'louvain_{resolution}'

            if self.dataset_name == "DLPFC":

                non_na_mask = ~data.obs["Region"].isna()

                masked_data_clusters = sc.tl.louvain(data[non_na_mask], resolution=resolution, key_added=key_added, copy=True)

                data.obs[f'louvain_{resolution}'] = np.zeros_like(data.obs.index, dtype=int) - 1  # Initialize with -1 for non-NA regions
                data.obs.loc[non_na_mask, f'louvain_{resolution}'] = masked_data_clusters.obs[f'louvain_{resolution}']
                print(data.obs[f'louvain_{resolution}'].unique())
                print(resolution)
            else:

                sc.tl.louvain(data, resolution=resolution, key_added=key_added)

            # # calculate embedding
            # get_embedding(data, embedding, **kwargs)

            # # plot embedding
            # _ = plot_embedding(data, key_added, embedding, **kwargs)


            self.target_dir_setter("Louvain", resolution=resolution)
            os.makedirs(self.target_dir, exist_ok=True)
            
            # Extracting row, col, and cluster values from the dataframe
            rows = torch.tensor(data.obs["row"].astype(int))
            cols = torch.tensor(data.obs["col"].astype(int))
            clusters = torch.tensor(data.obs[f'louvain_{resolution}'].astype(int))
            num_clusters = clusters.unique().size(0)

            num_rows = max(rows) + 1
            num_cols = max(cols) + 1

            cluster_grid = torch.zeros((num_rows, num_cols), dtype=torch.int)

            cluster_grid[rows, cols] = torch.tensor(clusters, dtype=torch.int) + 1

            if self.dataset_name == "SYNTHETIC":
                colormap = plt.cm.get_cmap('viridis', num_clusters)
            elif self.dataset_name == "DLPFC":
                colors = plt.cm.get_cmap('viridis', num_clusters)
                grey_color = [0.5, 0.5, 0.5, 1]  # Medium gray for unused cluster
                colormap_colors = np.vstack((grey_color, colors(np.linspace(0, 1, num_clusters-1))))
                colormap = ListedColormap(colormap_colors)
            else:
                colors = plt.cm.get_cmap('viridis', num_clusters + 1)
                colormap_colors = np.vstack(([[1, 1, 1, 1]], colors(np.linspace(0, 1, num_clusters))))
                colormap = ListedColormap(colormap_colors)

            plt.figure(figsize=(6, 6))
            if self.dataset_name == "DLPFC":

                rows, cols, clusters = rows.cpu(), cols.cpu(), clusters.cpu()

                # Create mapping between region names and integer codes
                region_to_int = {name: code + 1 for code, name in enumerate(data.obs["Region"].cat.categories)}
                int_to_region = {code + 1: name for code, name in enumerate(data.obs["Region"].cat.categories)}

                # Scatter plot
                plt.scatter(cols, rows, c=clusters+1, cmap=colormap, marker='h', s=12)#, edgecolors='white')

                # Calculate padding for the axis limits
                x_padding = (cols.max() - cols.min()) * 0.02  # 2% padding
                y_padding = (rows.max() - rows.min()) * 0.02        # 2% padding

                # Set axis limits with padding
                plt.xlim(cols.min() - x_padding, cols.max() + x_padding)
                plt.ylim(rows.min() - y_padding, rows.max() + y_padding)

                # Force square appearance by stretching the y-axis
                plt.gca().set_aspect((cols.max() - cols.min() + 2 * x_padding) / 
                                    (rows.max() - rows.min() + 2 * y_padding))  # Adjust for padded ranges
                plt.gca().invert_yaxis()  # Maintain spatial orientation
                # Add colorbar and title
                plt.colorbar(ticks=range(num_clusters), label="True Label").set_ticklabels(["NA"] + list(int_to_region.values()))
                plt.tight_layout()  # Minimize padding around the plot
            else:

                plt.imshow(cluster_grid.cpu(), cmap=colormap, interpolation='nearest', origin='lower')
                plt.colorbar(ticks=range(num_clusters + 1), label='Cluster Values')
            plt.title(f'Cluster Assignment with Louvain ($\gamma$ = {resolution})')

            plt.savefig(
                f"{self.target_dir}/clusters_RES={resolution}.png"
            )

        return {resolution: data.obs[f'louvain_{resolution}'].values.astype(int) for resolution in resolutions}

    def Hierarchical(
            self,
            data: ad.AnnData,
            num_clusters: int = 3,
            groupby: List[str] = ["spot_number"],
            embedding: str = "umap",
            include_spatial = True,
            use_pca = False,
            **kwargs
        ):

        key_added = f'dendrogram_{groupby}'

        if use_pca:
            try:
                init_data = data.obsm["X_pca"]
            except KeyError:
                warnings.warn("PCA matrix was not found in this anndata object. Using expression dataset X instead.")
                init_data = data.X
        else:
            init_data = data.X
        
        # calculate cluster assignment
        if include_spatial:
            # Normalize spatial coordinates to have a similar scale to the gene expression data
            norm_row = (data.obs['row'].astype(int) - np.min(data.obs['row'].astype(int))) / np.ptp(data.obs['row'].astype(int))
            norm_col = (data.obs['col'].astype(int) - np.min(data.obs['col'].astype(int))) / np.ptp(data.obs['col'].astype(int))

            # Create a temporary copy of X and append normalized spatial coordinates
            temp_X = np.concatenate([init_data, np.array(norm_row)[:, np.newaxis], np.array(norm_col)[:, np.newaxis]], axis=1)

            # Now perform the clustering with the temporary X
            # Create a list of PC components based on the number of variables in init_data
            var = data.var.copy()
            if use_pca:
                num_vars = init_data.shape[1]
                pc_components = [f"PC{i+1}" for i in range(num_vars)]
                var = pd.DataFrame(index=pc_components)
                var = pd.concat((var, pd.DataFrame(index=['norm_row', 'norm_col'])), axis=1)
            temp_data = sc.AnnData(X=temp_X, obs=data.obs.copy(), var=var, obsm=data.obsm.copy())

            # Calculate dendrogram
            sc.tl.dendrogram(temp_data, groupby=groupby, key_added=key_added, use_rep="X_pca" if use_pca else "X")
            linkage_matrix = temp_data.uns[key_added]['linkage']
        else:
            sc.tl.dendrogram(data, groupby=groupby, key_added=key_added, use_rep="X_pca" if use_pca else "X")
            linkage_matrix = data.uns[key_added]['linkage']

        # Form clusters from the dendrogram
        cluster_labels = sch.fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

        # Assign cluster labels to observations
        data.obs[key_added] = cluster_labels

        # plot dendrogram
        # sc.pl.dendrogram(data, groupby=groupby)

        # # calculate embedding
        # get_embedding(data, embedding, **kwargs)

        # # plot embedding
        # _ = plot_embedding(data, key_added, embedding, **kwargs)

        self.target_dir_setter("Hierarchical", K=num_clusters)
        os.makedirs(self.target_dir, exist_ok=True)
            
        # Extracting row, col, and cluster values from the dataframe
        rows = torch.tensor(data.obs["row"].astype(int))
        cols = torch.tensor(data.obs["col"].astype(int))
        clusters = torch.tensor(data.obs[key_added].astype(int))
        num_clusters = clusters.unique().size(0)

        num_rows = max(rows) + 1
        num_cols = max(cols) + 1

        cluster_grid = torch.zeros((num_rows, num_cols), dtype=torch.int)

        cluster_grid[rows, cols] = torch.tensor(clusters, dtype=torch.int)

        if self.dataset_name == "SYNTHETIC":
            colormap = plt.cm.get_cmap('viridis', num_clusters + 1)
        else:
            colors = plt.cm.get_cmap('viridis', num_clusters + 1)
            colormap_colors = np.vstack(([[1, 1, 1, 1]], colors(np.linspace(0, 1, num_clusters))))
            colormap = ListedColormap(colormap_colors)

        plt.figure(figsize=(6, 6))
        plt.imshow(cluster_grid.cpu(), cmap=colormap, interpolation='nearest', origin='lower')
        plt.colorbar(ticks=range(num_clusters + 1), label='Cluster Values')
        plt.title(f'Cluster Assignment with Hierarchical')

        plt.savefig(
            f"{self.target_dir}/clusters_K={num_clusters}.png"
        )

        return data.obs[key_added].values.astype(int) - 1

    def KMeans(
            self,
            data: ad.AnnData,
            K: int = 3,
            include_spatial=True,
            normalize=True,
            save_plot=True,
            use_pca = False,
        ):
            
            if use_pca:
                try:
                    spatial_init_data = data.obsm["X_pca"]
                except KeyError:
                    spatial_init_data = data.X
            else:
                spatial_init_data = data.X

            if include_spatial:

                spatial_locations = data.obs[["row", "col"]]

                spatial_init_data = np.concatenate((spatial_locations, spatial_init_data), axis=1)

            if normalize:

                spatial_init_data = StandardScaler().fit_transform(spatial_init_data)

            if self.dataset_name == "DLPFC":

                non_na_mask = ~data.obs["Region"].isna()

                kmeans = KMeans(n_clusters=K).fit(spatial_init_data[non_na_mask])

                cluster_assignments = kmeans.predict(spatial_init_data[non_na_mask])

                data.obs["cluster"] = np.zeros_like(data.obs.index, dtype=int) - 1  # Initialize with -1 for non-NA regions
                data.obs.loc[non_na_mask, "cluster"] = cluster_assignments

            else:

                kmeans = KMeans(n_clusters=K).fit(spatial_init_data)

                cluster_assignments = kmeans.predict(spatial_init_data)

                data.obs["cluster"] = cluster_assignments

            self.target_dir_setter("K-Means", K=K)
            os.makedirs(self.target_dir, exist_ok=True)
                
            # Extracting row, col, and cluster values from the dataframe
            rows = torch.tensor(data.obs["row"].astype(int))
            cols = torch.tensor(data.obs["col"].astype(int))
            clusters = torch.tensor(data.obs["cluster"].astype(int))
            num_clusters = clusters.unique().size(0)

            num_rows = max(rows) + 1
            num_cols = max(cols) + 1

            cluster_grid = torch.zeros((num_rows, num_cols), dtype=torch.int)

            cluster_grid[rows, cols] = torch.tensor(clusters, dtype=torch.int) + 1

            if self.dataset_name == "SYNTHETIC":
                colormap = plt.cm.get_cmap('viridis', num_clusters)
            elif self.dataset_name == "DLPFC":
                colors = plt.cm.get_cmap('viridis', num_clusters)
                grey_color = [0.5, 0.5, 0.5, 1]  # Medium gray for unused cluster
                colormap_colors = np.vstack((grey_color, colors(np.linspace(0, 1, num_clusters-1))))
                colormap = ListedColormap(colormap_colors)
            else:
                colors = plt.cm.get_cmap('viridis', num_clusters + 1)
                colormap_colors = np.vstack(([[1, 1, 1, 1]], colors(np.linspace(0, 1, num_clusters))))
                colormap = ListedColormap(colormap_colors)


            plt.figure(figsize=(6, 6))
            if self.dataset_name == "DLPFC":

                rows, cols, clusters = rows.cpu(), cols.cpu(), clusters.cpu()

                # Create mapping between region names and integer codes
                region_to_int = {name: code + 1 for code, name in enumerate(data.obs["Region"].cat.categories)}
                int_to_region = {code + 1: name for code, name in enumerate(data.obs["Region"].cat.categories)}

                # Scatter plot
                plt.scatter(cols, rows, c=clusters+1, cmap=colormap, marker='h', s=12)#, edgecolors='white')

                # Calculate padding for the axis limits
                x_padding = (cols.max() - cols.min()) * 0.02  # 2% padding
                y_padding = (rows.max() - rows.min()) * 0.02        # 2% padding

                # Set axis limits with padding
                plt.xlim(cols.min() - x_padding, cols.max() + x_padding)
                plt.ylim(rows.min() - y_padding, rows.max() + y_padding)

                # Force square appearance by stretching the y-axis
                plt.gca().set_aspect((cols.max() - cols.min() + 2 * x_padding) / 
                                    (rows.max() - rows.min() + 2 * y_padding))  # Adjust for padded ranges
                plt.gca().invert_yaxis()  # Maintain spatial orientation
                # Add colorbar and title
                plt.colorbar(ticks=range(num_clusters), label="True Label").set_ticklabels(["NA"] + list(int_to_region.values()))
                plt.tight_layout()  # Minimize padding around the plot
            else:

                plt.imshow(cluster_grid.cpu(), cmap=colormap, interpolation='nearest', origin='lower')
                plt.colorbar(ticks=range(num_clusters + 1), label='Cluster Values')
            plt.title(f'Cluster Assignment with K-Means')

            plt.savefig(
                f"{self.target_dir}/clusters_K={K}.png"
            )

            return data.obs["cluster"]
            
    def BayesSpace(
        self,
        data: ad.AnnData,
        init_method: str = "mclust",
        num_pcs: int = 15,
        K: int = 3,
        grid_search: bool = True,
    ):

        def run_r_script(script_path: str, *args):
            """
            Function to run an R script with optional arguments.
            
            Parameters:
            script_path (str): Path to the R script.
            *args: Additional arguments to pass to the R script.
            """
            command = ["Rscript", script_path] + list(args)
            subprocess.run(command, check=True, capture_output=True)

        run_r_script("xenium_BayesSpace.R", self.dataset_name, f"{self.SPOT_SIZE}", f"{init_method}", f"{num_pcs}", f"{K}", f"{grid_search}")
        self.target_dir_setter("BayesSpace", num_pcs=num_pcs, K=K, INIT=init_method)
        os.makedirs(self.target_dir, exist_ok=True)
        gammas = np.linspace(1, 3, 9) if grid_search else [2]
        for gamma in gammas:
            self.target_dir = os.path.join(self.target_dir, f"{gamma:.2f}")
            os.makedirs(self.target_dir, exist_ok=True)
            BayesSpace_clusters = pd.read_csv(f"{self.target_dir}/{gamma:.2f}/clusters_K={K}.csv", index_col=0)
            data.obs["cluster"] = np.array(BayesSpace_clusters["BayesSpace cluster"])
            # Extracting row, col, and cluster values from the dataframe
            rows = torch.tensor(data.obs["row"].astype(int))
            cols = torch.tensor(data.obs["col"].astype(int))
            clusters = torch.tensor(data.obs["cluster"].astype(int))
            num_clusters = clusters.unique().size(0)

            num_rows = max(rows) + 1
            num_cols = max(cols) + 1

            cluster_grid = torch.zeros((num_rows, num_cols), dtype=torch.int)

            if self.dataset_name == "SYNTHETIC":
                cluster_grid[rows, cols] = torch.tensor(clusters, dtype=torch.int)
                colormap = plt.cm.get_cmap('viridis', num_clusters)
            else:
                cluster_grid[rows, cols] = torch.tensor(clusters, dtype=torch.int) + 1
                colors = plt.cm.get_cmap('viridis', num_clusters + 1)
                colormap_colors = np.vstack(([[1, 1, 1, 1]], colors(np.linspace(0, 1, num_clusters))))
                colormap = ListedColormap(colormap_colors)

            plt.figure(figsize=(6, 6))
            plt.imshow(cluster_grid, cmap=colormap, interpolation='nearest', origin='lower')
            plt.colorbar(ticks=range(num_clusters + 1), label='Cluster Values')
            plt.title(f'Cluster Assignment with BayesSpace ($\gamma$ = {gamma})')

            plt.savefig(
                os.path.join(self.target_dir, f"clusters_K={K}.png")
            )

        return data.obs["cluster"].values.astype(int)
    
    def mclust(
        self,
        data: ad.AnnData,
        G: int = 17,
        model_name: str = "EEE",
        temp_dir: str = "temporary_pca_file.csv"
    ):
        """
            G: if int, will check for all clusters 1:G. If list, will only check values in list.
        """

        def run_r_script(script_path: str, *args):
            """
            Function to run an R script with optional arguments and capture its output.
            
            Parameters:
            script_path (str): Path to the R script.
            *args: Additional arguments to pass to the R script.
            
            Returns:
            str: The standard output from the R script.
            """
            command = ["Rscript", script_path] + list(args)
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                return result.stdout
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while running the R script: {e.output}")
                raise
        try:
            np.savetxt(temp_dir, data.obsm["X_pca"], delimiter=",")
        except KeyError:
            raise KeyError("PCA is required for this model to work.")
        self.target_dir_setter("mclust", num_pcs=data.obsm['X_pca'].shape[1], K=G)
        num_output_clusters = run_r_script("mclust.R", temp_dir, f"{G}", f"{data.obsm['X_pca'].shape[1]}", f"{self.SPOT_SIZE}", self.dataset_name, self.target_dir)

        mclust_clusters = pd.read_csv(f"{self.target_dir}/clusters_K={G}.csv", index_col=0)
        data.obs["cluster"] = np.array(mclust_clusters["mclust cluster"])
    
        # Extracting row, col, and cluster values from the dataframe
        rows = torch.tensor(data.obs["row"].astype(int))
        cols = torch.tensor(data.obs["col"].astype(int))
        clusters = torch.tensor(data.obs["cluster"].astype(int))
        num_clusters = clusters.unique().size(0)
        num_rows = max(rows) + 1
        num_cols = max(cols) + 1

        cluster_grid = torch.zeros((num_rows, num_cols), dtype=torch.int)

        cluster_grid[rows, cols] = torch.tensor(clusters, dtype=torch.int) + 1

        if self.dataset_name == "SYNTHETIC":
            cluster_grid[rows, cols] = torch.tensor(clusters, dtype=torch.int)
            colormap = plt.cm.get_cmap('viridis', num_clusters)
        else:
            cluster_grid[rows, cols] = torch.tensor(clusters, dtype=torch.int) + 1
            colors = plt.cm.get_cmap('viridis', num_clusters + 1)
            colormap_colors = np.vstack(([[1, 1, 1, 1]], colors(np.linspace(0, 1, num_clusters))))
            colormap = ListedColormap(colormap_colors)

        plt.figure(figsize=(6, 6))
        plt.imshow(cluster_grid.cpu(), cmap=colormap, interpolation='nearest', origin='lower')
        plt.colorbar(ticks=range(num_clusters + 1), label='Cluster Values')
        plt.title(f'Cluster Assignment with mclust')

        plt.savefig(
            f"{self.target_dir}/clusters_K={G}.png"
        )

        return data.obs["cluster"].values