import torch
import zuko
import numpy as np
import pandas as pd
from torch import Size, Tensor
import pyro
import copy
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import Adam, ClippedAdam, PyroOptim, SGD

import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.distributions.transforms import Spline, ComposeTransform
from pyro.distributions import TransformedDistribution

import torch_geometric as pyg
import torch.nn as nn
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, CuGraphSAGEConv, GATConv, GINConv
from torch_geometric.data import Data
from torch_geometric import EdgeIndex

from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

# Utility imports
import GPUtil
import math
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
import os
import json
from utils import ARI, NMI

# Custom module imports
from xenium_cluster import XeniumCluster
from data import prepare_DLPFC_data, prepare_synthetic_data, prepare_Xenium_data
from zuko_flow import setup_zuko_flow, ZukoToPyro
from omegaconf import OmegaConf

class GCNFlowModel(nn.Module):
    def __init__(self, original_graph, in_features, out_features, conv_type="GCN"):
        super().__init__()
        self.x = original_graph.x
        self.edge_index = original_graph.edge_index
        self.set_layers(in_features, out_features, conv_type)

    def set_layers(self, in_features, out_features, conv_type):
        match conv_type:
            case "GCN":
                self.layers = nn.ModuleList([
                    GCNModule(in_features, 256, conv_type),  # First layer with reduced output size
                    nn.Tanh(),  # Using ReLU activation for better performance
                    # GCNModule(256, 256, conv_type),  # Second layer with same output size for deeper representation
                    # nn.Tanh(),  # Another ReLU activation
                    GCNModule(256, out_features, conv_type)  # Final layer to output the desired features
                ])
            case "SGCN":
                self.layers = nn.ModuleList([
                    GCNModule(in_features, out_features, conv_type),
                ])
            case "SAGE" | "cuSAGE":
                self.layers = nn.ModuleList([
                    GCNModule(in_features, 256, conv_type),
                    nn.Tanh(),
                    GCNModule(256, out_features, conv_type)
                ])
            case "GAT":
                NUM_HEADS = 8
                self.layers = nn.ModuleList([
                    GCNModule(in_features, 128, neighborhood_size=1, conv="GAT", num_heads=NUM_HEADS),
                    nn.Tanh(),
                    GCNModule(128 * NUM_HEADS, out_features, neighborhood_size=1, conv="GAT", num_heads=1)
                ])
            case "GIN":
                self.layers = nn.ModuleList([
                    GCNModule(in_features, 256, conv_type),
                    nn.Tanh(),
                    GCNModule(256, out_features, conv_type)
                ])
            case _:
                raise NotImplementedError(f"{conv_type} not supported yet.")
        
        # Initialize weights with Xavier initialization
        for layer in self.layers:
            if isinstance(layer, GCNModule):
                for param in layer.parameters():
                    if param.dim() > 1:  # Only apply to weights
                        nn.init.xavier_uniform_(param)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GCNModule):
                x = layer(x, self.edge_index)
            else:
                x = layer(x)
        return x


class GCNModule(nn.Module):
    def __init__(self, in_channels, out_channels, neighborhood_size=1, conv="GCN", num_heads=8):
        super().__init__()
        match conv:
            case "GCN":
                self.conv = GCNConv(in_channels, out_channels)
            case "SGCN":
                self.conv = SGConv(in_channels, out_channels, K=neighborhood_size*3)
            case "SAGE":
                self.conv = SAGEConv(in_channels, out_channels)
            case "cuSAGE":
                self.conv = CuGraphSAGEConv(in_channels, out_channels)
            case "GAT":
                self.conv = GATConv(in_channels, out_channels, heads=num_heads, concat=True)
                self.conv.batch_size = None  # Allow for handling of additional batches
            case "GIN":
                self.conv = GINConv(
                    nn.Sequential(
                        nn.Linear(in_channels, out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels, out_channels)
                    ), 
                    train_eps=True,

                )
            case _:
                raise NotImplementedError("This convolutional layer does not have support.")

        # Initialize weights with Xavier initialization
        for param in self.parameters():
            if param.dim() > 1:  # Only apply to weights
                nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

def custom_cluster_initialization(original_adata, method, K=17, num_pcs=3, resolution=0.45, dataset="SYNTHETIC", neighborhood_size=1):

    # this is important for graph based methods
    original_adata.generate_neighborhood_graph(original_adata.xenium_spot_data, n_neighbors=1+4*config.data.neighborhood_size, n_pcs=num_pcs, plot_pcas=False)

    # This function initializes clusters based on the specified method
    if method == "K-Means":
        if dataset == "DLPFC":
            initial_clusters = original_adata.KMeans(original_adata.xenium_spot_data, save_plot=False, K=K, include_spatial=False, use_pca=True)
        else:
            initial_clusters = original_adata.KMeans(original_adata.xenium_spot_data, save_plot=False, K=K, include_spatial=False, use_pca=False)            
    elif method == "Hierarchical":
        if dataset == "DLPFC":
            initial_clusters = original_adata.Hierarchical(original_adata.xenium_spot_data, save_plot=True, num_clusters=K, use_pca=True, include_spatial=False)
        else:
            initial_clusters = original_adata.Hierarchical(original_adata.xenium_spot_data, save_plot=True, num_clusters=K, use_pca=True, include_spatial=False)
    elif method == "Leiden":
        initial_clusters = original_adata.Leiden(original_adata.xenium_spot_data, resolutions=[resolution], save_plot=False, K=K)[resolution]
    elif method == "Louvain":
        initial_clusters = original_adata.Louvain(original_adata.xenium_spot_data, resolutions=[resolution], save_plot=False, K=K)[resolution]
    elif method == "mclust":
        original_adata.pca(original_adata.xenium_spot_data, num_pcs)
        initial_clusters = original_adata.mclust(original_adata.xenium_spot_data, G=K, model_name = "EEE")
    elif method == "random":
        initial_clusters = np.random.randint(0, K, size=original_adata.xenium_spot_data.X.shape[0])
    else:
        raise ValueError(f"Unknown method: {method}")

    return initial_clusters

def prepare_data(config):

    if config.data.dataset == "SYNTHETIC":
        gene_data, spatial_locations, original_adata, TRUE_PRIOR_WEIGHTS = prepare_synthetic_data(
            num_clusters=config.data.num_clusters, 
            data_dimension=config.data.data_dimension
        )
    elif config.data.dataset == "DLPFC":
        gene_data, spatial_locations, original_adata = prepare_DLPFC_data(151673, num_pcs=config.data.num_pcs)
        TRUE_PRIOR_WEIGHTS = None

    # clamping
    MIN_CONCENTRATION = config.VI.min_concentration

    gene_data = StandardScaler().fit_transform(gene_data)
    data = torch.tensor(gene_data).float()
    num_obs, data_dim = data.shape

    rows = spatial_locations["row"].astype(int)
    columns = spatial_locations["col"].astype(int)

    num_rows = max(rows) + 1
    num_cols = max(columns) + 1

    if config.data.init_method !=  "None":
        initial_clusters = custom_cluster_initialization(original_adata, config.data.init_method, K=config.data.num_clusters, num_pcs=config.data.num_pcs, resolution=config.data.resolution, dataset=config.data.dataset, neighborhood_size=config.data.neighborhood_size)

        if config.data.dataset == "SYNTHETIC":
            ari = ARI(initial_clusters, TRUE_PRIOR_WEIGHTS.argmax(axis=-1))
            nmi = NMI(initial_clusters, TRUE_PRIOR_WEIGHTS.argmax(axis=-1))
            cluster_metrics = {
                "ARI": round(ari, 3),
                "NMI": round(nmi, 3)
            }

            with open(f"{original_adata.target_dir}/cluster_metrics.json", 'w') as fp:
                json.dump(cluster_metrics, fp)

        elif config.data.dataset == "DLPFC":
            # Create a DataFrame for easier handling
            cluster_data = pd.DataFrame({
                'ClusterAssignments': initial_clusters,
                'Region': original_adata.xenium_spot_data.obs["Region"]
            })

            # Drop rows where 'Region' is NaN
            filtered_data = cluster_data.dropna(subset=['Region'])
            ari = ARI(filtered_data['ClusterAssignments'], filtered_data['Region'])
            nmi = NMI(filtered_data['ClusterAssignments'], filtered_data['Region'])

            cluster_metrics = {
                "ARI": round(ari, 3),
                "NMI": round(nmi, 3)
            }

            with open(f"{original_adata.target_dir}/cluster_metrics.json", 'w') as fp:
                json.dump(cluster_metrics, fp)

    empirical_prior_means = torch.randn(config.data.num_clusters, gene_data.shape[1])
    empirical_prior_scales = 1.0 + torch.abs(torch.randn(config.data.num_clusters, gene_data.shape[1]))
    assert sum(np.unique(initial_clusters) >= 0) == config.data.num_clusters, "K doesn't match initial number of unique detected clusters."
    if config.VI.empirical_prior:
        for i in range(config.data.num_clusters):
            cluster_data = gene_data[initial_clusters == i]
            if len(cluster_data) > 1:  # Check if there are any elements in the cluster_data
                empirical_prior_means[i] = torch.tensor(cluster_data.mean(axis=0))
                empirical_prior_scales[i] = torch.tensor(cluster_data.std(axis=0))
            else:
                raise ValueError("Not all clusters have a data point.")
    cluster_probs_prior = torch.zeros((initial_clusters.shape[0], config.data.num_clusters))
    cluster_probs_prior[torch.arange(initial_clusters.shape[0]), initial_clusters] = 1.

    locations_tensor = torch.tensor(spatial_locations.to_numpy())

    # Compute the number of elements in each dimension
    num_spots = cluster_probs_prior.shape[0]

    # Initialize an empty tensor for spatial concentration priors
    spatial_cluster_probs_prior = torch.zeros_like(cluster_probs_prior, dtype=torch.float64)

    spot_locations = KDTree(locations_tensor.cpu())  # Ensure this tensor is in host memory
    neighboring_spot_indexes = spot_locations.query_ball_point(locations_tensor.cpu(), r=config.data.neighborhood_size, p=1, workers=8)

    # Iterate over each spot
    for i in tqdm(range(num_spots)):

        # Select priors in the neighborhood
        priors_in_neighborhood = cluster_probs_prior[neighboring_spot_indexes[i]]

        # Compute the sum or mean, or apply a custom weighting function
        if config.data.neighborhood_agg == "mean":
            neighborhood_priors = priors_in_neighborhood.mean(dim=0)
        else:
            locations = original_adata.xenium_spot_data.obs[["x_location", "y_location", "z_location"]].values
            neighboring_locations = locations[neighboring_spot_indexes[i]].astype(float)
            distances = torch.tensor(np.linalg.norm(neighboring_locations - locations[i], axis=1))
            def distance_weighting(x):
                weight = 1/(1 + x/1)
                # print(weight)
                return weight / weight.sum()
            neighborhood_priors = (priors_in_neighborhood * distance_weighting(distances).reshape(-1, 1)).sum(dim=0)
        # Update the cluster probabilities
        spatial_cluster_probs_prior[i] += neighborhood_priors

    spatial_cluster_probs_prior = spatial_cluster_probs_prior.clamp(MIN_CONCENTRATION)
    sample_for_assignment_options = [True, False]

    for sample_for_assignment in sample_for_assignment_options:

        if sample_for_assignment:
            cluster_assignments_prior_TRUE = pyro.sample("cluster_assignments", dist.Categorical(spatial_cluster_probs_prior).expand_by([config.VI.num_prior_samples])).detach().mode(dim=0).values
            cluster_assignments_prior = cluster_assignments_prior_TRUE
        else:
            cluster_assignments_prior_FALSE = spatial_cluster_probs_prior.argmax(dim=1)
            cluster_assignments_prior = cluster_assignments_prior_FALSE

        # Load the data
        data = torch.tensor(gene_data).float()

        cluster_grid_PRIOR = torch.zeros((num_rows, num_cols), dtype=torch.long)

        cluster_grid_PRIOR[rows, columns] = cluster_assignments_prior + 1

    return data, spatial_locations, original_adata, spatial_cluster_probs_prior, empirical_prior_means, empirical_prior_scales, TRUE_PRIOR_WEIGHTS, cluster_grid_PRIOR

def train(
        model, 
        guide, 
        data,
        config,
        true_prior_weights=None
    ):

    NUM_EPOCHS = config.flows.num_epochs
    NUM_BATCHES = int(math.ceil(data.shape[0] / config.flows.batch_size))

    def per_param_callable(param_name):
        if param_name == 'cluster_means_q_mean':
            return {"lr": 0.0005, "betas": (0.9, 0.999)}
        elif param_name == 'cluster_scales_q_mean':
            return {"lr": 0.0005, "betas": (0.9, 0.999)}
        else:
            return {"lr": config.flows.lr, "betas": (0.9, 0.999), "weight_decay": 1e-6}

    # Create a scheduler using ClippedAdam with a parameter callable
    scheduler = ClippedAdam(per_param_callable)
    
    # Retrieve parameter names and their associated optimizer hyperparameters
    print("Hyperparameters:")
    param_store = pyro.get_param_store()

    for param_name in param_store.keys():
        param_value = param_store[param_name]
        # Get the optimizer associated with this parameter
        optim_state = scheduler.optim_objs[param_value.unconstrained()]
        hyperparams = optim_state["optim_args"]
        print(f"Parameter: {param_name}")
        print(f"  lr: {hyperparams['lr']}, betas: {hyperparams['betas']}, weight_decay: {hyperparams.get('weight_decay', 0)}")

    # Setup the inference algorithm
    svi = SVI(model, guide, scheduler, loss=TraceMeanField_ELBO(num_particles=config.VI.num_particles, vectorize_particles=True))

    epoch_pbar = tqdm(range(NUM_EPOCHS))
    current_min_loss = float('inf')
    patience_counter = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        running_loss = 0.0
        for step in range(NUM_BATCHES):
            loss = svi.step(data)
            running_loss += loss / config.flows.batch_size

        epoch_pbar.set_description(f"Epoch {epoch} : loss = {round(running_loss, 4)}")

        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():  # Ensure no backpropagation graphs are used
                cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_flow_dist(data)).to_event(1))
                cluster_assignments_posterior = torch.argmax(torch.softmax(cluster_logits, dim=1), dim=1)

                if config.data.dataset == "DLPFC":
                    true_assignments = original_adata.xenium_spot_data.obs.loc[~original_adata.xenium_spot_data.obs["Region"].isna(), "Region"]
                    print(f"ARI: {ARI(cluster_assignments_posterior.cpu(), true_assignments):.3f}, NMI: {NMI(cluster_assignments_posterior.cpu(), true_assignments):.3f}")

        if running_loss > current_min_loss:
            patience_counter += 1
        else:
            current_min_loss = running_loss
            patience_counter = 0
            best_params = pyro.get_param_store().get_state()
        if patience_counter >= config.flows.patience:
            break

        torch.cuda.empty_cache()

    return best_params

def save_filepath(config):
    total_file_path = os.path.join(
        "results", config.data.dataset, "XenNF", 
        f"DATA_DIM={config.data.data_dimension}", 
        f"K={config.data.num_clusters}", 
        f"INIT={config.data.init_method}", 
        f"NEIGHBORSIZE={config.data.neighborhood_size}" if config.data.dataset == "SYNTHETIC" else f"RADIUS={config.data.radius}", 
        f"PRIOR_FLOW_TYPE={config.flows.prior_flow_type}", 
        f"PRIOR_FLOW_LENGTH={config.flows.prior_flow_length}" if config.flows.prior_flow_type != 'CNF' else '', 
        f"GCONV={config.flows.gconv_type}", 
        f"POST_FLOW_TYPE={config.flows.posterior_flow_type}", 
        f"POST_FLOW_LENGTH={config.flows.posterior_flow_length}" if config.flows.posterior_flow_type != 'CNF' else '', 
        f"HIDDEN_LAYERS={config.flows.hidden_layers}"
    )
    return total_file_path

def edit_flow_nn(config, flow, graph, graph_conv="GCN"):
    match type(flow):
        case zuko.flows.continuous.CNF:
            network = GCNFlowModel(
                original_graph=graph,
                in_features=flow.transform.ode[0].weight.shape[1],
                out_features=config.data.num_clusters,
                conv_type=graph_conv,
            )
            flow.transform.ode = network
        case zuko.flows.autoregressive.MAF:
            network = GCNFlowModel(
                original_graph=graph,
                in_features=config.data.num_clusters,
                out_features=config.data.num_clusters*2,
                conv_type=graph_conv,
            )

            # flow.transform.transforms.insert(0, copy.deepcopy(flow.transform.transforms[0]))
            flow.transform.transforms[0].hyper = network
        case zuko.flows.spline.NSF:
            network = GCNFlowModel(
                original_graph=graph,
                in_features=config.data.num_clusters,  # Corrected in_features for neural spline flow
                out_features=config.data.num_clusters,      # Corrected out_features for neural spline flow
                conv_type=graph_conv,
            )

            flow.transform.transforms.insert(0, copy.deepcopy(flow.transform.transforms[0]))
            flow.transform.transforms[0].hyper = network
        case _:
            # Handle default case
            pass
    return flow

def posterior_eval(
        data, 
        spatial_locations,  
        original_adata,
        config,
        spatial_cluster_probs_prior, 
        true_prior_weights=None,
        loading_trained_model=False
    ):

    cluster_probs_samples = []
    batch_size = config.flows.batch_size

    if loading_trained_model:
        pyro.module("posterior_flow", cluster_probs_flow_dist, update_module_params=True)

    rows = spatial_locations["row"].astype(int)
    columns = spatial_locations["col"].astype(int)

    num_rows = max(rows) + 1
    num_cols = max(columns) + 1

    cluster_grid = torch.zeros((num_rows, num_cols), dtype=torch.long)

    with torch.no_grad():
        current_power = 0
        for sample_num in range(1, config.VI.num_posterior_samples + 1):
            cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_flow_dist(data)).to_event(1))

            # Make the logits numerically stable
            max_logit = torch.max(cluster_logits, dim=-1, keepdim=True).values
            stable_logits = cluster_logits - max_logit
            cluster_probs_sample = torch.nn.functional.softmax(stable_logits, dim=-1)
            torch.cuda.empty_cache()
            cluster_probs_samples.append(cluster_probs_sample)
            del cluster_probs_sample  # Explicitly delete

            if sample_num == config.VI.num_posterior_samples or sample_num == 10**current_power:

                cluster_probs_avg = torch.stack(cluster_probs_samples).mean(dim=0)
                cluster_assignments_posterior = cluster_probs_avg.argmax(dim=-1)

                if config.data.dataset == "DLPFC":
                    original_adata.xenium_spot_data.obs["cluster"] = -1
                    original_adata.xenium_spot_data.obs.loc[~original_adata.xenium_spot_data.obs["Region"].isna(), "cluster"] = cluster_assignments_posterior.cpu().numpy()
                    cluster_assignments_posterior = torch.tensor(original_adata.xenium_spot_data.obs["cluster"])

                cluster_grid[rows, columns] = cluster_assignments_posterior + 1

                if config.data.dataset == "SYNTHETIC":
                    colormap = plt.cm.get_cmap('rainbow', config.data.num_clusters + 1)
                elif config.data.dataset == "DLPFC":
                    colors = plt.cm.get_cmap('rainbow', config.data.num_clusters)
                    grey_color = [0.5, 0.5, 0.5, 1]  # Medium gray for unused cluster
                    colormap_colors = np.vstack((grey_color, colors(np.linspace(0, 1, config.data.num_clusters))))
                    colormap = ListedColormap(colormap_colors)
                else:
                    colors = plt.cm.get_cmap('rainbow', config.data.num_clusters + 1)
                    colormap_colors = np.vstack(([[1, 1, 1, 1]], colors(np.linspace(0, 1, config.data.num_clusters))))
                    colormap = ListedColormap(colormap_colors)

                plt.figure(figsize=(6, 6))
                if config.data.dataset == "DLPFC":

                    # Create mapping between region names and integer codes
                    region_to_int = {name: code + 1 for code, name in enumerate(original_adata.xenium_spot_data.obs["Region"].cat.categories)}
                    int_to_region = {code + 1: name for code, name in enumerate(original_adata.xenium_spot_data.obs["Region"].cat.categories)}

                    # Scatter plot
                    plt.scatter(columns, rows, c=cluster_assignments_posterior.cpu()+1, cmap=colormap, marker='h', s=12)#, edgecolors='white')

                    # Calculate padding for the axis limits
                    x_padding = (columns.max() - columns.min()) * 0.02  # 2% padding
                    y_padding = (rows.max() - rows.min()) * 0.02        # 2% padding

                    # Set axis limits with padding
                    plt.xlim(columns.min() - x_padding, columns.max() + x_padding)
                    plt.ylim(rows.min() - y_padding, rows.max() + y_padding)

                    # Force square appearance by stretching the y-axis
                    plt.gca().set_aspect((columns.max() - columns.min() + 2 * x_padding) / 
                                        (rows.max() - rows.min() + 2 * y_padding))  # Adjust for padded ranges
                    plt.gca().invert_yaxis()  # Maintain spatial orientation
                    # Add colorbar and title
                    plt.colorbar(ticks=range(config.data.num_clusters+1), label="True Label").set_ticklabels(["NA"] + list(int_to_region.values()))
                    plt.tight_layout()  # Minimize padding around the plot
                else:

                    plt.imshow(cluster_grid.cpu(), cmap=colormap, interpolation='nearest', origin='lower')
                    plt.colorbar(ticks=range(config.data.num_clusters + 1), label='Cluster Values')
                plt.title('Posterior Cluster Assignment with XenNF')

                if not os.path.exists(xennf_clusters_filepath := save_filepath(config)):
                    os.makedirs(xennf_clusters_filepath)
                _ = plt.savefig(
                    f"{xennf_clusters_filepath}/result_nsamples={sample_num}.png"
                )

                # no axis version save for writeup
                plt.figure(figsize=(6, 6))
                if config.data.dataset == "DLPFC":

                    # Create mapping between region names and integer codes
                    region_to_int = {name: code + 1 for code, name in enumerate(original_adata.xenium_spot_data.obs["Region"].cat.categories)}
                    int_to_region = {code + 1: name for code, name in enumerate(original_adata.xenium_spot_data.obs["Region"].cat.categories)}

                    # Scatter plot
                    plt.scatter(columns, rows, c=cluster_assignments_posterior.cpu()+1, cmap=colormap, marker='h', s=12)#, edgecolors='white')

                    # Calculate padding for the axis limits
                    x_padding = (columns.max() - columns.min()) * 0.02  # 2% padding
                    y_padding = (rows.max() - rows.min()) * 0.02        # 2% padding

                    # Set axis limits with padding
                    plt.xlim(columns.min() - x_padding, columns.max() + x_padding)
                    plt.ylim(rows.min() - y_padding, rows.max() + y_padding)

                    # Force square appearance by stretching the y-axis
                    plt.gca().set_aspect((columns.max() - columns.min() + 2 * x_padding) / 
                                        (rows.max() - rows.min() + 2 * y_padding))  # Adjust for padded ranges
                    plt.gca().invert_yaxis()  # Maintain spatial orientation
                    # Add colorbar and title
                    plt.colorbar(ticks=range(config.data.num_clusters+1), label="True Label").set_ticklabels(["NA"] + list(int_to_region.values()))
                    plt.tight_layout()  # Minimize padding around the plot
                else:

                    plt.imshow(cluster_grid.cpu(), cmap=colormap, interpolation='nearest', origin='lower')
                    plt.colorbar(ticks=range(config.data.num_clusters + 1), label='Cluster Values')
                plt.axis('off')

                if not os.path.exists(xennf_clusters_filepath := save_filepath(config)):
                    os.makedirs(xennf_clusters_filepath)
                _ = plt.savefig(
                    f"{xennf_clusters_filepath}/simple_result_nsamples={sample_num}.png",
                    bbox_inches='tight',
                    pad_inches=0
                )

                if config.data.dataset == "SYNTHETIC":
                    ari = ARI(cluster_assignments_posterior.cpu(), true_prior_weights.argmax(axis=-1))
                    nmi = NMI(cluster_assignments_posterior.cpu(), true_prior_weights.argmax(axis=-1))
                
                elif config.data.dataset == "DLPFC":
                    # Create a DataFrame for easier handling
                    cluster_data = pd.DataFrame({
                        'ClusterAssignments': cluster_assignments_posterior.cpu().numpy(),
                        'Region': original_adata.xenium_spot_data.obs["Region"]
                    })

                    # Drop rows where 'Region' is NaN
                    filtered_data = cluster_data.dropna(subset=['Region'])
                    ari = ARI(filtered_data['ClusterAssignments'], filtered_data['Region'])
                    nmi = NMI(filtered_data['ClusterAssignments'], filtered_data['Region'])

                cluster_metrics = {
                    "ARI": round(ari, 3),
                    "NMI": round(nmi, 3)
                }
                with open(f"{xennf_clusters_filepath}/cluster_metrics_nsamples={sample_num}.json", 'w') as fp:
                    json.dump(cluster_metrics, fp)

                current_power += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the normalizing flow or load existing parameters.")
    parser.add_argument("-l", "--load_model", action="store_true", default=False, help="Load a pre-trained model.")
    parser.add_argument("-n", "--config_name", default="0", help="The name of the config to use.")
    args = parser.parse_args()

    # cuda setup
    if torch.cuda.is_available():
        print("YAY! GPU available :3")
        
        # Get all available GPUs sorted by memory usage (lowest first)
        available_gpus = GPUtil.getAvailable(order='memory', limit=1)
        
        if available_gpus:
            selected_gpu = available_gpus[0]
            
            # Set the GPU with the lowest memory usage
            torch.cuda.set_device(selected_gpu)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            
            print(f"Using GPU: {selected_gpu} with the lowest memory usage.")
        else:
            print("No GPUs available with low memory usage.")
    else:
        print("No GPU available :(")

    # setup 
    torch.set_printoptions(sci_mode=False)
    expected_total_param_dim = 2 # K x D
    warnings.filterwarnings("ignore")
    config = OmegaConf.load(f'config/config_DLPFC/config{args.config_name}.yaml')

    (
        data, 
        spatial_locations, 
        original_adata, 
        spatial_cluster_probs_prior, 
        empirical_prior_means, 
        empirical_prior_scales, 
        true_prior_weights, 
        cluster_states
    ) = prepare_data(config)

    print(empirical_prior_means, empirical_prior_scales)

    cluster_probs_graph_flow_dist = setup_zuko_flow(
        flow_type=config.flows.prior_flow_type,
        flow_length=config.flows.prior_flow_length,
        num_clusters=config.data.num_clusters,
        context_length=0,
        hidden_layers=(128, 128)
    )

    # handle DLPFC missingness
    if config.data.dataset == "DLPFC":

        non_na_mask = ~original_adata.xenium_spot_data.obs["Region"].isna()
        data = data[non_na_mask]

        # setup the data graph
        positions = torch.tensor(spatial_locations[non_na_mask].to_numpy()).float()
        # edge_index = pyg.nn.knn_graph(positions, k=1+4*config.data.neighborhood_size, loop=True)
        edge_index = pyg.nn.radius_graph(positions, r=config.data.radius, loop=True)
        
        # Print summary metrics about the graph
        num_nodes = positions.shape[0]
        num_edges = edge_index.shape[1]
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        print(f"Number of nodes in the graph: {num_nodes}")
        print(f"Number of edges in the graph: {num_edges}")
        print(f"Average degree of the graph: {avg_degree:.2f}")
        print(f"Density of the graph: {density:.4f}")

        input_x = torch.tensor(original_adata.xenium_spot_data.X[non_na_mask], dtype=torch.float32)
        graph = Data(x=data, edge_index=edge_index)
    else:

        # setup the data graph
        positions = torch.tensor(spatial_locations.to_numpy()).float()
        edge_index = pyg.nn.knn_graph(positions, k=1+4*config.data.neighborhood_size, loop=True)

        input_x = torch.tensor(original_adata.xenium_spot_data.X, dtype=torch.float32)
        graph = Data(x=data, edge_index=edge_index)

    # we are modelling over the whole graph
    if config.flows.batch_size == -1:
        config.flows.batch_size = len(data)

    # update the flow to use the gcn as the hypernet
    cluster_probs_graph_flow_dist = edit_flow_nn(config, cluster_probs_graph_flow_dist, graph, config.flows.gconv_type)

    def model(data):

        pyro.module("prior_flow", cluster_probs_graph_flow_dist)
        
        with pyro.plate("clusters", config.data.num_clusters):

            # Define the means and variances of the Gaussian components
            cluster_means = pyro.sample("cluster_means", dist.Normal(empirical_prior_means, 10.0).to_event(1))
            cluster_scales = pyro.sample("cluster_scales", dist.LogNormal(empirical_prior_scales, 10.0).to_event(1))

        cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_graph_flow_dist()).expand([len(data)]).to_event(1))

        with pyro.plate("data", len(data)):

            if cluster_means.dim() == expected_total_param_dim:
                pyro.sample(f"obs", dist.MixtureOfDiagNormals(
                        cluster_means.unsqueeze(0).expand(config.flows.batch_size, -1, -1), 
                        cluster_scales.unsqueeze(0).expand(config.flows.batch_size, -1, -1), 
                        cluster_logits.squeeze(1)
                    ), 
                    obs=data
                )
            # likelihood for batch WITH vectorization of particles
            else:
                pyro.sample(f"obs", dist.MixtureOfDiagNormals(
                        cluster_means.unsqueeze(1).expand(-1, config.flows.batch_size, -1, -1), 
                        cluster_scales.unsqueeze(1).expand(-1, config.flows.batch_size, -1, -1), 
                        cluster_logits.squeeze(1)
                    ), 
                    obs=data
                )

        # # Define priors for the cluster assignment probabilities and Gaussian parameters
        # with pyro.plate("data", len(data), subsample_size=config.flows.batch_size) as ind:
        #     batch_data = data[ind]
        #     batch_logits = cluster_logits[..., ind, :]
        #     # likelihood for batch
        #     if cluster_means.dim() == expected_total_param_dim:
        #         pyro.sample(f"obs", dist.MixtureOfDiagNormals(
        #                 cluster_means.unsqueeze(0).expand(config.flows.batch_size, -1, -1), 
        #                 cluster_scales.unsqueeze(0).expand(config.flows.batch_size, -1, -1), 
        #                 batch_logits.squeeze(1)
        #             ), 
        #             obs=batch_data
        #         )
        #     # likelihood for batch WITH vectorization of particles
        #     else:
        #         pyro.sample(f"obs", dist.MixtureOfDiagNormals(
        #                 cluster_means.unsqueeze(1).expand(-1, config.flows.batch_size, -1, -1), 
        #                 cluster_scales.unsqueeze(1).expand(-1, config.flows.batch_size, -1, -1), 
        #                 batch_logits.squeeze(1)
        #             ), 
        #             obs=batch_data
        #         )

    cluster_probs_flow_dist = setup_zuko_flow(
        flow_type=config.flows.posterior_flow_type,
        flow_length=config.flows.posterior_flow_length,
        num_clusters=config.data.num_clusters,
        context_length=config.data.data_dimension,
        hidden_layers=config.flows.hidden_layers
    )
 
    def guide(data):
        
        pyro.module("posterior_flow", cluster_probs_flow_dist)

        with pyro.plate("clusters", config.data.num_clusters):
            # Global variational parameters for cluster means and scales
            cluster_means_q_mean = pyro.param("cluster_means_q_mean", empirical_prior_means + torch.randn_like(empirical_prior_means) * 0.05)
            cluster_scales_q_mean = pyro.param("cluster_scales_q_mean", empirical_prior_scales + torch.randn_like(empirical_prior_scales) * 0.01, constraint=dist.constraints.positive)
            if config.VI.learn_global_variances:
                cluster_means_q_scale = pyro.param("cluster_means_q_scale", torch.ones_like(empirical_prior_means) * 1.0, constraint=dist.constraints.positive)
                cluster_scales_q_scale = pyro.param("cluster_scales_q_scale", torch.ones_like(empirical_prior_scales) * 0.25, constraint=dist.constraints.positive)
                cluster_means = pyro.sample("cluster_means", dist.Normal(cluster_means_q_mean, cluster_means_q_scale).to_event(1))
                cluster_scales = pyro.sample("cluster_scales", dist.LogNormal(cluster_scales_q_mean, cluster_scales_q_scale).to_event(1))
            else:
                cluster_means = pyro.sample("cluster_means", dist.Delta(cluster_means_q_mean).to_event(1))
                cluster_scales = pyro.sample("cluster_scales", dist.Delta(cluster_scales_q_mean).to_event(1))
        if config.flows.posterior_flow_type == "UMNN":
            cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_flow_dist(data, constant=0.0)).to_event(1))
        else:
            cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_flow_dist(data)).to_event(1))
        # print("GUIDE ", cluster_logits.shape)

    model_save_path = os.path.join(
        "/nfs/turbo/lsa-regier/scratch", 
        "roko/nf_results",
        save_filepath(config)
    )

    # Set CUDA launch blocking for better error reporting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if args.load_model:
        print("Loading pre-trained model...")
        try:
            model_file = os.path.join(model_save_path, 'model.save')
            # Ensure the model parameters are loaded to the current GPU
            pyro.get_param_store().load(model_file, map_location=torch.device(f'cuda:{selected_gpu}'))
        except FileNotFoundError:
            raise FileNotFoundError("This model version doesn't have a saved version yet. You need to train it.")

    else:
        pyro.clear_param_store()
        best_params = train(model, guide, data, config, true_prior_weights)
        pyro.get_param_store().set_state(best_params)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)
        model_file = os.path.join(model_save_path, 'model.save')
        pyro.get_param_store().save(model_file)

    posterior_eval(data, spatial_locations, original_adata, config, cluster_states, true_prior_weights)