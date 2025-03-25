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
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, CuGraphSAGEConv, GATConv, GINConv, GMMConv
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

# Set seeds for all libraries
seed = 42
np.random.seed(seed)

# If using PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class GCNFlowModel(nn.Module):
    def __init__(self, original_graph, in_features, out_features, depth=2, width=512, conv_type="GCN"):
        super().__init__()
        self.x = original_graph.x
        self.edge_index = original_graph.edge_index
        self.edge_attr = original_graph.edge_attr
        self.edge_weights = original_graph.edge_weights
        self.set_layers(in_features, out_features, conv_type, depth, width)
        self.conv_type = conv_type

    def set_layers(self, in_features, out_features, conv_type, depth, width):
        match conv_type:
            case "GCN":
                self.layers = nn.ModuleList()
                for _ in range(depth):
                    self.layers.append(GCNModule(in_features, width, conv_type))
                    self.layers.append(retrieve_activation(config))  # Changed to use retrieve_activation
                    in_features = width
                self.layers.append(GCNModule(width, out_features, conv_type))
            case "SGCN":
                self.layers = nn.ModuleList([
                    GCNModule(in_features, out_features, conv_type, hops=depth),
                ])
            case "SAGE" | "cuSAGE":
                self.layers = nn.ModuleList()
                for _ in range(depth):
                    self.layers.append(GCNModule(in_features, width, conv_type))
                    self.layers.append(retrieve_activation(config))  # Changed to use retrieve_activation
                    in_features = width
                self.layers.append(GCNModule(width, out_features, conv_type))
            case "GAT":
                NUM_HEADS = 8
                self.layers = nn.ModuleList([
                    GCNModule(in_features, width, neighborhood_size=1, conv="GAT", num_heads=NUM_HEADS),
                    retrieve_activation(config),  # Changed to use retrieve_activation
                    GCNModule(width * NUM_HEADS, out_features, neighborhood_size=1, conv="GAT", num_heads=1)
                ])
            case "GIN":
                self.layers = nn.ModuleList()
                for _ in range(depth):
                    self.layers.append(GCNModule(in_features, width, conv_type))
                    self.layers.append(retrieve_activation(config))  # Changed to use retrieve_activation
                    in_features = width
                self.layers.append(GCNModule(width, out_features, conv_type))
            case "MoNeT":
                self.layers = nn.ModuleList()
                # for _ in range(depth):
                #     self.layers.append(GCNModule(in_features, width, conv_type))
                #     self.layers.append(retrieve_activation(config))  # Changed to use retrieve_activation
                #     in_features = width
                # self.layers.append(GCNModule(width, out_features, conv_type))
                self.layers.append(GCNModule(in_features, out_features, conv_type))
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
                if self.conv_type == "MoNeT":
                    x = layer(x, self.edge_index, edge_attr=self.edge_attr)
                # elif self.conv_type == "GCN":
                #     x = layer(x, self.edge_index, edge_weights=self.edge_weights)
                else:
                    x = layer(x, self.edge_index)
            else:
                x = layer(x)
        return x


class GCNModule(nn.Module):
    def __init__(self, in_channels, out_channels, conv="GCN", hops=1, num_heads=8, num_clusters=7):
        super().__init__()
        self.conv_type = conv
        match conv:
            case "GCN":
                self.conv = GCNConv(in_channels, out_channels)
            case "SGCN":
                self.conv = SGConv(in_channels, out_channels, K=hops)
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
                        retrieve_activation(config),  # Changed to use retrieve_activation
                        nn.Linear(out_channels, out_channels)
                    ), 
                    train_eps=True,
                )
            case "MoNeT":
                self.conv = GMMConv(in_channels, out_channels, dim=2, kernel_size=num_clusters, separate_gaussians=True)
            case _:
                raise NotImplementedError("This convolutional layer does not have support.")

        # Initialize weights with Xavier initialization
        for param in self.parameters():
            if param.dim() > 1:  # Only apply to weights
                nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, edge_attr=None, edge_weights=None):
        if self.conv_type == "MoNeT":
            return self.conv(x, edge_index, edge_attr=edge_attr)
        elif self.conv_type in ["GCN"]:
            return self.conv(x, edge_index, edge_weight=edge_weights)
        return self.conv(x, edge_index)

def retrieve_activation(config):
    if not hasattr(config.flows, 'activation'):
        return nn.Tanh()

    match config.flows.activation:
        case "ReLU":
            return nn.ReLU()
        case "LeakyReLU":
            return nn.LeakyReLU()
        case "Tanh":
            return nn.Tanh()
        case "Sigmoid":
            return nn.Sigmoid()
        case "ELU":
            return nn.ELU()
        case "SELU":
            return nn.SELU()
        case _:
            return nn.Tanh()

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

def prepare_data(config, dlpfc_sample):

    if config.data.dataset == "SYNTHETIC":
        gene_data, spatial_locations, original_adata, TRUE_PRIOR_WEIGHTS = prepare_synthetic_data(
            num_clusters=config.data.num_clusters, 
            data_dimension=config.data.data_dimension
        )
    elif config.data.dataset == "DLPFC":
        gene_data, spatial_locations, original_adata = prepare_DLPFC_data(dlpfc_sample, num_pcs=config.data.num_pcs)
        TRUE_PRIOR_WEIGHTS = None

    gene_data = StandardScaler().fit_transform(gene_data)
    data = torch.tensor(gene_data).float()

    if config.data.init_method != "None":
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
    # empirical_prior_means = torch.ones(config.data.num_clusters, gene_data.shape[1])
    empirical_prior_scales = torch.abs(torch.randn(config.data.num_clusters, gene_data.shape[1])).clamp(min=0.1)
    # empirical_prior_scales = torch.ones(config.data.num_clusters, gene_data.shape[1]) * 0.1
    if config.VI.empirical_prior:
        assert sum(np.unique(initial_clusters) >= 0) == config.data.num_clusters, "K doesn't match initial number of unique detected clusters."
        for i in range(config.data.num_clusters):
            cluster_data = gene_data[initial_clusters == i]
            if len(cluster_data) > 1:  # Check if there are any elements in the cluster_data
                empirical_prior_means[i] = torch.tensor(cluster_data.mean(axis=0))
                empirical_prior_scales[i] = torch.tensor(cluster_data.std(axis=0))
            else:
                raise ValueError("Not all clusters have a data point.")

    return data, spatial_locations, original_adata, empirical_prior_means, empirical_prior_scales, TRUE_PRIOR_WEIGHTS

def train(
        model, 
        guide, 
        data,
        config,
        true_prior_weights=None
    ):

    NUM_EPOCHS = config.flows.num_epochs
    NUM_BATCHES = int(math.ceil(data.shape[0] / config.flows.batch_size))
    MIN_ANNEAL = 0.001

    def per_param_callable_single_optim(param_name):
        if 'cluster_means_q_mean' in param_name:
            return dict(config.flows.lr.cluster_means_q_mean)
        elif 'cluster_scales_q_mean' in param_name:
            return dict(config.flows.lr.cluster_scales_q_mean)
        else:
            return dict(config.flows.lr.default)

    scheduler = ClippedAdam(per_param_callable_single_optim)

    # Setup the inference algorithm
    svi = SVI(model, guide, scheduler, loss=TraceMeanField_ELBO(num_particles=config.VI.num_particles, vectorize_particles=True))
    
    # ATTEMPT AT CYCLING WHAT TO LEARN
    # def per_param_callable_mean_optim(param_name):
    #     if 'cluster_means_q_mean' in param_name:
    #         return dict(config.flows.lr.cluster_means_q_mean)
    #     else:
    #         return {"lr": 0.0}
    # def per_param_callable_scale_optim(param_name):
    #     if 'cluster_scales_q_mean' in param_name:
    #         return dict(config.flows.lr.cluster_scales_q_mean)
    #     else:
    #         return {"lr": 0.0}
    # def per_param_callable_flow_optim(param_name):
    #     if 'flow' in param_name:
    #         return dict(config.flows.lr.default)
    #     else:
    #         return {"lr": 0.0}
            
    # flow_optimizer = ClippedAdam(per_param_callable_flow_optim)
    # mean_optimizer = ClippedAdam(per_param_callable_mean_optim)
    # scale_optimizer = ClippedAdam(per_param_callable_scale_optim)
    # svi_flow = SVI(model, guide, flow_optimizer, loss=TraceMeanField_ELBO(num_particles=config.VI.num_particles, vectorize_particles=True))
    # svi_means = SVI(model, guide, mean_optimizer, loss=TraceMeanField_ELBO(num_particles=config.VI.num_particles, vectorize_particles=True))
    # svi_scales = SVI(model, guide, scale_optimizer, loss=TraceMeanField_ELBO(num_particles=config.VI.num_particles, vectorize_particles=True))

    epoch_pbar = tqdm(range(NUM_EPOCHS))
    current_min_loss = float('inf')
    patience_counter = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        annealing_factor = 1.0 if not config.VI.kl_annealing else min(1.0, max(epoch / config.VI.kl_annealing, MIN_ANNEAL))
        # print("ANNEALING FACTOR", annealing_factor)
        running_loss = 0.0
        for step in range(NUM_BATCHES):
            loss = svi.step(data, annealing_factor=annealing_factor)  # Pass annealing_factor to svi.step
            running_loss += (loss / config.flows.batch_size)
            # if epoch % 40 > 20:
            #     loss = svi_flow.step(data)
            # else:
            #     loss = svi_means.step(data)
            #     loss = svi_scales.step(data)

        epoch_pbar.set_description(f"Epoch {epoch} : loss = {round(running_loss, 4)}")

        # print(
        #     f"MEAN LR", scheduler.optim_objs[pyro.param("cluster_means_q_mean")].state_dict()['param_groups'][0]['lr'], "\n", 
        #     f"SCALE LR", scheduler.optim_objs[pyro.param("cluster_scales_q_mean").unconstrained()].state_dict()['param_groups'][0]['lr'], "\n", 
        #     f"FLOW LR", scheduler.optim_objs[pyro.param("posterior_flow$$$transform.ode.0.weight")].state_dict()['param_groups'][0]['lr']
        # )
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():  # Ensure no backpropagation graphs are used
                cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_flow_dist(data)).to_event(1))
                cluster_assignments_posterior = torch.argmax(cluster_logits, dim=1)

                if config.data.dataset == "DLPFC":
                    true_assignments = original_adata.xenium_spot_data.obs.loc[~original_adata.xenium_spot_data.obs["Region"].isna(), "Region"]
                    print("Total Clusters Used:", torch.unique(cluster_assignments_posterior).numel(), "OUT OF", len(true_assignments.unique()))
                    # Print the cluster proportion breakdown for the posterior and the ground truth
                    posterior_proportions = torch.bincount(cluster_assignments_posterior).float() / len(cluster_assignments_posterior)
                    true_proportions = torch.bincount(torch.tensor(true_assignments.astype('category').cat.codes)).float() / len(true_assignments)
                    print(f"Posterior Cluster Proportions: {posterior_proportions}")
                    print(f"True Cluster Proportions: {true_proportions}")
                    current_ari = ARI(cluster_assignments_posterior.cpu(), true_assignments)
                    print(f"ARI: {current_ari:.3f}, NMI: {NMI(cluster_assignments_posterior.cpu(), true_assignments):.3f}")
                    
                    if 'max_ari' not in locals():
                        max_ari = current_ari
                    else:
                        max_ari = max(max_ari, current_ari)
                    
                    print(f"Maximum ARI so far: {max_ari:.3f}")
                # posterior_eval(
                #     data,
                #     spatial_locations,
                #     original_adata,
                #     config,
                #     num_samples=5
                # )

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
        f"GDEPTH={config.graphs.depth}",
        f"GWIDTH={config.graphs.width}",
        f"GCONV={config.graphs.gconv_type}", 
        f"POST_FLOW_TYPE={config.flows.posterior_flow_type}", 
        f"POST_FLOW_LENGTH={config.flows.posterior_flow_length}" if config.flows.posterior_flow_type != 'CNF' else '', 
        f"HIDDEN_LAYERS={config.flows.hidden_layers}",
        f"ACTIVATION={'Tanh' if not hasattr(config.flows, 'activation') else config.flows.activation}",
        f"KL_ANNEAL={config.VI.kl_annealing}",
        f"LEARN_GLOBAL_VARS={config.VI.learn_global_variances}"
    )
    return total_file_path

def edit_flow_nn(config, flow, graph):
    match type(flow):
        case zuko.flows.continuous.CNF:
            network = GCNFlowModel(
                original_graph=graph,
                in_features=flow.transform.ode[0].weight.shape[1],
                out_features=config.data.num_clusters,
                conv_type=config.graphs.gconv_type,
                depth=config.graphs.depth,
                width=config.graphs.width,
            )
            flow.transform.ode = network
        case zuko.flows.autoregressive.MAF:
            network = GCNFlowModel(
                original_graph=graph,
                in_features=config.data.num_clusters,
                out_features=config.data.num_clusters*2,
                conv_type=config.graphs.gconv_type,
                depth=config.graphs.depth,
                width=config.graphs.width,
            )

            # flow.transform.transforms.insert(0, copy.deepcopy(flow.transform.transforms[0]))
            flow.transform.transforms[0].hyper = network
        case zuko.flows.spline.NSF:
            network = GCNFlowModel(
                original_graph=graph,
                in_features=config.data.num_clusters,  # Corrected in_features for neural spline flow
                out_features=config.data.num_clusters,      # Corrected out_features for neural spline flow
                conv_type=config.graphs.gconv_type,
                depth=config.graphs.depth,
                width=config.graphs.width,
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
        num_samples=1000,
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
        for sample_num in range(1, num_samples + 1):
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

def pick_device(gpu_arg="auto", max_load=0.10, max_mem=0.15):
    """
    Picks a device based on the gpu_arg:
      - 'cpu' -> CPU
      - 'auto' -> The GPU with usage under (max_load, max_mem), else the least loaded GPU
      - integer string -> That GPU index if available
    Returns a torch.device or raises an error if not possible.
    """
    if not torch.cuda.is_available() or gpu_arg.lower() in ["cpu", "none"]:
        print("CUDA not available or CPU requested. Using CPU.")
        return torch.device("cpu")

    if gpu_arg.lower() == "auto":
        # Try to pick a GPU that's under thresholds for load/memory
        # so we don't overshadow another job if possible.
        best_gpus = GPUtil.getAvailable(
            order="memory",
            limit=1,
            maxLoad=max_load,
            maxMemory=max_mem
        )
        if best_gpus:
            # We found a GPU that meets the threshold
            chosen = best_gpus[0]
            print(f"Auto-select: Using GPU {chosen} (meets load/mem thresholds)")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
            return torch.device("cuda:0")  # Now the chosen GPU is "cuda:0" in this environment

        # If none meets the threshold, pick the single GPU with the lowest usage
        # (still better than a random pick).
        print("No GPU under thresholds. Picking the GPU with lowest usage.")
        best_gpus = GPUtil.getAvailable(order="memory", limit=1)
        if not best_gpus:
            print("No GPUs found by GPUtil; falling back to CPU.")
            return torch.device("cpu")
        chosen = best_gpus[0]
        print(f"Auto-select: Using GPU {chosen} anyway (busy but minimal).")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
        return torch.device("cuda:0")

    # Otherwise, the user gave an explicit GPU index
    try:
        gpu_index = int(gpu_arg)
        print(f"User specified GPU {gpu_index}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        return torch.device("cuda:0")
    except ValueError:
        raise ValueError(f"Invalid gpu argument: {gpu_arg}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the normalizing flow or load existing parameters.")
    parser.add_argument("-l", "--load_model", action="store_true", default=False, help="Load a pre-trained model.")
    parser.add_argument("-n", "--config_name", default="0", help="The name of the config to use.")
    parser.add_argument("-d", "--dlpfc_sample", type=int, default=151673, help="DLPFC Sample #")
    args = parser.parse_args()
    device = pick_device("auto")

    # cuda setup
    if torch.cuda.is_available():
        print("YAY! GPU available :3")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
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
    # don't run model again if it exists 
    if os.path.exists(save_filepath(config)):
        print("Directory already exists. Ending execution.")
        exit()  # Safely end execution if the directory exists

    (
        data, 
        spatial_locations, 
        original_adata,  
        empirical_prior_means, 
        empirical_prior_scales, 
        true_prior_weights
    ) = prepare_data(config, args.dlpfc_sample)

    cluster_probs_graph_flow_dist = setup_zuko_flow(
        flow_type=config.flows.prior_flow_type,
        flow_length=config.flows.prior_flow_length,
        num_clusters=config.data.num_clusters,
        context_length=0,
        hidden_layers=(256, 256),
        activation="Tanh" if not hasattr(config.flows, "activation") else config.flows.activation
    )

    # handle DLPFC missingness
    if config.data.dataset == "DLPFC":

        non_na_mask = ~original_adata.xenium_spot_data.obs["Region"].isna()
        data = data[non_na_mask]

        # setup the data graph
        positions = torch.tensor(spatial_locations[non_na_mask].to_numpy()).float()
        # edge_index = pyg.nn.knn_graph(positions, k=1+4*config.data.neighborhood_size, loop=True)
        edge_index = pyg.nn.radius_graph(positions, r=config.data.radius, loop=True)

        # Calculate edge attributes: polar angle and distance
        edge_attr = []
        node1_positions = positions[edge_index[0]]
        node2_positions = positions[edge_index[1]]
        distances = torch.norm(node2_positions - node1_positions, dim=1) / config.data.radius # Calculate distances
        angles = torch.atan2(node2_positions[:, 1] - node1_positions[:, 1], node2_positions[:, 0] - node1_positions[:, 0]) / (torch.pi / 2)  # Calculate polar angles
        degrees = torch.bincount(edge_index.flatten(), minlength=positions.shape[0])
        edge_attr = torch.stack((angles, distances, degrees[edge_index[0]], degrees[edge_index[1]]), dim=1)  # Store angles and distances as tensor

        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        # if using edge weights, set a max value (1.0) so that weights don't explode
        edge_weights = (1.0 / distances.clamp(min=1e-6)).clamp(max=1.0)
        
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
        graph = Data(x=data, edge_index=edge_index, edge_attr=edge_attr, edge_weights=edge_weights)
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
    cluster_probs_graph_flow_dist = edit_flow_nn(config, cluster_probs_graph_flow_dist, graph)

    MIN_CONCENTRATION = 0.01

    def model(data, annealing_factor=1.0):

        pyro.module("prior_flow", cluster_probs_graph_flow_dist)
        
        with pyro.plate("clusters", config.data.num_clusters):
            if config.VI.kl_annealing:
                with pyro.poutine.scale(scale=annealing_factor):
                    print(annealing_factor)
                    # Define the means and variances of the Gaussian components
                    cluster_means = pyro.sample("cluster_means", dist.Normal(empirical_prior_means, 10.0).to_event(1))
                    cluster_scales = pyro.sample("cluster_scales", dist.LogNormal(empirical_prior_scales, 10.0).to_event(1))
            else:
                # Define the means and variances of the Gaussian components
                cluster_means = pyro.sample("cluster_means", dist.Normal(empirical_prior_means, 10.0).to_event(1))
                cluster_scales = pyro.sample("cluster_scales", dist.LogNormal(empirical_prior_scales, 10.0).to_event(1))
        if config.VI.kl_annealing:
            with pyro.poutine.scale(scale=annealing_factor):
                cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_graph_flow_dist()).expand([len(data)]).to_event(1))
                max_logit = torch.max(cluster_logits, dim=-1, keepdim=True).values
                z_min = np.log((MIN_CONCENTRATION / (1 - MIN_CONCENTRATION)) * (config.data.num_clusters - 1))
                stable_logits = (cluster_logits - max_logit).clamp(min=z_min)
        else:
            cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_graph_flow_dist()).expand([len(data)]).to_event(1))
            max_logit = torch.max(cluster_logits, dim=-1, keepdim=True).values
            z_min = np.log((MIN_CONCENTRATION / (1 - MIN_CONCENTRATION)) * (config.data.num_clusters - 1))
            stable_logits = (cluster_logits - max_logit).clamp(min=z_min)    

        with pyro.plate("data", len(data)):

            if cluster_means.dim() == expected_total_param_dim:
                pyro.sample(f"obs", dist.MixtureOfDiagNormals(
                        cluster_means.unsqueeze(0).expand(config.flows.batch_size, -1, -1), 
                        cluster_scales.unsqueeze(0).expand(config.flows.batch_size, -1, -1), 
                        stable_logits.squeeze(1)
                    ), 
                    obs=data
                )
            # likelihood for batch WITH vectorization of particles
            else:
                pyro.sample(f"obs", dist.MixtureOfDiagNormals(
                        cluster_means.unsqueeze(1).expand(-1, config.flows.batch_size, -1, -1), 
                        cluster_scales.unsqueeze(1).expand(-1, config.flows.batch_size, -1, -1), 
                        stable_logits.squeeze(1)
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
        hidden_layers=config.flows.hidden_layers,
        activation="Tanh" if not hasattr(config.flows, "activation") else config.flows.activation
    )
 
    def guide(data, annealing_factor=1.0):
        
        pyro.module("posterior_flow", cluster_probs_flow_dist)

        with pyro.plate("clusters", config.data.num_clusters):
            # Global variational parameters for cluster means and scales
            cluster_means_q_mean = pyro.param("cluster_means_q_mean", empirical_prior_means + torch.randn_like(empirical_prior_means) * 0.15)
            cluster_scales_q_mean = pyro.param("cluster_scales_q_mean", empirical_prior_scales + torch.randn_like(empirical_prior_scales) * 0.03, constraint=dist.constraints.positive)
            if config.VI.learn_global_variances:
                cluster_means_q_scale = pyro.param("cluster_means_q_scale", torch.ones_like(empirical_prior_means) * 0.05, constraint=dist.constraints.positive)
                cluster_scales_q_scale = pyro.param("cluster_scales_q_scale", torch.ones_like(empirical_prior_scales) * 0.025, constraint=dist.constraints.positive)
                
                # Check if KL annealing is enabled
                if config.VI.kl_annealing:
                    with pyro.poutine.scale(scale=annealing_factor):
                        cluster_means = pyro.sample("cluster_means", dist.Normal(cluster_means_q_mean, cluster_means_q_scale).to_event(1))
                        cluster_scales = pyro.sample("cluster_scales", dist.LogNormal(cluster_scales_q_mean, cluster_scales_q_scale).to_event(1))
                else:
                    cluster_means = pyro.sample("cluster_means", dist.Normal(cluster_means_q_mean, cluster_means_q_scale).to_event(1))
                    cluster_scales = pyro.sample("cluster_scales", dist.LogNormal(cluster_scales_q_mean, cluster_scales_q_scale).to_event(1))
            else:
                # Check if KL annealing is enabled for Delta sampling
                if config.VI.kl_annealing:
                    with pyro.poutine.scale(scale=annealing_factor):
                        cluster_means = pyro.sample("cluster_means", dist.Delta(cluster_means_q_mean).to_event(1))
                        cluster_scales = pyro.sample("cluster_scales", dist.Delta(cluster_scales_q_mean).to_event(1))
                else:
                    cluster_means = pyro.sample("cluster_means", dist.Delta(cluster_means_q_mean).to_event(1))
                    cluster_scales = pyro.sample("cluster_scales", dist.Delta(cluster_scales_q_mean).to_event(1))
        
        if config.flows.posterior_flow_type == "UMNN":
            # Handle the case when kl_annealing is false
            if config.VI.kl_annealing:
                with pyro.poutine.scale(scale=annealing_factor):
                    cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_flow_dist(data, constant=0.0)).to_event(1))
            else:
                cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_flow_dist(data, constant=0.0)).to_event(1))
        else:
            # Handle the case when kl_annealing is false
            if config.VI.kl_annealing:
                with pyro.poutine.scale(scale=annealing_factor):
                    cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_flow_dist(data)).to_event(1))
            else:
                cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_flow_dist(data)).to_event(1))
            

        # # We add the penalty factor here
        # cluster_probs = torch.softmax(cluster_logits, dim=-1)  # [N, K]
        # usage = cluster_probs.mean(dim=0)                      # [K]
        
        # # KL(p||Unif) = \sum p*log(p/(1/K)) = \sum p*log(p*K)
        # alpha = 0.1
        # penalty = alpha * torch.sum(usage * torch.log((usage * config.data.num_clusters) + 1e-30))
        
        # # pyro.factor expects a name and a scalar tensor (negative log-factor):
        # print("UNIFORM PENALTY", -penalty)
        # pyro.factor("cluster_balance_factor", -penalty, has_rsample=True)
        # print(f"Mean Norm: {torch.norm(cluster_means_q_mean - empirical_prior_means)}")
        # print(f"Scale Norm: {torch.norm(cluster_scales_q_mean - empirical_prior_scales)}")

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
            pyro.get_param_store().load(model_file, map_location=torch.device(f'cuda:{0}'))
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

    posterior_eval(data, spatial_locations, original_adata, config, num_samples=config.VI.num_posterior_samples, true_prior_weights=true_prior_weights)