import torch
import zuko
import numpy as np
from torch import Size, Tensor
import pyro
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import Adam, PyroOptim

import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.distributions.transforms import Spline, ComposeTransform
from pyro.distributions import TransformedDistribution

from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

# Utility imports
import GPUtil
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import os
import json
from utils import ARI, NMI

# Custom module imports
from xenium_cluster import XeniumCluster
from data import prepare_DLPFC_data, prepare_synthetic_data, prepare_Xenium_data
from zuko_flow import setup_zuko_flow, ZukoToPyro
from omegaconf import OmegaConf

class Potts2D(dist.Distribution):

    def __init__(self, current_state, batch_idx, num_clusters=7, radius=1, gamma=1.5):
        super().__init__()
        self.current_state = current_state
        self.batch_idx = batch_idx
        self.num_clusters = num_clusters
        self.radius = radius
        self.gamma = gamma
        self.num_rows, self.num_cols = current_state.shape

    @property
    def batch_shape(self):
        # The shape of the grid
        return torch.Size([len(self.batch_idx)])

    @property
    def event_shape(self):
        # No event dimensions (this is over the whole grid)
        return torch.Size([])

    def expand(self, batch_shape, _instance=None):
        # Create a new instance of Potts2D with the same parameters but a new batch shape
        new_instance = Potts2D(
            current_state=self.current_state,
            batch_idx=self.batch_idx,
            num_clusters=self.num_clusters,
            radius=self.radius,
            gamma=self.gamma,
        )
        return new_instance

    def __call__(self):
        return self.sample()

    def get_neighbors(self, i, j):
        neighbors = []
        for x in range(max(0, i - self.radius), min(self.current_state.shape[0], i + self.radius + 1)):
            for y in range(max(0, j - self.radius), min(self.current_state.shape[1], j + self.radius + 1)):
                if abs(x - i) + abs(y - j) <= self.radius:
                    neighbors.append((x, y))
        return neighbors

    def sample(self):
        num_rows, num_cols = self.current_state.shape

        new_state = self.current_state.clone() 
        new_soft_state = torch.zeros(new_state.size(0), new_state.size(1), self.num_clusters)
        
        for i in range(num_rows):
            for j in range(num_cols):
            # Compute conditional probabilities for site i
                probs = torch.zeros(self.num_clusters)
                for k in range(1, self.num_clusters + 1):
                    # Compute the contribution of neighbors
                    probs[k-1] = torch.sum(
                        torch.tensor(
                            [2 * self.gamma if new_state[a][b] == k else 0.0 for (a, b) in self.get_neighbors(i, j)]
                        )
                    )
                
                # Normalize to get valid probabilities
                probs = torch.exp(probs - torch.max(probs))  # Avoid numerical issues
                probs /= probs.sum()
                new_soft_state[i][j] = probs
                new_state[i][j] = torch.multinomial(probs, 1).item()

        self.current_state = new_state
        
        return new_soft_state.reshape(-1, self.num_clusters)[self.batch_idx]

    def log_prob(self, cluster_probs):

        if cluster_probs.dim() == 2:

            cluster_state_flattened = self.current_state.reshape(-1,1)[self.batch_idx]

            # -1 is for indexing purposes. The 0 cluster is for empty cells.
            # print(cluster_probs.shape)
            # print(range(cluster_state_flattened.size(0)))
            # print(cluster_state_flattened.flatten() - 1)
            cluster_prob_tensor = cluster_probs[range(cluster_state_flattened.size(0)), cluster_state_flattened.flatten() - 1]

            log_prob_tensor = torch.log(cluster_prob_tensor)

        else:

            cluster_state_flattened = self.current_state.reshape(-1,1)[self.batch_idx]

            # -1 is for indexing purposes. The 0 cluster is for empty cells.
            cluster_prob_tensor = cluster_probs[:, range(cluster_state_flattened.size(0)), cluster_state_flattened.flatten() - 1]

            log_prob_tensor = torch.log(cluster_prob_tensor)

        return log_prob_tensor  # Return the sum of all values in log_prob_tensor

def custom_cluster_initialization(original_adata, method, K=17, num_pcs=3):

    original_adata.generate_neighborhood_graph(original_adata.xenium_spot_data, plot_pcas=False)

    # This function initializes clusters based on the specified method
    if method == "K-Means":
        initial_clusters = original_adata.KMeans(original_adata.xenium_spot_data, save_plot=False, K=K, include_spatial=False)
    elif method == "Hierarchical":
        initial_clusters = original_adata.Hierarchical(original_adata.xenium_spot_data, save_plot=True, num_clusters=K)
    elif method == "Leiden":
        initial_clusters = original_adata.Leiden(original_adata.xenium_spot_data, resolutions=[0.35], save_plot=False, K=K)[0.35]
    elif method == "Louvain":
        initial_clusters = original_adata.Louvain(original_adata.xenium_spot_data, resolutions=[0.35], save_plot=False, K=K)[0.35]
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

    initial_clusters = custom_cluster_initialization(original_adata, config.data.init_method, K=config.data.num_clusters, num_pcs=config.data.num_pcs)

    empirical_prior_means = torch.zeros(config.data.num_clusters, gene_data.shape[1])
    empirical_prior_scales = torch.ones(config.data.num_clusters, gene_data.shape[1])
    if config.VI.empirical_prior:
        for i in range(config.data.num_clusters):
            cluster_data = gene_data[initial_clusters == i]
            if cluster_data.size > 0:  # Check if there are any elements in the cluster_data
                empirical_prior_means[i] = torch.tensor(cluster_data.mean(axis=0))
                empirical_prior_scales[i] = torch.tensor(cluster_data.std(axis=0))
    cluster_probs_prior = torch.zeros((initial_clusters.shape[0], config.data.num_clusters))
    cluster_probs_prior[torch.arange(initial_clusters.shape[0]), initial_clusters - 1] = 1.

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

        colors = plt.cm.get_cmap('viridis', config.data.num_clusters)

        plt.figure(figsize=(6, 6))
        plt.imshow(cluster_grid_PRIOR.cpu(), cmap=colors, interpolation='nearest', origin='lower')
        plt.colorbar(ticks=range(1, config.data.num_clusters + 1), label='Cluster Values')
        plt.title('Prior Cluster Assignment with XenNF')

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
            return {"lr": config.flows.lr, "betas": (0.9, 0.999)}
        elif param_name == 'cluster_scales_q_mean':
            return {"lr": config.flows.lr, "betas": (0.9, 0.999)}
        elif "logit" in param_name:
            return {"lr": config.flows.lr, "betas": (0.9, 0.999)}
        else:
            return {"lr": config.flows.lr, "betas": (0.9, 0.999)}

    scheduler = Adam(per_param_callable)

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
        # if (epoch % 10 == 0 or epoch == 1) and config.data.dataset == "SYNTHETIC":
        #     ari = ARI(cluster_assignments_posterior, true_prior_weights.argmax(dim=-1))
        #     nmi = NMI(cluster_assignments_posterior, true_prior_weights.argmax(dim=-1))
        #     print(f"ARI = {round(ari, 2)} NMI = {round(nmi, 2)}")
        if running_loss > current_min_loss:
            patience_counter += 1
        else:
            current_min_loss = running_loss
            patience_counter = 0
        if patience_counter >= config.flows.patience:
            break 

def save_filepath(config):
    total_file_path = (
        f"results/{config.data.dataset}/XenNF/DATA_DIM={config.data.data_dimension}/"
        f"K={config.data.num_clusters}/INIT={config.data.init_method}/NEIGHBORSIZE={config.data.neighborhood_size}/"
        f"FLOW_TYPE={config.flows.flow_type}/FLOW_LENGTH={config.flows.flow_length}/HIDDEN_LAYERS={config.flows.hidden_layers}"
    )
    return total_file_path

def posterior_eval(
        data, 
        spatial_locations,  
        config, 
        true_prior_weights=None
    ):

    cluster_probs_samples = []
    batch_size = config.flows.batch_size

    with torch.no_grad():
        current_power = 0
        for sample_num in range(1, config.VI.num_posterior_samples + 1):
            cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_flow_dist(data)), infer={'is_auxiliary': True})
            temperature = pyro.param("temperature")

            # Make the logits numerically stable
            max_logit = torch.max(cluster_logits, dim=-1, keepdim=True).values
            stable_logits = cluster_logits - max_logit
            with pyro.plate("data", len(data)):
                cluster_probs_sample = pyro.sample(
                    "cluster_probs",
                    dist.RelaxedOneHotCategorical(temperature=temperature, logits=stable_logits, validate_args=False)
                )
            torch.cuda.empty_cache()
            cluster_probs_samples.append(cluster_probs_sample)
            del cluster_probs_sample  # Explicitly delete

            if sample_num == config.VI.num_posterior_samples or sample_num == 10**current_power:

                cluster_probs_avg = torch.stack(cluster_probs_samples).mean(dim=0)
                cluster_assignments_posterior = cluster_probs_avg.argmax(dim=-1)

                rows = spatial_locations["row"].astype(int)
                columns = spatial_locations["col"].astype(int)

                num_rows = max(rows) + 1
                num_cols = max(columns) + 1

                cluster_grid = torch.zeros((num_rows, num_cols), dtype=torch.long)

                cluster_grid[rows, columns] = cluster_assignments_posterior + 1

                colors = plt.cm.get_cmap('viridis', config.data.num_clusters + 1)

                plt.figure(figsize=(6, 6))
                plt.imshow(cluster_grid.cpu(), cmap=colors, interpolation='nearest', origin='lower')
                plt.colorbar(ticks=range(1, config.data.num_clusters + 1), label='Cluster Values')
                plt.title('Posterior Cluster Assignment with XenNF')

                if not os.path.exists(xennf_clusters_filepath := save_filepath(config)):
                    os.makedirs(xennf_clusters_filepath)
                _ = plt.savefig(
                    f"{xennf_clusters_filepath}/result_nsamples={sample_num}.png"
                )

                if config.data.dataset == "SYNTHETIC":
                    ari = ARI(cluster_assignments_posterior.cpu(), true_prior_weights.argmax(axis=-1))
                    nmi = NMI(cluster_assignments_posterior.cpu(), true_prior_weights.argmax(axis=-1))
                    cluster_metrics = {
                        "ARI": ari,
                        "NMI": nmi
                    }
                    with open(f"{xennf_clusters_filepath}/cluster_metrics_nsamples={sample_num}.json", 'w') as fp:
                        json.dump(cluster_metrics, fp)

                current_power += 1


if __name__ == "__main__":

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
    pyro.clear_param_store()
    expected_total_param_dim = 2 # K x D
    warnings.filterwarnings("ignore")
    config = OmegaConf.load('config/config1.yaml')

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

    # model, flow, and guide setup
    def model(data):

        global cluster_states

        with pyro.plate("clusters", config.data.num_clusters):

            # Define the means and variances of the Gaussian components
            cluster_means = pyro.sample("cluster_means", dist.Normal(empirical_prior_means, 10.0).to_event(1))
            cluster_scales = pyro.sample("cluster_scales", dist.LogNormal(empirical_prior_scales, 10.0).to_event(1))

        # Define priors for the cluster assignment probabilities and Gaussian parameters
        with pyro.plate("data", len(data), subsample_size=config.flows.batch_size) as ind:
            batch_data = data[ind]
            print(batch_data[:5])
            prior_dist = Potts2D(cluster_states, ind, radius=config.data.neighborhood_size, num_clusters=config.data.num_clusters)
            cluster_probs = pyro.sample("cluster_probs", prior_dist)
            cluster_states = prior_dist.current_state
            # likelihood for batch
            if cluster_means.dim() == expected_total_param_dim:
                pyro.sample(f"obs", dist.MixtureOfDiagNormals(
                        cluster_means.unsqueeze(0).expand(config.flows.batch_size, -1, -1), 
                        cluster_scales.unsqueeze(0).expand(config.flows.batch_size, -1, -1), 
                        torch.log(cluster_probs)
                    ), 
                    obs=batch_data
                )
            # likelihood for batch WITH vectorization of particles
            else:
                pyro.sample(f"obs", dist.MixtureOfDiagNormals(
                        cluster_means.unsqueeze(1).expand(-1, config.flows.batch_size, -1, -1), 
                        cluster_scales.unsqueeze(1).expand(-1, config.flows.batch_size, -1, -1), 
                        torch.log(cluster_probs)
                    ), 
                    obs=batch_data
                )

    cluster_probs_flow_dist = setup_zuko_flow(
        flow_type=config.flows.flow_type,
        flow_length=config.flows.flow_length,
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

        with pyro.plate("data", len(data), subsample_size=config.flows.batch_size) as ind:
            batch_data = data[ind]

            cluster_logits = pyro.sample("cluster_logits", ZukoToPyro(cluster_probs_flow_dist(batch_data)), infer={'is_auxiliary': True})
            temperature = pyro.param(
                "temperature",
                torch.tensor(0.75),
                constraint=dist.constraints.positive
            )

            # Make the logits numerically stable
            max_logit = torch.max(cluster_logits, dim=-1, keepdim=True).values
            stable_logits = cluster_logits - max_logit

            # Sample cluster_probs using RelaxedOneHotCategorical
            cluster_probs = pyro.sample(
                "cluster_probs",
                dist.RelaxedOneHotCategorical(temperature=temperature, logits=stable_logits)
            )

    # execution
    train(model, guide, data, config, true_prior_weights)
    posterior_eval(data, spatial_locations, config, true_prior_weights)