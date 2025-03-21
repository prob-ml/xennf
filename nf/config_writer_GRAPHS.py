import itertools
import yaml
import os
import shutil

use_empirical_params = [False]
prior_flow_type = ["MAF"]
posterior_flow_type = ["CNF"]
prior_flow_length_by_type = {
    "CNF": [32],  # CNF only needs one "flow length" since it doesn't use one
    "MAF": [3],
    # "NSF": [1, 3]
}
posterior_flow_length_by_type = {
    "CNF": [32],  # CNF only needs one "flow length" since it doesn't use one
    "MAF": [16, 32, 64],
    "NSF": [16, 32, 64]
}
init_method = ["K-Means"]
hidden_layers = [
    [2048, 1024, 512, 256, 128, 64, 32, 16],
]
neighborhood_size = [1]
radius_size = [2, 2.25, 2.5, 2.75, 3]
graph_depth = [1]
graph_features = [512]
graph_conv = ["GCN", "SAGE"]
activations = ["Tanh"]
KL_Annealing = [150, 1000, False]

DATASET = "DLPFC"
config_filepath = f"config/config_{DATASET}"

if os.path.exists(config_filepath):
    shutil.rmtree(config_filepath)
os.makedirs(config_filepath, exist_ok=True)

# Generate all possible combinations
all_combinations = []
for prior_ft in prior_flow_type:
    for posterior_ft in posterior_flow_type:
        all_combinations.extend(list(itertools.product(
            (prior_ft,), # Wrap prior_ft in a tuple
            (posterior_ft,), # Wrap posterior_ft in a tuple
            prior_flow_length_by_type[prior_ft],
            posterior_flow_length_by_type[posterior_ft],
            init_method,
            hidden_layers,
            neighborhood_size,
            radius_size,
            graph_depth,
            graph_features,
            graph_conv,
            use_empirical_params,
            activations,
            KL_Annealing,
        )))

print(f"THERE ARE {len(all_combinations)} COMBOS.")

def get_resolution(dataset, init_method, data_dimension=5, num_clusters=7):
    match dataset:
        case "SYNTHETIC":
            if init_method == "Leiden":
                return 0.47
            return 0.65
        case "DLPFC":
            if init_method == "Louvain":
                return 0.195
            return 0.25

for i, combo in enumerate(all_combinations):

    prior_flow_type = combo[0]
    posterior_flow_type = combo[1]
    prior_flow_length = combo[2]
    posterior_flow_length = combo[3]
    init_method = combo[4]
    hidden_layers = combo[5]
    neighborhood_size = combo[6]
    radius_size = combo[7]
    graph_depth = combo[8]
    graph_width = combo[9]
    graph_conv = combo[10]
    use_empirical_params = combo[11]
    activation = combo[12]
    kl_anneal = combo[13]

    config_yaml = f"""
    data:
        dataset: {DATASET}
        data_dimension: {8 if DATASET == "DLPFC" else 5}
        num_clusters: 7
        resolution: {get_resolution(DATASET, init_method)}
        num_pcs: {8 if DATASET == "DLPFC" else 5}
        init_method: {init_method if  use_empirical_params else "None"}
        neighborhood_size: {neighborhood_size}
        neighborhood_agg: "mean"
        radius: {radius_size}
    VI:
        empirical_prior: {use_empirical_params}
        kl_annealing: {kl_anneal}
        learn_global_variances: False
        min_concentration: 0.001
        num_prior_samples: 1000
        num_posterior_samples: 1000
        num_particles: 3
    graphs:
        depth: {graph_depth}
        width: {graph_width}
        gconv_type: {graph_conv}
    flows:
        prior_flow_type: {prior_flow_type}
        prior_flow_length: {prior_flow_length}
        posterior_flow_type: {posterior_flow_type}
        posterior_flow_length: {posterior_flow_length}
        hidden_layers: {hidden_layers}
        num_epochs: 10000
        batch_size: -1
        patience: {10 if use_empirical_params else 25}
        activation: {activation}
        lr: 
          cluster_means_q_mean: 
            lr: 0.001
            betas: 
              - 0.9
              - 0.999
            lrd: 1.0
          cluster_scales_q_mean: 
            lr: 0.001
            betas: 
              - 0.9
              - 0.999
            lrd: 1.0
          default: 
            lr: 0.001
            betas:
              - 0.9
              - 0.999
            lrd: 1.0
            weight_decay: 1e-6
    """

    config_file = yaml.safe_load(config_yaml)

    with open(f'{config_filepath}/config{i}.yaml', 'w') as file:
        yaml.dump(config_file, file)


