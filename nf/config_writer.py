import itertools
import yaml
import os

flow_type = ["CNF", "MAF", "NSF"]
flow_length_by_type = {
    "CNF": [32],  # CNF only needs one flow length
    "MAF": [2, 8, 32],
    "NSF": [2, 8, 32]
}
init_method = ["mclust", "Leiden", "Louvain", "K-Means"]
hidden_layers = [
    [128, 128, 128],
    [512, 512, 512]
]
gamma = [1.5, 2.0, 2.5, 3.0]
neighborhood_size = [1,2]

DATASET = "SYNTHETIC"
config_filepath = f"config/config_{DATASET}"

os.makedirs(config_filepath, exist_ok=True)

# Generate all possible combinations
all_combinations = []
for ft in flow_type:
    all_combinations.extend(list(itertools.product(
        [ft],
        flow_length_by_type[ft],
        init_method,
        hidden_layers,
        gamma,
        neighborhood_size
    )))

for i, combo in enumerate(all_combinations):

    flow_type, flow_length, init_method, hidden_layers, gamma, neighborhood_size = combo

    config_yaml = f"""
    data:
        dataset: "SYNTHETIC"
        data_dimension: 5
        num_clusters: 7
        resolution: {0.65 if init_method == "Louvain" else 0.47} # 0.47 for Leiden, 0.65 for Louvain when K = 7
        num_pcs: 3
        init_method: {init_method}
        neighborhood_size: {neighborhood_size}
        neighborhood_agg: "mean"
    VI:
        empirical_prior: True
        learn_global_variances: False
        min_concentration: 0.001
        num_prior_samples: 100
        num_posterior_samples: 2500
        num_particles: 25
        gamma: {gamma}
    flows:
        flow_type: {flow_type}
        flow_length: {flow_length}
        hidden_layers: {hidden_layers}
        num_epochs: 1000
        batch_size: 512
        patience: 25
        lr: 0.001
    """

    config_file = yaml.safe_load(config_yaml)

    with open(f'{config_filepath}/config{i}.yaml', 'w') as file:
        yaml.dump(config_file, file)


