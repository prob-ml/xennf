import itertools
import yaml
import os

flow_type = ["CNF", "MAF", "NSF"]
flow_length_by_type = {
    "CNF": [32],  # CNF only needs one "flow length" since it doesn't use one
    "MAF": [2, 4, 8],
    "NSF": [2, 4, 8]
}
init_method = ["Louvain", "K-Means"]
hidden_layers = [
    [512, 512, 512],
    [512, 512, 512, 512]
]
neighborhood_size = [1]
graph_conv = ["GCN", "SAGE"]

DATASET = "DLPFC"
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
        neighborhood_size,
        graph_conv,
    )))

def get_resolution(dataset, init_method, data_dimension=5, num_clusters=7):
    match dataset:
        case "SYNTHETIC":
            if init_method == "Leiden":
                return 0.47
            return 0.65
        case "DLPFC":
            if init_method == "Louvain":
                return 0.175
            return 0.25

for i, combo in enumerate(all_combinations):

    flow_type, flow_length, init_method, hidden_layers, neighborhood_size, graph_conv = combo

    config_yaml = f"""
    data:
        dataset: {DATASET}
        data_dimension: {8 if DATASET == "DLPFC" else 5}
        num_clusters: 7
        resolution: {get_resolution(DATASET, init_method)}
        num_pcs: {8 if DATASET == "DLPFC" else 5}
        init_method: {init_method}
        neighborhood_size: {neighborhood_size}
        neighborhood_agg: "mean"
    VI:
        empirical_prior: True
        learn_global_variances: False
        min_concentration: 0.001
        num_prior_samples: 1000
        num_posterior_samples: 2500
        num_particles: 3
    flows:
        gconv_type: {graph_conv}
        prior_flow_type: CNF
        posterior_flow_type: {flow_type}
        flow_length: {flow_length}
        hidden_layers: {hidden_layers}
        num_epochs: 2500
        batch_size: 512
        patience: 50
        lr: 0.001
    """

    config_file = yaml.safe_load(config_yaml)

    with open(f'{config_filepath}/config{i}.yaml', 'w') as file:
        yaml.dump(config_file, file)


