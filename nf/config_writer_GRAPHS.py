import itertools
import yaml
import os

prior_flow_type = ["MAF"]
posterior_flow_type = ["CNF"]
prior_flow_length_by_type = {
    "CNF": [32],  # CNF only needs one "flow length" since it doesn't use one
    "MAF": [1, 3, 5],
    "NSF": [2, 4, 8]
}
posterior_flow_length_by_type = {
    "CNF": [32],  # CNF only needs one "flow length" since it doesn't use one
    "MAF": [16, 32, 64],
    "NSF": [16, 32, 64]
}
init_method = ["K-Means", "mclust"]
hidden_layers = [
    [2048, 1024, 512, 256, 128, 64, 32, 16],
]
neighborhood_size = [1]
radius_size = [2, 2.25, 2.5, 2.75, 3]
graph_conv = ["GCN", "SAGE"]

DATASET = "DLPFC"
config_filepath = f"config/config_{DATASET}"

os.makedirs(config_filepath, exist_ok=True)

# Generate all possible combinations
all_combinations = []
for prior_ft in prior_flow_type:
    for posterior_ft in posterior_flow_type:
        all_combinations.extend(list(itertools.product(
            (prior_ft,),  # Wrap prior_ft in a tuple
            (posterior_ft,),  # Wrap posterior_ft in a tuple
            prior_flow_length_by_type[prior_ft],
            posterior_flow_length_by_type[posterior_ft],
            init_method,
            hidden_layers,
            neighborhood_size,
            radius_size,
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
    graph_conv = combo[8]

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
        radius: {radius_size}
    VI:
        empirical_prior: True
        learn_global_variances: False
        min_concentration: 0.001
        num_prior_samples: 1000
        num_posterior_samples: 2500
        num_particles: 3
    flows:
        gconv_type: {graph_conv}
        prior_flow_type: {prior_flow_type}
        prior_flow_length: {prior_flow_length}
        posterior_flow_type: {posterior_flow_type}
        posterior_flow_length: {posterior_flow_length}
        hidden_layers: {hidden_layers}
        num_epochs: 10000
        batch_size: -1
        patience: 25
        lr: 0.00075
    """

    config_file = yaml.safe_load(config_yaml)

    with open(f'{config_filepath}/config{i}.yaml', 'w') as file:
        yaml.dump(config_file, file)


