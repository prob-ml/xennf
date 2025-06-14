from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score


def NMI(cluster_pred_1, cluster_pred_2):
    return normalized_mutual_info_score(cluster_pred_1, cluster_pred_2)

def ARI(cluster_pred_1, cluster_pred_2):
    return adjusted_rand_score(cluster_pred_1, cluster_pred_2)

def AMI(cluster_pred_1, cluster_pred_2):
    return adjusted_mutual_info_score(cluster_pred_1, cluster_pred_2)