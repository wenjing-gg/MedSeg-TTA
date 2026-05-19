import os
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
from pathlib import Path

from utils.clustering import run_kmeans

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    script_dir = Path(__file__).resolve().parent
    artifact_dir = script_dir.parent / "artifacts"
    parser = argparse.ArgumentParser(description="cluster STDR source anchors")
    parser.add_argument(
        "--features-path",
        type=str,
        default=str(artifact_dir / "features_t1c" / "WCH_dataset_objective_vectors_512.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(artifact_dir / "Reference_points_3dunet_t1c"),
    )
    parser.add_argument("--ncentroids", type=int, default=30)
    args = parser.parse_args()
    seed = 36
    # fix random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # load and transition to 24966*(19*256)
    CAU = torch.load(args.features_path)
    print("CAU.shape = ", CAU.shape)
    x = np.reshape(CAU, (CAU.shape[0], CAU.shape[1] * CAU.shape[2])).astype('float32')
    print("CAU.shape = ", x.shape)

    # kmeans
    ncentroids = args.ncentroids
    cluster_centroids, cluster_index, cluster_loss = run_kmeans(x, ncentroids, verbose=True)

    '''
    # origin cluster
    ncentroids = 10
    niter = 20
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=True, gpu=True)
    kmeans.train(x)
    # get the result
    cluster_result = kmeans.centroids
    cluster_loss = kmeans.obj
    '''
    print("cluster_centroids = ",cluster_centroids)
    print("len(cluster_centroids) = ", len(cluster_centroids))
    # print("cluster_index = ",cluster_index)
    print("len(cluster_index) = ",len(cluster_index))
    print("cluster_loss = ",cluster_loss)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(cluster_centroids, os.path.join(args.output_dir, 'SPH_cluster256_centroids_full_%d.pkl' % ncentroids))
    '''
    # import cluster
    nmb = 10
    deepcluster = clustering.Kmeans(nmb)
    clustering_loss = deepcluster.cluster(CAU, verbose=True)
    '''


if __name__ == '__main__':
    main()
