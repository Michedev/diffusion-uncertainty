# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""k-NN precision and recall."""

import numpy as np
import torch
from time import time

#----------------------------------------------------------------------------

def batch_pairwise_distances(U, V):
    """Compute pairwise distances between two batches of feature vectors."""
    # Squared norms of each row in U and V.
    norm_u = torch.sum(U**2, dim=1)
    norm_v = torch.sum(V**2, dim=1)
        
    # norm_u as a column and norm_v as a row vectors.
    norm_u = norm_u.view(-1, 1)
    norm_v = norm_v.view(1, -1)

    # Pairwise squared Euclidean distances.
    D = torch.clip(norm_u - 2*torch.matmul(U, V.t()) + norm_v, min=0.0)

    return D

#----------------------------------------------------------------------------

class DistanceBlock():
    """Provides multi-GPU support to calculate pairwise distances between two batches of feature vectors."""
    def __init__(self, num_features, num_gpus):
        self.num_features = num_features
        self.num_gpus = num_gpus

    def pairwise_distances(self, U, V):
        """Evaluate pairwise distances between two batches of feature vectors."""
        U_split = torch.split(U, U.shape[0] // self.num_gpus, dim=0)
        V_split = torch.split(V, V.shape[0] // self.num_gpus, dim=0)

        distances_split = []
        for gpu_idx in range(self.num_gpus):
            with torch.cuda.device(gpu_idx):
                distances_split.append(batch_pairwise_distances(U_split[gpu_idx].cuda(), V_split[gpu_idx].cuda()))

        distances = torch.cat(distances_split, dim=1).cpu()
        return distances

#----------------------------------------------------------------------------

class ManifoldEstimator():
    """Estimates the manifold of given feature vectors."""

    def __init__(self, distance_block, features, row_batch_size=25000, col_batch_size=50000,
                 nhood_sizes=[3], clamp_to_percentile=None, eps=1e-5):
        """Estimate the manifold of given feature vectors.
        
            Args:
                distance_block: DistanceBlock object that distributes pairwise distance
                    calculation to multiple GPUs.
                features (np.array/torch.Tensor): Matrix of feature vectors to estimate their manifold.
                row_batch_size (int): Row batch size to compute pairwise distances
                    (parameter to trade-off between memory usage and performance).
                col_batch_size (int): Column batch size to compute pairwise distances.
                nhood_sizes (list): Number of neighbors used to estimate the manifold.
                clamp_to_percentile (float): Prune hyperspheres that have radius larger than
                    the given percentile.
                eps (float): Small number for numerical stability.
        """
        num_images = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features
        self._distance_block = distance_block

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        self.D = torch.zeros([num_images, self.num_nhoods], dtype=torch.float32)
        distance_batch = torch.zeros([row_batch_size, num_images], dtype=torch.float32)
        seq = torch.arange(max(self.nhood_sizes) + 1, dtype=torch.int64)

        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0:end1-begin1, begin2:end2] = self._distance_block.pairwise_distances(row_batch, col_batch)
    
            # Find the k-nearest neighbor from the current batch.
            self.D[begin1:end1, :] = torch.topk(distance_batch[0:end1-begin1, :], k=max(self.nhood_sizes)+1, dim=1, largest=False, sorted=True)[0][:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D.numpy(), clamp_to_percentile, axis=0)
            self.D[self.D > torch.from_numpy(max_distances)] = 0

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        """Evaluate if new feature vectors are at the manifold."""
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        distance_batch = torch.zeros([self.row_batch_size, num_ref_images], dtype=torch.float32)
        batch_predictions = torch.zeros([num_eval_images, self.num_nhoods], dtype=torch.int32)
        max_realism_score = torch.zeros([num_eval_images,], dtype=torch.float32)
        nearest_indices = torch.zeros([num_eval_images,], dtype=torch.int64)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0:end1-begin1, begin2:end2] = self._distance_block.pairwise_distances(feature_batch, ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0:end1-begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = torch.any(samples_in_manifold, dim=1).int()

            max_realism_score[begin1:end1] = torch.max(self.D[:, 0] / (distance_batch[0:end1-begin1, :] + self.eps), dim=1)[0]
            nearest_indices[begin1:end1] = torch.argmin(distance_batch[0:end1-begin1, :], dim=1)

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions
    


def knn_precision_recall_features(ref_features, eval_features, nhood_sizes=[3],
                                  row_batch_size=10000, col_batch_size=50000, num_gpus=1):
    """Calculates k-NN precision and recall for two sets of feature vectors.
    
        Args:
            ref_features (np.array/tf.Tensor): Feature vectors of reference images.
            eval_features (np.array/tf.Tensor): Feature vectors of generated images.
            nhood_sizes (list): Number of neighbors used to estimate the manifold.
            row_batch_size (int): Row batch size to compute pairwise distances
                (parameter to trade-off between memory usage and performance).
            col_batch_size (int): Column batch size to compute pairwise distances.
            num_gpus (int): Number of GPUs used to evaluate precision and recall.

        Returns:
            State (dict): Dict that contains precision and recall calculated from
            ref_features and eval_features.
    """
    state = dict()
    num_images = ref_features.shape[0]
    num_features = ref_features.shape[1]

    # Initialize DistanceBlock and ManifoldEstimators.
    distance_block = DistanceBlock(num_features, num_gpus)
    ref_manifold = ManifoldEstimator(distance_block, ref_features, row_batch_size, col_batch_size, nhood_sizes) 
    eval_manifold = ManifoldEstimator(distance_block, eval_features, row_batch_size, col_batch_size, nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    print('Evaluating k-NN precision and recall with %i samples...' % num_images)
    start = time()

    # Precision: How many points from eval_features are in ref_features manifold.
    precision = ref_manifold.evaluate(eval_features)
    state['precision'] = precision.mean(dim=0)

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features)
    state['recall'] = recall.mean(dim=0)

    print('Evaluated k-NN precision and recall in: %gs' % (time() - start))

    return state
