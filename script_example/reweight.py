#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize
from pyblock.blocking import reblock, find_optimal_block
import warnings
from collections import Counter
from numpy_indexed import group_by as group_by_
import time
import pickle
import os
from argparse import ArgumentParser


# utility functions

def group_by(keys: np.ndarray,
             values: np.ndarray = None,
             reduction: callable = None):
    if reduction is not None:
        values = np.ones_like(keys) / len(keys) if values is None else values

        if values.squeeze().ndim > 1:

            return np.stack([i[-1] for i in group_by_(keys=keys, values=values, reduction=reduction)])

        else:
            return np.asarray(group_by_(keys=keys, values=values, reduction=reduction))[:, -1]

    values = np.arange(len(keys)) if values is None else values

    return group_by_(keys).split_array_as_list(values)


def save_dict(file, dict):
    with open(file, "wb") as handle:
        pickle.dump(dict, handle)
    return None


def reindex_list(unsorted_list: list, indices: "list or np.ndarray"):
    return list(map(unsorted_list.__getitem__, to_numpy(indices).astype(int)))


def to_numpy(x: "int, list or array"):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (int, float, np.int64, np.int32, np.float32, np.float64)):
        return np.array([x])
    if isinstance(x, list):
        return np.asarray(x)
    if isinstance(x, (map, filter, tuple)):
        return np.asarray(list(x))


def process_ids(ids):
    types = np.array(["_".join(i.split("_")[:-1]) for i in ids])
    indices_list = group_by(types)
    indices = reindex_list(indices_list, np.argsort(np.fromiter(map(np.mean, indices_list), int)))
    return indices


def rmse(x, y):
    return np.sqrt(np.power(x.flatten() - y.flatten(), 2).mean())


def block_error(x: np.ndarray):
    """
    x : (d, N) numpy array with d features and N measurements
    """
    n = x.shape[-1]
    blocks = reblock(x)
    optimal_indices = np.asarray(find_optimal_block(n, blocks))
    isnan = np.isnan(optimal_indices)
    mode = Counter(optimal_indices[~isnan].astype(int)).most_common()[0][0]
    optimal_indices[isnan] = mode
    return np.asarray([blocks[i].std_err[j] for j, i in enumerate(optimal_indices.astype(int))])


class MaxEntropyReweight():
    def __init__(self,
                 constraints: list,
                 targets: list,
                 sigma_md: list = None,
                 sigma_reg: list = None,
                 target_kish: float = 10):

        """
        constraints : list of numpy arrays each with shape (N_observations, ).
                      Each array should be paired with a target. 
                      Optimization is performed to find a set of weights (N_observations)
                      that will result in a weighted average for each constraint that equals the corresponding target.

        targets : list of targets for each constraint. 

        sigma_md : error of each constraint data type estimated from blocking (correlated time series data)

        sigma_reg : regularization parameter for each constraint, class method optimize_sigma_reg will find these

        target_kish : minimum kish required when searching for sigma_reg for each data type.
                      Will not necessarily match the kish of the final reweighting of all constraints combined.

        """

        self.constraints = np.asarray(constraints)
        self.targets = np.asarray(targets)
        self.lambdas0 = np.zeros(len(constraints))
        self.n_samples = len(constraints[0])
        self.n_constraints = len(self.lambdas0)

        # regularizations
        self.target_kish = target_kish

        # result status
        self.has_result = False
        self.weights = None
        self.lambdas = None

        # error in comp data
        self.sigma_md = block_error(np.asarray(constraints)) if sigma_md is None else np.copy(sigma_md)

        # regularization hyperparameter (one per data type)
        self.sigma_reg = np.zeros(self.n_constraints) if sigma_reg is None else np.copy(sigma_reg)

    def compute_weights(self, lambdas, constraints: np.ndarray = None):
        constraints = self.constraints if constraints is None else constraints
        logits = 1 - np.dot(constraints.T, lambdas)
        # Normalize exponents to avoid overflow
        weights = np.exp(logits - logits.max())
        # return weights
        return weights / np.sum(weights)

    def compute_entropy(self, weights: np.ndarray = None, *args):
        if weights is None:
            assert self.weights is not None, "Must provide weights if class attribute 'weights' is None"
            weights = self.weights
        entropy = -np.sum(weights * np.log(weights + 1e-12))  # Small offset to avoid log(0)
        return entropy

    def compute_weighted_mean(self, weights: np.ndarray = None):
        if weights is None:
            assert self.weights is not None, "Must provide weights if class attribute 'weights' is None"
            weights = self.weights
        return self.constraints @ weights

    def lagrangian(self,
                   lambdas,
                   constraints: np.ndarray,
                   targets: np.ndarray,
                   regularize: bool = False,
                   sigma_reg: np.ndarray = None,
                   sigma_md: np.ndarray = None):

        logits = 1 - np.dot(constraints.T, lambdas)
        shift = logits.max()
        unnormalized_weights = np.exp(logits - shift)
        norm = unnormalized_weights.sum()
        weights = unnormalized_weights / norm

        L = np.log(norm / self.n_samples) + shift - 1 + np.dot(lambdas, targets)
        dL = targets - np.dot(constraints, weights)

        if regularize:
            L += 0.5 * np.sum(np.power(sigma_reg * lambdas, 2) + np.power(sigma_md * lambdas, 2))
            dL += np.power(sigma_reg, 2) * lambdas + np.power(sigma_md, 2) * lambdas

        return L, dL

    def reweight(self,
                 regularize: bool = False,
                 sigma_reg: list = None,
                 data_indices: list = None,
                 store_result: bool = False
                 ):

        args = []

        if data_indices is not None:
            assert isinstance(data_indices, (np.ndarray, list)), "data_indices must be type np.ndarray or list"
            data_indices = np.asarray(data_indices) if isinstance(data_indices, list) else data_indices
            constraints, targets, lambdas0 = [getattr(self, i)[data_indices] for i in
                                              ["constraints", "targets", "lambdas0"]]

        else:
            constraints, targets, lambdas0 = self.constraints, self.targets, self.lambdas0

        args.extend([constraints, targets])

        if regularize:
            assert sigma_reg is not None or self.sigma_reg is not None, (
                "Must provide sigma_reg (regularization parameter)"
                "as an argument or upon instantiation")
            args.extend([regularize,
                         np.asarray(sigma_reg) if sigma_reg is not None else self.sigma_reg[data_indices].squeeze(),
                         self.sigma_md[data_indices].squeeze()])

        else:
            args.extend([False, None, None])  # not necessary

        result = minimize(
            self.lagrangian,
            lambdas0,
            method='L-BFGS-B',
            jac=True,
            args=tuple(args)
        )

        weights = self.compute_weights(result.x, constraints)

        if store_result:
            if data_indices is not None:
                warnings.warn("Storing parameters and weights from reweighting performed on a subset of the data.")
            self.lambdas = result.x
            self.weights = weights
            self.has_result = True

        weighted_averages = constraints @ weights

        return dict(lambdas=result.x,
                    weights=weights,
                    kish=self.compute_kish(weights),
                    regularize=args[-2],
                    sigma_reg=args[-1],
                    data_indices=data_indices,
                    weighted_averages=weighted_averages,
                    targets=targets,
                    rmse=rmse(weighted_averages, targets)
                    )

    def reset(self):
        self.weights = None
        self.lambdas = None
        self.has_result = False
        return

    def compute_kish(self, weights: np.ndarray = None):
        if weights is None:
            assert self.weights is not None, "Must provide weights if class attribute 'weights' is None"
            weights = self.weights
        return 100 / (self.n_samples * np.power(weights, 2).sum())

    def kish_scan(self,
                  data_indices: list = None,
                  target_kish: float = None,
                  sigma_reg_l: float = 0.001,
                  sigma_reg_u: float = 20,
                  steps: int = 200,
                  scale: np.array = 1,
                  store_sigma: bool = False):

        if data_indices is not None:

            assert isinstance(data_indices, (np.ndarray, list)), "data_indices must be type np.ndarray or list"
            data_indices = np.asarray(data_indices) if isinstance(data_indices, list) else data_indices
        else:
            data_indices = np.arange(self.n_constraints)

        if target_kish is not None:
            self.target_kish = target_kish

        kish = lambda sigma: self.reweight(regularize=True,
                                           sigma_reg=sigma,
                                           data_indices=data_indices,
                                           store_result=False)["kish"]
        reached_target = False
        sigma_optimal = sigma_reg_u * scale
        for sigma in np.linspace(sigma_reg_l, sigma_reg_u, steps)[::-1]:

            sigma = scale * sigma

            if kish(sigma) < self.target_kish:
                reached_target = True
                break
                
            sigma_optimal = sigma

        if not reached_target: print("Did not find optimal kish")
        if store_sigma: self.sigma_reg[data_indices] = sigma_optimal

        return sigma_optimal

    def optimize_sigma_reg(self,
                           indices_list: list,
                           single_sigma_reg_l: float = 0.001,
                           single_sigma_reg_u: float = 20,
                           single_steps: int = 200,
                           global_sigma_reg_l: float = 0.01,
                           global_sigma_reg_u: float = 20,
                           global_steps: int = 60,
                           ):

        single_regs = np.concatenate([self.kish_scan(i,
                                                     sigma_reg_l=single_sigma_reg_l,
                                                     sigma_reg_u=single_sigma_reg_u,
                                                     steps=single_steps) * np.ones(len(i))
                                      for i in indices_list])

        self.kish_scan(scale=single_regs,
                       store_sigma=True,
                       sigma_reg_l=global_sigma_reg_l,
                       sigma_reg_u=global_sigma_reg_u,
                       steps=global_steps,
                       )


if __name__ == "__main__":

    parser = ArgumentParser(description="Maximum Entropy Reweighting")

    parser.add_argument("--comp_data", required=True, type=str,
                        help="path to a .npy file containing an D (observables) by N (samples) array"
                        )

    parser.add_argument("--exp_data", required=True, type=str,
                        help=("path to a .npy file containing a 1-Dimensional array of length D (observables),"
                              "experimental average for each observable")
                        )

    parser.add_argument("--data_ids", required=True, type=str,
                        help=("path to a .npy file containing a 1-Dimensional string array of length D (observables),"
                              "name of each observable underscore index, e.g. 'CA_#', so observables of the same type can be grouped")
                        )

    parser.add_argument("--sigma_md", required=False, type=str, default=None,
                        help=("path to a .npy file containing a 1-Dimensional array of length D (observables),"
                              "block error for each experimental observable")
                        )

    parser.add_argument("--sigma_reg", required=False, type=str, default=None,
                        help=("path to a .npy file containing a 1-Dimensional array of length D (observables),"
                              "regularization parameter for each experimental observable")
                        )

    parser.add_argument("--target_kish", required=False, type=float, default=10,
                        help=("Target kish score to use in regularization optimization")
                        )

    parser.add_argument("--result_path", required=False, type=str, default=None,
                        help="directory to save pickled dictionary of reweighting results to"
                        )
    parser.add_argument("--description", required=False, type=str, default=None,
                        help="Description to add to reweighting results file name"
                        )

    args = parser.parse_args()

    dic = vars(args)

    array_files = ["comp_data", "exp_data", "data_ids", "sigma_md", "sigma_reg"]

    constraints, targets, data_ids, sigma_md, sigma_reg = [np.load(dic[i]) if dic[i] is not None else None for i in array_files]

    data_type_indices = process_ids(data_ids)

    reweight = MaxEntropyReweight(constraints,
                                  targets.flatten(),
                                  sigma_md=sigma_md,
                                  sigma_reg=sigma_reg,
                                  target_kish=args.target_kish)

    if sigma_reg is None:
        reweight.optimize_sigma_reg(data_type_indices)

    result = reweight.reweight(regularize=True)

    path = os.getcwd() if args.result_path is None else args.result_path

    file = f"reweighting_result_dict_{args.description}" if args.description is not None else "reweighting_result_dict"

    save_dict(f"{path}/{file}", result)
