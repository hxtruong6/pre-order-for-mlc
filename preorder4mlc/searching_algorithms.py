"""ILP search for bipartite ordered preference orders (BOPOs).

:class:`Search_BOPreOs` and :class:`Search_BOParOs` solve, per test
instance, the integer linear program that turns pairwise probabilistic
predictions into either a pre-order or a partial-order plus its derived
binary vector and partial-abstention prediction. The two classes expose
``PRE_ORDER`` / ``PARTIAL_ORDER`` methods, each supporting four
``(target_metric, height)`` configurations corresponding to the eight
inference algorithms IA1-IA8 reported in the paper.
"""

import os

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from numpy import array

from preorder4mlc.constants import TargetMetric
from preorder4mlc.solvers import solve_milp
from preorder4mlc.utils.suppress import suppress_output

# Parallelism for per-instance ILP solves. Loky backend (process-based) is
# safe because cvxopt/GLPK keeps solver state per process. Default n_jobs=1
# preserves the previous sequential behavior; set PREORDER_SEARCH_N_JOBS=-1
# (or any joblib-compatible value) to parallelize across test instances.
# Parallel pays a per-task pickle/IPC cost (~ms) so it's only a win when
# n_instances and the per-solve cost are both non-trivial.
_SEARCH_N_JOBS = int(os.environ.get("PREORDER_SEARCH_N_JOBS", "1"))


class Search_BOPreOs:
    """
    Search for the best preference order using the BOPreOs model.
    """

    # For ilp with cvxopt.glpk, see https://gist.github.com/nipunbatra/7059160?fbclid=IwY2xjawEgKndleHRuA2FlbQIxMAABHfelujAwksDZjWb9Pn-Nfakv52P-ltg295HO8m0F_XnTjbp9_TvAlZxcjA_aem_ZA3BrIWlMvWu2HlK8bKKaA

    def __init__(
        self,
        pairwise_probabilistic_predictions,
        n_labels,
        n_instances,
        target_metric: TargetMetric,
        height,
    ):
        self.pairwise_probabilistic_predictions = pairwise_probabilistic_predictions
        self.n_labels = n_labels
        self.n_instances = n_instances
        self.height = height
        self.target_metric = target_metric

    # PRE-ORDER
    # subset - height = 2
    # subset - height = None
    # hamming - height = 2
    # hamming - height = None

    # PARTIAL-ORDER
    # subset - height = 2
    # subset - height = None
    # hamming - height = 2
    # hamming - height = None

    def PRE_ORDER(self):
        indices_vector = {}
        indVec = 0
        for i in range(self.n_labels - 1):
            for j in range(i + 1, self.n_labels):
                for l in range(4):
                    indices_vector[f"{i}_{j}_{l}"] = indVec
                    indVec += 1
        G, h, A, b, I, B = self._encode_parameters_PRE_ORDER(indices_vector)  # type: ignore
        predicted_Y = []
        predicted_preorders = []
        prediction_with_partial_abstentions = []
        # Precompute upper-triangular pair indices once. np.triu_indices(K, k=1)
        # yields (i, j) pairs in row-major order matching the prior
        # `for i in range(K-1): for j in range(i+1, K)` loop. Flattening
        # `pred[ii, jj, n, :]` then walks (i, j, l) in the same sequence as
        # the old append-per-l loop, so `vector` ordering — and hence the
        # downstream `indices_vector` mapping — is unchanged.
        ii, jj = np.triu_indices(self.n_labels, k=1)
        # Per-instance ILP solves are independent — wrap in joblib.Parallel.
        # With _SEARCH_N_JOBS=1 (default) this is a thin wrapper that runs
        # in-process, equivalent to the prior for-loop. With higher n_jobs,
        # loky forks worker processes; cvxopt/GLPK is process-safe.
        # batch_size=1: per-instance solve times are wildly uneven (sub-second to
        # tens of seconds on enron K=53). One task per dispatch gives proper
        # work-stealing instead of joblib's 'auto' lumping; the ~ms IPC cost is
        # negligible vs multi-second solves. NB: on enron this only modestly helps
        # (~4->4.5x effective) — the ceiling is the workload (few instances are
        # slow at once), not batching or CPU count (cores are NOT cgroup-confined;
        # the node grants all SLURM_CPUS_ON_NODE). Scheduling-only — results identical.
        results = Parallel(n_jobs=_SEARCH_N_JOBS, batch_size=1)(
            delayed(self._solve_one_PRE_ORDER)(
                n, ii, jj, indices_vector, G, h, A, b, I, B
            )
            for n in range(self.n_instances)
        )
        predicted_Y = [r[0] for r in results]
        predicted_preorders = [r[1] for r in results]
        prediction_with_partial_abstentions = [r[2] for r in results]
        return (
            predicted_Y,
            predicted_preorders,
            indices_vector,
            prediction_with_partial_abstentions,
        )

    def _solve_one_PRE_ORDER(self, n, ii, jj, indices_vector, G, h, A, b, I, B):
        """Compute the cost vector for instance n and run the ILP reasoning.

        Extracted to a method so joblib.Parallel can dispatch it per-instance.
        G/h/A/b/I/B are loop-invariant inputs (built once outside the loop in
        PRE_ORDER); indices_vector / ii / jj are precomputed mappings.
        """
        pair_slice = self.pairwise_probabilistic_predictions[ii, jj, n, :]
        if self.target_metric == TargetMetric.Hamming:
            vector = (-pair_slice).flatten()
        elif self.target_metric == TargetMetric.Subset:
            vector = (-np.log(np.clip(pair_slice, 1e-12, 1.0))).flatten()
        else:
            raise ValueError(f"Unknown target metric: {self.target_metric}")
        return self._reasoning_procedure_PRE_ORDER(
            vector, indices_vector, self.n_labels, G, h, A, b, I, B
        )

    def _encode_parameters_PRE_ORDER(self, indices_vector):
        assert self.n_labels is not None
        n = self.n_labels
        n_vars = n * (n - 1) * 2
        n_ineq = n * (n - 1) * (n - 2)
        n_eq = n * (n - 1) // 2
        h = np.ones((n_ineq, 1))
        b = np.ones((n_eq, 1))
        I = set()
        B = set(range(n_vars))

        A_rows, A_cols, A_vals = [], [], []
        rowA = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                for l in range(4):
                    A_rows.append(rowA)
                    A_cols.append(indices_vector[f"{i}_{j}_{l}"])
                    A_vals.append(1.0)
                rowA += 1
        A = sp.csr_matrix(
            (A_vals, (A_rows, A_cols)), shape=(n_eq, n_vars), dtype=np.float64
        )

        G_rows, G_cols, G_vals = [], [], []

        if not self.height:
            rowG = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    for k in range(i):
                        keys1 = [f"{i}_{j}_0", f"{i}_{j}_3", f"{k}_{i}_1", f"{k}_{i}_3", f"{k}_{j}_0", f"{k}_{j}_3"]
                        vals1 = [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
                        for key, v in zip(keys1, vals1):
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(v)
                        rowG += 1
                        keys2 = [f"{i}_{j}_1", f"{i}_{j}_3", f"{k}_{i}_0", f"{k}_{i}_3", f"{k}_{j}_1", f"{k}_{j}_3"]
                        vals2 = [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
                        for key, v in zip(keys2, vals2):
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(v)
                        rowG += 1
                    for k in range(i + 1, j):
                        keys1 = [f"{i}_{j}_0", f"{i}_{j}_3", f"{i}_{k}_0", f"{i}_{k}_3", f"{k}_{j}_0", f"{k}_{j}_3"]
                        vals1 = [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
                        for key, v in zip(keys1, vals1):
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(v)
                        rowG += 1
                        keys2 = [f"{i}_{j}_1", f"{i}_{j}_3", f"{i}_{k}_1", f"{i}_{k}_3", f"{k}_{j}_1", f"{k}_{j}_3"]
                        vals2 = [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
                        for key, v in zip(keys2, vals2):
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(v)
                        rowG += 1
                    for k in range(j + 1, n):
                        keys1 = [f"{i}_{j}_0", f"{i}_{j}_3", f"{i}_{k}_0", f"{i}_{k}_3", f"{j}_{k}_1", f"{j}_{k}_3"]
                        vals1 = [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
                        for key, v in zip(keys1, vals1):
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(v)
                        rowG += 1
                        keys2 = [f"{i}_{j}_1", f"{i}_{j}_3", f"{i}_{k}_1", f"{i}_{k}_3", f"{j}_{k}_0", f"{j}_{k}_3"]
                        vals2 = [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
                        for key, v in zip(keys2, vals2):
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(v)
                        rowG += 1
            G = sp.csr_matrix(
                (G_vals, (G_rows, G_cols)), shape=(n_ineq, n_vars), dtype=np.float64
            )
            return G, h, A, b, I, B

        elif self.height == 2:
            rowG = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    for k in range(i):
                        for key in [f"{k}_{i}_1", f"{k}_{i}_3", f"{k}_{j}_0", f"{k}_{j}_3"]:
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(1.0)
                        rowG += 1
                        for key in [f"{k}_{i}_0", f"{k}_{i}_3", f"{k}_{j}_1", f"{k}_{j}_3"]:
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(1.0)
                        rowG += 1
                    for k in range(i + 1, j):
                        for key in [f"{i}_{k}_0", f"{i}_{k}_3", f"{k}_{j}_0", f"{k}_{j}_3"]:
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(1.0)
                        rowG += 1
                        for key in [f"{i}_{k}_1", f"{i}_{k}_3", f"{k}_{j}_1", f"{k}_{j}_3"]:
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(1.0)
                        rowG += 1
                    for k in range(j + 1, n):
                        for key in [f"{i}_{k}_0", f"{i}_{k}_3", f"{j}_{k}_1", f"{j}_{k}_3"]:
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(1.0)
                        rowG += 1
                        for key in [f"{i}_{k}_1", f"{i}_{k}_3", f"{j}_{k}_0", f"{j}_{k}_3"]:
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(1.0)
                        rowG += 1
            G = sp.csr_matrix(
                (G_vals, (G_rows, G_cols)), shape=(n_ineq, n_vars), dtype=np.float64
            )
            return G, h, A, b, I, B
        else:
            raise ValueError("The height is not supported")

    def _reasoning_procedure_PRE_ORDER(self, vector, indices_vector, n_labels, G, h, A, b, I, B):
        c = np.zeros((n_labels * (n_labels - 1) * 2, 1))

        for ind in range(len(vector)):
            c[ind, 0] = vector[ind]

        with suppress_output():
            _, x = solve_milp(c, G, h, A, b, I, B)

        optX = array(x)

        scores_d = [0 for x in range(n_labels)]  # label i-th dominates at least one label
        scores_n = [0 for x in range(n_labels)]  # no label dominates label i-th
        for i in range(n_labels):
            for k in range(0, i):
                scores_d[i] += optX[indices_vector[f"{k}_{i}_{1}"], 0]
                scores_n[i] += optX[indices_vector[f"{k}_{i}_{0}"], 0]
            for j in range(i + 1, n_labels):
                scores_d[i] += optX[indices_vector[f"{i}_{j}_{0}"], 0]
                scores_n[i] += optX[indices_vector[f"{i}_{j}_{1}"], 0]
        hard_prediction = [0 for x in range(n_labels)]
        for ind in range(n_labels):
            if scores_d[ind] > 0 or scores_n[ind] == 0:
                hard_prediction[ind] = 1

        prediction_with_partial_abstention = [0 for x in range(n_labels)]
        for ind in range(n_labels):
            if scores_d[ind] > 0:
                prediction_with_partial_abstention[ind] = 1
            elif scores_n[ind] == 0 and scores_d[ind] == 0:
                prediction_with_partial_abstention[ind] = -1
        # optX: [n*(n-1)*2, 1] -> [1]
        #  [[0.1], [0.2],] => [0.1, 0.2]
        predicted_partial_order = optX.flatten()
        return hard_prediction, predicted_partial_order, prediction_with_partial_abstention


class Search_BOParOs:
    def __init__(
        self,
        pairwise_probabilistic_predictions,
        n_labels,
        n_instances,
        target_metric,
        height,
    ):
        self.pairwise_probabilistic_predictions = pairwise_probabilistic_predictions
        self.n_labels = n_labels
        self.n_instances = n_instances
        self.height = height
        self.target_metric = target_metric

    def PARTIAL_ORDER(self):
        indices_vector = {}
        indVec = 0
        assert self.n_labels is not None

        for i in range(self.n_labels - 1):
            for j in range(i + 1, self.n_labels):
                for l in range(3):
                    indices_vector[f"{i}_{j}_{l}"] = indVec
                    indVec += 1
        G, h, A, b, I, B = self._encode_parameters_PARTIAL_ORDER(indices_vector)  # type: ignore
        predicted_Y = []
        predicted_partial_orders = []
        prediction_with_partial_abstentions = []
        # See PRE_ORDER above for the triu_indices vectorisation rationale.
        ii, jj = np.triu_indices(self.n_labels, k=1)
        # Per-instance ILP solves wrapped in joblib.Parallel — see PRE_ORDER above.
        # batch_size=1 for the same load-balancing reason (scheduling-only).
        results = Parallel(n_jobs=_SEARCH_N_JOBS, batch_size=1)(
            delayed(self._solve_one_PARTIAL_ORDER)(
                n, ii, jj, indices_vector, G, h, A, b, I, B
            )
            for n in range(self.n_instances)
        )
        predicted_Y = [r[0] for r in results]
        predicted_partial_orders = [r[1] for r in results]
        prediction_with_partial_abstentions = [r[2] for r in results]
        return (
            predicted_Y,
            predicted_partial_orders,
            indices_vector,
            prediction_with_partial_abstentions,
        )

    def _solve_one_PARTIAL_ORDER(self, n, ii, jj, indices_vector, G, h, A, b, I, B):
        """Per-instance helper for PARTIAL_ORDER joblib dispatch.

        See _solve_one_PRE_ORDER for the same pattern in the sibling class.
        """
        pair_slice = self.pairwise_probabilistic_predictions[ii, jj, n, :]
        if self.target_metric == TargetMetric.Hamming:
            vector = (-pair_slice).flatten()
        elif self.target_metric == TargetMetric.Subset:
            vector = (-np.log(np.clip(pair_slice, 1e-12, 1.0))).flatten()
        else:
            raise ValueError(f"Unknown target metric: {self.target_metric}")
        return self._reasoning_procedure_PARTIAL_ORDER(
            vector, indices_vector, G, h, A, b, I, B
        )

    def _encode_parameters_PARTIAL_ORDER(self, indices_vector):
        assert self.n_labels is not None
        n = self.n_labels
        n_vars_p = int(n * (n - 1) * 1.5)
        n_ineq = n * (n - 1) * (n - 2)
        n_eq = n * (n - 1) // 2
        h = np.ones((n_ineq, 1))
        b = np.ones((n_eq, 1))
        I = set()
        B = set(range(n_vars_p))

        A_rows, A_cols, A_vals = [], [], []
        rowA = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                for l in range(3):
                    A_rows.append(rowA)
                    A_cols.append(indices_vector[f"{i}_{j}_{l}"])
                    A_vals.append(1.0)
                rowA += 1
        A = sp.csr_matrix(
            (A_vals, (A_rows, A_cols)), shape=(n_eq, n_vars_p), dtype=np.float64
        )

        G_rows, G_cols, G_vals = [], [], []

        if not self.height:
            rowG = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    for k in range(i):
                        for key, v in zip(
                            [f"{i}_{j}_0", f"{k}_{i}_1", f"{k}_{j}_0"],
                            [-1.0, 1.0, 1.0],
                        ):
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(v)
                        rowG += 1
                        for key, v in zip(
                            [f"{i}_{j}_1", f"{k}_{i}_0", f"{k}_{j}_1"],
                            [-1.0, 1.0, 1.0],
                        ):
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(v)
                        rowG += 1
                    for k in range(i + 1, j):
                        for key, v in zip(
                            [f"{i}_{j}_0", f"{i}_{k}_0", f"{k}_{j}_0"],
                            [-1.0, 1.0, 1.0],
                        ):
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(v)
                        rowG += 1
                        for key, v in zip(
                            [f"{i}_{j}_1", f"{i}_{k}_1", f"{k}_{j}_1"],
                            [-1.0, 1.0, 1.0],
                        ):
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(v)
                        rowG += 1
                    for k in range(j + 1, n):
                        for key, v in zip(
                            [f"{i}_{j}_0", f"{i}_{k}_0", f"{j}_{k}_1"],
                            [-1.0, 1.0, 1.0],
                        ):
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(v)
                        rowG += 1
                        for key, v in zip(
                            [f"{i}_{j}_1", f"{i}_{k}_1", f"{j}_{k}_0"],
                            [-1.0, 1.0, 1.0],
                        ):
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(v)
                        rowG += 1
            G = sp.csr_matrix(
                (G_vals, (G_rows, G_cols)), shape=(n_ineq, n_vars_p), dtype=np.float64
            )
            return G, h, A, b, I, B

        elif self.height == 2:
            rowG = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    for k in range(i):
                        for key in [f"{k}_{i}_1", f"{k}_{j}_0"]:
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(1.0)
                        rowG += 1
                        for key in [f"{k}_{i}_0", f"{k}_{j}_1"]:
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(1.0)
                        rowG += 1
                    for k in range(i + 1, j):
                        for key in [f"{i}_{k}_0", f"{k}_{j}_0"]:
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(1.0)
                        rowG += 1
                        for key in [f"{i}_{k}_1", f"{k}_{j}_1"]:
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(1.0)
                        rowG += 1
                    for k in range(j + 1, n):
                        for key in [f"{i}_{k}_0", f"{j}_{k}_1"]:
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(1.0)
                        rowG += 1
                        for key in [f"{i}_{k}_1", f"{j}_{k}_0"]:
                            G_rows.append(rowG)
                            G_cols.append(indices_vector[key])
                            G_vals.append(1.0)
                        rowG += 1
            G = sp.csr_matrix(
                (G_vals, (G_rows, G_cols)), shape=(n_ineq, n_vars_p), dtype=np.float64
            )
            return G, h, A, b, I, B
        else:
            raise ValueError("The height is not supported")

    def _reasoning_procedure_PARTIAL_ORDER(self, vector, indices_vector, G, h, A, b, I, B):
        c = np.zeros((self.n_labels * (self.n_labels - 1) * 2, 1))
        for ind in range(len(vector)):
            c[ind, 0] = vector[ind]
        with suppress_output():
            _, x = solve_milp(c, G, h, A, b, I, B)
        optX = array(x)

        # Let both partial and preorder make the hard predictions in similar ways ...
        scores_d = [0 for x in range(self.n_labels)]  # label i-th dominates at least one label
        scores_n = [0 for x in range(self.n_labels)]  # no label dominates label i-th
        for i in range(self.n_labels):
            for k in range(0, i):
                scores_d[i] += optX[indices_vector[f"{k}_{i}_{1}"], 0]
                scores_n[i] += optX[indices_vector[f"{k}_{i}_{0}"], 0]
            for j in range(i + 1, self.n_labels):
                scores_d[i] += optX[indices_vector[f"{i}_{j}_{0}"], 0]
                scores_n[i] += optX[indices_vector[f"{i}_{j}_{1}"], 0]
        hard_prediction = [0 for x in range(self.n_labels)]
        for ind in range(self.n_labels):
            if scores_d[ind] > 0 or scores_n[ind] == 0:
                hard_prediction[ind] = 1

        prediction_with_partial_abstention = [0 for x in range(self.n_labels)]
        for ind in range(self.n_labels):
            if scores_d[ind] > 0:
                prediction_with_partial_abstention[ind] = 1
            elif scores_n[ind] == 0 and scores_d[ind] == 0:
                prediction_with_partial_abstention[ind] = -1
        # optX: [n*(n-1)*2, 1] -> [1]
        #  [[0.1], [0.2],] => [0.1, 0.2]
        predicted_partial_order = optX.flatten()
        return hard_prediction, predicted_partial_order, prediction_with_partial_abstention
