#!/usr/bin/env python3
"""Generate gene regulatory network and time series expression data.

"""
import sys
import numpy as np
from scipy.integrate import odeint
from scipy.stats import norm
import pandas as pd
import argparse
import os


def main(argv):
    num_times = 6
    max_in_deg = 3
    rand_seed = None
    parser = argparse.ArgumentParser()
    parser.add_argument("adjmat", help="adjacency matrix")
    parser.add_argument("exp", help="expression level file")
    parser.add_argument("-c", "--create",
                        help="create new adjacency matrix",
                        action="store_true")
    parser.add_argument("-d", "--design",
                        help="path to save design file to",
                        default="")
    parser.add_argument(
        "--num-core-genes",
        help="number of core genes (only with -c)", type=int,
        default=5
        )
    parser.add_argument(
        "--num-genes",
        help="total number of genes (only with -c)",
        type=int, default=20
        )
    parser.add_argument("-s", "--snr",
                        help="signal to noise ratio",
                        type=float, default=1)
    parser.add_argument(
        "--gamma",
        help="fraction of power of condition dependent noise",
        type=float, default=0
        )
    parser.add_argument("-m", "--method", choices=['phi', 'glm'],
                        help="data generation method", default='phi')
    parser.add_argument("--num-cond", help="number of conditions",
                        type=int, default=10)
    parser.add_argument("--num-rep", help="number of replicates",
                        type=int, default=3)
    args = parser.parse_args(argv)
    # The regulation coefficients have variance one regardless
    # of the margin.  So the SNR is [CITATION NEEDED]
    # num_times*max_in_degree/36/sigma**2.
    sigma = np.sqrt(max_in_deg*num_times/36/args.snr)
    sigma_c = np.sqrt(args.gamma)*sigma
    sigma_b = np.sqrt(1-args.gamma)*sigma
    adj_mat_file = args.adjmat
    if args.create:
        # Generate a random adjacency matrix file.
        margin = 0.5
        adj_mat = gen_adj_mat(args.num_core_genes, max_in_deg, margin)
        np.savetxt(adj_mat_file, adj_mat)
    gen_planted_edge_data(
        args.num_genes, adj_mat_file, sigma_c, sigma_b,
        args.num_cond, args.exp, args.design, args.num_rep,
        num_times, rand_seed, method=args.method
        )
    return


def gen_planted_edge_data(
        num_genes, adj_mat, sigma_c, sigma_b, num_experiments,
        csv_exp_file, csv_design_file, num_replicates, num_times,
        rand_seed, true_time=True, method='phi', noise=0.0
        ):
    """Generate time series expression data.

    Args:
        num_genes: Number of genes.  Must be at least as large
            as the number of rows (or columns) of the adjacency
            matrix.  If num_genes is strictly larger than the
            adjacency matrix, isolated genes are padded.
        adj_mat: Adjacency matrix.
            This variable can be either the path to the adjacency
            matrix file as a str, or the matrix itself as a numpy
            array.  The sign of the (i, j)th element for i not equal
            to j indicates the type of regulation of j by i, and the
            absolute value the strength of the regulation.  The
            diagonal elements can be either self-regulation or
            negative of the degradation rate.
        sigma_c: Condition-dependent variation level.
            If scalar, it applies to all genes.  Otherwise the
            size must be equal to gene number and each element
            applies to a single gene.
        sigma_b: Condition-independent variation level.
            If scalar, it applies to all genes.  Otherwise the
            size must be equal to gene number and each element
            applies to a single gene.
        num_experiments: Number of experiments.
        csv_exp_file: Path to output expression file.
            Return the DataFrame if csv_exp_file is an empty string.
        csv_design_file: Path to output design file.
            If an empty string, no design file is written.
            If file exists, do not overwrite it.
        num_replicates: Number of replicates.
        num_times: Number of sample times.
        rand_seed: Seed for random number generation.  None for the
            default clock seed (see
            https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState).
        true_time: Indicator of using different individual for
            each sample (i.e., one-shot sampling).
        method: Dynamics.  Can be 'phi' for Phi-net dynamics or 'glm'
            for Gaussian linear model dynamics.  Default is Phi-net.
        noise: Observation noise level.

    Returns:
        Write an expression file (csv_exp_file) and a
        design file (csv_design_file, if given) in CSV format.

    """
    # Load adjacency matrix.
    if isinstance(adj_mat, str):
        adj_mat = np.loadtxt(adj_mat, delimiter=' ')
    # Check size of adjacency matrix.
    num_genes_in_adj_mat = adj_mat.shape[0]
    if num_genes < num_genes_in_adj_mat:
        print('The specified number of genes is smaller than the '
              'size of the adjacency matrix.')
        return 1
    if num_genes > num_genes_in_adj_mat:
        adj_mat_big = np.zeros((num_genes, num_genes))
        adj_mat_big[
            :num_genes_in_adj_mat, :num_genes_in_adj_mat
            ] = adj_mat
        adj_mat = adj_mat_big
    np.random.seed(rand_seed)
    expressions = []
    for i in range(num_experiments):
        # Generate the condition-dependent standard variation.
        noise_c = np.random.randn(num_genes, num_times)
        if true_time:
            x = np.empty((num_genes, 0, num_replicates))
            for t in range(1, num_times+1):
                # Generate new independent trajectory up to time t.
                x_rep = gen_traj(num_replicates, t, adj_mat, sigma_b,
                                 sigma_c, noise_c, method)
                x_sample = x_rep[:, -1, :]
                x_sample_3d = x_sample[:, np.newaxis, :]
                x = np.concatenate((x, x_sample_3d), axis=1)
        else:
            x = gen_traj(num_replicates, num_times, adj_mat, sigma_b,
                         sigma_c, noise_c, method)
        expressions.append(x)
    # Output expression file.
    sample_ids = ['Sample'+str(i) for i in
                  range(num_replicates*num_experiments*num_times)]
    genes = ['Gene'+str(i) for i in range(num_genes)]
    flattened_exp = np.empty((num_genes, 0))
    for i in range(num_experiments):
        flattened_exp = np.concatenate((
            flattened_exp, expressions[i].reshape(
                num_genes, num_times*num_replicates
                )
            ), axis=1)
    df = pd.DataFrame(data=flattened_exp, columns=sample_ids,
                      index=genes)
    df = df+np.random.randn(*df.shape)*noise
    # write design file if it is not empty str and it does not exist
    # already.
    if csv_design_file and not os.path.isfile(csv_design_file):
        with open(csv_design_file, 'w') as f:
            idx_sample = 0
            for i in range(num_experiments):
                for j in range(num_times):
                    for k in range(num_replicates):
                        # Write the sample ID, condition, and the
                        # sample time to each line.
                        f.write(
                            sample_ids[idx_sample]+','+str(i)+','
                            +str(j)+'\n'
                            )
                        idx_sample += 1
    if csv_exp_file:
        df.to_csv(csv_exp_file)
        return 0
    else:
        return df


def phi_input(x_t_minus_1, adj_mat, sigma_b, sigma_c, noise_c_st,
              method='phi'):
    """Gene regulatory input function.

    Args:
        x_t_minus_1: An n-by-r array of expression levels at
            time t-1, where n is the number of genes and r is
            the number of replicates.
        adj_mat: An n-by-n array of the adjacency matrix.
        sigma_b: Condition-independent noise level.
            If scalar, it applies to all genes.  Otherwise the
            size must be equal to gene number and each element
            applies to a single gene.
        sigma_c: Condition-dependent noise level.
            If scalar, it applies to all genes.  Otherwise the
            size must be equal to gene number and each element
            applies to a single gene.
        noise_c_st: Standard condition-dependent noise.
            An n-dim array.
        method: Dynamic model.  Can be 'phi' or 'glm'.

    Returns:
        An n-by-r array of expression levels at time t.
    """
    num_genes, num_replicates = x_t_minus_1.shape
    # Discrete AWGN with noise level or noise level array sigma_b.
    # For array sigma_b, the following broadcasting happens:
    #     standard_noise: R x n
    #     sigma_b:            n
    #     product:        R x n
    # See https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
    noise = (np.random.normal(size=(num_replicates, num_genes))
             * np.asarray(sigma_b))
    noise_c = (noise_c_st.reshape(1, num_genes)
               * np.asarray(sigma_c))
    if method == 'phi':
        # Influence of the regulating genes with mean subtracted.
        influence = (x_t_minus_1.T-0.5).dot(adj_mat)
        # Standard deviations of the sum of influence and noise.
        sd_lin_expressions = np.sqrt(
            np.diag(adj_mat.T.dot(adj_mat))/12 + sigma_b**2
            + sigma_c**2
            )
        # Standardization of the linear expressions is done via
        # broadcasting.
        standardized_lin_expressions = (
            (influence+noise+noise_c) / sd_lin_expressions
            )
        # Map the linear expressions back to [0, 1] by the CDF of
        # standard Gaussian (a.k.a. the Phi function).
        x_t = norm.cdf(standardized_lin_expressions).T
    else:
        x_t = adj_mat.T.dot(x_t_minus_1)+noise.T+noise_c.T
    return x_t


def gen_traj(num_replicates, num_times, adj_mat, sigma_b,
             sigma_c, noise_c, method='phi'):
    """Generate a new conditionally independent trajectory from time 0 up
    to time T.

    The same condition-dependent noise is shared across the replicates.

    Args:
        num_replicates: Number of replicates.
        num_times: Number of sample times.
        adj_mat: An n-by-n array of the adjacency matrix.
        sigma_b: Condition-independent noise level.
            If scalar, it applies to all genes.  Otherwise the
            size must be equal to gene number and each element
            applies to a single gene.
        sigma_c: Condition-dependent noise level.
            If scalar, it applies to all genes.  Otherwise the
            size must be equal to gene number and each element
            applies to a single gene.
        noise_c: Standard condition-dependent noise.  An n-by-T array.
        method: Dynamic model.  Can be 'phi' or 'glm'.

    Returns:
        An n-by-T-by-r array of the expression level trajectory
        for n genes and r replicates.

    """
    num_genes = adj_mat.shape[0]
    if method == 'phi':
        # Generate constant 1/2 expression levels for all genes at
        # time 0 for the Phi-net model.
        x = np.ones((num_genes, 1, num_replicates))/2
    else:
        # Generate constant 0 expression levels for all genes at
        # time 0 for the Gaussian linear model.
        x = np.zeros((num_genes, 1, num_replicates))
    for t in range(1, num_times+1):
        x_new = phi_input(x[:, t-1, :], adj_mat, sigma_b, sigma_c,
                          noise_c[:, t-1], method)
        x = np.concatenate((x, x_new[:, np.newaxis, :]), axis=1)
    return x[:, 1:, :]


def gen_adj_mat(num_genes, max_in_deg, margin):
    """Generate adjacency matrix.

    Assume the in-degree is uniformly distributed over 0, 1, 2,
    ..., max_in_deg.  Regulation strength coefficients are
    Gaussian shifted away from the origin by the margin with
    variance one.  No self regulation is generated; i.e., the diagonal
    elements are zeros.

    Args:
        num_genes: The number of genes.
        max_in_deg: The maximum in-degree.
        margin: The margin of regulation strength coefficients
            from zero.
            margin must be between 0 and 1.
            The standard deviation of the Gaussian distribution
            before the shift is then determined by the margin so
            that the actual variance stays one.

    Returns:
        A 2-d array of the adjacency matrix of the generated
        network.

    """
    adj_mat = np.zeros((num_genes, num_genes))
    in_degrees = np.random.randint(max_in_deg+1, size=num_genes)
    # Standard deviation of the unshifted Gaussians.
    sd = np.sqrt(1-(1-2/np.pi)*margin**2)-np.sqrt(2/np.pi)*margin
    for i in range(num_genes):
        other_genes = [x for x in range(num_genes) if x != i]
        regulators = np.random.choice(
            other_genes, size=in_degrees[i], replace=False
            )
        st_gaussians = np.random.randn(in_degrees[i])
        coeffs = sd*st_gaussians + margin*np.sign(st_gaussians)
        adj_mat[regulators, i] = coeffs
    return adj_mat


if __name__ == "__main__":
    main(sys.argv[1:])
