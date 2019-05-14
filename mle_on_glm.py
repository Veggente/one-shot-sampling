#!/usr/bin/env python
"""MLE for self-regulation coefficient with Gaussian linear model.

Functions:
    main: Evaluate the MLE for a group of conditions.
    script_isit_plot: Script for plotting the sign errors.
    main_gen_tables: Script for generating error tables.
    plot_errors: Plot sign errors with confidence intervals.
    gen_cov_mat: Generate covariance matrix.
    mle_single_gene: MLE for single gene reconstruction.
    single_gene_llh: Calculate adjusted negative log-likelihood
        function for single gene.
    adj_neg_llh: Calculate adjusted negative log-likelihood
        function for general jointly Guassian distribution.
    geom_sum: Partial sum of geometric series.
    mle_eval: Evaluate the MLE for GLM.
    gen_error_tables: Generate error tables.
    remove_zeros: Remove trailing zeros from float numbers.
    parse_args: Argument parser.
    test_mle_single_gene: Test mle_single_gene().
    likelihood_counter_example: A counter example for
        the 6:47 PM Jan 7, 2019 email.
    test_glrt_single_gene: Test for glrt_single_gene().
    single_gene_llh_cmpt: Compact likelihood function.
    joint_mle_4: Joint A-gamma-sigma-sigma_Z MLE.
    joint_dena_eval: Joint A-gamma-sigma-sigma_Z estimation
        evaluation.
    joint_mle: Joint A-gamma MLE with knowledge of
        variation/noise levels.
    single_gene_llh_cmpt_2: Likelihood function for joint_mle().
    eval_joint_sign_error: Evaluate sign error rate of joint
        estimation with unknown sigmas.
    scatter_hist_joint: Do scatter or histogram plots for joint
        estimation.
    script_mle_eval: Script for joint MLE evaluation.
    plot_plos_one_fig_4: Plotting PLOS ONE submission Fig. 4.
"""
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import norm
from scipy.stats import pearsonr
import argparse
import sys
import pickle
# Plotting module.
if sys.platform == 'darwin':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
elif sys.platform in ['linux', 'linux2']:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
else:
    print("No support for Windows.")
    exit(1)

from bslr import bar_plot


def main(self_reg_coeff, rep_list, one_shot, num_sims,
         num_samples, gamma_vec, noise, plot_hist=False,
         joint=False):
    """Evaluate the MLE for a group of conditions.

    Args:
        self_reg_coeff: Self regulation coefficient.
        rep_list: list
            A list of numbers of replicates.
        one_shot: Indicator for one-shot sampling.
        num_sims: Number of simulations.
        num_samples: Total budget in number of samples.
        gamma_vec: List of gamma values.
        noise: Observation noise level.
        plot_hist: bool, default False
            Plot and save the histograms.
        joint: bool, default False
            Joint estimation of A and gamma.

    Returns:
        Write a file of error rates and multiple files of
        histograms of estimated self regulation coefficient.
    """
    sigma = 1
    num_times = 6
    for num_reps in rep_list:
        num_conditions = int(num_samples/num_reps/num_times)
        errors = []
        for gamma in tqdm(gamma_vec):
            if plot_hist:
                histogram_file = (
                    'mle-hist-a{self_reg_coeff}'
                    '-b{num_reps}'
                    '-o{one_shot}'
                    '-s{num_sims}'
                    '-g{gamma}'
                    '-e{noise}-j{joint}.pdf'.format(
                        self_reg_coeff=self_reg_coeff,
                        num_reps=num_reps,
                        one_shot=one_shot,
                        num_sims=num_sims,
                        gamma=gamma,
                        noise=noise,
                        joint=joint
                        )
                    )
            else:
                histogram_file = ''
            errors.append(1-mle_eval(
                num_sims, num_conditions, num_times, num_reps,
                self_reg_coeff, sigma, gamma,
                method='nelder-mead-dual',
                histogram_file=histogram_file, one_shot=one_shot,
                noise=noise, noise_alg=noise, joint=joint,
                num_restarts=3
                ))
        data = np.asarray([gamma_vec, errors]).T
        np.savetxt('mle-error-a{self_reg_coeff}'
                   '-b{num_reps}'
                   '-s{sigma}'
                   '-o{one_shot}'
                   '-n{num_sims}'
                   '-e{noise}-j{joint}.csv'.format(
                       self_reg_coeff=self_reg_coeff,
                       num_reps=num_reps,
                       sigma=sigma,
                       one_shot=one_shot,
                       num_sims=num_sims,
                       noise=noise,
                       joint=joint
                       ), data)
    return


def script_isit_plot():
    self_reg_coeff = 0.1
    num_sims = 10000
    path = ('/Users/veggente/Documents/workspace/python/'
            'mle_glm_single_gene/budget-180/isit/final')
    rep_vec = [1, 2, 3, 5, 6, 10, 15, 30]
    noise_level = 0.3
    one_shot = True
    line_plot = False
    figsize = (5, 2)
    ylim = [-0.02, 0.6]
    loc = 'hidden'
    plot_errors(self_reg_coeff, num_sims, path, path+'/glm', rep_vec,
                noise_level, one_shot, line_plot=line_plot,
                figsize=figsize, ylim=ylim, loc=loc)
    return


def main_gen_tables():
    self_reg_coeff_vec = [0.1]
    gamma_vec = np.linspace(0, 1, 6)
    b_vec = [1]
    path = ('/Users/veggente/Documents/workspace/python/'
            'mle_glm_single_gene/temp/')
    gen_error_tables(self_reg_coeff_vec, b_vec, path,
                     latex=True)
    return


def plot_errors(self_reg_coeff, num_sims, error_file_prefix,
                output_prefix, num_rep_vec, noise, one_shot,
                line_plot=False, figsize=(18, 10),
                display=False, ylim=None, loc=''):
    """Plot errors with confidence intervals.

    Args:
        self_reg_coeff: Self regulation coefficient.
        num_sims: Number of simulations for each setting.
        error_file_prefix: Prefix of error files (no dash at
            the end).
        output_prefix: Output prefix.
        num_rep_vec: Number of replicate vector.
        noise: Observation noise level.
        one_shot: One shot sampling.
        line_plot: Do bar plot if false, and line plot if true.
        figsize: Figure size, a 2-tuple of int.
        display: Indicator for showing the figure.
        ylim: Y-axis limits.
        loc: Legend location.  Can be '' (default), 'hidden',
            'outside' or any location string in the package.

    Returns:
        Saves plot to file.
    """
    # Confidence interval for estimation of p from a binomial
    # r.v. using Gaussian approximation.
    alpha = 0.95
    yerr = norm.ppf((alpha+1)/2)/2/np.sqrt(num_sims)
    errors = pd.DataFrame(columns=num_rep_vec)
    gamma_vec_candidates = []
    for b in num_rep_vec:
        error_file = (
            '{prefix}-a{self_reg_coeff}'
            '-b{num_rep}'
            '-s1-o{one_shot}'
            '-n{num_sims}'
            '-e{noise}.csv'.format(
                prefix=error_file_prefix,
                self_reg_coeff=self_reg_coeff,
                num_rep=b,
                num_sims=num_sims,
                noise=noise,
                one_shot=one_shot
                )
            )
        x = np.genfromtxt(error_file)
        errors.loc[:, b] = x[:, 1]
        gamma_vec_candidates.append(x[:, 0])
    for a in gamma_vec_candidates:
        for b in gamma_vec_candidates:
            assert(np.array_equal(a, b))
    gamma_vec = gamma_vec_candidates[0]
    # Plotting.
    fig, ax = plt.subplots(figsize=figsize)
    rects = []
    if line_plot:
        for i in range(len(num_rep_vec)):
            rects.append(ax.plot(
                gamma_vec, errors[num_rep_vec[i]], '-o'
                ))
        plot_str = '-line'
    else:
        ind = np.arange(len(gamma_vec))
        width = 0.6/len(num_rep_vec)
        offset = 0
        my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # A darkish cyan.
        my_colors.append('#4CD3D3')
        ax.set_prop_cycle(color=my_colors)
        for i in range(len(num_rep_vec)):
            rects.append(ax.bar(ind+offset*width,
                                errors[num_rep_vec[i]],
                                width, yerr=yerr))
            offset += 1
        ax.set_xticks(ind+width/2*(len(num_rep_vec)-1))
        ax.set_xticklabels(gamma_vec)
        plot_str = '-bar'
    ax.set_ylabel('sign error rate')
    ax.set_xlabel(r'$\gamma$')
    rep_legend = ['1 replicate']+[str(b)+' replicates'
                                  for b in num_rep_vec[1:]]
    if loc == '':
        lgd = ax.legend(tuple(r[0] for r in rects), rep_legend)
    elif loc == 'hidden':
        lgd = None
    elif loc == 'outside':
        lgd = ax.legend(tuple(r[0] for r in rects),
                        rep_legend,
                        bbox_to_anchor=(0., 1.02, 1., .102),
                        loc=3, ncol=2, borderaxespad=0.)
    else:
        lgd = ax.legend(tuple(r[0] for r in rects), rep_legend,
                        loc=loc)
    if ylim:
        ax.set_ylim(*ylim)
    if display:
        fig.show()
    if lgd:
        fig.savefig(
            output_prefix+plot_str+(
                '-a{self_reg_coeff}'
                '-n{num_sims}'
                '-o{one_shot}'
                '-e{noise}.pdf'.format(
                    self_reg_coeff=self_reg_coeff,
                    num_sims=num_sims,
                    one_shot=one_shot,
                    noise=noise
                    )
                ), bbox_extra_artists=(lgd,), bbox_inches='tight'
            )
    else:
        fig.savefig(
            output_prefix+plot_str+(
                '-a{self_reg_coeff}'
                '-n{num_sims}'
                '-o{one_shot}'
                '-e{noise}.pdf'.format(
                    self_reg_coeff=self_reg_coeff,
                    num_sims=num_sims,
                    one_shot=one_shot,
                    noise=noise
                    )
                ), bbox_inches='tight'
            )
    return


def gen_cov_mat(self_reg_coeff, sigma, gamma, num_cond, num_time,
                num_rep, one_shot=True, noise=0):
    """Generate covariance matrix for the observations of a single
    gene under Gaussian linear model with a full factorial design.

    Args:
        self_reg_coeff: Self-regulation coefficient.
        sigma: Standard deviation of the variation term in dynamics.
        gamma: Correlation coefficient of the variation terms of two
            plants under the same condition.
        num_cond: Number of experimental conditions.
        num_time: Number of sampling times.
        num_rep: Number of replicates.
        one_shot: Indicator of one-shot sampling.
        noise: Observation noise level.

    Returns:
        A numpy array of covariance matrix.
    """
    num_samples = num_cond*num_time*num_rep
    # A T-by-T matrix of the discounted covariance, where T is the
    # number of times.
    standard_block = np.empty((num_time, num_time))
    for i in range(num_time):
        for j in range(num_time):
            standard_block[i, j] = (
                geom_sum(self_reg_coeff**2, min(i, j)+1)
                * self_reg_coeff**np.abs(i-j)
                )
    if one_shot:
        # Block with replicates.
        block_w_rep = np.kron(standard_block,
                              np.ones((num_rep, num_rep)))
        # Different individuals in the same condition are discounted
        # with a condition effect of gamma.
        condition_effect = (
            (1-gamma)*np.identity(num_time*num_rep)
            + gamma*np.ones((num_time*num_rep, num_time*num_rep))
            )
        # Final block for a single condition.
        block = block_w_rep*condition_effect
    else:
        # Small matrix of condition effect.
        condition_effect_small = (
            (1-gamma)*np.identity(num_rep)
            + gamma*np.ones((num_rep, num_rep))
            )
        block = np.kron(standard_block, condition_effect_small)
    cov_mat = (
        sigma**2*np.kron(np.identity(num_cond), block)
        + noise**2*np.identity(num_samples)
        )
    return cov_mat


def mle_single_gene(y, sigma, gamma, num_cond, num_time, num_rep,
                    method='nelder-mead-dual',
                    options={'xatol': 1e-3},
                    options_bf={
                        'llh_file': '', 'lower': -2, 'upper': 2,
                        'num_edges': 81
                        }, one_shot=True, noise=0):
    """ML estimator for single gene self regulation.

    Args:
        y: Observation as a numpy array.
        sigma: Standard deviation of the variation term in dynamics.
        gamma: Correlation coefficient of two plants under the same
            condition.
        num_cond: Number of experimental conditions.
        num_time: Number of sampling times.
        num_rep: Number of replicates.
        method: Optimization algorithm.  Could be 'nelder-mead',
            'nelder-mead-dual' or 'brute-force'.
        options: Options for 'nelder-mead' and 'nelder-mead-dual'.
            See scipy.optimize.minimize.
        options_bf: Options for 'brute-force',
            llh_file: Adjusted log-likelihood file to be saved.
                Default is not saving adjusted log-likelihood file.
            lower: Lower limit of optimization range.
            upper: Upper limit of optimization range.
            num_edges: Number of values considered in range.
        one_shot: Indicator of one-shot sampling.
        noise: Observation noise level.

    Returns:
        The MLE.
    """
    # If variations depend entirely on condition, the replicates are
    # identical.  Then we only need to calculate the MLE for the
    # single-replicate data.
    #
    # Note the adjusted log-likelihood function, if plotted, is for
    # the single replicate case rather than the original
    # multiple-replicate case.
    if gamma == 1 and num_rep > 1:
        return mle_single_gene(
            y[::num_rep], sigma, gamma, num_cond, num_time, 1,
            method=method, options=options, options_bf=options_bf
            )
    if method == 'brute-force':
        a_vec = np.linspace(
            options_bf['lower'], options_bf['upper'],
            options_bf['num_edges']
            )
        adj_llh = []
        for a in a_vec:
            cov_mat = gen_cov_mat(a, sigma, gamma, num_cond,
                                  num_time, num_rep, one_shot,
                                  noise)
            # Adjusted log-likelihood (multiplied by -1/2 with
            # offset).
            adj_llh.append(
                np.log(np.linalg.det(cov_mat))
                + y.dot(np.linalg.inv(cov_mat)).dot(y)
                )
        if options_bf['llh_file']:
            plt.figure()
            plt.plot(a_vec, adj_llh)
            plt.savefig(options_bf['llh_file'])
        # Randomize over minimum adjusted log-likelihood.
        indices = [idx for idx, x in enumerate(adj_llh)
                   if x == min(adj_llh)]
        min_idx = indices[np.random.randint(len(indices))]
        return a_vec[min_idx]
    elif method == 'nelder-mead':
        self_reg_0 = np.random.uniform(-1, 1)
        res_nm = minimize(single_gene_llh, self_reg_0, args=(
            sigma, gamma, num_cond, num_time, num_rep, y,
            one_shot, noise
            ), method='nelder-mead', options=options)
        return res_nm.x[0]
    elif method == 'nelder-mead-dual':
        self_reg_left = -1
        res_left = minimize(
            single_gene_llh, self_reg_left, args=(
                sigma, gamma, num_cond, num_time, num_rep,
                y, one_shot, noise
                ), method='nelder-mead', options=options
            )
        self_reg_right = 1
        res_right = minimize(
            single_gene_llh, self_reg_right, args=(
                sigma, gamma, num_cond, num_time, num_rep,
                y, one_shot, noise
                ), method='nelder-mead', options=options
            )
        if res_left.fun > res_right.fun:
            return res_right.x[0]
        elif res_left.fun < res_right.fun:
            return res_left.x[0]
        else:
            if np.random.randint(2):
                return res_right.x[0]
            else:
                return res_left.x[0]
    else:
        print('Unknown optimization algorithm.')
        exit(1)


def single_gene_llh(self_reg, sigma, gamma, num_cond, num_time,
                    num_rep, y, one_shot=True, noise=0):
    """Calculate adjusted negative likelihood for single gene GLM
    model.

    Args:
        self_reg: Self regulation coefficient.
        sigma: Standard deviation of the variation term in dynamics.
        gamma: float
            Correlation coefficient of the variation terms of two
            plants under the same condition.  Must be strictly
            less than 1 when num_rep > 1 and noise == 0.
        num_cond: Number of experimental conditions.
        num_time: Number of sampling times.
        num_rep: Number of replicates.
        y: Observation.
        one_shot: Indicator of one-shot sampling.
        noise: Observation noise level.

    Returns:
        Adjusted negative likelihood.
    """
    cov_mat = gen_cov_mat(self_reg, sigma, gamma, num_cond,
                          num_time, num_rep, one_shot, noise)
    return adj_neg_llh(cov_mat, y)


def adj_neg_llh(cov_mat, y):
    """Adjusted negative log-likelihood function.

    This function calculates the following function
        \log(\det(cov_mat))+y^T*(cov_mat^{-1})*y
        instead of the actual log-likelihood
        -1/2*\log(2*\pi*\det(cov_mat))-1/2*y^T*(cov_mat^{-1})*y.

    Args:
        cov_mat: Covariance matrix as a two-dimensional Numpy array.
        y: Observation as a one-dimensional Numpy array.

    Returns:
        The adjusted negative log-likelihood value.
    """
    a = np.linalg.slogdet(cov_mat)[1]
    b = y.dot(np.linalg.inv(cov_mat)).dot(y)
    return a+b


def geom_sum(a, k):
    """Partial sum of the geometric series 1+a+a**2+...+a**(k-1)."""
    if a == 1:
        return k
    else:
        return (1-a**k)/(1-a)


def mle_eval(num_sims, num_conditions, num_times, num_replicates,
             self_reg_coeff, sigma, gamma, histogram_file='',
             method='nelder-mead', options={'xatol': 1e-3},
             options_bf={
                 'lower': -2, 'upper': 2, 'num_edges': 81
                 }, one_shot=True, noise=0, noise_alg=0,
             joint=False, num_restarts=1):
    """Evaluation of MLE on a single gene.

    Args:
        num_sims: Number of simulations.
        num_conditions: Number of conditions.
        num_times: Number of times.
        num_replicates: Number of replicates.
        self_reg_coeff: Self-regulation coefficient.
        sigma: Standard deviation of the variation term in dynamics.
        gamma: Correlation coefficient of the variation terms of two
            plants under the same condition.
        histogram_file: Path to histogram file to be saved.  Default
            is not saving the histogram.
        method: Optimization algorithm.  Can be 'nelder-mead',
            'nelder-mead-dual' or 'brute-force'.
        options: Options for 'nelder-mead'.
        options_bf: Options for 'brute-force'.
        one_shot: Indicator for one-shot sampling as opposed to
            multi-shot sampling.
        noise: Observation noise level.
        noise_alg: Observation noise level used in algorithm.
        joint: Indicating joint estimation of A and gamma.
        num_restarts: Number of restarts in search.

    Returns:
        If more than one simulations is done, returns the accuracy.
            An estimate of zero is treated as half-accurate.
        If only one simulation is done, returns the MLE.
    """
    num_samples = num_conditions*num_times*num_replicates
    true_cov_mat = gen_cov_mat(
        self_reg_coeff, sigma, gamma, num_conditions, num_times,
        num_replicates, one_shot, noise
        )
    mle_vec = []
    if num_sims == 1:
        is_single_sim = True
    else:
        is_single_sim = False
    for i in range(num_sims):
        # Generate one-shot observation.
        y = np.random.multivariate_normal(
            np.zeros(num_samples), true_cov_mat
            )
        options_bf['llh_file'] = ''
        if is_single_sim:
            print('sigma =', sigma)
            print('gamma =', gamma)
            print('y =', y)
            options_bf['llh_file'] = (
                'adj_llh-s{}-g{}-b{}.pdf'.format(
                    sigma, gamma, num_replicates
                    )
                )
        if joint:
            mle_vec.append(
                joint_mle(
                    y, num_conditions, num_times, num_replicates,
                    one_shot, num_restarts, sigma, noise_alg
                    )[0]
                )
        else:
            mle_vec.append(mle_single_gene(
                y, sigma, gamma, num_conditions, num_times,
                num_replicates, method=method, options=options,
                options_bf=options_bf, one_shot=one_shot,
                noise=noise_alg
                ))
    if not is_single_sim:
        if histogram_file:
            plt.figure()
            # Add one bin to count the rightmost edge.
            if method == 'brute-force':
                lower = options_bf['lower']
                upper = options_bf['upper']
                num_edges = options_bf['num_edges']
                plt.hist(mle_vec, bins=np.linspace(
                    lower,
                    upper + (upper-lower)/(num_edges-1),
                    num_edges+1
                    ))
            elif method in ['nelder-mead', 'nelder-mead-dual']:
                plt.hist(mle_vec)
            else:
                print('Unknown optimization algorithm.')
                exit(1)
            plt.savefig(histogram_file)
        mle_arr = np.asarray(mle_vec)
        if self_reg_coeff > 0:
            acc = (sum(mle_arr > 0)+sum(mle_arr == 0)/2)/num_sims
        elif self_reg_coeff < 0:
            acc = (sum(mle_arr < 0)+sum(mle_arr == 0)/2)/num_sims
        else:
            print('No regulation in ground truth.')
            exit(1)
        return acc
    else:
        return mle_vec[0]


def gen_error_tables(self_reg_coeff_vec, b_vec,
                     path, latex=False):
    """Generate error rate tables.

    Args:
        self_reg_coeff_vec: Self regulation coefficient vector.
        b_vec: Batch (replicate size) vector.
        path: Path to the error files.
        latex: Indicator to print latex table entries.

    Returns:
        Prints the tables.
    """
    x = {}
    gamma_vec = {}
    for b in b_vec:
        for a in self_reg_coeff_vec:
            error_file = (path+'mle-error-a{}-b{}-s1.csv'.format(
                a, b
                ))
            data = np.genfromtxt(error_file)
            x[a] = data[:, 1]
            gamma_vec[a] = data[:, 0]
        for key_a in gamma_vec:
            for key_b in gamma_vec:
                assert(np.array_equal(
                    gamma_vec[key_a], gamma_vec[key_b]
                    ))
        errors = pd.DataFrame(
            data=x, index=gamma_vec[self_reg_coeff_vec[0]]
            )
        print('b =', b)
        if latex:
            print(remove_zeros(errors))
        else:
            print(errors)
    return


def remove_zeros(x):
    """Remove trailing zeros from float numbers and format as
    LaTeX table for the most part.

    Args:
        x: A pandas DataFrame of float numbers.

    Returns:
        A pandas Dataframe of string representations of the
        float numbers.

    """
    y = pd.DataFrame(index=x.index, columns=x.columns)
    for col in x:
        for idx in x[col].index:
            if col == 0.9:
                y[col][idx] = '$%g$\\\\'%(x[col][idx])
            else:
                y[col][idx] = '$%g$ &'%(x[col][idx])
    gamma_str = []
    for gamma in y.index:
        gamma_str.append('$%g$ &'%(gamma))
    y.index = gamma_str
    return y


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--self-reg-coeff",
        help="self regulation coefficient",
        type=float, default=0.5)
    parser.add_argument(
        "-r", "--num-reps", help="number of replicates",
        type=int, default=3)
    parser.add_argument("-o", "--one-shot",
                        help="one-shot sampling",
                        action="store_true")
    parser.add_argument("-s", "--num-sims",
                        help="number of simulations",
                        type=int, default=100)
    parser.add_argument("-k", "--num-samples",
                        help="budget in number of samples",
                        type=int, default=180)
    parser.add_argument("-g", "--gamma-vec",
                        help="list of gamma values",
                        type=float, nargs="+", default=[
                            0, 0.2, 0.4, 0.6, 0.8, 1
                            ])
    parser.add_argument("-e", "--noise",
                        help="observation noise level",
                        type=float, default=0)
    return parser.parse_args(argv)


def test_mle_single_gene(self_reg_coeff, num_cond, num_time,
                         num_reps, gamma, one_shot, y=None,
                         saveas=False, noise=0):
    sigma = 1
    num_samp = num_cond*num_time*num_reps
    true_cov_mat = gen_cov_mat(
        self_reg_coeff, sigma, gamma, num_cond, num_time,
        num_reps, one_shot, noise
        )
    if y is None:
        y = np.random.multivariate_normal(
            np.zeros(num_samp), true_cov_mat
            )
    mle = mle_single_gene(
        y, sigma, gamma, num_cond, num_time, num_reps,
        one_shot=one_shot
        )
    print(mle)
    a_vec = np.linspace(-2, 2, 101)
    llh = [single_gene_llh(
        a, sigma, gamma, num_cond, num_time, num_reps, y,
        one_shot
        ) for a in a_vec]
    plt.figure()
    reverse_llh = llh.copy()
    reverse_llh.reverse()
    plt.plot(a_vec, llh, a_vec, reverse_llh)
    plt.axvline(x=mle, color='k')
    if saveas:
        plt.savefig(saveas)
    return y


def likelihood_counter_example():
    """An example where l(A) = l(-A) for multiple values of A."""
    self_reg_coeff, num_cond, num_time, num_reps, gamma, one_shot = (
        0.9, 4, 3, 2, 0.5, True
        )
    y_counter_example = np.asarray([
        0.47249251,  1.09651484, -1.10655747, -0.91039473, -0.69909359,
       -0.94916349,  0.25979171, -0.22746461,  0.0942397 ,  1.31538243,
        0.56332587,  2.59745066, -0.15654428, -0.382863  ,  2.98033385,
        2.11137429,  1.01076027, -0.80187295,  0.20700742, -0.12052936,
       -1.33710031,  0.2097227 , -0.14146918,  0.23988047
        ])
    _ = test_mle_single_gene(self_reg_coeff, num_cond, num_time,
                             num_reps, gamma, one_shot,
                             y=y_counter_example,
                             saveas='counter.pdf')
    return


def test_glrt_single_gene(self_reg_coeff, sigma, num_cond, num_time,
                          num_reps, gamma, one_shot, y=None,
                          saveas=False, noise=0, num_restarts=1):
    num_samp = num_cond*num_time*num_reps
    true_cov_mat = gen_cov_mat(
        self_reg_coeff, sigma, gamma, num_cond, num_time,
        num_reps, one_shot, noise
        )
    if y is None:
        y = np.random.multivariate_normal(
            np.zeros(num_samp), true_cov_mat
            )
    mle = joint_mle_4(
        y, num_cond, num_time, num_reps,
        one_shot=one_shot, num_restarts=num_restarts
        )
    print(mle)
    a_vec = np.linspace(-1, 1, 101)
    llh = [single_gene_llh(
        a, sigma, gamma, num_cond, num_time, num_reps, y,
        one_shot, noise
        ) for a in a_vec]
    plt.figure()
    plt.plot(a_vec, llh)
    plt.axvline(x=mle[0], color='k')
    if saveas:
        plt.savefig(saveas)
    return y


def single_gene_llh_cmpt(x, num_cond, num_time, num_rep, y, one_shot):
    """Compact version of the adjusted negative log-likelihood
    for single gene GLM model.

    Args:
        x: Four parameters.
            x[0]: Self regulation coefficient.
            x[1]: sigma, the (total) variation level.
            x[2]: gamma, the condition dependency factor.
            x[3]: sigma_Z, the observation noise level.
        num_cond: Number of experimental conditions.
        num_time: Number of sampling times.
        num_rep: Number of replicates.
        y: Observation.
        one_shot: Indicator of one-shot sampling.

    Returns:
        Adjusted negative log-likelihood.
    """
    return single_gene_llh(
        x[0], x[1], x[2], num_cond, num_time, num_rep, y,
        one_shot, x[3]
        )


def joint_mle_4(y, num_cond, num_time, num_rep,
                one_shot=True, num_restarts=1):
    """Joint estimation of all four parameters using MLE.

    Args:
        y: Observation.
        num_cond: Number of conditions.
        num_time: Number of times.
        num_rep: Number of replicates.
        one_shot: Indicator one-shot sampling rather than multi-shot.
        num_restarts: Number of restarts in search.

    Returns:
        The MLE of self-regulation coefficient, variation level,
        gamma and noise level.
    """
    # Generate initial guess uniformly over
    # [-1, 1]*[0, 1]*[0, 1]*[0.1, 1].
    lower = [-1, 0, 0, 0.1]
    upper = [1, 1, 1, 1]
    res_vec = []
    for i in range(num_restarts):
        x0 = np.random.uniform(lower, upper)
        res_vec.append(minimize(
            single_gene_llh_cmpt, x0, args=(
                num_cond, num_time, num_rep, y, one_shot
                ), bounds=tuple(zip(lower, upper))
            ))
    fun_vec = [res.fun for res in res_vec]
    indices = [idx for idx, x in enumerate(fun_vec)
               if x == min(fun_vec)]
    min_idx = indices[np.random.randint(len(indices))]
    return res_vec[min_idx].x


def joint_dena_eval(num_sims, num_cond, num_time, num_rep,
                    sigma, noise, one_shot, scatter, restart,
                    fixed_a=None, fixed_gamma=None, saveas='',
                    save_data=''):
    """Evaluate the correlation of A and A hat, and that of gamma
    and gamma hat.

    Args:
        num_sims: Number of simulations.
        num_cond: Number of conditions.
        num_time: Number of times.
        num_rep: Number of replicates.
        sigma: Variation level.
        noise: Observation noise level.
        one_shot: Indicator for one-shot sampling rather than
            multi-shot sampling.
        scatter: Indicator for scatter plots.
        restart: Number of restarts in optimization.
        fixed_a: The fixed value of self regulation coefficient.
            None indicates i.i.d. random values for different
            simulations.
        fixed_gamma: The fixed value of gamma.
            None indicates i.i.d. random values for different
            simulations.
        saveas: Prefix for output.
        save_data: Save data in pickle file.

    Returns:
        Pearson correlation coefficients for A vs. A hat, and
        for gamma vs. gamma hat.

    """
    if fixed_a is None:
        a_vec = np.random.uniform(-1, 1, num_sims)
    else:
        a_vec = np.ones(num_sims)*fixed_a
    if fixed_gamma is None:
        gamma_vec = np.random.uniform(0, 1, num_sims)
    else:
        gamma_vec = np.ones(num_sims)*fixed_gamma
    num_samples = num_cond*num_time*num_rep
    a_hat_vec = []
    gamma_hat_vec = []
    for i in tqdm(range(num_sims)):
        true_cov_mat = gen_cov_mat(
            a_vec[i], sigma, gamma_vec[i], num_cond, num_time,
            num_rep, one_shot, noise
            )
        y = np.random.multivariate_normal(
            np.zeros(num_samples), true_cov_mat
            )
        a_hat, _, gamma_hat, _ = joint_mle_4(
            y, num_cond, num_time, num_rep, one_shot, restart
            )
        a_hat_vec.append(a_hat)
        gamma_hat_vec.append(gamma_hat)
    if save_data:
        pickle.dump({
            'a_hat_vec': a_hat_vec, 'a_vec': a_vec,
            'gamma_vec': gamma_vec,
            'gamma_hat_vec': gamma_hat_vec
            }, open(save_data, 'wb'))
    if scatter:
        plt.figure()
        plt.scatter(a_vec, a_hat_vec)
        if saveas:
            plt.savefig(saveas+'-a.pdf')
        plt.figure()
        plt.scatter(gamma_vec, gamma_hat_vec, c='b')
        if saveas:
            plt.savefig(saveas+'-g.pdf')
    pa = pearsonr(a_vec, a_hat_vec)
    pg = pearsonr(gamma_vec, gamma_hat_vec)
    if saveas:
        with open(saveas+'-p.tsv', 'w') as f:
            f.write('{}\t{}\n{}\t{}\n'.format(pa[0], pa[1],
                                              pg[0], pg[1]))
    return pa, pg


def joint_mle(y, num_cond, num_time, num_rep,
              one_shot, num_restarts, sigma, noise,
              sing_margin=0.001):
    """Joint estimation of self regulation coefficient and
    gamma with known variation and noise levels.

    To avoid complication of comparison with degenerate
    distributions, we restrict gamma to be less than
    1-sing_margin.

    Args:
        y: Observation.
        num_cond: Number of conditions.
        num_time: Number of times.
        num_rep: Number of replicates.
        one_shot: Indicator one-shot sampling rather than multi-shot.
        num_restarts: Number of restarts in search.
        sigma: Variation level.
        noise: Noise level.
        sing_margin: float, optional, default 0.001
            Margin of upper bound of gamma from 1 to guard the
            covariance matrix from singularity.

    Returns:
        The MLE of self-regulation coefficient and variation level.
    """
    # Generate initial guess uniformly.
    lower = [-1, 0]
    upper = [1, 1-sing_margin]
    res_vec = []
    for i in range(num_restarts):
        x0 = np.random.uniform(lower, upper)
        res_vec.append(minimize(
            single_gene_llh_cmpt_2, x0, args=(
                num_cond, num_time, num_rep, y, one_shot,
                sigma, noise
                ), bounds=tuple(zip(lower, upper))
            ))
    fun_vec = [res.fun for res in res_vec]
    indices = [idx for idx, x in enumerate(fun_vec)
               if x == min(fun_vec)]
    min_idx = indices[np.random.randint(len(indices))]
    return res_vec[min_idx].x


def single_gene_llh_cmpt_2(x, num_cond, num_time, num_rep, y,
                           one_shot, sigma, noise):
    """Compact version of the adjusted negative log-likelihood
    for single gene GLM model.

    Args:
        x: Four parameters.
            x[0]: Self regulation coefficient.
            x[1]: gamma, the condition dependency factor.
        num_cond: Number of experimental conditions.
        num_time: Number of sampling times.
        num_rep: Number of replicates.
        y: Observation.
        one_shot: Indicator of one-shot sampling.
        sigma: Variation level.
        noise: Noise level.

    Returns:
        Adjusted negative log-likelihood.
    """
    return single_gene_llh(
        x[0], sigma, x[1], num_cond, num_time, num_rep, y,
        one_shot, noise
        )


def eval_joint_sign_error(
        self_reg_coeff, num_reps, one_shot, num_sims,
        num_samples, gamma_vec, noise, num_restarts
        ):
    """Evaluate the sign error rate of GLRT with full joint
    estimation.

    Multiple gammas are used.  Result is written to file.

    Args:
        self_reg_coeff: Self regulation coefficient.
        num_reps: Number of replicates.
        one_shot: Indicator for one-shot sampling.
        num_sims: Number of simulations.
        num_samples: Total budget in number of samples.
        gamma_vec: List of gamma values.
        noise: Observation noise level.
        num_restarts: Number of restarts.

    Returns:
        Write a file of error rates.

    """
    print("Self regulation coefficient: {}\n"
          "Number of replicates: {}\n"
          "One-shot sampling: {}\n"
          "Number of simulations: {}\n"
          "Number of samples: {}\n"
          "Gamma values: {}\n"
          "Noise: {}\n"
          "Number of restarts: {}\n".format(
              self_reg_coeff, num_reps, one_shot, num_sims,
              num_samples, gamma_vec, noise, num_restarts
              ))
    sigma = 1
    num_times = 6
    num_conditions = int(num_samples/num_reps/num_times)
    errors = []
    for gamma in tqdm(gamma_vec):
        true_cov_mat = gen_cov_mat(
            self_reg_coeff, sigma, gamma, num_conditions,
            num_times, num_reps, one_shot, noise
            )
        mle_vec = []
        for i in range(num_sims):
            # Generate one-shot observation.
            y = np.random.multivariate_normal(
                np.zeros(num_samples), true_cov_mat
                )
            a_hat, _, _, _ = joint_mle_4(
                y, num_conditions, num_times, num_reps,
                one_shot, num_restarts
                )
            mle_vec.append(a_hat)
        mle_arr = np.asarray(mle_vec)
        if self_reg_coeff > 0:
            acc = (sum(mle_arr > 0)+sum(mle_arr == 0)/2)/num_sims
        elif self_reg_coeff < 0:
            acc = (sum(mle_arr < 0)+sum(mle_arr == 0)/2)/num_sims
        else:
            print('No regulation in ground truth.')
            exit(1)
        errors.append(1-acc)
    data = np.asarray([gamma_vec, errors]).T
    np.savetxt('jglrt-sign-error-a{self_reg_coeff}'
               '-b{num_reps}'
               '-s{sigma}'
               '-o{one_shot}'
               '-n{num_sims}'
               '-e{noise}.csv'.format(
                   self_reg_coeff=self_reg_coeff,
                   num_reps=num_reps,
                   sigma=sigma,
                   one_shot=one_shot,
                   num_sims=num_sims,
                   noise=noise
                   ), data)
    return


def scatter_hist_joint(pickle_file, prefix, fixed_a, fixed_g, alpha):
    """Do scatter plots and/or histograms of joint MLE estimation.

    Args:
        pickle_file: pickle file of the dictionary of A, A hat,
            gamma, gamma hat.
        prefix: Prefix for saving the plots.
        fixed_a: Fixed value of A.  Do histogram of A hat
            instead of scatter plot.
        fixed_g: Fixed value of gamma. Do hitsogram of gamma
            hat instead of scatter plot.
        alpha: Alpha channel transparency.

    Returns:
        Saves plots for A and gamma.
    """
    data_dict = pickle.load(open(pickle_file, 'rb'))
    figsize = (4, 3)
    plt.figure(figsize=figsize)
    color = plt.rcParams['axes.prop_cycle'].by_key()
    if fixed_a is None:
        plt.scatter(data_dict['a_vec'], data_dict['a_hat_vec'],
                    color=color['color'][0], alpha=alpha, s=10,
                    linewidth=0)
        plt.ylim([-1.06, 1.06])
        plt.xlim([-1.06, 1.06])
        plt.xlabel(r'$A$')
        plt.ylabel(r'$\widehat A$')
    else:
        plt.hist(data_dict['a_hat_vec'], range=(-1, 1),
                 color=color['color'][0])
    plt.savefig(prefix+'-a.pdf', bbox_inches='tight')
    plt.figure(figsize=figsize)
    if fixed_g is None:
        plt.scatter(data_dict['gamma_vec'],
                    data_dict['gamma_hat_vec'],
                    color=color['color'][1], alpha=alpha,
                    s=10, linewidth=0)
        plt.ylim([-0.03, 1.03])
        plt.xlim([-0.03, 1.03])
        plt.xlabel(r'$\gamma$')
        plt.ylabel(r'$\widehat\gamma$')
    else:
        plt.hist(data_dict['gamma_hat_vec'], range=(0, 1),
                 color=color['color'][1])
    plt.savefig(prefix+'-g.pdf', bbox_inches='tight')
    return


def script_mle_eval(one_shot, fixed_a, fixed_g):
    num_sims = 1000
    num_cond = 10
    num_time = 6
    num_rep = 3
    sigma = 1
    noise = 0.3
    scatter = False
    restart = 3
    alpha = 0.8
    extra_str = ''
    if fixed_a:
        extra_str = '-a'+str(fixed_a)
    if fixed_g:
        extra_str = '-g'+str(fixed_g)
    pickle_file = (
        'mle-n{num_sims}-b{num_rep}-e{noise}-o{one_shot}'
        '{extra_str}.pkl'.format(
            num_sims=num_sims,
            num_rep=num_rep,
            noise=noise,
            one_shot=one_shot,
            extra_str=extra_str
            )
        )
    pa, pg = joint_dena_eval(
        num_sims, num_cond, num_time, num_rep, sigma, noise,
        one_shot, scatter, restart, save_data=pickle_file,
        fixed_a=fixed_a, fixed_gamma=fixed_g
        )
    with open(pickle_file[:-4]+'.txt', 'w') as f:
        f.write("Pearson correlation coefficient for A: ")
        f.write(str(pa[0]))
        f.write("\np-value for A: ")
        f.write(str(pa[1]))
        f.write("\nPearson correlation coefficient for gamma: ")
        f.write(str(pg[0]))
        f.write("\np-value for gamma: ")
        f.write(str(pg[1]))
        f.write("\n")
    scatter_hist_joint(
        pickle_file, pickle_file[:-4], fixed_a, fixed_g, alpha
        )
    return


def plot_plos_one_fig_4(
        num_sims=10000, rep_list=[1, 2, 3, 5, 6, 10, 15, 30],
        gamma_list=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        self_reg_list=[0.1, 0.5], one_shot_list=[True, False],
        noise_joint_list=[(0.0, False), (1.0, True)],
        figsize=(7, 4), ylim=[-0.02, 0.6]
        ):
    """Plot PLOS ONE submission Figure 4.

    TODO:
        Eliminate intermediate file generation.

    Args:
        num_sims: int, default 10000
            Number of simulations.
        rep_list: list, default is the list of all divisors of 30
            List of replicates.
        gamma_list: list, default is 0 to 1 with interval 0.2
            List of gamma values.  Need to have at least two
            elements.
        self_reg_list: list, default [0.1, 0.5]
            The self-regulation coefficients.
        one_shot_list: list, default [True, False]
            Indicator of one-shot sampling.
        noise_joint_list: list, default [(0.0, False), (1.0, True)]
            A list of tuples of observation noise level and
            indicator of joint A-gamma estimation.
        figsize: tuple, default (7, 4)
            Figure size.
        ylim: list, default [-0.02, 0.6]
            Y-axis range.  Can be None for an automatic range.

    Returns:
        Saves error data and figures to files.
    """
    # Confidence interval for estimation of p from a binomial
    # r.v. using Gaussian approximation.
    alpha = 0.95
    yerr = norm.ppf((alpha+1)/2)/2/np.sqrt(num_sims)
    # Generate data.
    total_num_samples = 180
    legend_pos = 'outside'
    rep_text = (lambda x: '1 replicate' if x == 1
                else '{} replicates'.format(x))
    rep_legend = [rep_text(b) for b in rep_list]
    for self_reg in self_reg_list:
        for one_shot in one_shot_list:
            for noise, joint in noise_joint_list:
                main(self_reg, rep_list, one_shot, num_sims,
                     total_num_samples, gamma_list, noise,
                     joint=joint)
                # Load data from files.
                errors = pd.DataFrame(index=gamma_list,
                                      columns=rep_legend)
                gamma_vec_candidates = []
                for idx_b, b in enumerate(rep_list):
                    error_file = (
                        'mle-error-a{self_reg_coeff}'
                        '-b{num_reps}'
                        '-s{sigma}'
                        '-o{one_shot}'
                        '-n{num_sims}'
                        '-e{noise}'
                        '-j{joint}.csv'.format(
                            self_reg_coeff=self_reg,
                            num_reps=b,
                            sigma=1,
                            one_shot=one_shot,
                            num_sims=num_sims,
                            noise=noise,
                            joint=joint
                            )
                        )
                    x = np.genfromtxt(error_file)
                    errors.loc[:, rep_legend[idx_b]] = x[:, 1]
                    gamma_vec_candidates.append(x[:, 0])
                # All files should have the same list of gamma
                # values.
                for a in gamma_vec_candidates:
                    for b in gamma_vec_candidates:
                        assert(np.array_equal(a, b))
                gamma_vec = gamma_vec_candidates[0]
                filename = (
                    'glm-bar-a{self_reg}-n{num_sims}-o{one_shot}'
                    '-e{noise}-j{joint}.pdf'.format(
                        self_reg=self_reg, num_sims=num_sims,
                        one_shot=one_shot, noise=noise,
                        joint=joint
                        )
                    )
                bar_plot(num_sims, errors.T, 'sign error rate',
                         legend_pos, ylim=ylim,
                         save_file=filename, figsize=figsize,
                         x_axis_name=r'$\gamma$', colors=8)
                # Only the first figure shows legend.
                legend_pos = 'hidden'


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(**vars(args))
