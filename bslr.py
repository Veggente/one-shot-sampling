#!/usr/bin/env python
"""BSLR evaluation.

Functions:
    eval_bslr: Evaluate BSLR.
    plot_plos_one_fig_7: Plot PLOS ONE Fig 7.
    bar_plot: Generic bar plot.
    plot_vary_gamma_sampling_and_noise: Bar plot for varying
        gamma, sampling method and noise level.
    plot_vary_replicate: Bar plot for varying number of
        replicates.
    plot_hetero_gamma: Bar plot of loss functions for heterogeneous
        gamma case.
    plot_plos_one_fig_5_6: Plot PLOS ONE Figs 5 and 6.
    eval_bslr_on_locke: Evaluate BSLR using Locke model.
    close: Unparameterize function.
    locke_drift: Drift coefficient of Locke model.
    locke_drift_lite: Drift for one variable.
    hill: Hill function.
    diff_coeff: Diffusion coefficient.
    get_locke_params: Get parameters in Locke paper.
    eval_bslr_multi_sims: Evaluate BSLR with multiple simulations.
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import norm
import sys
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
import pickle
import sdeint
import json

# CausNet commit 425e3a9.
import causnet
import bio_data_gen
from perf_eval import get_sas


def eval_bslr(sigma, gamma, num_replicates, one_shot, noise,
              sig_level, num_sims, num_samples=180):
    """Evaluate BSLR with averaging.

    Args:
        sigma: Variation level.
        gamma: Condition correlation coefficient.
            Can be a scalar or an array of gene size.
            For an array, the gene size must be even and the
            first half gamma's are equal to a higher value,
            while the second a lower value.
        num_replicates: Number of replicates.
        one_shot: Indicator for one-shot sampling.
        noise: Observation noise level.
        sig_level: Significance level for Granger F-test.
        num_sims: int
            Number of simulations.  Simulations with undefined
            performance metric (e.g., one with a recall of 0 due
            to an empty ground-truth network) do not count.
        num_samples: int, default 180
            Number of total samples.

    Returns: tuple
        A 3-tuple of lists of recall, precision and specificity
        for scalar gamma, or a 3-tuple of lists of quartets of
        recall, precision and specificity for heterogeneous
        gamma (10 highs, 10 lows).
    """
    num_genes = 20
    max_in_deg = 3
    margin = 0.5
    sigma_c = sigma*np.sqrt(gamma)
    sigma_b = sigma*np.sqrt(1-gamma)
    num_times = 6
    num_experiments = int(num_samples/num_replicates/num_times)
    # Empty CSV file indicates the expression array is returned as a
    # variable.
    csv_exp_file = ''
    csv_design_file = 'design-b{}.csv'.format(num_replicates)
    # Default time seed.
    rand_seed = None
    dynamics = 'glm'
    # Evaluate performance in recall, precision and specificity.
    r_vec = []
    p_vec = []
    s_vec = []
    r = np.nan
    p = np.nan
    s = np.nan
    # Repeat this simulation until the recall, precision or
    # specificity do not contain nan.
    while np.isnan(r).any() or np.isnan(p).any() or np.isnan(s).any():
        # Generate network.
        adj_mat = bio_data_gen.gen_adj_mat(num_genes, max_in_deg,
                                           margin)
        # Generate data with design file.
        exp_df = bio_data_gen.gen_planted_edge_data(
            num_genes, adj_mat, sigma_c, sigma_b, num_experiments,
            csv_exp_file, csv_design_file, num_replicates,
            num_times, rand_seed, one_shot, dynamics, noise
            )
        parser_dict = causnet.load_parser(csv_design_file)
        # Reconstruct network with edge signs only.
        adj_mat_sign_rec = causnet.bslr(
            parser_dict, exp_df, num_experiments, num_times,
            num_genes, num_replicates, max_in_deg, sig_level
            )

        if np.isscalar(gamma):
            r, p, s = get_sas(adj_mat_sign_rec, np.sign(adj_mat))
        else:
            # Assume the first half of gamma are equal and the second
            # half are equal.
            group_size = int(num_genes/2)
            r1, p1, s1 = get_sas(
                adj_mat_sign_rec[0:group_size, 0:group_size],
                np.sign(adj_mat)[0:group_size, 0:group_size]
                )
            r2, p2, s2 = get_sas(
                adj_mat_sign_rec[0:group_size, group_size:num_genes],
                np.sign(adj_mat)[0:group_size, group_size:num_genes]
                )
            r3, p3, s3 = get_sas(
                adj_mat_sign_rec[group_size:num_genes, 0:group_size],
                np.sign(adj_mat)[group_size:num_genes, 0:group_size]
                )
            r4, p4, s4 = get_sas(
                adj_mat_sign_rec[group_size:num_genes,
                                 group_size:num_genes],
                np.sign(adj_mat)[group_size:num_genes,
                                 group_size:num_genes]
                )
            r = [r1, r2, r3, r4]
            p = [p1, p2, p3, p4]
            s = [s1, s2, s3, s4]
    r_vec.append(r)
    p_vec.append(p)
    s_vec.append(s)
    # Repeat for another num_sims-1 times.
    for i in tqdm(range(num_sims-1)):
        r = np.nan
        p = np.nan
        s = np.nan
        while np.isnan(r).any() or np.isnan(p).any() or np.isnan(s).any():
            adj_mat = bio_data_gen.gen_adj_mat(
                num_genes, max_in_deg, margin
                )
            exp_df = bio_data_gen.gen_planted_edge_data(
                num_genes, adj_mat, sigma_c, sigma_b,
                num_experiments, csv_exp_file, csv_design_file,
                num_replicates, num_times, rand_seed, one_shot,
                dynamics, noise
                )
            adj_mat_sign_rec = causnet.bslr(
                parser_dict, exp_df, num_experiments, num_times,
                num_genes, num_replicates, max_in_deg,
                sig_level
                )
            if np.isscalar(gamma):
                r, p, s = get_sas(adj_mat_sign_rec, np.sign(adj_mat))
            else:
                # Assume the first half of gamma are equal and the second
                # half are equal.
                r1, p1, s1 = get_sas(
                    adj_mat_sign_rec[0:group_size, 0:group_size],
                    np.sign(adj_mat)[0:group_size, 0:group_size]
                    )
                r2, p2, s2 = get_sas(
                    adj_mat_sign_rec[
                        0:group_size, group_size:num_genes
                        ], np.sign(adj_mat)[
                            0:group_size, group_size:num_genes
                            ]
                    )
                r3, p3, s3 = get_sas(
                    adj_mat_sign_rec[
                        group_size:num_genes, 0:group_size
                        ], np.sign(adj_mat)[
                            group_size:num_genes, 0:group_size
                            ]
                    )
                r4, p4, s4 = get_sas(
                    adj_mat_sign_rec[
                        group_size:num_genes,
                        group_size:num_genes
                        ], np.sign(adj_mat)[
                            group_size:num_genes,
                            group_size:num_genes
                            ]
                    )
                r = [r1, r2, r3, r4]
                p = [p1, p2, p3, p4]
                s = [s1, s2, s3, s4]
        r_vec.append(r)
        p_vec.append(p)
        s_vec.append(s)
    return r_vec, p_vec, s_vec


def plot_plos_one_fig_7(num_sims=1000):
    """Plot PLOS ONE submission Figure 4.

    Args:
        num_sims: int, default 1000
            Number of simulations.

    Returns:
        Saves figures to files.
    """
    rpf_dict_full = {}
    rpf_dict = {}
    gamma_list = np.concatenate((np.ones(10)*0.8, np.zeros(10)))
    alpha = 0.95
    figsize = (7, 4)
    ylim = [-0.05, 1.05]
    one_shot_labels = ['one-shot', 'multi-shot']
    for idx, one_shot in enumerate([True, False]):
        rpf_dict[one_shot_labels[idx]] = []
        rpf_dict_full[one_shot_labels[idx]] = eval_bslr(
            1, gamma_list, 3, one_shot, 1, 0.05, num_sims
            )
        for perf_arr in rpf_dict_full[one_shot_labels[idx]]:
            perf_list_no_nan = []
            counter = 0
            for row in perf_arr:
                if not np.isnan(row).any():
                    counter += 1
                    perf_list_no_nan.append(row)
            rpf_dict[one_shot_labels[idx]].append(np.mean(
                perf_list_no_nan, axis=0
                ))
    rec_dict = {key: rpf_dict[key][0] for key in rpf_dict}
    prec_dict = {key: rpf_dict[key][1] for key in rpf_dict}
    spec_dict = {key: rpf_dict[key][2] for key in rpf_dict}
    columns = ['DEG to DEG', 'DEG to non-DEG', 'non-DEG to DEG',
               'non-DEG to non-DEG']
    rec_df = pd.DataFrame.from_dict(
        rec_dict, orient='index', columns=columns
        )
    prec_df = pd.DataFrame.from_dict(
        prec_dict, orient='index', columns=columns
        )
    spec_df = pd.DataFrame.from_dict(
        spec_dict, orient='index', columns=columns
        )
    bar_plot(num_sims, 1-rec_df, 'FNR', 'outside',
             save_file=(
                 'hetero-gamma-fnr-n{}'
                 '.pdf'.format(num_sims)
                 ), figsize=figsize, ylim=ylim)
    bar_plot(num_sims, 1-prec_df, 'FDR', 'outside',
             save_file=(
                 'hetero-gamma-fdr-n{}'
                 '.pdf'.format(num_sims)
                 ), figsize=figsize, ylim=ylim)
    bar_plot(num_sims, 1-spec_df, 'FPR', 'outside',
             save_file=(
                 'hetero-gamma-fpr-n{}'
                 '.pdf'.format(num_sims)
                 ), figsize=figsize, ylim=ylim)
    return


def bar_plot(num_sims, data, data_name, loc,
             bbox_to_anchor=(0., 1.02, 1., .102), ylim=None,
             display=False, save_file=None, figsize=None,
             colors=None, alpha=0.95, x_axis_name=None):
    """Generic bar plotting with confidence intervals.

    Args:
        num_sims: int
            Number of samples/simulations.
        data: pandas.DataFrame
            The index is for different colors of bars shown in
            the legend, and column names different conditions
            in x-axis of the bar plot.
        data_name: str
            The name of the quantities in data.
        loc: str or None
            Location of legend.  Can be 'hidden', 'outside',
            None or other default location string.
        bbox_to_anchor: array-like, default (0., 1.02, 1., .102)
            A 4-tuple for the location of the legend box outside
            the plot.
        ylim: array-like or None
            Limits for y-axis.
        display: bool
            Show figure.
        save_file: str or None
            Path to write file.
        figsize: array-like or None
            A 2-tuple for figure size.  Can be default by
            setting to None.
        colors: int, array-like or None
            Specify colors used for bars.  Can be None (cycles
            of default 7 colors), 8 (one extra color), or an
            array of colors.
        alpha: float, default 0.95
            Confidence level for the confidence interval based
            on independen Gaussian samples.
        x_axis_name: str or None
            The name of the x-axis labels.

    Returns: None
        Saves figure if save_file is not None.
    """
    yerr = norm.ppf((alpha+1)/2)/2/np.sqrt(num_sims)
    fig, ax = plt.subplots(figsize=figsize)
    rects = []
    num_colors, num_clusters = data.shape
    width = 0.6/num_colors
    ind = np.arange(num_clusters)
    offset = 0
    if colors:
        if colors == 8:
            my_colors = plt.rcParams[
                'axes.prop_cycle'
                ].by_key()['color']
            # A darkish cyan.
            my_colors.append('#4CD3D3')
            ax.set_prop_cycle(color=my_colors)
        else:
            ax.set_prop_cycle(color=colors)
    for idx in data.index:
        rects.append(ax.bar(ind+offset*width,
                            data.loc[idx],
                            width, yerr=yerr))
        offset += 1
    ax.set_xticks(ind+width/2*(num_colors-1))
    ax.set_xticklabels(data.columns)
    ax.set_ylabel(data_name)
    if x_axis_name:
        ax.set_xlabel(x_axis_name)
    if loc == '':
        lgd = ax.legend(tuple(r[0] for r in rects), data.index)
    elif loc == 'hidden':
        lgd = None
    elif loc == 'outside':
        lgd = ax.legend(tuple(r[0] for r in rects),
                        data.index,
                        bbox_to_anchor=bbox_to_anchor,
                        loc=3, ncol=2, borderaxespad=0.)
    else:
        lgd = ax.legend(tuple(r[0] for r in rects), data.index,
                        loc=loc)
    if ylim:
        ax.set_ylim(*ylim)
    if display:
        fig.show()
    if lgd:
        fig.savefig(save_file, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
    else:
        fig.savefig(save_file, bbox_inches='tight')
    return


def plot_vary_gamma_sampling_and_noise():
    filename = (
        '/Users/veggente/Documents/workspace/python/'
        'bslr_20_gene/vary-noise-and-gamma/'
        'glm-bslr-g{}-o{}.pkl'
        )
    rec_dict = {}
    prec_dict = {}
    spec_dict = {}
    one_shot_str_dict = {True: 'one-shot', False: 'multi-shot'}
    for gamma in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        for one_shot in [True, False]:
            data_dict = pickle.load(open(filename.format(
                gamma, one_shot
                ), 'rb'))
            for data_key in data_dict:
                (gamma_in_file, one_shot_in_file,
                 noise_in_file) = data_key
                if gamma != gamma_in_file:
                    print('Gamma value {} does not match '
                          'in-file value {}'.format(
                              gamma, gamma_in_file
                              ))
                    break
                if one_shot != one_shot_in_file:
                    print('One-shot sampling indicator {} '
                          'does not match in-file value '
                          '{}'.format(
                              one_shot, one_shot_in_file
                              ))
                    break
                plot_key = r'{} $\sigma_Z = {}$'.format(
                    one_shot_str_dict[one_shot], noise_in_file
                    )
                if plot_key in rec_dict:
                    rec_dict[plot_key].append(
                        data_dict[data_key][0]
                        )
                else:
                    rec_dict[plot_key] = [data_dict[data_key][0]]
                if plot_key in prec_dict:
                    prec_dict[plot_key].append(
                        data_dict[data_key][1]
                        )
                else:
                    prec_dict[plot_key] = [data_dict[data_key][1]]
                if plot_key in spec_dict:
                    spec_dict[plot_key].append(
                        data_dict[data_key][2]
                        )
                else:
                    spec_dict[plot_key] = [data_dict[data_key][2]]
    columns = [0, 0.2, 0.4, 0.6, 0.8, 1]
    rec_df = pd.DataFrame.from_dict(
        rec_dict, orient='index', columns=columns
        )
    bar_plot(1000, rec_df, 'recall', 'hidden',
             save_file='glm-bslr-recall.pdf',
             x_axis_name=r'$\gamma$')
    prec_df = pd.DataFrame.from_dict(
        prec_dict, orient='index', columns=columns
        )
    bar_plot(1000, prec_df, 'precision',
             'outside', save_file='glm-bslr-precision.pdf',
             x_axis_name=r'$\gamma$')
    spec_df = pd.DataFrame.from_dict(
        spec_dict, orient='index', columns=columns
        )
    bar_plot(1000, spec_df, 'specificity',
             'hidden', save_file='glm-bslr-specificity.pdf',
             x_axis_name=r'$\gamma$')
    return


def plot_vary_replicate(noise):
    """Bar plot with varying replicates.

    Args:
        noise: int
            Observation noise level for data retrieval.  Can
            be either 0 (noiseless) or 1 (noisy).

    Returns: None
        Saves figures to files.
    """
    data_dir = ('/Users/veggente/Documents/workspace/python/'
                'bslr_20_gene/vary-replicates/')
    metric_list = ['recall', 'precision', 'specificity']
    loss_names = ['false negative rate',
                  'false discovery rate',
                  'false positive rate']
    loss_names_short = ['fnr', 'fdr', 'fpr']
    gamma_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    num_rep_list = [1, 2, 3, 5, 6, 10, 15]
    num_rep_names = [str(x)+' replicates' for x in num_rep_list]
    num_rep_names[0] = num_rep_names[0][:-1]
    for idx, metric in enumerate(metric_list):
        for one_shot in [True, False]:
            loss_dict = {}
            for idx_rep, num_rep in enumerate(num_rep_list):
                loss_dict[num_rep_names[idx_rep]] = []
                for gamma in gamma_list:
                    data_file = (
                        data_dir+'glm-bslr-g{}-o{}-b{}-n1000'
                        '.pkl'.format(gamma, one_shot, num_rep)
                        )
                    data = pickle.load(open(data_file, 'rb'))
                    loss_dict[num_rep_names[idx_rep]].append(
                        1-data[(gamma, one_shot,
                                noise, num_rep)][idx]
                        )
            loss_df = pd.DataFrame.from_dict(
                loss_dict, orient='index', columns=gamma_list
                )
            if metric == 'precision' and one_shot == True:
                loc = 'outside'
            else:
                loc = 'hidden'
            bar_plot(
                1000, loss_df, loss_names[idx], loc,
                ylim=[-0.05, 1.05], x_axis_name=r'$\gamma$',
                figsize=(5, 2), save_file=(
                    'bslr-{}-o{}-e{}.pdf'.format(
                        loss_names_short[idx], one_shot, noise
                        )
                    )
                )
    return


def plot_hetero_gamma():
    rpf_dict_full = pickle.load(open(
        '/Users/veggente/Documents/workspace/jupyter/notebooks/'
        'hetero.pkl', 'rb'
        ))
    num_sims = 1000
    figsize = (7, 4)
    columns = ['DEG to DEG', 'DEG to non-DEG', 'non-DEG to DEG',
               'non-DEG to non-DEG']
    fdr_dict = {key: 1-np.mean(rpf_dict_full[key][1], axis=0)
                for key in rpf_dict_full}
    fdr_df = pd.DataFrame.from_dict(
        fdr_dict, orient='index', columns=columns
        )
    fnr_dict = {key: 1-np.mean(rpf_dict_full[key][0], axis=0)
                for key in rpf_dict_full}
    fnr_df = pd.DataFrame.from_dict(
        fnr_dict, orient='index', columns=columns
        )
    fpr_dict = {key: 1-np.mean(rpf_dict_full[key][2], axis=0)
                for key in rpf_dict_full}
    fpr_df = pd.DataFrame.from_dict(
        fpr_dict, orient='index', columns=columns
        )
    bar_plot(num_sims, fdr_df, 'FDR', 'outside', figsize=figsize,
             ylim=[-0.05, 1.05], save_file=(
                 'hetero-gamma-fdr-n{}'
                 '.pdf'.format(num_sims)
                 ))
    bar_plot(num_sims, fnr_df, 'FNR', 'hidden', figsize=figsize,
             ylim=[-0.05, 1.05], save_file=(
                 'hetero-gamma-fnr-n{}'
                 '.pdf'.format(num_sims)
                 ))
    bar_plot(num_sims, fpr_df, 'FPR', 'hidden', figsize=figsize,
             ylim=[-0.05, 1.05], save_file=(
                 'hetero-gamma-fpr-n{}'
                 '.pdf'.format(num_sims)
                 ))
    return


def plot_plos_one_fig_5_6(
        num_sims=1000, rep_list=[1, 2, 3, 5, 6, 10, 15],
        gamma_list=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        one_shot_list=[True, False],
        noise_list=[0, 1],
        figsize=(5, 2), ylim=[-0.05, 1.05]
        ):
    """Plot PLOS ONE submission Figures 5 and 6.

    Args:
        num_sims: int, default 1000
            Number of simulations.
        rep_list: list, default is the list of all divisors of
            30 except 30 itself
            List of replicates.
        gamma_list: list, default is 0 to 1 with interval 0.2
            List of gamma values.  Need to have at least two
            elements.
        one_shot_list: list, default [True, False]
            Indicator of one-shot sampling.
        noise_list: list, default [0.0, 1.0]
            Noise levels.
        figsize: tuple, default (5, 2)
            Figure size.
        ylim: list, default [-0.05, 1.05]
            Y-axis range.  Can be None for an automatic range.

    Returns:
        Saves figures to files.
    """
    sigma = 1
    sig_level = 0.05
    legend_loc = 'outside'
    rep_text = (lambda x: '1 replicate' if x == 1
                else '{} replicates'.format(x))
    rep_legend = [rep_text(b) for b in rep_list]
    for one_shot in one_shot_list:
        for noise in noise_list:
            fdr_dict = {}
            fnr_dict = {}
            fpr_dict = {}
            for idx_r, num_rep in enumerate(rep_list):
                rep_name = rep_legend[idx_r]
                fdr_dict[rep_name] = []
                fnr_dict[rep_name] = []
                fpr_dict[rep_name] = []
                for gamma in gamma_list:
                    rec, prec, spec = eval_bslr(
                        sigma, gamma, num_rep, one_shot, noise,
                        sig_level, num_sims
                        )
                    fdr_dict[rep_name].append(1-np.mean(prec))
                    fnr_dict[rep_name].append(1-np.mean(rec))
                    fpr_dict[rep_name].append(1-np.mean(spec))
            fdr_df = pd.DataFrame.from_dict(
                fdr_dict, orient='index', columns=gamma_list
                )
            bar_plot(num_sims, fdr_df, 'FDR', legend_loc,
                     save_file='bslr-fdr-o{}-e{}.pdf'.format(
                         one_shot, noise
                         ), x_axis_name=r'$\gamma$',
                     figsize=figsize, ylim=ylim)
            legend_loc = 'hidden'
            fnr_df = pd.DataFrame.from_dict(
                fnr_dict, orient='index', columns=gamma_list
                )
            bar_plot(num_sims, fnr_df, 'FNR', legend_loc,
                     save_file='bslr-fnr-o{}-e{}.pdf'.format(
                         one_shot, noise
                         ), x_axis_name=r'$\gamma$',
                     figsize=figsize, ylim=ylim)
            fpr_df = pd.DataFrame.from_dict(
                fpr_dict, orient='index', columns=gamma_list
                )
            bar_plot(num_sims, fpr_df, 'FPR', legend_loc,
                     save_file='bslr-fpr-o{}-e{}.pdf'.format(
                         one_shot, noise
                         ), x_axis_name=r'$\gamma$',
                     figsize=figsize, ylim=ylim)
    return


def eval_bslr_on_locke(
        sampling_times, num_cond, num_rep, one_shot, sigma_co,
        sigma_bi, write_file, rand_seed=0, sig_level=0.05,
        output='', num_integration_interval=100, max_in_deg=3,
        rep_avg=True
        ):
    """Evaluate BSLR using network in Locke et al. MSB 2005.

    One environmental condition is modeled by the same set of
    initial conditions (values) of the gene expression levels,
    as well as the same condition-dependent nominal production
    variations.

    Args:
        sampling_times: array
            Sampling times as evenly spaced nonnegative
            numbers in an increasing order.
        num_cond: int
            Number of conditions.
        num_rep: int
            Number of replicates per single time.
        one_shot: bool
            True if one-shot, False if multi-shot.
        sigma_co: float
            Condition-dependent production variation level.
        sigma_bi: float
            Biological production variation level.
        write_file: bool
            Writes xml file if True.  Returns the adjacency
            matrix if False.
        rand_seed: int
            Random number generator seed.
        sig_level: float
            Significance level.
        output: str
            Output filename.
        num_integration_interval: int
            Number of integration intervals for the Ito
            integral, evenly spaced over
            [0, sampling_times[-1]].
        max_in_deg: int
            Maximum in-degree used in BSLR.
        rep_avg: bool
            Do replicate averaging if True.  Otherwise take
            replicates as different conditions.

    Returns:
        Saves graph file or return adjacency matrix.
    """
    # Create a shallow copy of the default parameters.
    param_test = get_locke_params()
    param_test['sigma_co'] = sigma_co*np.ones((3, 4))
    param_test['sigma_bi'] = sigma_bi*np.ones((3, 4))
    np.random.seed(rand_seed)
    # Generate data file.
    num_time = len(sampling_times)
    num_genes = 4
    mrna = np.empty((num_genes, num_cond*num_rep*num_time))
    tspan = np.linspace(0, sampling_times[-1],
                        num_integration_interval+1)
    if one_shot:
        num_rep_per_traj = num_rep*num_time
    else:
        num_rep_per_traj = num_rep
    for idx_cond in range(num_cond):
        # Generate the same 12-dimensional initial
        # conditions for all replicates.
        exp_init_per_rep = np.random.rand(12)
        exp_init = np.empty(12*num_rep_per_traj)
        for idx_rep in range(num_rep_per_traj):
            exp_init[idx_rep+np.arange(12)*num_rep_per_traj] = (
                exp_init_per_rep
                )
        # Entire solution over the fine tspan as a T-by-12R
        # matrix, where T = len(tspan) and R = num_rep_per_traj.
        exp_sol = sdeint.itoint(
            close(locke_drift, param_test),
            close(diff_coeff, param_test),
            exp_init, tspan
            )
        # Sampled expression levels at the coarse times,
        # approximated by the closest time in tspan.
        exp_sampled = exp_sol[[
            int(round(x)) for x in
            np.asarray(sampling_times) / sampling_times[-1]
            * num_integration_interval
            ], :]
        # Reshape the array.
        for i in range(num_genes):
            for j in range(num_time):
                start = idx_cond*num_rep*num_time+j*num_rep
                if one_shot:
                    mrna[i, start:start+num_rep] = (
                        exp_sampled[
                            j,
                            i*num_rep_per_traj+j*num_rep:
                            i*num_rep_per_traj+(j+1)*num_rep
                            ]
                        )
                else:
                    mrna[i, start:start+num_rep] = (
                        exp_sampled[j, i*num_rep:(i+1)*num_rep]
                        )
    sample_ids = ['c{}_t{}_r{}'.format(k, i, j)
                  for k in range(num_cond)
                  for i in range(num_time)
                  for j in range(num_rep)]
    mrna_df = pd.DataFrame(data=mrna, columns=sample_ids,
                           index=['G1', 'G2', 'G3', 'G4'])
    mrna_df.to_csv('exp-locke.csv')
    # Generate gene list file.
    np.savetxt('gene-list-locke.csv',
               [['G1', 'LHY'], ['G2', 'TOC1'],
                ['G3', 'X'], ['G4', 'Y']],
               fmt='%s', delimiter=',')
    # Generate condition list file.
    num_rep_alg = num_rep
    num_cond_alg = num_cond
    if rep_avg:
        conditions = list(range(num_cond))
    else:
        num_rep_alg = 1
        num_cond_alg = num_cond*num_rep
        conditions = list(range(num_cond_alg))
    json.dump([conditions, list(range(num_time))],
              open('cond-locke.json', 'w'), indent=4)
    # Generate design file.
    samples_df = pd.DataFrame(data=sample_ids)
    if rep_avg:
        samples_df['cond'] = samples_df[0].apply(
            lambda x: x.split('_')[0][1:]
            )
    else:
        samples_df['cond'] = samples_df[0].apply(
            lambda x: int(x.split('_')[0][1:])*num_rep
            + int(x.split('_')[2][1:])
            )
    samples_df['time'] = samples_df[0].apply(
        lambda x: x.split('_')[1][1:]
        )
    samples_df.to_csv('design-locke.csv', header=False,
                      index=False)

    if write_file:
        if not output:
            output = (
                'test-t{num_times}-c{num_cond}-bslr'
                '-s{sig_level}-r{rand_seed}.xml'.format(
                    num_times=num_time, sig_level=sig_level,
                    rand_seed=rand_seed, num_cond=num_cond_alg
                    )
                )
        # Run BSLR.
        causnet.main(
            '-c cond-locke.json '
            '-i gene-list-locke.csv -g {output} '
            '-x exp-locke.csv '
            '-P design-locke.csv '
            '-f {sig_level} '
            '-m {max_in_deg}'.format(
                output=output, num_times=num_time,
                sig_level=sig_level, max_in_deg=max_in_deg
                ).split()
            )
        return
    else:
        parser_dict = causnet.load_parser('design-locke.csv')
        adj_mat_sign_rec = causnet.bslr(
            parser_dict, mrna_df, num_cond_alg, num_time,
            num_genes, num_rep_alg, max_in_deg, sig_level
            )
        return adj_mat_sign_rec


def close(func, *args):
    """A nested function to convert parameterized function
    to an unparameterized one."""
    def newfunc(x, t):
        return func(x, t, *args)
    return newfunc


def locke_drift(x, t, p):
    """Drift array of the SDE adapted from Locke et al. MSB 2005
    for one condition.

    Args:
        x: array
            Gene expression levels (mRNA abundances and protein
            concentrations).  len(x) must be a multiple of 12.
            x[0]: mRNA abundance of LHY, rep 1.
            x[1]: mRNA abundance of LHY, rep 2.
            ...
            x[r-1]: mRNA abundance of LHY, rep r.
            x[r]: mRNA abundance of TOC1, rep 1.
            x[r+1]: mRNA abundance of TOC1, rep 2.
            ...
            x[2r-1]: mRNA abundance of TOC1, rep r.
            x[2r]: mRNA abundance of X, rep 1.
            ...
            ...
            x[4r-1]: mRNA abundance of Y, rep r.
            x[4r]: cytoplasmic protein concentration of LHY, rep 1.
            ...
            ...
            x[8r-1]: cytoplasmic protein concentration of Y, rep r.
            x[8r]: nuclear protein concentration of LHY, rep 1.
            ...
            ...
            x[12r-1]: nuclear protein concentration of Y, rep r.
        t: float
            Time.
        p: dict
            Parameters.
    """
    assert((len(x)/3/4).is_integer())
    num_rep = int(len(x)/3/4)
    drift = np.empty(len(x))
    # Replicates.
    for idx_rep in range(num_rep):
        # Extract relevant data for the replicate.
        mrna = x[idx_rep+np.arange(4)*num_rep]
        cprot = x[4*num_rep+idx_rep+np.arange(4)*num_rep]
        nprot = x[8*num_rep+idx_rep+np.arange(4)*num_rep]
        # Three levels of expression (mRNA, cytoplasmic protein,
        # nuclear protein).
        for idx_lvl in range(3):
            # Four genes.
            for idx_gene in range(4):
                drift[idx_lvl*4*num_rep
                      + idx_gene*num_rep
                      + idx_rep] = locke_drift_lite(
                          idx_lvl, idx_gene, mrna, cprot,
                          nprot, p
                          )
    return np.asarray(drift)


def locke_drift_lite(idx_lvl, idx_gene, mrna, cprot, nprot, p):
    """Scalar drift of Locke et al. MSB 2005.

    Args:
        idx_lvl: int
            Index of expression level.
                0: mRNA.
                1: cytoplasmic protein.
                2: nuclear protein.
        idx_gene: int
            Index of gene.
        mrna: array
            mRNA abundances.
        cprot: array
            Cytoplasmic protein concentrations.
        nprot: array
            Nuclear protein concentrations.
        p: dict
            Parameters.

    Returns: float
        Drift including production and degradation.
    """
    if idx_lvl == 0:
        # mRNA.
        if idx_gene == 0:
            drift = (
                hill(nprot[2], p['g1'], p['a'], p['n1'], True)
                - hill(mrna[0], p['k1'], 1, p['m1'], True)
                )
        elif idx_gene == 1:
            drift = (
                hill(nprot[3], p['g2'], p['b'], p['n2'], True)
                *hill(nprot[0], p['g3'], p['c'], 1, False)
                - hill(mrna[1], p['k4'], 1, p['m4'], True)
                )
        elif idx_gene == 2:
            drift = (
                hill(nprot[1], p['g4'], p['d'], p['n3'], True)
                - hill(mrna[2], p['k7'], 1, p['m9'], True)
                )
        elif idx_gene == 3:
            drift = (
                hill(nprot[1], p['g5'], p['e'], p['n5'], False)
                *hill(nprot[0], p['g6'], p['f'], 1, False)
                - hill(mrna[3], p['k10'], 1, p['m12'], True)
                )
        else:
            raise ValueError('Unrecognized gene')
    elif idx_lvl == 1:
        # Cytoplasmic protein.
        if idx_gene == 0:
            drift = (
                p['p1']*mrna[0]
                - p['r1']*cprot[0]
                + p['r2']*nprot[0]
                - hill(cprot[0], p['k2'], 1, p['m2'], True)
                )
        elif idx_gene == 1:
            drift = (
                p['p2']*mrna[1]
                - p['r3']*cprot[1]
                + p['r4']*nprot[1]
                - ((p['m5']+p['m6'])
                   *hill(cprot[1], p['k5'], 1, 1, True))
                )
        elif idx_gene == 2:
            drift = (
                p['p3']*mrna[2]
                - p['r5']*cprot[2]
                + p['r6']*nprot[2]
                - hill(cprot[2], p['k8'], 1, p['m10'], True)
                )
        elif idx_gene == 3:
            drift = (
                p['p4']*mrna[3]
                - p['r7']*cprot[3]
                + p['r8']*nprot[3]
                - hill(cprot[3], p['k11'], 1, p['m13'], True)
                )
        else:
            raise ValueError('Unrecognized gene')
    elif idx_lvl == 2:
        # Nuclear protein.
        if idx_gene == 0:
            drift = (
                p['r1']*cprot[0]
                - p['r2']*nprot[0]
                - hill(nprot[0], p['k3'], 1, p['m3'], True)
                )
        elif idx_gene == 1:
            drift = (
                p['r3']*cprot[1]
                - p['r4']*nprot[1]
                - hill(nprot[1], p['k6'], 1, p['m7']+p['m8'],
                       True)
                )
        elif idx_gene == 2:
            drift = (
                p['r5']*cprot[2]
                - p['r6']*nprot[2]
                - hill(nprot[2], p['k9'], 1, p['m11'], True)
                )
        elif idx_gene == 3:
            drift = (
                p['r7']*cprot[3]
                - p['r8']*nprot[3]
                - hill(nprot[3], p['k12'], 1, p['m14'], True)
                )
        else:
            raise ValueError('Unrecognized gene')
    return drift


def hill(x, k, h, beta, activation):
    """Hill function.

    Args:
        x: float
            Concentration.
        k: float
            Michaelis-Menten coefficient.
        h: float
            Hill coefficient.
        beta: float
            Maximum activation.
        activation: bool
            True: activation.
            False: repression.

    Returns: float
        Activation level.
        """
    if activation:
        act_lvl = x**h/(k**h+x**h)*beta
    else:
        act_lvl = k**h/(k**h+x**h)*beta
    return act_lvl


def diff_coeff(x, t, p):
    """Diffusion coefficient.

    Args:
        x: array
            Gene expression levels (mRNA abundances and protein
                concentrations).
            The dimension is 12r, where r is the number of
                replicates.
            x[0]: mRNA abundance of LHY, rep 1.
            x[1]: mRNA abundance of LHY, rep 2.
            ...
            x[r-1]: mRNA abundance of LHY, rep r.
            x[r]: mRNA abundance of TOC1, rep 1.
            x[r+1]: mRNA abundance of TOC1, rep 2.
            ...
            x[2r-1]: mRNA abundance of TOC1, rep r.
            x[2r]: mRNA abundance of X, rep 1.
            ...
            ...
            x[4r-1]: mRNA abundance of Y, rep r.
            x[4r]: cytoplasmic protein concentration of LHY,
                rep 1.
            ...
            ...
            x[8r-1]: cytoplasmic protein concentration of Y,
                rep r.
            x[8r]: nuclear protein concentration of LHY, rep 1.
            ...
            ...
            x[12r-1]: nuclear protein concentration of Y,
                rep r.
        t: float
            Time.
        p: dict
            Parameters.

    Returns: array
        The N-by-(4*N/3) matrix for the condition, where
        N = len(x).
    """
    assert((len(x)/3/4).is_integer())
    num_rep = int(len(x)/3/4)
    diff_mat = np.zeros((len(x), 12*(num_rep+1)))
    lambda_mat = np.empty((num_rep, num_rep+1))
    lambda_mat[:, 0] = np.ones(num_rep)
    lambda_mat[:, 1:] = np.identity(num_rep)
    for idx_lvl in range(3):
        for idx_gene in range(4):
            start_pos = idx_lvl*4*num_rep+idx_gene*num_rep
            exp_lvl = x[start_pos:start_pos+num_rep]
            horizontal_start_pos = int(
                start_pos*(num_rep+1)/num_rep
                )
            sigma_mat = np.zeros((num_rep+1, num_rep+1))
            sigma_mat[0, 0] = p['sigma_co'][idx_lvl,
                                                idx_gene]
            sigma_mat[1:, 1:] = (
                p['sigma_bi'][idx_lvl, idx_gene]
                * np.identity(num_rep)
                )
            diff_mat[
                start_pos:start_pos+num_rep,
                horizontal_start_pos:
                horizontal_start_pos+num_rep+1
                ] = np.diag(exp_lvl).dot(lambda_mat).dot(
                    sigma_mat
                    )
    return diff_mat


def get_locke_params():
    """Get parameters for the SDE from the supplementary table
    of Locke et al.

    Args: None

    Returns: dict
        Parameters.
    """
    return {'q1': 2.4514,
            'n1': 5.1694,
            'a': 3.3064,
            'g1': 0.8767,
            'm1': 1.5283,
            'k1': 1.8170,
            'p1': 0.8295,
            'r1': 16.8363,
            'r2': 0.1687,
            'm2': 20.4400,
            'k2': 1.5644,
            'm3': 3.6888,
            'k3': 1.2765,
            'n2': 3.0087,
            'b': 1.0258,
            'g2': 0.0368,
            'g3': 0.2658,
            'c': 1.0258,
            'm4': 3.8231,
            'k4': 2.5734,
            'p2': 4.3240,
            'r3': 0.3166,
            'r4': 2.1509,
            'm5': 0.0013,
            'm6': 3.1741,
            'k5': 2.7454,
            'm7': 0.0492,
            'm8': 4.0424,
            'k6': 0.4033,
            'n3': 0.2431,
            'd': 1.4422,
            'g4': 0.5388,
            'm9': 10.1132,
            'k7': 6.5585,
            'p3': 2.1470,
            'r5': 1.0352,
            'r6': 3.3017,
            'm10': 0.2179,
            'k8': 0.6632,
            'm11': 3.3442,
            'k9': 17.1111,
            'q2': 2.4017,
            'n4': 0.0857,
            'n5': 0.1649,
            'g5': 1.1780,
            'g6': 0.0645,
            'e': 3.6064,
            'f': 1.0237,
            'm12': 4.2970,
            'k10': 1.7303,
            'p4': 0.2485,
            'r7': 2.2123,
            'r8': 0.2002,
            'm13': 0.1347,
            'k11': 1.8258,
            'm14': 0.6114,
            'k12': 1.8066,
            'p5': 0.5000,
            'k13': 1.2000,
            'm15': 1.2000,
            'q3': 1.0000,
            # Multipliers of condition-dependent and biological
            # components of nominal production variations.
            'sigma_co': 0.1*np.ones((3, 4)),
            'sigma_bi': 0.1*np.ones((3, 4))}


def eval_bslr_multi_sims(num_sims, one_shot, sigma_co,
                         sigma_bi, num_reps=3, rep_avg=True):
    """Evaluate BSLR with multiple simulations.

    Args:
        num_sims: int
            Number of simulations.
        one_shot: bool
            True for one-shot sampling.  False for multi-shot.
        sigma_co: float
            Condition-dependent variation level.
        sigma_bi: float
            Biologically variation level.
        num_reps: int
            Number of replicates.
        rep_avg: bool
            Do replicate averaging if True.  Otherwise take
            replicates as different conditions.

    Returns: None
        Prints number of defined FDR, and average FDR, FNR, FPR.
    """
    # Each row is for the same regulator.  See, e.g., line 325
    # of bio_data_gen.py (CausNet commit 0f67f57) for details.
    adj_mat_true = np.asarray([
        [0, -1, 0, -1], [0, 0, 1, -1],
        [1, 0, 0, 0], [0, 1, 0, 0]
        ]).astype(float)
    fdr = []
    fnr = []
    fpr = []
    for i in tqdm(range(num_sims)):
        # Use None as rand_seed to have different result for
        # each simulation.
        adj_mat_rec = eval_bslr_on_locke(
            list(range(0, 12, 2)), 1, num_reps, one_shot,
            sigma_co, sigma_bi, False, rand_seed=None,
            num_integration_interval=1000, max_in_deg=2,
            rep_avg=rep_avg
            )
        r, p, s = get_sas(adj_mat_rec, adj_mat_true)
        if not np.isnan(p):
            fdr.append(1-p)
        fnr.append(1-r)
        fpr.append(1-s)
    print(len(fdr))
    print(np.mean(fdr), np.mean(fnr), np.mean(fpr))
    return
