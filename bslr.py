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
