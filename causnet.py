#!/usr/bin/env python3
"""Gene regulatory network analysis.
"""
import networkx as nx
import matplotlib.pyplot as plt
import sys
import getopt
import numpy as np
from tqdm import tqdm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os
import json
import ast
import math
import scipy.stats
import itertools

import causnet_bslr as ca


def main(argv):
    # TODO: Remove default expression file.
    expression_file = 'expression-2011.csv'
    # Indicator for perturbation analysis.
    is_perturb = False
    # TODO: Remove default factor selection.
    photoperiod_set = ['LD', 'SD', 'Sh']
    strain_set = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']
    time_point_set = ['T1', 'T3', 'T5']
    cond_list = [
        [16, 25, 32],
        ['LD', 'SD', 'Sh'],
        [1, 2, 3, 4, 5],
        ['II1', 'II2', 'II3', 'II4', 'II5', 'II6']
        ]
    # Self-regulation in Granger causality inference.
    self_reg = True
    num_time_lags = 1
    # TODO: only required by BSLR and oCSE, not SBL.
    max_in_degree = 3
    significance_level = 0.05
    gene_list_file = ''
    # TODO: Remove obsolete NetworkX graphing and its options.
    font_size = 14
    width_max = 2
    node_size = 500
    node_color = 'w'
    node_shape = 's'
    border_width = 0
    edge_threshold = 0.0
    is_plotting = False
    # TODO: Test compatibility of gene clustering.
    clustering_indicator = False
    # TODO: Remove default number of replicates; extract from data.
    num_replicates = 3
    # TODO: Remove obsolete position file.
    pos_file = ''
    # TODO: Remove obsolete JSON input and output files.
    json_out_file = ''
    json_in_file = ''
    # TODO: Test validity of time-dependent standardization.
    tds = False
    graphml_file = 'grn.xml'
    # Virtual time shift.
    vts = 0
    f_test = False
    # Experiment design file (a.k.a. sample ID parser file).
    parser_file = ''
    # TODO: Use argparse instead of getopt.
    # TODO: Remove obsolete options and document new options.
    # Network inference algorithm.
    algorithm = 'causnet'
    # Number of permutations in the permutation test for oCSE.
    num_perm = 100
    sparsity_threshold = 3.0
    epsilon = 0.4
    try:
        opts, args = getopt.getopt(
            argv, 'p:c:i:sl:m:f:r:u:o:e:a:t:ndT:M:E:g:v:Fx:P:A:R:S:I:'
            )
    except getopt.GetoptError as e:
        print(str(e))
        # TODO: Move Arabidopsis flowering data generation to
        # bio-data-gen.py.
        print(
            """Usage: ./soybean.py [-r] <random_seed> [-c] <cond_list_file>
            [-p] <num_perturb> [-l] <num_time_lags> [-m]
            <max_in_degree> [-f] <significance_level> [-i]
            <gene_list_file> [-u] <clustering_file>
            [-o] <position_out_file> [-e] <json_out_file>
            [-a] <json_in_file> [-t] <edge_threshold> [-d] [-T]
            <num_times> [-M] <num_experiments> [-E] <num_extra_genes>
            [-g] <GraphML_file> [-v] <num_virtual_times> [-F] [-x]
            <expression_file> [-P] <parser_file> [-A] <algorithm>
            [-R] <num_perm> [-S] <sparsity_threshold> [-I]
            <espilon_sigma>

            -r      Pseudorandom number generator seed.
            -c      Condition list file.
                    This is a JSON format file of a list of lists
                    specifying the conditions of the samples to do
                    network analysis on. The order of the lists should
                    be compatible with the parser.
            -l      The number of time lags for network inference. The
                    default is 1.
            -m      The maximum in-degree ofthe network. The default is
                    3.
            -f      The significance level for edge rejection based on
                    Granger causality. The default is 0.05.
            -u      Input file to specify gene clusters.
            -v      Virtual time shift: replicate the first times and
                    append them to the end in order to close the loop
                    from the last time to the first times the next
                    day. Default is one virtual time.
            -F      F-test for one-way ANOVA.
            -x      Normalized gene expression file in CSV format.
            -P      Sample ID parser file.
                    Parse the sample IDs to get the conditions.
                    The last condition must be the sample time.
                    The other are the unstructured conditions to
                    create experimental perturbation, which can
                    also be selected as subset of data. At least
                    two replicates are needed for each sample
                    condition for the perturbation analysis.
            -A      Select network inference algorithm.
                    Can be 'causnet', 'ocse' or 'sbl'.  Default is
                    'causnet'.
            -R      Number of permutations in the permutation test.
                    Only required for 'ocse' algorithm.
            -S      Sparsity threshold for SBL algorithm.
            -I      Epsilon for sigma squared estimation in SBL
                    algorithm.
            """
            )
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-r':
            np.random.seed(int(arg))
        elif opt == '-p':
            is_perturb = True
            num_perturb = int(arg)
        # Single photoperiod condition.
        elif opt == '-c':
            with open(arg, 'r') as f:
                cond_list = json.load(f)
                # Convert numericals to strings.
                cond_list_temp = []
                for sub_list in cond_list:
                    cond_list_temp.append([
                        str(element) for element in sub_list
                        ])
                cond_list = cond_list_temp
        #elif opt == '-s':
        #    self_reg = True
        elif opt == '-l':
            num_time_lags = int(arg)
        elif opt == '-m':
            max_in_degree = int(arg)
        elif opt == '-f':
            significance_level = float(arg)
        elif opt == '-i':
            gene_list_file = arg
        elif opt == '-u':
            clustering_indicator = True
            clustering_file = arg
            # Clustering turns off position outputing.
            pos_file = ''
        elif opt == '-o':
            pos_file = arg
            assert(not os.path.exists(pos_file))
            # Position outputing turns off clustering.
            clustering_indicator = False
            clustering_file = ''
        elif opt == '-e':
            json_out_file = arg
            assert(not os.path.exists(json_out_file))
            json_in_file = ''
        elif opt == '-a':
            json_in_file = arg
            assert(os.path.exists(json_in_file))
            json_out_file = ''
        elif opt == '-t':
            edge_threshold = float(arg)
        elif opt == '-d':
            tds = True
        elif opt == '-T':
            num_times = int(arg)
        elif opt == '-M':
            num_exp = int(arg)
            # Settings for Arabidopsis artificial data.
            photoperiod_set = ['PP'+str(idx_exp+1) for idx_exp in
                               np.arange(num_exp)]
            strain_set = ['G1']
            time_point_set = ['T'+str(idx_time+1) for idx_time in
                              np.arange(num_times)]
        elif opt == '-E':
            num_extra_genes = int(arg)
            expression_file = (
                'data/bio-data-gen' + '-t' + str(num_times) + '-m'
                + str(num_exp) + '-e' + str(num_extra_genes) + '.csv'
                )
        elif opt == '-g':
            graphml_file = arg
        elif opt == '-v':
            # The default number of virtual time shifts is 1.
            # Otherwise specify using an argument.
            if arg:
                vts = int(arg)
            else:
                vts = 1
        elif opt == '-F':
            f_test = True
        elif opt == '-x':
            expression_file = arg
        elif opt == '-P':
            parser_file = arg
        elif opt == '-A':
            algorithm = arg
            alg_set = ['causnet', 'ocse', 'sbl']
            if not algorithm in alg_set:
                print('Inference algorithm not recognized.  '
                      'Please choose one of the following.')
                for alg in alg_set:
                    print(alg)
                sys.exit(1)
        elif opt == '-R':
            num_perm = int(arg)
        elif opt == '-S':
            sparsity_threshold = float(arg)
        elif opt == '-I':
            epsilon = float(arg)
        # No other cases.
    # TODO: Add more input checking.
    if not gene_list_file:
        print('Please specify a gene list file using [-i].')
        exit(1)
    gene_list = read_gene(gene_list_file)
    if not json_in_file:
        parser_dict = load_parser(parser_file)
        # TODO: Don't need data_var if not doing perturbation
        # analysis.
        data_cell, data_var = extract_data_new(
            gene_list, expression_file, cond_list, parser_dict
            )
        if vts:
            data_cell = virtual_time_shift(data_cell, vts)
            data_var = virtual_time_shift(data_var, vts)
    if clustering_indicator:
        # Read from clustering_file.
        with open(clustering_file, 'r') as f:
            for line in f:
                # Make a copy of the original gene list.
                genes_temp = gene_list.genes[:]
                words = line.strip().split()
                # Each line should consist of two gene IDs, a gene pair name
                # and two positioning coordinates.
                assert(len(words) == 5)
                cluster_pair = words[0:2]
                cluster_name = words[2]
                cluster_pos = words[3:]
                # Find the two indices for the cluster.
                genes_extract = []
                cluster_indices = []
                for idx_gene, gene in enumerate(gene_list.genes):
                    if gene.id in cluster_pair:
                        genes_extract.append(gene)
                        genes_temp.remove(gene)
                        cluster_indices.append(idx_gene)
                # Modify gene list.
                cluster_id_list = [gene.id for gene in genes_extract]
                cluster_id_list.sort()
                cluster_id = '/'.join(cluster_id_list)
                cluster = Gene(cluster_id, cluster_name,
                               float(cluster_pos[0]), float(cluster_pos[1]))
                genes_temp.append(cluster)
                gene_list.genes = genes_temp
                assert(len(cluster_indices) == 2)
                if not json_in_file:
                    # Need to use index to modify data_cell and data_var.
                    for idx_data in range(len(data_cell)):
                        # For each experiment, extract those two columns in
                        # the cluster.
                        mean_mat = data_cell[idx_data]
                        var_mat = data_var[idx_data]
                        mean_extract = mean_mat[:, cluster_indices]
                        var_extract = var_mat[:, cluster_indices]
                        mean_mat = np.delete(mean_mat, cluster_indices, 1)
                        var_mat = np.delete(var_mat, cluster_indices, 1)
                        # Calculate new columns to append.
                        mean_new = (mean_extract[:, 0]
                                    + mean_extract[:, 1]) / 2
                        # Estimator of the variance of the mean.
                        var_new = (
                            (var_extract[:, 0]+var_extract[:, 1])
                            * (num_replicates-1) / (4*num_replicates-2)
                            + (mean_extract[:, 0]-mean_extract[:, 1])**2
                            / (8*num_replicates-4)
                            )
                        mean_mat = np.concatenate((
                            mean_mat, mean_new.reshape([mean_mat.shape[0],
                                                        1])
                            ), axis=1)
                        var_mat = np.concatenate((
                            var_mat, var_new.reshape([var_mat.shape[0], 1])
                            ), axis=1)
                        data_cell[idx_data] = mean_mat
                        data_var[idx_data] = var_mat
    # TODO: Remove the now obsolete -d and -g options.
    if json_in_file:
        # Import from file.
        data = load_dict(json_in_file)
        # data should contain a dictionary for weight and another one for
        # sign.
        assert(len(data) == 2)
        weight, sign_dict = data
    else:
        # TODO: Can this be combined with the previous "not json_in_file"
        # case?
        if not is_perturb:
            if algorithm == 'causnet':
                caspian_out = ca.caspian(
                    data_cell, num_time_lags, max_in_degree,
                    significance_level, self_reg, tds
                    )
            elif algorithm == 'ocse':
                caspian_out = ca.ocse(
                    data_cell, num_perm, significance_level,
                    max_in_degree
                    )
            elif algorithm == 'sbl':
                caspian_out = ca.sbl_grn(
                    data_cell,
                    sparsity_threshold=sparsity_threshold,
                    sigma_eps=epsilon
                    )
            else:
                print("Unknown algorithm.")
                return 1
            # TODO: Make returning p-values is compatible with the
            # ocse algorithm.
            if significance_level:
                parents, signs = caspian_out
            else:
                parents, signs, p_values = caspian_out
            weight = {}
            sign_dict = {}
            for gene, parents_for_gene in enumerate(parents):
                for idx_p, parent in enumerate(parents_for_gene):
                    target_id = gene_list.genes[gene].id
                    parent_id = gene_list.genes[parent].id
                    if significance_level:
                        weight[target_id, parent_id] = 1.0
                    else:
                        # TODO: Remove magic number.
                        weight[target_id, parent_id] = thickness(
                            p_values[gene][idx_p], 0.05
                            )
                    sign_dict[target_id, parent_id] = (
                        float(signs[gene][idx_p])
                        )
        else:
            edge_count = {}
            sign_count = {}
            sign_cnt_total = {}
            if not significance_level:
                p_values_agg = {}
            for idx_perturb in tqdm(range(num_perturb)):
                perturbed_data_cell = []
                for idx_page, data_page in enumerate(data_cell):
                    random_matrix = np.random.randn(data_page.shape[0],
                                                    data_page.shape[1])
                    perturbed_data_cell.append(
                        data_page
                        + np.sqrt(data_var[idx_page])*random_matrix
                        )
                if algorithm == 'causnet':
                    caspian_out = ca.caspian(
                        perturbed_data_cell, num_time_lags, max_in_degree,
                        significance_level, self_reg, tds
                        )
                elif algorithm == 'ocse':
                    caspian_out = ca.ocse(
                        perturbed_data_cell, num_perm,
                        significance_level, max_in_degree
                        )
                elif algorithm == 'sbl':
                    caspian_out = ca.sbl_grn(
                        perturbed_data_cell,
                        sparsity_threshold=sparsity_threshold,
                        sigma_eps=epsilon
                        )
                # TODO: Make returning p-values is compatible with the
                # ocse algorithm.
                if significance_level:
                    parents, signs = caspian_out
                else:
                    parents, signs, p_values = caspian_out
                for gene, parents_for_gene in enumerate(parents):
                    for idx_parent, parent in enumerate(parents_for_gene):
                        # TODO: Fix uniform weight for p-value thickness.
                        if (gene, parent) in edge_count:
                            edge_count[gene, parent] += 1
                        else:
                            edge_count[gene, parent] = 1
                        if (gene, parent) in sign_count:
                            sign_count[gene, parent] += (
                                signs[gene][idx_parent]
                                )
                            sign_cnt_total[gene, parent] += 1
                        else:
                            sign_count[gene, parent] = (
                                signs[gene][idx_parent]
                                )
                            sign_cnt_total[gene, parent] = 1
                        if not significance_level:
                            if (gene, parent) in p_values_agg:
                                p_values_agg[gene, parent].append(
                                    p_values[gene][idx_parent]
                                    )
                            else:
                                p_values_agg[gene, parent] = [
                                    p_values[gene][idx_parent]
                                    ]
            weight = {}
            for gene, parent in edge_count:
                target_id = gene_list.genes[gene].id
                parent_id = gene_list.genes[parent].id
                if significance_level:
                    weight[target_id, parent_id] = (
                        edge_count[gene, parent]/num_perturb
                        )
                else:
                    weight[target_id, parent_id] = thickness(fisher(
                        p_values_agg[gene, parent]
                        ), 0.05)
            sign_dict = {(gene_list.genes[u].id, gene_list.genes[v].id):
                         float(np.sign(sign_count[u, v]))
                         for u, v in sign_count}
        # Export JSON format file.
        if json_out_file:
            dump_dict([weight, sign_dict], json_out_file)
    if graphml_file:
        gene_network = create_graph(gene_list, weight, sign_dict)
        #add_source(gene_network)
        nx.write_graphml(gene_network, graphml_file)
    if is_plotting:
        if significance_level:
            thickness_type = 'multiplicity-ratio'
        else:
            thickness_type = 'p-value'
        draw_gene_network(gene_list, weight, sign_dict, font_size,
                          width_max, node_size, node_color, node_shape,
                          border_width, pos_file, edge_threshold,
                          thickness_type)


def generate_xlsx(
        gene_list_filename, expression_filename, exp_output_file='',
        output_format='', is_perturb=False, photoperiod_set='LD'
        ):
    """OBSOLETE: Generate the .xlsx or the .csv file for CaSPIAN."""
    if is_perturb:
        print('Generating perturbed data...')
    else:
        print('Generating unperturbed data...')
    num_replicates = 3
    # Read the list of gene names.
    genes_n_labels = read_flower_genes(gene_list_filename)
    genes = genes_n_labels[::2]
    labels = genes_n_labels[1::2]
    # Get the expression levels from Minglei.
    expressions = {}
    strain_set = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']
    time_point_set = ['T1', 'T3', 'T5']
    with open(expression_filename, 'r') as f:
        # Remove trailing newline and leading empty ID.
        sample_ids = f.readline().strip().split(',')[1:]
    for index, gene in enumerate(genes):
        with open(expression_filename, 'r') as f:
            for line in f:
                if gene in line:
                    # Here we assume each gene maps to a unique line.
                    expressions[gene] = (line.strip().split(','))
                    break
    avg_expression_level = {}
    for sid_index, sid in enumerate(sample_ids):
        id_split = sid.split('_')
        photoperiod = id_split[0]
        strain = id_split[1]
        sample_time = id_split[2]
        for gene in genes:
            expression_key = (gene, photoperiod, strain, sample_time)
            if expression_key in avg_expression_level:
                avg_expression_level[expression_key].append(
                    float(expressions[gene][sid_index+1])
                    )
            else:
                avg_expression_level[expression_key] = [
                    float(expressions[gene][sid_index+1])
                    ]
    if output_format == 'xlsx':
        # Output the xlsx file.
        for photoperiod in photoperiod_set:
            for strain in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']:
                with open(photoperiod+strain+'.csv', 'w') as f:
                    for gene in genes:
                        f.write(
                            ','.join([
                                str(avg_expression_level[(
                                    gene, photoperiod, strain, 'T1'
                                    )]),
                                str(avg_expression_level[(
                                    gene, photoperiod, strain, 'T3'
                                    )]),
                                str(avg_expression_level[(
                                    gene, photoperiod, strain, 'T5'
                                    )])
                                ])+'\n'
                            )
        return None
    elif output_format == 'csv':
        num_time_points = 3
        with open(exp_output_file, 'w') as f:
            for photoperiod in photoperiod_set:
                for strain in strain_set:
                    f.write(str(num_time_points)+'\n')
                    for gene in genes:
                        exp_lvl_1 = avg_expression_level[
                            (gene, photoperiod, strain, 'T1')
                            ]
                        exp_lvl_2 = avg_expression_level[
                            (gene, photoperiod, strain, 'T3')
                            ]
                        exp_lvl_3 = avg_expression_level[
                            (gene, photoperiod, strain, 'T5')
                            ]
                        avg_exp_lvl_1 = np.mean(exp_lvl_1)
                        avg_exp_lvl_2 = np.mean(exp_lvl_2)
                        avg_exp_lvl_3 = np.mean(exp_lvl_3)
                        if is_perturb:
                            avg_exp_lvl_1 += (
                                np.random.normal()
                                * np.sqrt(sample_mean_var(exp_lvl_1))
                                )
                            avg_exp_lvl_2 += (
                                np.random.normal()
                                * np.sqrt(sample_mean_var(exp_lvl_2))
                                )
                            avg_exp_lvl_3 += (
                                np.random.normal()
                                * np.sqrt(sample_mean_var(exp_lvl_3))
                                )
                        f.write(','.join([
                            str(avg_exp_lvl_1), str(avg_exp_lvl_2),
                            str(avg_exp_lvl_3)
                            ])+'\n')
        return None
    # If output_format is an empty string, the Python-integrated script
    # caspian.py is used instead of the MATLAB script written by Amin Emad.
    # No file output is necessary. A tuple of two lists of matrices are
    # returned.
    elif not output_format:
        num_photoperiods = len(photoperiod_set)
        num_strains = len(strain_set)
        num_time_points = len(time_point_set)
        num_genes = len(genes)
        data_cell = []
        data_var = []
        exp_count = 0
        for idx_p, pp in enumerate(photoperiod_set):
            for idx_s, strain in enumerate(strain_set):
                data_cell.append(np.empty((num_time_points, num_genes)))
                data_var.append(np.empty((num_time_points, num_genes)))
                for idx_g, gene in enumerate(genes):
                    for idx_t, time_point in enumerate(time_point_set):
                        expression_key = (gene, pp, strain, time_point)
                        replicates = avg_expression_level[expression_key]
                        data_cell[exp_count][idx_t, idx_g] = np.mean(
                            replicates
                            )
                        data_var[exp_count][idx_t, idx_g] = sample_mean_var(
                            replicates
                            )
                exp_count += 1
    return data_cell, data_var


def read_flower_genes(filename):
    """OBSOLETE: Read flower genes from file."""
    first_two_cols = []
    with open(filename, 'r') as f:
        data = f.read().split()
        for i in range(len(data)//4):
            first_two_cols.append(data[i*4])
            first_two_cols.append(data[i*4+1])
    return first_two_cols


def read_gene_positions(filename):
    """OBSOLETE: Read gene positions from file."""
    pos = {}
    with open(filename, 'r') as f:
        data = f.read().split()
        for i in range(len(data)//4):
            pos[data[i*4+1]] = (int(data[i*4+2]), int(data[i*4+3]))
    return pos


def draw_gene_network(labels, gene_pos, adjlist_file='', zero_based_list=[],
                      weight_dict={}, sign_count={}):
    """OBSOLETE: Draw a gene network."""
    # If no adjacency list files are given, a zero-based list of parents
    # or a weight dictionary must be provided.
    if not adjlist_file:
        if zero_based_list:
            num_genes = len(zero_based_list)
            gene_network = nx.DiGraph()
            # Use 1-based nodes.
            nodes = [str(i) for i in range(1, num_genes + 1)]
            gene_network.add_nodes_from(nodes)
            for idx_gene, parents in enumerate(zero_based_list):
                for parent in parents:
                    # Compensate for the 1-based nodes.
                    gene_network.add_edge(str(parent + 1), str(idx_gene + 1))
        else:
            gene_network = nx.DiGraph()
            for gene, parent in weight_dict:
                gene_network.add_edge(
                    str(parent+1), str(gene+1),
                    weight=weight_dict[gene, parent]
                    )
    else:
        # Read graph from adjacency list.
        reverse_gene_network = nx.read_adjlist(
            adjlist_file, create_using=nx.DiGraph(), delimiter=' '
            )
        gene_network = reverse_gene_network.reverse()
    # Draw graph.
    gene_network_relabeled = nx.relabel_nodes(gene_network, labels)
    inv_labels = {v: k for k, v in labels.items()}
    nx.draw_networkx_nodes(gene_network_relabeled, gene_pos, node_size=500,
                           node_color='w', node_shape='s', linewidths=0)
    if weight_dict:
        weights_relabeled = []
        edge_colors = []
        for u, v in gene_network_relabeled.edges():
            gene, parent = int(inv_labels[v]) - 1, int(inv_labels[u]) - 1
            # Weights are between 0 and 2.
            weight = weight_dict[gene, parent] * 2
            weights_relabeled.append(weight)
            if sign_count[gene, parent] > 0:
                edge_colors.append('b')
            else:
                edge_colors.append('r')
        nx.draw_networkx_edges(gene_network_relabeled, gene_pos,
                               width=weights_relabeled,
                               edge_color=edge_colors)
    else:
        nx.draw_networkx_edges(gene_network_relabeled, gene_pos)
    nx.draw_networkx_labels(gene_network_relabeled, gene_pos, font_size=14)
    plt.show()


def file_suffix(path):
    """OBSOLETE: Generate a file suffix."""
    return ''


def generate_exp_filename(path, gene_file):
    """OBSOLETE: Generate gene expression level filename."""
    return (
        path+'gene-expression-level'
        + file_suffix(gene_file)+'.csv'
        )


def sample_var(x):
    """Sample variance of a list."""
    if len(x) > 1:
        return np.var(x)/(1-1/len(x))
    else:
        return 0.0


def sample_mean_var(x):
    """Variance of the sample mean."""
    return sample_var(x)/len(x)


def extract_data(gene_list, expression_file, photoperiod_set, strain_set,
                 time_point_set, clustering_indicator, f_test,
                 num_replicates):
    """Extract gene expression data for a list of genes."""
    # The expression dictionary will have 4-tuple keys and list values.  The
    # key is (gene, photoperiod, strain, sample_time) and the value is
    # [replicate 1, replicate 2, replicate 3, ...].
    expression = {}
    # Create a (shallow copy of the) gene list for counting purpose.
    gene_ids_remaining = gene_list.make_gene_id_set()
    with open(expression_file, 'r') as f:
        # The first line in the expression file is the sample IDs.  An ID
        # could look like 'LD_G1_T1_A'.
        # Remove trailing newline and leading empty ID.
        sample_ids_str = f.readline().strip().split(',')[1:]
        # Each element in sample_ids is a list of strings that identify a
        # sample ID.  E.g., one element could be ['LD', 'G1', 'T1', 'A'].
        sample_ids = [sid_str.split('_') for sid_str in sample_ids_str]
        # The rest of the lines are gene expression levels.
        for line in f:
            exp = line.strip().split(',')
            gene_id = exp[0]
            exp_str_list = exp[1:]
            if gene_id in gene_ids_remaining:
                # We assert that this gene has not been recorded in the
                # expression dictionary.  We take the first values of
                # photoperiods, strains and time points for sanity check.
                key_check = (gene_id, photoperiod_set[0], strain_set[0],
                             time_point_set[0])
                assert(key_check not in expression)
                for idx_sid, sid in enumerate(sample_ids):
                    photoperiod = sid[0]
                    strain = sid[1]
                    sample_time = sid[2]
                    # sid[3] denotes replicates (i.e., A, B, C, etc.), which
                    # we do not use more than stacking the gene expression
                    # levels for different replicates as a list in the
                    # expression dictionary.
                    exp_key = (gene_id, photoperiod, strain, sample_time)
                    # Append the expression level to the dictionary if
                    # exp_key exists; create it and set it to the
                    # single-element list containing the expression level if
                    # it does not.
                    if exp_key in expression:
                        expression[exp_key].append(
                            float(exp_str_list[idx_sid])
                            )
                    else:
                        expression[exp_key] = [
                            float(exp_str_list[idx_sid])
                            ]
                # Remove the gene from the remaining gene list.
                gene_ids_remaining.remove(gene_id)
        # After scanning the entire gene expression file, we assert that
        # the expression levels for the gene in the list are obtained; i.e.,
        # nothing is left in the list of remaining genes.
        assert(not gene_ids_remaining)
    # Do one-way ANOVA using F-test.
    # TODO: Separate F-test from extract_data().
    if f_test:
        print('Doing one-way ANOVA...')
        print('Gene list:', gene_list.make_gene_id_set())
        for gene in gene_list.genes:
            print('Gene ID is', gene.id)
            f_stat, dof = anova_old(
                expression, gene.id, photoperiod_set, strain_set,
                time_point_set, num_replicates
                )
            print('F-statistic is', f_stat)
            print('DOF is', dof)
            print('p-value is', 1-scipy.stats.f.cdf(f_stat, dof[0], dof[1]),
                  '\n')
        print('Done one-way ANOVA.')
        exit(0)
    if clustering_indicator:
        # Normalize the different photoperiod, time, strain and replicate
        # for each single gene.
        # TODO: Remove unnecessary normalization.
        for gene in gene_list.genes:
            exp_sum = 0
            exp_sq_sum = 0
            exp_count = 0
            for pp in photoperiod_set:
                for strain in strain_set:
                    for sample_time in time_point_set:
                        exp_key = (gene.id, pp, strain, sample_time)
                        exp_sum += sum(expression[exp_key])
                        exp_sq_sum += sum([
                            x**2 for x in expression[exp_key]
                            ])
                        exp_count += len(expression[exp_key])
            exp_mean = exp_sum / exp_count
            exp_centered_norm = np.sqrt(exp_sq_sum-exp_sum**2/exp_count)
            for pp in photoperiod_set:
                for strain in strain_set:
                    for sample_time in time_point_set:
                        exp_key = (gene.id, pp, strain, sample_time)
                        expression[exp_key] = [
                            (x-exp_mean) / exp_centered_norm
                            for x in expression[exp_key]
                            ]
    # Generate the average expression profiles.
    num_photoperiods = len(photoperiod_set)
    num_strains = len(strain_set)
    num_time_points = len(time_point_set)
    num_genes = len(gene_list.genes)
    data_cell = []
    data_var = []
    exp_count = 0
    for idx_p, pp in enumerate(photoperiod_set):
        for idx_s, strain in enumerate(strain_set):
            # Each photoperiod-strain pair is considered an individual
            # experiment.
            data_cell.append(np.empty((num_time_points, num_genes)))
            data_var.append(np.empty((num_time_points, num_genes)))
            for idx_g, gene in enumerate(gene_list.genes):
                for idx_t, time_point in enumerate(time_point_set):
                    expression_key = (gene.id, pp, strain, time_point)
                    replicates = expression[expression_key]
                    data_cell[exp_count][idx_t, idx_g] = np.mean(
                        replicates
                        )
                    data_var[exp_count][idx_t, idx_g] = sample_mean_var(
                        replicates
                        )
            exp_count += 1
    return data_cell, data_var


class Gene:
    """Gene with ID, name, and positions."""
    def __init__(self, id, name, pos_x, pos_y):
        self.id = id
        self.name = name
        self.pos_x = pos_x
        self.pos_y = pos_y


class GeneList:
    """Gene list."""
    def __init__(self):
        self.genes = []
        self.has_pos = True

    def make_gene_id_set(self):
        """Make a gene ID set from all the genes in the gene list."""
        gene_id_set = set()
        for gene in self.genes:
            gene_id_set.add(gene.id)
        return gene_id_set


def read_gene(gene_list_file):
    """Read from a gene list file.

    Args:
        gene_list_file: Path to the input file.  The input file should be
            in CSV format with multiple rows and two columns.  The first
            column are the gene IDs and the second column are the gene
            names.

    Returns:
        A GeneList object.
    """
    gene_list = GeneList()
    with open(gene_list_file, 'r') as f:
        for line in f:
            # Remove trailing newline or white spaces and split by comma.
            data = line.rstrip().split(',')
            # OBSOLETE: Gene positions are no longer needed since we use
            # Cytoscape for network visualization.
            if len(data) == 4:
                gene = Gene(data[0], data[1], float(data[2]),
                            float(data[3]))
            else:
                # Use the gene name if it is provided. Otherwise use the
                # gene ID as the gene name.
                if data[1]:
                    gene = Gene(data[0], data[1], 0, 0)
                else:
                    gene = Gene(data[0], data[0], 0, 0)
                gene_list.has_pos = False
            gene_list.genes.append(gene)
    return gene_list


def draw_gene_network(gene_list, weight, sign, font_size, width_max,
                      node_size, node_color, node_shape, border_width,
                      pos_file, edge_threshold, thickness):
    """Draw a gene regulatory network with weights and signs.

    Arguments:
    gene_list -- A GeneList object storing the genes of interest.
    weight -- A dictionary with 2-tuples of strings (<target gene ID>,
        <parent gene ID>) as keys and floats <weight> between 0 and
        1 as values, indicating the weight of the directed edge
        from <parent gene ID> to <target gene ID> is <weight>.
    sign -- A dictionary with 2-tuples of strings (<target gene ID>,
        <parent gene ID>) as keys and floats <sign ratio> between
        -1 and 1 as values, indicating the normalized count of the
        instances in perturbation analysis with definite sign on
        the edge from <parent gene ID> to <target gene ID>.  E.g.,
        <sign ratio> = 1.0 means all of the perturbations that
        recover this edge give a plus sign, and <sign ratio> = -1
        means all of the perturbations that recover this edge give
        a minus sign.  A <sign ratio> value close to 0 means the
        numbers of perturbations with plus and minus sign for this
        edge are close.
    font_size -- The font size of the gene names.
    width_max -- The maximum width of the edges.
    node_size -- The node size.
    node_color -- The node color.
    node_shape -- The node shape.
    border_width -- The width of the border of the nodes.
    pos_file -- The output file for the positions.  Do not output if empty.
    edge_threshold -- The minimum weight of edge to display in the graph.
    thickness -- the edge thickness type.  Could be either
        multiplicity-ratio or p-value.
    """
    gene_network = nx.DiGraph()
    gene_network.add_nodes_from(gene_list.make_gene_id_set())
    # Each edge should have a weight and a sign.
    assert(weight.keys() == sign.keys())
    for target, parent in weight:
        if weight[target, parent] >= edge_threshold:
            gene_network.add_edge(
                parent, target, weight=weight[target, parent],
                sign=sign[target, parent]
                )
    if gene_list.has_pos:
        gene_pos = {gene.id: (gene.pos_x, gene.pos_y)
                    for gene in gene_list.genes}
    else:
        gene_pos = nx.spring_layout(gene_network)
    # Output position file.
    if pos_file:
        # Check existence.
        assert(not gene_list.has_pos)
        with open(pos_file, 'w') as f:
            for gene in gene_list.genes:
                line = ' '.join([gene.id, gene.name,
                                 str(gene_pos[gene.id][0]),
                                 str(gene_pos[gene.id][1])])
                f.write(line + '\n')
    fig = plt.figure()
    ax = plt.subplot(111)
    # Draw nodes without labels.
    nx.draw_networkx_nodes(
        gene_network, gene_pos, node_size=node_size, node_color=node_color,
        node_shape=node_shape, linewidths=border_width
        )
    # Draw edges.
    edges = gene_network.edges()
    widths = [gene_network[u][v]['weight']*width_max for u, v in edges]
    color_values = [gene_network[u][v]['sign'] for u, v in edges]
    colors = color_map_rgb(color_values)
    nx.draw_networkx_edges(gene_network, gene_pos, edgelist=edges,
                           width=widths, edge_color=colors)
    # Draw labels.
    gene_names = {gene.id: gene.name for gene in gene_list.genes}
    nx.draw_networkx_labels(gene_network, gene_pos, labels=gene_names,
                            font_size=font_size)
    # The widths of the arrowheads are four times those of the lines.
    if thickness == 'multiplicity-ratio':
        reg_100 = mlines.Line2D([], [], color='black',
                                linewidth=width_max*4,
                                label='100% regulator')
        reg_60 = mlines.Line2D([], [], color='black',
                               linewidth=width_max*4*0.6,
                               label='60% regulator')
        reg_20 = mlines.Line2D([], [], color='black',
                               linewidth=width_max*4*0.2,
                               label='20% regulator')
    elif thickness == 'p-value':
        reg_100 = mlines.Line2D([], [], color='black',
                                linewidth=width_max*4,
                                label='p-value = 0.0')
        reg_60 = mlines.Line2D([], [], color='black',
                               linewidth=width_max*4*0.5,
                               label='p-value = 0.05')
        reg_20 = mlines.Line2D([], [], color='black',
                               linewidth=width_max*4*0.25,
                               label='p-value = 0.1')
    else:
        print('Wrong thickness type')
        exit(1)
    plus_100 = mpatches.Patch(color=(0, 0, 1, 1), label='100% activator')
    plus_60 = mpatches.Patch(color=(0.4, 0.4, 1, 1), label='60% activator')
    plus_20 = mpatches.Patch(color=(0.8, 0.8, 1, 1), label='20% activator')
    minus_20 = mpatches.Patch(color=(1, 0.8, 0.8, 1), label='20% repressor')
    minus_60 = mpatches.Patch(color=(1, 0.4, 0.4, 1), label='60% repressor')
    minus_100 = mpatches.Patch(color=(1, 0, 0, 1), label='100% repressor')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    plt.legend(
        handles=[reg_100, reg_60, reg_20, plus_100, plus_60, plus_20,
                 minus_20, minus_60, minus_100],
        loc='center left', bbox_to_anchor=(1.02, 0.5)
        )
    plt.show()


def color_map_rgb(color_values):
    """Map numbers to RGBA colors with three polars.

    -1 maps to red.
    0 maps to white.
    1 maps to blue.
    Use linear interpolation in between.
    The alpha value is always 1.

    Argument:
    color_values -- A list of numbers normalized between -1 and 1.
    Return value:
    colors -- A list of 4-tuples representing the RGBA color.
    """
    colors = []
    for value in color_values:
        assert(value >= -1.0 and value <= 1.0)
        if value <= 0.0:
            rgba = (1, value + 1, value + 1, 1)
        else:
            rgba = (1 - value, 1 - value, 1, 1)
        colors.append(rgba)
    return colors


def dump_dict(dict_list, filename):
    assert(filename)
    dict_list_str_key = []
    # Convert tuple keys to string keys.
    for my_dict in dict_list:
        my_dict_str_key = {str(key): my_dict[key] for key in my_dict}
        dict_list_str_key.append(my_dict_str_key)
    with open(filename, 'w') as f:
        json.dump(dict_list_str_key, f)


def load_dict(json_in_file):
    with open(json_in_file, 'r') as f:
        data = json.load(f)
        dict_list = []
        for my_dict in data:
            dict_list.append(
                {ast.literal_eval(key): my_dict[key] for key in my_dict}
                )
    return dict_list


def thickness(p_value, significance_level):
    """Exponentially decreasing thickness.

    This is a different function to convert p-values to edge
    reliability scores.  The default is the threshold function.

    Args:
        p_value: The p-value of the used test statistic on the edge.
        significance_level: The significance level for the test.

    Returns:
        An exponentially decreasing reliability score.
    """
    return math.exp(-math.log(2) / significance_level * p_value)


def fisher(p_values):
    """Fisher's method."""
    chi2 = -2 * sum([math.log(p) for p in p_values])
    dof = 2 * len(p_values)
    return 1 - scipy.stats.chi2.cdf(chi2, dof)


def create_graph(gene_list, weight, sign):
    """Create a directed graph with attributes.

    gene_list -- a GeneList object.
    weight -- a dictionary of edge weights.
    sign -- a dictionary of edge signs.
    """
    gene_network = nx.DiGraph()
    for gene in gene_list.genes:
        gene_network.add_node(gene.id, name=gene.name)
    # Each edge should have a weight and a sign.
    assert(weight.keys() == sign.keys())
    for target, parent in weight:
        gene_network.add_edge(
            parent, target, weight=weight[target, parent],
            sign=sign[target, parent]
            )
    return gene_network


def add_source(gene_network):
    """Add source of genes to nodes' attributes.

    Source could be:
    minglei -- Minglei's differentially expressed genes under the control
               of E-genes.
    faqiang -- soybean core flowering genes in the literature by Faqiang.
    both -- in both Minglei's and Faqiang's lists; this could be either
            a) the same gene is in both lists, or b) the gene is a cluster
            containing at least one gene from each list.
    """
    genes = gene_network.nodes()
    source = {}
    with open('data/minglei.txt', 'r') as f:
        mingleis_genes = f.read().strip().split()
    with open('data/faqiang.txt', 'r') as f:
        faqiangs_genes = f.read().strip().split()
    for gene in genes:
        for m_gene in mingleis_genes:
            if m_gene in gene:
                source[gene] = 'minglei'
        for f_gene in faqiangs_genes:
            if f_gene in gene:
                if gene in source:
                    source[gene] = 'both'
                else:
                    source[gene] = 'faqiang'
    nx.set_node_attributes(gene_network, 'source', source)


def virtual_time_shift(data, extra=1):
    """Virtual time shift.

    Args:
        data: A list of matrices.
        extra: The number of virtual times. Default is 1.

    Returns:
        A list of matrices, each of which is appended with a number
        of rows that is identical to the first rows.
    """
    return [np.concatenate((x, x[:extra, :]), axis=0) for x in data]


def anova_old(expression, gene_id, photoperiod_set, strain_set,
              time_point_set, num_replicates):
    """One-way analysis of variance (ANOVA) using F-test."""
    num_groups = len(photoperiod_set)*len(strain_set)*len(time_point_set)
    group_size = num_replicates
    total_expression = 0
    # First scan: calculate overall average.
    for pp in photoperiod_set:
        for ss in strain_set:
            for tt in time_point_set:
                total_expression += sum(expression[(gene_id, pp, ss, tt)])
    overall_avg = total_expression/num_groups/group_size
    # Second scan: calculate variances.
    in_group_var = 0
    bt_group_var = 0
    for pp in photoperiod_set:
        for ss in strain_set:
            for tt in time_point_set:
                group = expression[(gene_id, pp, ss, tt)]
                group_avg = sum(group)/group_size
                in_group_var += group_size*(group_avg-overall_avg)**2
                for element in group:
                    bt_group_var += (element-group_avg)**2
    dof = (num_groups-1, group_size*num_groups-num_groups)
    f_stat = bt_group_var / dof[0] / in_group_var * dof[1]
    return f_stat, dof


def convert_csv(csv_white, csv_comma=''):
    """Convert white-space-delimited CSV to comma-delimited CSV.
    
    Args:
        csv_white: Original CSV file with white space as the delimiter.
        csv_comma: New CSV file with comma as the delimiter. Default
            use the same prefix as the original CSV.
    """
    if not csv_comma:
        csv_comma = csv_white[:-4]+'.csv'
    with open(csv_white, 'r') as f_white:
        with open(csv_comma, 'w') as f_comma:
            for line in f_white:
                f_comma.write(','.join(line.split()))
                f_comma.write('\n')
    return None


def extract_data_new(
    gene_list, expression_file, cond_list, parser_dict
    ):
    """Extract expression data from file.
    
    Args:
        gene_list: A GeneList object of the selected genes.
        expression_file: A path to the expression file.
        cond_list: A list of lists of conditions.
            The last list is for the sample times,
            which must be in order. For example,
            cond_list = [
                [16, 25],
                ['LD', 'SD'],
                [1, 2, 3, 4],
                ['I1', 'D0', 'I3', 'I4', 'I5', 'I6']
                ]
        parser_dict: A dictionary with sample IDs as key and tuples
            of conditions as values.

    Returns:
        A 2-tuple of lists of 2-d numpy arrays.
        Both lists are indexed by the unstructured
        condition in arbitrary order. The elements
        in both lists are 2-d numpy arrays with
        sample time index as the first axis and
        gene index as the second axis. The values
        of the arrays in the first list are the
        mean expression levels, and those in the
        second list are variance of the mean
        expression levels.
    """
    data_mean = []
    data_var = []
    # Create a shallow copy of the gene list for
    # counting purpose.
    gene_ids_remaining = gene_list.make_gene_id_set()
    # Dictionary with gene-time-condition 3-tuple as
    # key and expression list for all replicates as
    # value.
    expression = {}
    with open(expression_file, 'r') as f:
        # First line is the list of the sample IDs.
        # Remove trailing newline and leading empty
        # element.
        sample_ids_str = f.readline().strip().split(',')[1:]
        for line in f:
            exp = line.strip().split(',')
            gene_id = exp[0]
            exp_str_list = exp[1:]
            if gene_id in gene_ids_remaining:
                for idx_sid, sid_str in enumerate(sample_ids_str):
                    sid = parser_dict[sid_str]
                    # Check if current sample is in
                    # the condition list.
                    is_in_cond_list = True
                    for idx_cond, cond in enumerate(sid):
                        if cond not in cond_list[idx_cond]:
                            is_in_cond_list = False
                            break
                    if not is_in_cond_list:
                        continue
                    # Otherwise this sample is in the
                    # condition list. We now read the
                    # expression level into the 
                    # dictionary. We append to the
                    # dictionary if the key already
                    # exists, or create a new entry
                    # if it does not.
                    sample_time = sid[-1]
                    other_cond = sid[:-1]
                    exp_key = (
                        gene_id, sample_time, other_cond
                        )
                    if exp_key in expression:
                        expression[exp_key].append(
                            float(exp_str_list[idx_sid])
                            )
                    else:
                        expression[exp_key] = [
                            float(exp_str_list[idx_sid])
                            ]
                # Remove the gene from the remaining
                # gene list.
                gene_ids_remaining.remove(gene_id)
    # Generate average and variance.
    # Use itertools.product() and argument unpacking
    # instead of nested loops.
    time_point_set = cond_list[-1]
    num_time_points = len(time_point_set)
    num_genes = len(gene_list.genes)
    # Use single star to unpack the list into positional arguments.
    for idx_oc, other_cond in enumerate(
            itertools.product(*cond_list[:-1])
            ):
        data_mean.append(np.empty((
            num_time_points, num_genes
            )))
        data_var.append(np.empty((
            num_time_points, num_genes
            )))
        for idx_g, gene in enumerate(gene_list.genes):
            for idx_t, time_point in enumerate(
                    time_point_set
                    ):
                exp_key = (
                    gene.id, time_point, other_cond
                    )
                replicates = expression[exp_key]
                data_mean[idx_oc][idx_t, idx_g] = (
                    np.mean(replicates)
                    )
                data_var[idx_oc][idx_t, idx_g] = (
                    sample_mean_var(replicates)
                    )
    return data_mean, data_var


def rename_genes(old_graphml_file, gene_list_file, new_graphml_file):
    """Rename genes from a GraphML file and save to another file.

    Args:
        old_graphml_file: A graph in GraphML format where each node has
            a string of gene ID as the value and a string as an attribute
            called "name".

            For example:
            $ G.node
            {'Glyma.01G000100': {'name: 'My first gene'},
             'Glyma.01G000200': {'name': 'My second gene'}}

        gene_list_file: A CSV format file consisting a column of gene
            IDs and a corresponding column of gene names.

            For example:
            Glyma.01G000100,Gene 1
            Glyma.01G000200,Gene 2

        new_graphml_file: Path to the GraphML file with new gene names.

            For example:
            $ G.node
            {'Glyma.01G000100': {'name: 'Gene 1'},
             'Glyma.01G000200': {'name': 'Gene 2'}}

    Returns:
        None
    """
    gene_network = nx.read_graphml(old_graphml_file)
    gene_list = read_gene(gene_list_file)
    for gene in gene_list.genes:
        if gene.id in gene_network.nodes():
            gene_network.node[gene.id]['name'] = gene.name
        else:
            print('Gene {} is not found in {}.'.format(
                gene.id, old_graphml_file
                ))
    nx.write_graphml(gene_network, new_graphml_file)
    return


def find_cond_by_cond(conditions, cond_target, cond_given):
    """Find the set of target conditions given other conditions.

    Args:
        conditions: A list of tuples of sample conditions.
            E.g.,
            [(16, 'LD', 1, 'I1'), (16, 'SD', 2, 'I2'),
             (16, 'SD', 1, 'I1'), (16, 'Sh', 2, 'I1')]
        cond_target: A list of integers indicating the 0-based
            indices of the target conditions.
            E.g., [1].
        cond_given: A list of integers indicating the 0-based
            indices of the given conditions.
            E.g., [2, 3].

    Returns:
        A dictionary with tuples of given conditions as the keys
        and list of tuples of the target conditions as the values.
    """
    cond_dict = {}
    for cond in conditions:
        target = tuple(cond[i] for i in cond_target)
        given = tuple(cond[i] for i in cond_given)
        if given not in cond_dict:
            cond_dict[given] = [target]
        else:
            if target not in cond_dict[given]:
                cond_dict[given].append(target)
    return cond_dict


def load_parser(parser_table_file):
    """Load parser table into a dictionary.

    Args:
        parser_table_file: A CSV format file.
            The first element in each row is the sample ID. The rest
            are the conditions. The last condition is the sample time.

    Returns:
        A dictionary with sample IDs as the key and the tuple of
        conditions as the value.
    """
    parser_dict = {}
    with open(parser_table_file, 'r') as f:
        for line in f:
            words = line.strip().split(',')
            parser_dict[words[0]] = tuple(words[1:])
    return parser_dict


def bslr(parser_dict, exp_df, num_experiments, num_times,
         num_genes, num_replicates, max_in_degree,
         significance_level):
    """BSLR with averaging.

    Args:
        parser_dict: A dictionary.
            Sample ID parser.
        exp_df: A dataframe.
            Gene expression levels with gene IDs as the
            index and the sample IDs as the column names.
        num_experiments: Number of experiments.
        num_times: Number of times.
        num_genes: Number of genes.
        num_replicates: Number of replicates.
        max_in_degree: Maximum in-degree.
        significance_level: Significance level for Granger
            causality F-test.

    Returns:
        A 2d array of the reconstructed network.
    """
    # Self-regulation in Granger causality inference.
    self_reg = True
    num_time_lags = 1
    # Time-dependent standardization off.
    tds = False
    data_cell = average_full_factorial(
        exp_df, parser_dict, num_experiments, num_times,
        num_genes, num_replicates
        )
    parents, signs = ca.caspian(
        data_cell, num_time_lags, max_in_degree,
        significance_level, self_reg, tds
        )
    adj_mat = np.zeros((num_genes, num_genes))
    for gene, parents_for_gene in enumerate(parents):
        for idx_p, parent in enumerate(parents_for_gene):
            adj_mat[parent, gene] = signs[gene][idx_p]
    return adj_mat


def average_full_factorial(exp_df, parser_dict, num_experiments,
                           num_times, num_genes, num_replicates):
    """Average replicates in a full factorial design.

    Conditions must be str of integers ('0', '1', ..., 'C-1'),
    and times must be str of integers ('0', '1', ..., 'T-1').

    Args:
        parser_dict: A dictionary.
            Sample ID parser.
        exp_df: A dataframe.
            Gene expression levels with gene IDs as the index
            and the sample IDs as the column names.
        num_experiments: Number of experiments.
        num_times: Number of times.
        num_genes: Number of genes.
        num_replicates: Number of replicates.

    Returns:
        A 3-d array of the average TPMs.
            Axis 0: condition.
            Axis 1: time.
            Axis 2: gene.
    """
    data_cell = np.zeros((num_experiments, num_times, num_genes))
    for s in parser_dict:
        condition = int(parser_dict[s][0])
        time = int(parser_dict[s][1])
        data_cell[condition, time] += exp_df[s]/num_replicates
    return data_cell


if __name__ == "__main__":
    main(sys.argv[1:])
