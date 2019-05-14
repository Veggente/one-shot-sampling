#!/usr/bin/env python3
"""CaSPIAN with brute force."""
import numpy as np
import scipy.stats
import itertools


def testing():
    data_cell_single = np.array(np.mat('3 4 5; 6 7 8'))
    data_cell = [
        np.array([[0.836334,  0.08015632,  0.001396,  0.64343897],
                  [0.63934881,  0.44191633,  0.70205871,  0.47928562],
                  [0.01185488,  0.69447034,  0.49794843,  0.73974141]]).T,
        np.array([[0.179556,  0.07568321,  0.66753137],
                  [0.62753963,  0.22615587,  0.03522489],
                  [0.39816049,  0.42056573,  0.10281385]])
        ]
    num_time_lags = 2
    num_gene = 3
    matrix_2 = get_shifted_matrix(data_cell, num_time_lags)
    for j in range(num_gene):
        for i in range(num_time_lags + 1):
            print('matrix_2[:, {0}, {1}] ='.format(j, i),
                  matrix_2[:, j, i])
    data_normalized = normalize(matrix_2)
    for j in range(num_gene):
        for i in range(num_time_lags + 1):
            print('data_normalized[:, {0}, {1}] ='.format(j, i),
                  data_normalized[:, j, i])
    phi = data_normalized
    potential_parents = compressive_sensing(phi)
    print(potential_parents)
    np.random.seed(0)
    print("Testing standardize():")
    data_cell_2 = [
        np.random.rand(3, 4), np.random.rand(3, 4), np.random.rand(3, 4)
        ]
    print("Before standardization:\n", data_cell_2)
    data_cell_2_st = standardize(data_cell_2)
    print("After standardization:\n", data_cell_2_st)


def caspian(data_cell, num_time_lags, max_in_degree, significance_level,
            self_reg=False, st=False):
    # Preprocess the data.
    # Standardization.
    if st:
        data_cell_processed = standardize(data_cell)
    else:
        data_cell_processed = data_cell
    shifted_data = get_shifted_matrix(data_cell_processed, num_time_lags)
    phi = normalize(shifted_data)
    # Find candidate links with brute force using numpy.linalg.inv.
    # The data of latest time point is the target.
    # The rest are used for prediction; i.e., the Phi matrix.
    # Compressive sensing with errors.
    potential_parents, errors = compressive_sensing(phi, self_reg,
                                                    max_in_degree)
    # Use Granger causality to remove any insignificant links.  Then return
    # the resulting graph with signs, and the p-values if the significance
    # level is set to be zero.
    return granger(phi, potential_parents, errors,
                   significance_level, self_reg)


def get_shifted_matrix(data_cell, num_time_lags, is_single_mat=False):
    """Return a 3-dimensional matrix for compressive sensing.

    Axis 0 -- distinct virtual experiments.
    Axis 1 -- genes.
    Axis 2 -- time lags.
    """
    # Generate a single block row for a single-matrix input.
    if is_single_mat:
        num_time_points, num_genes = data_cell.shape
        sliding_window_height = num_time_points - num_time_lags
        # Generate the uninitialized shifted matrix.
        shifted_matrix = np.empty([
            sliding_window_height, num_genes, num_time_lags + 1
            ])
        for pos in range(num_time_lags + 1):
            shifted_matrix[:, :, pos] = (
                data_cell[pos:pos+sliding_window_height, :]
                )
        return shifted_matrix
    else:  # data_cell is a list of multiple matrices.
        # Recursively run the single-matrix versions.
        num_genes = data_cell[0].shape[1]
        shifted_matrix = np.empty([0, num_genes, num_time_lags+1])
        for data_page in data_cell:
            new_block_row = get_shifted_matrix(
                data_page, num_time_lags, is_single_mat=True
                )
            shifted_matrix = np.concatenate((
                shifted_matrix, new_block_row
                ), axis=0)
        return shifted_matrix


def normalize(data):
    """Normalize the 3-dimensional data along first axis."""
    _, dim_1, dim_2 = data.shape
    data_normalized = np.empty(data.shape)
    for idx_1 in range(dim_1):
        for idx_2 in range(dim_2):
            data_vec_temp = data[:, idx_1, idx_2]
            data_vec_centered = data_vec_temp - np.mean(data_vec_temp)
            data_vec_norm = np.linalg.norm(data_vec_centered)
            # Normalize if the norm is not zero.
            if data_vec_norm:
                data_vec_normalized = data_vec_centered / data_vec_norm
            else:
                # Otherwise, keep the all-zero vectors unchanged.
                data_vec_normalized = data_vec_centered
            data_normalized[:, idx_1, idx_2] = data_vec_normalized
    return data_normalized


def compressive_sensing(phi, self_reg=False, max_in_degree=0):
    """Compressive sensing with brute force."""
    num_genes = phi.shape[1]
    parents = []
    errors = []
    # Return all other genes if max_in_degree is 0.
    if not max_in_degree:
        for idx_gene in range(num_genes):
            all_other_genes = (
                list(range(idx_gene))
                + list(range(idx_gene + 1, num_genes))
                )
            parents.append(all_other_genes)
            _, err = whole_gene_lsa(phi, idx_gene, all_other_genes,
                                    self_reg)
            # A single-element list is used in accordance with the
            # max_degree > 0 case.
            errors.append([err])
    else:
        # For each target gene, try all combinations of k = max_in_degree
        # parent genes. Pick the best combination.
        for idx_gene in range(num_genes):
            errors.append([])
            parents.append([])
            all_other_genes = [g for g in range(num_genes) if g != idx_gene]
            for combo_tuple in itertools.combinations(all_other_genes,
                                                      max_in_degree):
                combo = list(combo_tuple)
                _, err = whole_gene_lsa(phi, idx_gene, combo, self_reg)
                # Note that we distinguish the case of errors[idx_gene]
                # being False (the empty list, meaning that gene has not
                # been considered yet) and the case of errors[idx_gene]
                # being [0] (zero error).
                if errors[idx_gene]:
                    if errors[idx_gene][0] > err:
                        errors[idx_gene][0] = err
                        parents[idx_gene] = combo
                else:
                    errors[idx_gene].append(err)
                    parents[idx_gene] = combo
    return parents, errors


# TODO: Better interface.
def granger(phi, potential_parents, errors, significance_level=0.0,
            self_reg=False):
    # Setting the significance level to be the default 0.0 indicates
    # actual p-values to be returned instead of filtered parents, in which
    # case the returned p_values and signs are for all the
    # potential_parents.
    if significance_level:
        parents = []
    else:
        parents = potential_parents
        all_p_values = []
    signs = []
    num_experiments = phi.shape[0]
    num_time_lags = phi.shape[2] - 1
    for idx_gene, its_pot_parents in enumerate(potential_parents):
        if significance_level:
            parents.append([])
        else:
            all_p_values.append([])
        signs.append([])
        if self_reg:
            num_genes_unrestricted = len(its_pot_parents) + 1
        else:
            num_genes_unrestricted = len(its_pot_parents)
        dof = (
            num_time_lags, (
                num_experiments
                - num_genes_unrestricted * num_time_lags
                - 1
                )
            )
        assert(dof[0] > 0 and dof[1] > 0)
        for idx_parent in its_pot_parents:
            restricted_parents = [
                p for p in its_pot_parents if p != idx_parent
                ]
            _, restricted_error = whole_gene_lsa(
                phi, idx_gene, restricted_parents, self_reg
                )
            f_stat = (
                (restricted_error**2 - errors[idx_gene][0]**2)
                / dof[0]
                / (errors[idx_gene][0]**2)
                * dof[1]
                )
            p_value = 1 - scipy.stats.f.cdf(f_stat, dof[0], dof[1])
            if significance_level and p_value < significance_level:
                parents[idx_gene].append(idx_parent)
            if not significance_level:
                all_p_values[idx_gene].append(p_value)
        num_parents = len(parents[idx_gene])
        coeff, err = whole_gene_lsa(phi, idx_gene, parents[idx_gene],
                                    self_reg)
        coeff_2d = coeff.reshape(num_time_lags, num_parents)
        for idx_parent, parent in enumerate(parents[idx_gene]):
            if (coeff_2d[:, idx_parent] > 0).all():
                # Sign is positive (activation).
                signs[idx_gene].append(1)
            elif (coeff_2d[:, idx_parent] < 0).all():
                # Sign is negative (repression).
                signs[idx_gene].append(-1)
            else:
                # Sign is undetermined.
                signs[idx_gene].append(0)
    # TODO: Probably need to resolve the multiple-return-statement issue.
    if not significance_level:
        return parents, signs, all_p_values
    else:
        return parents, signs


def whole_gene_lsa(phi, this_gene, parent_genes, self_reg=False):
    size_phi = phi.shape
    num_experiments = size_phi[0]
    num_time_lags = size_phi[2] - 1
    regressand = phi[:, this_gene, num_time_lags]
    assert(this_gene not in parent_genes)
    # The previous time lags of the target gene are also used in regression.
    # Note the genes in the regressor are not ordered.
    if self_reg:
        regressor_3d = phi[:, np.array(parent_genes + [this_gene]), :-1]
        num_genes_to_fit_with = len(parent_genes) + 1
    else:
        regressor_3d = phi[:, parent_genes, :-1]
        num_genes_to_fit_with = len(parent_genes)
    regressor = regressor_3d.reshape(
        num_experiments, num_genes_to_fit_with * num_time_lags
        )
    coeff_all, residual = lsa(regressand, regressor)
    # Remove the coefficients for the target gene if self-regulation is on.
    if self_reg:
        assert(len(coeff_all)
               == num_genes_to_fit_with * num_time_lags)
        idx_keep = [x for x in range(len(coeff_all))
                    if (x+1) % num_genes_to_fit_with != 0]
        coeff = coeff_all[idx_keep]
    else:
        coeff = coeff_all
    return coeff, np.linalg.norm(residual)


def lsa(regressand, regressor):
    """Least-squares approximation to fit columns of regressor to those of
    regressand.
    """
    # Check the dimensions of regressand and regressor.
    assert(regressand.shape[0] == regressor.shape[0])
    if not regressor.size:
        coeff = np.empty((0, regressand.size//regressand.shape[0]))
        residual = regressand
    else:
        coeff = np.linalg.pinv(regressor).dot(regressand)
        residual = regressand-regressor.dot(coeff)
    return coeff, residual


def standardize(data_cell):
    """Time-dependent standardization.

    Note this is not applicable for varying time length data.
    """
    # Need at least two experiments for standardization.
    assert(len(data_cell) > 1)
    num_time_points, num_genes = data_cell[0].shape
    data_3d_array = np.empty([0, num_time_points, num_genes])
    for data_block in data_cell:
        assert((num_time_points, num_genes) == data_block.shape)
        data_3d_array = np.concatenate((
            data_3d_array, np.array([data_block])), axis=0
            )
    data_3d_array_st = normalize(data_3d_array)
    data_cell_st = []
    for idx in range(data_3d_array_st.shape[0]):
        data_cell_st.append(data_3d_array_st[idx, :, :])
    return data_cell_st


def ocse(data_cell, num_perm, alpha=0.05, sparsity=0):
    """Optimal causation entropy algorithm by Sun-Taylor-Bollt 2015.

    Note:
        Only works for single time lag.
        Only includes the discovery stage.
        The biased permutation test does not address the model
            selection uncertainty.

    Args:
        data_cell: A list of gene expression matrices.  The ith
            matrix is T_i-by-n where T_i is the number of sample
            times in experiment i, and n is the number of genes.
            Note that the permutation test in the algorithm will
            be purely temporal if the list contains only one
            element (single experiment).
        num_perm: The number of permutations per gene for the test.
        alpha: The significance level for the (biased) permutation
            test.
        sparsity: Maximum number of parents for each target vector.
            The default 0 indicates no sparsity limit (i.e.,
            sparsity is equal to the number of vectors).

    Returns:
        A 2-tuple sparse representation of the reconstructed network.
        The first element of the tuple is a list of lists of parent
        indices and the second element of the tuple is a list of
        lists of signs.
    """
    # Create an m-by-n-by-2 array by shifting a size-2 window in each
    # experiment to get multiple virtual experiments.  Here m is the
    # total number of virtual experiments and n the number of genes.
    # The first matrix [:, :, 0] is the predictor matrix and the
    # second matrix [:, :, 1] is the target matrix.
    shifted_data = get_shifted_matrix(data_cell, 1)
    # Center the predictors to eliminate the residual of the intercept.
    # TODO: Remove unnecessary standardization.
    phi = normalize(shifted_data)
    num_genes = data_cell[0].shape[1]
    if sparsity > 0:
        num_iterations = sparsity
    else:
        num_iterations = num_genes
    # Initialize the list of parents and signs for all the target
    #genes.
    parent_list = []
    sign_list = []
    for i in range(num_genes):
        # The orthonormal parent dictionary has parent indices as the
        # keys and vector-sign 2-tuples as the values.
        ortho_parent_dict = {}
        target = phi[:, i, 1]
        # Successively discover best predictors.  Note iteration
        # can terminate when no candidate is found or when candidate
        # does not pass the permutation test.  As a result, the final
        # number of parents may be smaller than num_iterations.
        for l in range(num_iterations):
            # Initialization of best candidate index.
            best_candidate_idx = -1
            # Initialization of best normalized innovation.
            best_cand_innov = np.zeros(target.shape)
            # Initialization of best candidate sign.
            best_sign = 0
            # Initialization of maximum absolute value of correlation
            # along the innovation direction.
            max_corr_abs = 0
            # For innovation calculation.
            ortho_vec_list = [
                ortho_parent_dict[key][0] for key in
                ortho_parent_dict
                ]
            # Find best candidate predictor.
            for j in range(num_genes):
                # Skip predictors that have already been selected.
                if j in ortho_parent_dict:
                    continue
                # Candidate predictor vector.
                candidate = phi[:, j, 0]
                # Normalized innovation of the candidate based on the
                # selected orthonormal predictors.
                cand_innov = get_norm_innov(
                    candidate, ortho_vec_list
                    )
                corr = np.inner(cand_innov, target)
                sign = np.sign(corr)
                corr_abs = np.absolute(corr)
                if corr_abs > max_corr_abs:
                    max_corr_abs = corr_abs
                    best_candidate_idx = j
                    best_cand_innov = cand_innov
                    best_sign = sign
            # Initialize the significance indicator.
            is_significant = False
            # Do permutation test if a vector is found.
            if best_candidate_idx >= 0:
                # Biased permutation test.
                counter = 0
                for k in range(num_perm):
                    perm_cand = np.random.permutation(
                        phi[:, best_candidate_idx, 0]
                        )
                    perm_cand_innov = get_norm_innov(
                        perm_cand, ortho_vec_list
                        )
                    perm_corr_abs = np.absolute(
                        np.inner(perm_cand_innov, target)
                        )
                    if perm_corr_abs > max_corr_abs:
                        counter += 1
                        # Terminate as soon as the count for permuted
                        # vectors that are more significant than our
                        # best vector is larger than the threshold.
                        # This only saves some computation.
                        if counter >= num_perm*alpha:
                            break
                if counter < num_perm*alpha:
                    is_significant = True
            # Candidate passed the permutation test.
            if is_significant:
                ortho_parent_dict[best_candidate_idx] = (
                    best_cand_innov, best_sign
                    )
            # No candidate found, or candidate did not pass the test.
            else:
                break
        parent_set = []
        sign_set = []
        for parent in ortho_parent_dict:
            parent_set.append(parent)
            sign_set.append(ortho_parent_dict[parent][1])
        parent_list.append(parent_set)
        sign_list.append(sign_set)
    return parent_list, sign_list


def get_norm_innov(x, ortho_vec, eps=1e-10):
    """Get normalized innovation with respect to a list of
    orthonormal vectors.

    This is a single step in the Gram-Schmidt process.

    Args:
        x: An array representing the target vector.
        ortho_vec: A list of arrays representing the orthonormal
            vectors.
        eps: Epsilon for comparison between norm and zero.  Default
            value is 1e-10.

    Returns:
        The normalized vector of the residual of x from the span of
        ortho_vec if x is not in the span of ortho_vec, or the zero
        vector otherwise.
    """
    residual = x
    for vec in ortho_vec:
        residual = residual - np.inner(x, vec)*vec
    if np.linalg.norm(residual) > eps:
        norm_res = residual / np.linalg.norm(residual)
    else:
        norm_res = np.zeros(x.shape)
    return norm_res


def sbl(y, phi, sigma_sq, epsilon=1e-4, max_iter=1000):
    """The SBL algorithm in Chandra Murthy's slides.

    Args:
        y: The observed signal as an array of length m.
        phi: The sensing matrix as an m-by-N array.
        sigma_sq: The variance of the additive Gaussian noise.
        epsilon: The tolerance for convergence of mu in terms of
            the 2-norm of the difference.
        max_iter: The maximum number of iterations.

    Returns:
        A 3-tuple of the array of hyperparameters (gammas), the
        array of the signal (x's), and the total number of steps
        used.
    """
    y.shape = (len(y), 1)
    gamma = np.identity(phi.shape[1])
    sigma_0 = phi.T.dot(phi)/sigma_sq
    mu = None
    for i in range(max_iter):
        mu_old = mu
        sigma_mat = np.linalg.inv(sigma_0+np.linalg.inv(gamma))
        mu = 1/sigma_sq*sigma_mat.dot(phi.T).dot(y)
        gamma = np.diag(np.diag(mu.dot(mu.T)+sigma_mat))
        if (mu_old is not None and
            np.linalg.norm(mu-mu_old) < epsilon):
            break
    return np.diag(gamma), mu.reshape(len(mu)), i+1


def sbl_grn(data_cell, sigma_sq_0=0.01, sigma_eps=0.4,
            sigma_max_iter=10, sparsity_threshold=1, **sbl_kargs):
    """Sparse Bayesian learning algorithm for gene regulatory
    network reconstruction.

    Args:
        data_cell: A list of gene expression matrices.  The ith
            matrix is T_i-by-n where T_i is the number of sample
            times in experiment i, and n is the number of genes.
        sigma_sq_0: The initial estimated noise variance.
        sigma_eps: The log relative tolerance for convergence of
            the iteratively estimated noise variance in absolute
            value.
        sigma_max_iter: The maximum number of iterations for the
            iteratively estimated noise variance.
        sparsity_threshold: The ratio of the signal sparsity
            threshold to the mean signal magnitude.
        sbl_kargs: Keyword arguments for convergence of SBL.

    Returns:
        A 2-tuple sparse representation of the reconstructed network.
        The first element of the tuple is a list of lists of parent
        indices and the second element of the tuple is a list of
        lists of signs.
    """
    # Create an m-by-n-by-2 array by shifting a size-2 window in each
    # experiment to get multiple virtual experiments.  Here m is the
    # total number of virtual experiments and n the number of genes.
    # The first matrix [:, :, 0] is the predictor matrix and the
    # second matrix [:, :, 1] is the target matrix.
    shifted_data = get_shifted_matrix(data_cell, 1)
    # Standardize the data.
    standardized_data = normalize(shifted_data)
    num_genes = data_cell[0].shape[1]
    parent_list = []
    sign_list = []
    for i in range(num_genes):
        y = standardized_data[:, i, 1]
        phi = standardized_data[:, :, 0]
        sigma_sq_new = sigma_sq_0
        for j in range(sigma_max_iter):
            sigma_sq_old = sigma_sq_new
            _, x, _ = sbl(y, phi, sigma_sq_old, **sbl_kargs)
            res = y-phi.dot(x[:, None])
            # Estimate the new noise variance by the mean
            # squared error.
            sigma_sq_new = np.mean(res*res)
            sigma_log_ratio = np.log(sigma_sq_new/sigma_sq_old)
            if np.absolute(sigma_log_ratio) < sigma_eps:
                break
        x_abs = np.absolute(x)
        x_threshold = np.mean(x_abs)*sparsity_threshold
        x_s = x
        x_s[x_abs < x_threshold] = 0
        parent_set = []
        sign_set = []
        for gene, reg in enumerate(x_s):
            if reg:
                parent_set.append(gene)
                sign_set.append(np.sign(reg))
        parent_list.append(parent_set)
        sign_list.append(sign_set)
    return parent_list, sign_list


if __name__ == "__main__":
    testing()
