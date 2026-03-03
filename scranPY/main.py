import multiprocessing as mp
import scanpy as sc
from anndata import AnnData
import numpy as np
import pandas as pd
import scipy as sp
import cvxpy as cp
from scipy.sparse import csc_matrix
from scipy.optimize import curve_fit
import time


def _create_linear_system(ngenes, cur_cells, cur_exprs, sphere, sizes, use_ave_cell):
    row_dex, col_dex, output = [], [], []
    last_row = 0
    for si in range(len(sizes)):
        out = forge_system(ngenes, cur_cells, cur_exprs, sphere, sizes[si], use_ave_cell)
        row_dex.append(out[0] + last_row)
        col_dex.append(out[1])
        output.append(out[2])
        last_row += cur_cells
    out = forge_system(ngenes, cur_cells, cur_exprs, sphere, 1, use_ave_cell)
    if isinstance(out, str):
        raise ValueError(out)
    si = len(row_dex)
    row_dex.append(out[0] + last_row)
    col_dex.append(out[1])
    output.append(out[2])
    low_weight = 0.000001
    output[si] = output[si] * np.sqrt(low_weight)
    eqn_values = np.repeat(np.repeat([1, np.sqrt(low_weight)], [si, 1]), [len(rd) for rd in row_dex])
    row_dex = np.concatenate(row_dex)
    col_dex = np.concatenate(col_dex)
    output = np.concatenate(output)
    design = csc_matrix((eqn_values, (col_dex, row_dex)), shape=(cur_cells, len(output)))
    return design, output

def forge_system(ng, nc, exprs, ordering, size, ref):
    ncells = int(nc)
    ngenes = int(ng)
    orptr = ordering
    SIZE = int(size)
    rptr = np.asarray(ref, dtype=float)
    eptrs = np.array(exprs, order='F').T
    out1 = np.empty(SIZE * ncells, dtype=int)
    out2 = np.empty(SIZE * ncells, dtype=int)
    out3 = np.empty(ncells, dtype=float)
    row_optr = out1
    col_optr = out2
    ofptr = out3
    combined = np.zeros(ngenes, dtype=float)
    for cell in range(ncells):
        combined.fill(0)
        for index in range(SIZE):
            curcell = orptr[index + cell]
            combined += np.squeeze(eptrs[:, curcell])
            row_optr[index * ncells + cell] = cell
            col_optr[index * ncells + cell] = curcell
        combined /= rptr
        combined = np.partition(combined, ngenes // 2)
        halfway = ngenes // 2
        if ngenes % 2 == 0:
            ofptr[cell] = (combined[halfway - 1] + combined[halfway]) / 2
        else:
            ofptr[cell] = combined[halfway]
    return out1, out2, out3

def clean_size_factors(size_factors, num_detected, control=None, iterations=3, nmads=3, *args, **kwargs):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        keep = size_factors > 0
        if keep.all():
            return size_factors
        if len(size_factors) != len(num_detected):
            raise ValueError("'size_factors' and 'num_detected' should be the same length")
        X = size_factors[keep]
        Y = num_detected[keep]
        if len(X) < 3:
            raise ValueError("need at least three positive values for trend fitting")
        lower = np.median(X) > X
        def linear_func(x, a):
            return a * x
        A, _ = curve_fit(linear_func, X[lower], Y[lower])
        below = np.where(Y < A * X)[0]
        top = below[np.argmax(Y[below])]
        B = A[0] / Y[top] - 1 / X[top]
        init = np.log([A[0], B])
        lY = np.log(Y)
        lX = np.log(X)
        weights = np.ones(len(Y))
        for i in range(iterations + 1):
            try:
                def model_func(x, logA, logB):
                    return np.log(np.exp(logA) * x + 1) - logB
                fit, _ = curve_fit(model_func, lX, lY, p0=init, sigma=weights, **kwargs)
            except:
                size_factors[~keep] = np.min(size_factors[keep])
                return size_factors
    
            init = fit
            resids = np.abs(lY - model_func(lX, *fit))
            bandwidth = max(1e-8, np.median(resids) * nmads)
            weights = (1 - np.minimum(resids / bandwidth, 1) ** 3) ** 3
    
        failed = num_detected[~keep]
        coefs = fit
        new_sf = failed / (np.exp(coefs[0]) - np.exp(coefs[1]) * failed)
        new_sf[new_sf < 0] = np.max(X)
        size_factors[~keep] = new_sf
        return size_factors

def limit_cluster_size(clusters, max_size):
    if max_size is None:
        return clusters
    unique_ids, counts = np.unique(clusters, return_counts=True)
    new_clusters = np.zeros(len(clusters), dtype=int)
    counter = 0
    for id, count in zip(unique_ids, counts):
        current = clusters == id
        if count <= max_size:
            new_clusters[current] = counter
            counter += 1
            continue
        mult = int(np.ceil(count / max_size))
        realloc = np.tile(np.arange(mult) + counter, max_size)[:count]
        new_clusters[current] = realloc
        counter += mult
  
    unique_clusters = np.unique(new_clusters.astype(str))
    new_categories = pd.Categorical.from_codes(new_clusters, categories=unique_clusters.astype(str))
    return new_categories

def generate_sphere(lib_sizes):
    nlibs = len(lib_sizes)
    o = sorted(range(nlibs), key=lambda i: lib_sizes[i])
    even = list(range(1, nlibs, 2))
    odd = list(range(0, nlibs, 2))
    out = [o[i] for i in odd] + [o[i] for i in even[::-1]]
    return out + out

def guess_min_mean(x, min_mean=None):
    if min_mean is None:
        mid_lib = np.median(np.sum(x, axis=0))
        if np.isnan(mid_lib):
            min_mean = 1
        elif mid_lib <= 50000:
            min_mean = 0.1
        elif mid_lib >= 100000:
            min_mean = 1
        else:
            min_mean = 0.1
            print("assuming UMI data when setting 'min.mean'")
    else:
        min_mean = max(min_mean, 1e-8)
    return min_mean

def rescale_clusters(mean_prof, ref_col, min_mean):
    n_clusters = len(mean_prof)
    rescaling = {}
    for clust in range(n_clusters):
        ref_prof = mean_prof[ref_col]
        cur_prof = mean_prof[clust]
        cur_libsize = np.sum(cur_prof)
        ref_libsize = np.sum(ref_prof)
        to_use = ((cur_prof / cur_libsize + ref_prof / ref_libsize) / 2 *
                  (cur_libsize + ref_libsize) / 2 >= min_mean)
        if not np.all(to_use):
            cur_prof = cur_prof[to_use]
            ref_prof = ref_prof[to_use]
        rescaling[clust] = np.nanmedian(cur_prof / ref_prof)
    return rescaling

def solve_quadratic_cvxpy(design, output, cur_cells, lower_bound):
    A = design
    G = np.diag(np.repeat(cur_cells, design.shape[1]))
    H = np.zeros(cur_cells)
    x = cp.Variable(design.shape[1], nonneg=True)
    objective = cp.Minimize(cp.sum_squares(A @ x - output))
    constraints = [G @ x >= H, x >= lower_bound]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=False)
    solution = np.array(x.value).flatten()
    return solution

def QR_decomposition(design, output):
    Q, R = np.linalg.qr(design.T.toarray())
    y = Q.T @ output
    coef, residuals, rank, s = np.linalg.lstsq(R, y, rcond=None)
    return coef


def process_cluster(args):
    clust, indices, exprs, lib_sizes, min_mean, sizes, algorithm, lower_bound = args
    curdex = indices[clust]
    cur_exprs = exprs[curdex]
    cur_libs = lib_sizes[curdex]
    cur_cells = len(curdex)
    ave_cell = np.mean(cur_exprs, axis=0) * np.mean(cur_libs)
    high_ave = min_mean <= ave_cell
    use_ave_cell = ave_cell
    if not all(high_ave):
        cur_exprs = cur_exprs[:, high_ave]
        use_ave_cell = use_ave_cell[high_ave]
    ngenes = np.sum(high_ave)
    sphere = generate_sphere(cur_libs)
    sizes = sizes[sizes <= exprs.shape[0]]
    design, output = _create_linear_system(ngenes, cur_cells, cur_exprs, sphere, sizes, use_ave_cell)
    if algorithm=='QR':
        final_nf = QR_decomposition(design, output)
    if algorithm=='CVXPY':
        final_nf = solve_quadratic_cvxpy(design.T, output, cur_cells, lower_bound)
    if all(final_nf > 0) == False:
        print('Not all size factors for clust = ', clust, 'are greater than 0. Cleaning size factors.')
        final_nf = clean_size_factors(final_nf, cur_exprs.sum(axis=1))
    return final_nf, ave_cell, np.mean(cur_libs)

def normalize_exprs_sparse(X, lib_sizes):
    """
    Normalize CSR or dense matrix by library sizes along rows.
    X: shape (n_cells, n_genes)
    lib_sizes: shape (n_cells,)
    """
    import scipy.sparse as sp
    lib_sizes = np.asarray(lib_sizes).ravel()  # shape (n_cells,)
    
    if sp.issparse(X):
        # CSR multiply broadcasts along rows safely as a 1D array
        if not sp.isspmatrix_csr(X):
            X = X.tocsr()
        return X.multiply(1.0 / lib_sizes)  # <--- pass 1D array, NOT [:, None]
    else:
        return X / lib_sizes[:, None]  # dense still needs [:, None]

def compute_sum_factors(
    adata=AnnData,
    sizes=np.arange(21, 102, 5),
    clusters=None,
    min_mean=None,
    max_size=3000,
    parallelize=True,
    algorithm='CVXPY',
    stopwatch=True,
    plotting=True,
    lower_bound=0.1,
    normalize_counts=False,
    log1p=False,
    layer='scranPY',
    save_plots_dir=None
):

    start_time = time.time()

    import scipy.sparse as sp

    # ------------------------------------------------------------------
    # Handle sparse safely
    # ------------------------------------------------------------------
    X = adata.X
    is_sparse = sp.issparse(X)

    if is_sparse:
        X = X.tocsr()

    # ------------------------------------------------------------------
    # Clustering setup
    # ------------------------------------------------------------------
    if clusters is None:
        clusters = pd.Series(['temp'] * X.shape[0], dtype='category')
        clusters.index = adata.obs_names
        print('Clusters is set to None')
    else:
        clusters = adata.obs[clusters].astype('category')
        print('Current smallest cluster = ', clusters.value_counts().min(),' cells.')

    if clusters.value_counts().min() < sizes.max():
        if 41 > clusters.value_counts().min():
            if 30 > clusters.value_counts().min():
                raise ValueError("You're passing a cluster that is too small. Minimum size is 30 cells.")
            else:
                print("Warning: small cluster (<=40 cells). Adjusting pool sizes.")
                sizes=np.arange(11, clusters.value_counts().min(), 5)
        else:
            print("Warning: cluster <100 cells. Adjusting pool sizes.")
            sizes=np.arange(21, clusters.value_counts().min(), 5)
    sizes = np.array(sizes, dtype=int)

    ncells = X.shape[0]

    if max_size is not None:
        clusters = limit_cluster_size(clusters, max_size=max_size)

    indices = [np.where(clusters == c)[0] for c in np.unique(clusters)]
    print('Using max_size = ',max_size, ', clusters split into ', len(indices), ' clusters.')

    # ------------------------------------------------------------------
    # Library size normalization (SPARSE SAFE)
    # ------------------------------------------------------------------
    lib_sizes = np.asarray(X.sum(axis=1)).ravel()
    lib_sizes = lib_sizes / np.mean(lib_sizes)

    exprs = normalize_exprs_sparse(X, lib_sizes)

    min_mean = guess_min_mean(X, min_mean=min_mean)
    print('min_mean = ', min_mean)

    clust_nf, clust_profile, clust_libsize = [], [], []

    # ------------------------------------------------------------------
    # Cluster processing
    # ------------------------------------------------------------------
    for clust in range(len(indices)):
        curdex = indices[clust]
        cur_exprs = exprs[curdex]
        cur_libs = lib_sizes[curdex]
        cur_cells = len(curdex)

        # Convert to dense ONLY here (linear system requires dense)
        if sp.issparse(cur_exprs):
            cur_exprs = cur_exprs.toarray()

        ave_cell = np.mean(cur_exprs, axis=0) * np.mean(cur_libs)
        high_ave = min_mean <= ave_cell
        use_ave_cell = ave_cell

        if not all(high_ave):
            cur_exprs = cur_exprs[:,high_ave]
            use_ave_cell = use_ave_cell[high_ave]

        ngenes = np.sum(high_ave)

        sphere = generate_sphere(cur_libs)
        sizes = sizes[sizes <= cur_cells]

        design, output = _create_linear_system(
            ngenes, cur_cells, cur_exprs, sphere, sizes, use_ave_cell
        )

        if algorithm=='QR':
            final_nf = QR_decomposition(design, output)
        if algorithm=='CVXPY':
            final_nf = solve_quadratic_cvxpy(design.T, output, cur_cells, lower_bound)

        if not all(final_nf > 0):
            print('Cleaning negative size factors for cluster', clust)
            final_nf = clean_size_factors(final_nf, cur_exprs.sum(axis=1))

        clust_nf.append(final_nf)
        clust_profile.append(ave_cell)
        clust_libsize.append(np.mean(cur_libs))

    # ------------------------------------------------------------------
    # Rescaling across clusters
    # ------------------------------------------------------------------
    non_zeroes = np.array([np.sum(x > 0) for x in clust_profile])
    ref_col = np.argmax(non_zeroes)

    rescaling_factors = rescale_clusters(
        mean_prof=clust_profile,
        ref_col=ref_col,
        min_mean=min_mean
    )

    for clust in range(len(indices)):
        clust_nf[clust] *= rescaling_factors[clust]

    final_sf = np.full(ncells, np.nan)
    final_sf[np.concatenate(indices)] = np.concatenate(clust_nf)
    final_sf = final_sf * lib_sizes

    is_pos = (final_sf > 0) & (~np.isnan(final_sf))
    final_sf = final_sf / np.mean(final_sf[is_pos])

    if stopwatch:
        print('---',round((time.time() - start_time)/60, 2),'mins ---')

    adata.obs['size_factors'] = final_sf
    print('size factor min = ', final_sf.min())
    print('size factor max = ', final_sf.max())

    # ------------------------------------------------------------------
    # Normalize counts (SPARSE SAFE)
    # ------------------------------------------------------------------
    if normalize_counts:
        print('Normalizing adata.X by size factors')

        if is_sparse:
            adata.X = adata.X.multiply(1.0 / final_sf[:, None])
        else:
            adata.X /= final_sf[:,None]

        if log1p:
            sc.settings.verbosity = 0
            sc.pp.log1p(adata)

    if normalize_counts and layer is not None:
        adata.layers[layer] = adata.X.copy()

    return final_sf
