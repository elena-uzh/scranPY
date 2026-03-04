import numpy as np
import scipy.sparse as sp
import anndata as ad
import scranPY


def test_sparse_normalization_shape_with_no_genes_passing_filter():
    # Create fake count matrix (cells x genes)
    n_cells = 100
    n_genes = 200

    X = sp.random(n_cells, n_genes, density=0.05, format="csr")
    adata = ad.AnnData(X)

    adata.obs["groups"] = ["A"] * n_cells

    # Should not raise shape errors
    scranPY.compute_sum_factors(
        adata,
        clusters="groups",
        parallelize=False,
        algorithm="CVXPY",
        max_size=3000,
        plotting=False,
        normalize_counts=False
    )

def test_sparse_normalization_shape():
    n_cells = 100
    n_genes = 200

    # Use high counts so genes pass min_mean filtering
    np.random.seed(42)
    dense = np.random.negative_binomial(n=20, p=0.5, size=(n_cells, n_genes)).astype(float)
    X = sp.csr_matrix(dense)
    
    adata = ad.AnnData(X)
    adata.obs["groups"] = ["A"] * n_cells

    scranPY.compute_sum_factors(
        adata,
        clusters="groups",
        parallelize=False,
        algorithm="CVXPY",
        max_size=3000,
        plotting=False,
        normalize_counts=False
    )