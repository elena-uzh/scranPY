import numpy as np
import scipy.sparse as sp
import anndata as ad
import scranPY


def test_sparse_normalization_shape():
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