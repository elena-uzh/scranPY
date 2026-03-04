import numpy as np
import scipy.sparse as sp
import anndata as ad
import scranPY


def test_no_genes_pass_filtering():
    n_cells = 100
    n_genes = 200

    # Very sparse low-count data so no genes pass min_mean filter
    np.random.seed(42)
    X = sp.random(n_cells, n_genes, density=0.05, format="csr")
    adata = ad.AnnData(X)
    adata.obs["groups"] = ["A"] * n_cells

    with pytest.raises(ValueError, match="No genes passed the min_mean filter."):
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