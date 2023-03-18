import torch as th

def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = th.sub(X[:, None, :], Y[None, :,  :])
    diff = th.square(diff)
    diff = th.sum(diff, dim=2)
    return th.exp(-diff / (2 * beta**2))


def is_positive_semi_definite(R):
    if not th.is_tensor(R):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a tensor, instead got : {}'.format(R))
    return th.all(th.real(th.linalg.eigvals(R))>0)

def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.
    """
    S, Q = th.linalg.eigh(G)
    eig_indices = th.flip(th.argsort(th.abs(S)), dims=(0, ))[:num_eig].tolist()
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S