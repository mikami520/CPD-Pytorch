from builtins import super
import numpy as np
import numbers
import torch as th
from .emregistration import EMRegistration
from .utils import gaussian_kernel

class DeformableRegistration(EMRegistration):
    """
    Deformable registration.
    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.
    beta: float(positive)
        Width of the Gaussian kernel.
    
    low_rank: bool
        Whether to use low rank approximation.
    
    num_eig: int
        Number of eigenvectors to use in lowrank calculation.
    """

    def __init__(self, alpha=None, beta=None, num_eig=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(alpha))

        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kerenl. Instead got: {}".format(beta))

        self.alpha = th.tensor(2) if alpha is None else alpha
        self.beta = th.tensor(2) if beta is None else beta
        self.W = th.zeros((self.M, self.D)).to(self.device)
        self.G = gaussian_kernel(self.Y, self.beta).to(self.device)
        self.num_eig = th.tensor(num_eig).to(self.device)

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.
        """
        A = th.mm(th.diag(self.P1.reshape(-1, )), self.G) + \
            self.alpha * self.sigma2 * th.eye(self.M).to(self.device)
        B = self.PX - th.mm(th.diag(self.P1.reshape(-1, )), self.Y)
        self.W = th.linalg.solve(A, B)


    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the deformable transformation.
        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.
                
        """
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + th.mm(G, self.W)
        else:
            self.TY = self.Y + th.mm(self.G, self.W)
            return


    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.
        """
        qprev = self.sigma2

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional will include terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        self.q = np.inf
        xPx = th.mm(self.Pt1.permute(1, 0), th.sum(th.mul(self.X, self.X), axis=1).reshape(-1, 1))
        yPy = th.mm(self.P1.permute(1, 0), th.sum(th.mul(self.TY, self.TY), axis=1).reshape(-1, 1))
        trPXY = th.sum(th.mul(self.TY, self.PX))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = th.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """
        Return the current estimate of the deformable transformation parameters.
        Returns
        -------
        self.G: numpy array
            Gaussian kernel matrix.
        self.W: numpy array
            Deformable transformation matrix.
        """
        return self.G, self.W
