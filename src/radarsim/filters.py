# -*- coding: utf-8 -*-
"""
Created on Tue Aug  28 15:25:59 2024

@author: harun
"""

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
import sys
import warnings

# Older versions of scipy do not support the allow_singular keyword. I could
# check the version number explicily, but perhaps this is clearer
_support_singular = True
try:
    multivariate_normal.logpdf(1, 1, 1, allow_singular=True)
except TypeError:
    warnings.warn(
        'You are using a version of SciPy that does not support the '\
        'allow_singular parameter in scipy.stats.multivariate_normal.logpdf(). '\
        'Future versions of FilterPy will require a version of SciPy that '\
        'implements this keyword',
        DeprecationWarning)
    _support_singular = False

class TempFilter:
    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = np.zeros((dim_x, 1))        # state
        self.P = np.eye(dim_x)               # uncertainty covariance
        self.Q = np.eye(dim_x)               # process uncertainty
        self.B = None                        # control transition matrix
        self.F = np.eye(dim_x)               # state transition matrix
        self.H = np.zeros((dim_z, dim_x))    # Measurement function
        self.R = np.eye(dim_z)               # state uncertainty
        self._alpha_sq = 1.                  # fading memory control
        self.M = np.zeros((dim_z, dim_z))    # process-measurement cross correlation
        self.z = np.array([[None]*self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z)) # kalman gain
        self.y = np.zeros((dim_z, 1))   # the residual
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty
        self.SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty
        
        ## Read Only
        self._log_likelihood = np.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        
    def logpdf(self, x, mean=None, cov=1, allow_singular=True):
        """
        Computes the log of the probability density function of the normal
        N(mean, cov) for the data x. The normal may be univariate or multivariate.

        Wrapper for older versions of scipy.multivariate_normal.logpdf which
        don't support support the allow_singular keyword prior to verion 0.15.0.

        If it is not supported, and cov is singular or not PSD you may get
        an exception.

        `x` and `mean` may be column vectors, row vectors, or lists.
        """

        if mean is not None:
            flat_mean = np.asarray(mean).flatten()
        else:
            flat_mean = None

        flat_x = np.asarray(x).flatten()

        if _support_singular:
            return multivariate_normal.logpdf(flat_x, flat_mean, cov, allow_singular)
        return multivariate_normal.logpdf(flat_x, flat_mean, cov)
    
    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = self.logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = np.exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood
    
    def reshape_z(self, z, dim_z, ndim):
        """ ensure z is a (dim_z, 1) shaped vector"""

        z = np.atleast_2d(z)
        if z.shape[1] == dim_z:
            z = z.T

        if z.shape != (dim_z, 1):
            raise ValueError('z must be convertible to shape ({}, 1)'.format(dim_z))

        if ndim == 1:
            z = z[:, 0]

        if ndim == 0:
            z = z[0, 0]

        return z

class KF(TempFilter):
    def __init__(self, dim_x, dim_z, dim_u=0):
        
        super().__init__(dim_x, dim_z, dim_u)

    
    def predict(self):
        # Compute the predicted mean x_prior and covariance matrix P_prior
        self.x = np.dot(self.F , self.x)
        self.P = self._alpha_sq * self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        self._likelihood = None
        self._log_likelihood = None
        z = self.reshape_z(z, self.dim_z, self.x.ndim)
        
        # Compute the innovation covariance matrix S and Kalman gain K
        PHT = self.P @ self.H.T
        self.S = self.H @ PHT + self.R
        self.SI = np.linalg.inv(self.S)
        self.K = PHT @ self.SI
        
        # Compute the residual y, posterior mean x_post and covariance matrix P_post
        self.y = z - np.dot(self.H, self.x)
        
        self.x = self.x + np.dot(self.K, self.y)
        I_KH = np.eye(self.dim_x) - self.K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T

class EKF(TempFilter):
    def __init__(self, dim_x, dim_z, dT, state_trans_func, state_trans_jacob_func, dim_u=0):
        
        super().__init__(dim_x, dim_z, dim_u)
        
        self.dT = dT
        self.state_trans_func = state_trans_func
        self.state_trans_jacob_func = state_trans_jacob_func
        
    def predict(self):
        # Compute the Jacobian of f at x_k−1|k−1
        self.F = self.state_trans_jacob_func(self.x, self.dT)
        
        # Compute predicted mean and covariance matrix
        self.x = self.state_trans_func(self.x, self.dT)
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    
    def update(self, z):
        self._likelihood = None
        self._log_likelihood = None
        z = self.reshape_z(z, self.dim_z, self.x.ndim)
        
        # Compute predicted measurement y, innovation covariance S and Kalman Gain K
        PHT = self.P @ self.H.T
        self.S = self.H @ PHT + self.R
        self.SI = np.linalg.inv(self.S)
        self.K = PHT @ self.SI
        
        # Compute posterior mean x_post and covariance matrix P_post
        self.y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(self.K, self.y)
        I_KH = np.eye(self.dim_x) - self.K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T
        

class IMMEstimator:
        def __init__(self, filters, mu, M):
            """ Implements an Interacting Multiple-Model (IMM) estimator.
    
            Parameters
            ----------
    
            filters : (N,) array_like of KalmanFilter objects
                List of N filters. filters[i] is the ith Kalman filter in the
                IMM estimator.
    
                Each filter must have the same dimension for the state `x` and `P`,
                otherwise the states of each filter cannot be mixed with each other.
    
            mu : (N,) array_like of float
                mode probability: mu[i] is the probability that
                filter i is the correct one.
    
            M : (N, N) ndarray of float
                Markov chain transition matrix. M[i,j] is the probability of
                switching from filter j to filter i.
    
    
            Attributes
            ----------
            x : numpy.array(dim_x, 1)
                Current state estimate. Any call to update() or predict() updates
                this variable.
    
            P : numpy.array(dim_x, dim_x)
                Current state covariance matrix. Any call to update() or predict()
                updates this variable.
    
            x_prior : numpy.array(dim_x, 1)
                Prior (predicted) state estimate. The *_prior and *_post attributes
                are for convienence; they store the  prior and posterior of the
                current epoch. Read Only.
    
            P_prior : numpy.array(dim_x, dim_x)
                Prior (predicted) state covariance matrix. Read Only.
    
            x_post : numpy.array(dim_x, 1)
                Posterior (updated) state estimate. Read Only.
    
            P_post : numpy.array(dim_x, dim_x)
                Posterior (updated) state covariance matrix. Read Only.
    
            N : int
                number of filters in the filter bank
    
            mu : (N,) ndarray of float
                mode probability: mu[i] is the probability that
                filter i is the correct one.
    
            M : (N, N) ndarray of float
                Markov chain transition matrix. M[i,j] is the probability of
                switching from filter j to filter i.
    
            cbar : (N,) ndarray of float
                Total probability, after interaction, that the target is in state j.
                We use it as the # normalization constant.
    
            likelihood: (N,) ndarray of float
                Likelihood of each individual filter's last measurement.
    
            omega : (N, N) ndarray of float
                Mixing probabilitity - omega[i, j] is the probabilility of mixing
                the state of filter i into filter j. Perhaps more understandably,
                it weights the states of each filter by:
                    x_j = sum(omega[i,j] * x_i)
    
                with a similar weighting for P_j
    
    
            Examples
            --------
    
            >>> import numpy as np
            >>> from filterpy.common import kinematic_kf
            >>> kf1 = kinematic_kf(2, 2)
            >>> kf2 = kinematic_kf(2, 2)
            >>> # do some settings of x, R, P etc. here, I'll just use the defaults
            >>> kf2.Q *= 0   # no prediction error in second filter
            >>>
            >>> filters = [kf1, kf2]
            >>> mu = [0.5, 0.5]  # each filter is equally likely at the start
            >>> trans = np.array([[0.97, 0.03], [0.03, 0.97]])
            >>> imm = IMMEstimator(filters, mu, trans)
            >>>
            >>> for i in range(100):
            >>>     # make some noisy data
            >>>     x = i + np.random.randn()*np.sqrt(kf1.R[0, 0])
            >>>     y = i + np.random.randn()*np.sqrt(kf1.R[1, 1])
            >>>     z = np.array([[x], [y]])
            >>>
            >>>     # perform predict/update cycle
            >>>     imm.predict()
            >>>     imm.update(z)
            >>>     print(imm.x.T)
    
            For a full explanation and more examples see my book
            Kalman and Bayesian Filters in Python
            https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    
    
            References
            ----------
    
            Bar-Shalom, Y., Li, X-R., and Kirubarajan, T. "Estimation with
            Application to Tracking and Navigation". Wiley-Interscience, 2001.
    
            Crassidis, J and Junkins, J. "Optimal Estimation of
            Dynamic Systems". CRC Press, second edition. 2012.
    
            Labbe, R. "Kalman and Bayesian Filters in Python".
            https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
            """
            # if len(filters) < 2:
            #     raise ValueError('filters must contain at least two filters')

            self.filters = filters
            self.mu = np.asarray(mu) / np.sum(mu)
            self.M = M

            x_shape = filters[0].x.shape
            for f in filters:
                if x_shape != f.x.shape:
                    raise ValueError(
                        'All filters must have the same state dimension')

            self.x = np.zeros(filters[0].x.shape)
            self.P = np.zeros(filters[0].P.shape)
            self.N = len(filters)  # number of filters
            self.likelihood = np.zeros(self.N)
            self.omega = np.zeros((self.N, self.N))
            if self.N != 1:
                self._compute_mixing_probabilities()

            # initialize imm state estimate based on current filters
            if self.N != 1:
                self._compute_state_estimate()
            self.x_prior = self.x.copy()
            self.P_prior = self.P.copy()
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()

        def update(self, z):
            """
            Add a new measurement (z) to the Kalman filter. If z is None, nothing
            is changed.

            Parameters
            ----------

            z : np.array
                measurement for this update.
            """
            if self.N == 1:
                self.filters[0].update(z)
                self.x = self.filters[0].x
                self.P = self.filters[0].P
                return

            # run update on each filter, and save the likelihood
            for i, f in enumerate(self.filters):
                f.update(z)
                self.likelihood[i] = f.likelihood

            # update mode probabilities from total probability * likelihood
            self.mu = self.cbar * self.likelihood
            self.mu /= np.sum(self.mu)  # normalize

            self._compute_mixing_probabilities()

            # compute mixed IMM state and covariance and save posterior estimate
            self._compute_state_estimate()
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()

        def predict(self, u=None):
            """
            Predict next state (prior) using the IMM state propagation
            equations.

            Parameters
            ----------

            u : np.array, optional
                Control vector. If not `None`, it is multiplied by B
                to create the control input into the system.
            """
            
            if self.N == 1:
                self.filters[0].predict()
                self.x = self.filters[0].x
                self.P = self.filters[0].P
                return

            # compute mixed initial conditions
            xs, Ps = [], []
            for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):
                x = np.zeros(self.x.shape)
                for kf, wj in zip(self.filters, w):
                    x += kf.x * wj
                xs.append(x)

                P = np.zeros(self.P.shape)
                for kf, wj in zip(self.filters, w):
                    y = kf.x - x
                    P += wj * (np.outer(y, y) + kf.P)
                Ps.append(P)

            #  compute each filter's prior using the mixed initial conditions
            for i, f in enumerate(self.filters):
                # propagate using the mixed state estimate and covariance
                f.x = xs[i].copy()
                f.P = Ps[i].copy()
                f.predict()

            # compute mixed IMM state and covariance and save posterior estimate
            self._compute_state_estimate()
            self.x_prior = self.x.copy()
            self.P_prior = self.P.copy()

        def _compute_state_estimate(self):
            """
            Computes the IMM's mixed state estimate from each filter using
            the the mode probability self.mu to weight the estimates.
            """
            self.x.fill(0)
            for f, mu in zip(self.filters, self.mu):
                self.x += f.x * mu

            self.P.fill(0)
            for f, mu in zip(self.filters, self.mu):
                y = f.x - self.x
                self.P += mu * (np.outer(y, y) + f.P)

        def _compute_mixing_probabilities(self):
            """
            Compute the mixing probability for each filter.
            """

            self.cbar = np.dot(self.mu, self.M)
            for i in range(self.N):
                for j in range(self.N):
                    self.omega[i, j] = (self.M[i, j]*self.mu[i]) / self.cbar[j]



def order_by_derivative(Q, dim, block_size):
    """
    Given a matrix Q, ordered assuming state space
        [x y z x' y' z' x'' y'' z''...]

    return a reordered matrix assuming an ordering of
       [ x x' x'' y y' y'' z z' y'']

    This works for any covariance matrix or state transition function

    Parameters
    ----------
    Q : np.array, square
        The matrix to reorder

    dim : int >= 1

       number of independent state variables. 3 for x, y, z

    block_size : int >= 0
        Size of derivatives. Second derivative would be a block size of 3
        (x, x', x'')


    """

    N = dim * block_size

    D = np.zeros((N, N))

    Q = np.array(Q)
    for i, x in enumerate(Q.ravel()):
        f = np.eye(block_size) * x

        ix, iy = (i // dim) * block_size, (i % dim) * block_size
        D[ix:ix+block_size, iy:iy+block_size] = f

    return D
    
def Q_kinematic(dim, dt=1., var=1., block_size=1, order_by_dim=True):
    """
    Returns the Q matrix for the Discrete Constant White Noise
    Model. dim may be either 2, 3, or 4 dt is the time step, and sigma
    is the variance in the noise.

    Q is computed as the G * G^T * variance, where G is the process noise per
    time step. In other words, G = [[.5dt^2][dt]]^T for the constant velocity
    model.

    Parameters
    -----------

    dim : int (2, 3, or 4)
        dimension for Q, where the final dimension is (dim x dim)

    dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations

    var : float, default=1.0
        variance in the noise

    block_size : int >= 1
        If your state variable contains more than one dimension, such as
        a 3d constant velocity model [x x' y y' z z']^T, then Q must be
        a block diagonal matrix.

    order_by_dim : bool, default=True
        Defines ordering of variables in the state vector. `True` orders
        by keeping all derivatives of each dimensions)

        [x x' x'' y y' y'']

        whereas `False` interleaves the dimensions

        [x y z x' y' z' x'' y'' z'']


    Examples
    --------
    >>> # constant velocity model in a 3D world with a 10 Hz update rate
    >>> Q_discrete_white_noise(2, dt=0.1, var=1., block_size=3)
    array([[0.000025, 0.0005  , 0.      , 0.      , 0.      , 0.      ],
           [0.0005  , 0.01    , 0.      , 0.      , 0.      , 0.      ],
           [0.      , 0.      , 0.000025, 0.0005  , 0.      , 0.      ],
           [0.      , 0.      , 0.0005  , 0.01    , 0.      , 0.      ],
           [0.      , 0.      , 0.      , 0.      , 0.000025, 0.0005  ],
           [0.      , 0.      , 0.      , 0.      , 0.0005  , 0.01    ]])

    References
    ----------

    Bar-Shalom. "Estimation with Applications To Tracking and Navigation".
    John Wiley & Sons, 2001. Page 274.
    """

    if not (dim == 2 or dim == 3 or dim == 4):
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [[.25*dt**4, .5*dt**3],
             [ .5*dt**3,    dt**2]]
    elif dim == 3:
        Q = [[.25*dt**4, .5*dt**3, .5*dt**2],
             [ .5*dt**3,    dt**2,       dt],
             [ .5*dt**2,       dt,        1]]
    else:
        Q = [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
             [(dt**5)/12, (dt**4)/4,  (dt**3)/2, (dt**2)/2],
             [(dt**4)/6,  (dt**3)/2,   dt**2,     dt],
             [(dt**3)/6,  (dt**2)/2 ,  dt,        1.]]

    if order_by_dim:
        return block_diag(*[Q]*block_size) * var
    return order_by_derivative(np.array(Q), dim, block_size) * var

"""
The MIT License (MIT)

Copyright (c) 2015 Roger R. Labbe Jr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""