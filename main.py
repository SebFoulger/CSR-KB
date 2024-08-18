import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
from typing import Literal
from scipy.linalg.lapack import dtrtri
from numpy.typing import ArrayLike
import warnings

class CSR_KB:
    """
    Class for continuous segmented regression with known breakpoints. 
    """    
    def __init__(self, 
                 endog: ArrayLike, 
                 breakpoints: ArrayLike, 
                 exog: ArrayLike = None, 
                 weights: ArrayLike = None, 
                 seg_weights: ArrayLike = None, 
                 hasconst: bool = False) -> None:
        """Initializer function for continuous segmented regression with known breakpoints class.

        Args:
            endog (ArrayLike): Endogenous variable.
            breakpoints (ArrayLike): Indices of breakpoints in exogenous data.
            exog (ArrayLike, optional): Exogenous variable. If set to None, exog will be 0, 1,..., nobs. Defaults to 
            None.
            weights (ArrayLike, optional): Weights for observations. Length of weights must be equal to the number of
            observations. Set to None if no observation weights are desired. Defaults to None.
            seg_weights (ArrayLike, optional): Weights for segments. Length of seg_weights must be equal to the number
            of segments. Set to None if no segment weights desired. Defaults to None.
            hasconst (bool, optional): Indicates whether exogenous variable contains a constant. Defaults to False.
        """             
        self.endog = endog
        # number of observations
        self.nobs = len(endog)
        self.breakpoints = np.sort(breakpoints)
        # Ensure breakpoints starts with 0
        if breakpoints[0] != 0:
            self.breakpoints = np.insert(self.breakpoints, 0, 0)
        # Ensure breakpoints ends with nobs
        if breakpoints[-1] != self.nobs:
            self.breakpoints = np.append(self.breakpoints, self.nobs)
        self.nsegs = len(breakpoints) - 1
        # Sizes of segments
        self.seg_sizes = np.diff(breakpoints)
        
        if exog is None:
            if hasconst:
                warnings.warn(('No exogenous variable provided but hasconst is set to True. Setting hasconst to False ' 
                               'and proceeding.'), RuntimeWarning)
                hasconst = False
            self.exog = np.arange(len(endog))
        else:
            if len(exog) != self.nobs:
                raise ValueError('Length of exog must be equal to length of endog') 
            self.exog = exog
        # exog_no_const is exogenous variable with no constant
        self.exog_no_const = self.exog
        if not hasconst:
            self.exog = sm.add_constant(self.exog)
        
        if weights is not None:
            if len(weights) != self.nobs:
                raise ValueError('Length of weights must be equal to length of endog')
            
        self.weights = weights
        # combined_weights combines the segment and observation weights
        self.combined_weights = weights

        if seg_weights is not None:
            if len(seg_weights) != self.nsegs:
                raise ValueError('Length of seg_weights must be equal to number of segments')
            # Segment weights are converted into a form so they can be treated as observation weights
            combined_seg_weights = np.array([])
            for i in range(self.nsegs):
                cur_seg_weight = np.repeat(seg_weights[i], self.seg_sizes[i])
                combined_seg_weights = np.append(combined_seg_weights, cur_seg_weight)

            if self.combined_weights is not None:
                # If both segment weights and observation weights have been supplied, combine them
                self.combined_weights = self.combined_weights * combined_seg_weights
            else:
                # If only segment weights have been supplied, use these as the combined weights
                self.combined_weights = combined_seg_weights
        self.seg_weights = seg_weights
        # Whiten exogenous and endogenous variables
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)
        # wexog_init stores the first observation of each segment
        self.wexog_init = self.wexog[self.breakpoints[:-1]]

    """Public Methods"""

    def whiten(self, 
               x: ArrayLike) -> ArrayLike:
        """Whitening method for model data.

        Args:
            x (ArrayLike): data to whiten. 

        Returns:
            ArrayLike: Whitened data (according to segment weights and observation weights).
        """        
        if self.combined_weights is not None:
            if x.ndim == 1:
                return np.sqrt(self.combined_weights) * x 
            else:
                return np.sqrt(self.combined_weights)[:, None] * x 
        else:
            return x

    def fit(self, 
            method: Literal['svd', 'qr', 'qr2', 'normal'] = 'svd') -> ArrayLike:
        """Fitting method.

        This method calculates (X.T @ X)^(-1) and (X.T @ X)^(-1) @ X.T @ y for exogenous data X and endogenous data y
        of each segment, passing these on to _calculate_betas which does the rest of the calculation.

        Args:
            method (Literal['svd', 'qr', 'qr2', 'normal'], optional): method to use for fitting. Defaults to 'svd'.

        Returns:
            ArrayLike: beta coefficients for the model.
        """
        m = self.nsegs
        # _beta will store (X.T @ X)^(-1) @ X.T @ y for exogenous data X and endogenous data y of each segment
        _beta = []
        
        if method == 'svd':
            # XTX will store (X.T @ X)^(-1) for exogenous data X of each segment
            XTX = []
            for i in range(m):
                # Data from current segment
                X = self.wexog[self.breakpoints[i]:self.breakpoints[i+1]]
                y = self.wendog[self.breakpoints[i]:self.breakpoints[i+1]]

                u, s, vt = np.linalg.svd(X, 0)
                v = vt.T

                sd = s * s
                vs = v / sd

                XTX.append(np.dot(vs, vt))
                _beta.append(np.dot(vs * s, np.dot(u.T, y)))

            XTX = np.array(XTX)
            kwargs = {'XTX': XTX}

        elif method == 'qr':
            # Q and R will store the Q and R components of the decompositions of exogenous data in each segment
            Q = []
            R = []

            for i in range(m):
                # Data from current segment
                X = self.wexog[self.breakpoints[i]:self.breakpoints[i+1]]
                y = self.wendog[self.breakpoints[i]:self.breakpoints[i+1]]
                Q_, R_ = np.linalg.qr(X)
                Q.append(Q_)
                R.append(R_)
                _beta.append(scipy.linalg.solve_triangular(R[i], np.dot(Q[i].T, y), check_finite = False))

            # XTX_x1 will store (X[i].T @ X[i])^(-1) x[i+1] for each segment. See documentation for more details
            XTX_x1 = []
            # XTX_x2 will store (X[i+1].T @ X[i+1])^(-1) x[i+1] for each segment.
            XTX_x2 = []

            for i in range(m - 1):
                x = self.wexog_init[i+1]

                _XTX_x1 = scipy.linalg.solve_triangular(R[i].T, x, lower = True, check_finite = False)
                XTX_x1.append(scipy.linalg.solve_triangular(R[i], _XTX_x1, check_finite = False))

                _XTX_x2 = scipy.linalg.solve_triangular(R[i+1].T, x, lower = True, check_finite = False)
                XTX_x2.append(scipy.linalg.solve_triangular(R[i+1], _XTX_x2, check_finite = False))

            XTX_x1 = np.array(XTX_x1)
            XTX_x2 = np.array(XTX_x2)

            kwargs = {'XTX_x1': XTX_x1, 'XTX_x2': XTX_x2}

        elif method == 'qr2':
            # XTX will store (X.T @ X)^(-1) for exogenous data X of each segment
            XTX = []
            for i in range(m):
                # Data from current segment
                X = self.wexog[self.breakpoints[i]:self.breakpoints[i+1]]
                y = self.wendog[self.breakpoints[i]:self.breakpoints[i+1]]

                Q, R = np.linalg.qr(X)
                # Invert upper triangular matrix R
                R_i = dtrtri(R)[0]

                XTX.append(np.dot(R_i, R_i.T))
                _beta.append(np.dot(XTX[i], np.dot(X.T, y)))

            XTX = np.array(XTX)
            kwargs = {'XTX': XTX}

        elif method == 'normal':
            # XTX will store (X.T @ X)^(-1) for exogenous data X of each segment
            XTX = []
            for i in range(m):
                # Data from current segment
                X = self.wexog[self.breakpoints[i]:self.breakpoints[i+1]]
                y = self.wendog[self.breakpoints[i]:self.breakpoints[i+1]]

                XTX.append(np.linalg.inv(np.dot(X.T, X)))
                _beta.append(np.linalg.multi_dot([XTX[i], X.T, y]))

            XTX = np.array(XTX)
            kwargs = {'XTX': XTX}

        else:
            raise ValueError('method has to be "svd", "normal", "qr", or "qr2"')
        
        _beta = np.array(_beta)
        # Finish calculating betas from current calculated data
        betas = self._calculate_betas(method = method, _beta = _beta, **kwargs)
        self.betas = betas
        # Calculate model in-sample predictions
        y_hat = np.array([])
        for i, (prev_break, next_break) in enumerate(zip(self.breakpoints[:-1], self.breakpoints[1:])):
            y_hat = np.append(y_hat, self.exog[prev_break:next_break, :].dot(self.betas[i]))
        self.y_hat = y_hat
        return betas

    def fit_regularized(self, 
                        mu: float, 
                        method: Literal['svd', 'normal'] = 'svd') -> ArrayLike:
        """Fits L2 regularised model.

        This method calculates (X.T @ X + mu * I)^(-1) and (X.T @ X + mu * I)^(-1) @ X.T @ y for exogenous data X and 
        endogenous data y of each segment, passing these on to _calculate_betas which does the rest of the calculation.

        Args:
            mu (float): L2 penalty parameter.
            method (Literal['svd', 'normal'], optional): Method for fitting regularized model. Defaults to 'svd'.

        Returns:
            ArrayLike: beta coefficients for the model.
        """        
        m = self.nsegs
        # XTX will store (X.T @ X)^(-1) for exogenous data X of each segment
        XTX = []
        # _beta will store (X.T @ X)^(-1) @ X.T @ y for exogenous data X and endogenous data y of each segment
        _beta = []

        if method == 'svd':

            for i in range(m):
                # Data from current segment
                X = self.wexog[self.breakpoints[i]:self.breakpoints[i+1]]
                y = self.wendog[self.breakpoints[i]:self.breakpoints[i+1]]

                u, s, vt = np.linalg.svd(X, 0)
                v = vt.T

                sd = s * s + mu[i]
                vs = v / sd

                XTX.append(np.dot(vs, vt))
                _beta.append(np.dot(vs * s, np.dot(u.T, y)))

        elif method == 'normal':
            
            for i in range(m):
                # Data from current segment
                X = self.wexog[self.breakpoints[i]:self.breakpoints[i+1]]
                y = self.wendog[self.breakpoints[i]:self.breakpoints[i+1]]

                XTX.append(np.linalg.inv(np.dot(X.T, X) + mu[i]*np.identity(X.shape[1])))
                _beta.append(np.linalg.multi_dot([XTX[i], X.T, y]))

        else:
            raise ValueError('method has to be "svd" or "normal"')

        XTX = np.array(XTX)
        _beta = np.array(_beta)
        # Finish calculating betas from current calculated data
        betas = self._calculate_betas(method = method, _beta = _beta, XTX = XTX)
        self.betas = betas
        # Calculate model in-sample predictions
        y_hat = np.array([])
        for i, (prev_break, next_break) in enumerate(zip(self.breakpoints[:-1], self.breakpoints[1:])):
            y_hat = np.append(y_hat, self.exog[prev_break:next_break, :].dot(self.betas[i]))
        self.y_hat = y_hat
        return betas

    def plot(self,
             plot_model: bool = True,
             plot_data: bool = True,
             model_color: str = None,
             data_color: str = None,
             title: str = 'CSR-KB plot',
             xlabel: str = None,
             ylabel: str = None,
             legend: bool = True,
             model_label: str = 'Model',
             data_label: str = 'Data',
             plot_breakpoints: bool = True) -> None:
        """Plot model and/or original data.

        Args:
            plot_model (bool, optional): set to True to plot model data. Defaults to True.
            plot_data (bool, optional): set to True to plot original data. Defaults to True.
            model_color (str, optional): color of model data line. Set to None for default color. Defaults to None.
            data_color (str, optional): color of original data line. Set to None for default color. Defaults to None.
            title (str, optional): title for plot. Defaults to 'CSR-KB plot'.
            xlabel (str, optional): x-axis label for plot. Set to None for no label. Defaults to None.
            ylabel (str, optional): y-axis label for plot. Set to None for no label. Defaults to None.
            legend (bool, optional): set to True to include legend in plot. Defaults to True.
            model_label (str, optional): model data line label. Defaults to 'Model'.
            data_label (str, optional): original data line label. Defaults to 'Data'.
            plot_breakpoints (bool, optional): set to True to plot breakpoints. Defaults to True.
        """ 

        if not plot_model and not plot_data:
            raise ValueError('Either plot_model or plot_data should be set to True.')

        fig, ax = plt.subplots() 

        # Plot original data
        if plot_data:
            ax.plot(self.exog_no_const, self.endog, color=data_color, label=data_label)
        # Check if betas have been calculated
        if not hasattr(self, 'betas'):
            if not plot_data:
                raise ValueError('No data to plot. Neither fit nor fit_regularised have been called.')
        else:
            ax.plot(self.exog_no_const, self.y_hat, color=model_color, label=model_label)

        if plot_breakpoints:
            ax.axvline(x = self.exog_no_const[self.breakpoints[1]], color = 'r', linestyle='dashed', linewidth=1, 
                       label='Breakpoints')
            for break_ in self.breakpoints[2:-1]:
                ax.axvline(x = self.exog_no_const[break_], color = 'r', linestyle='dashed', linewidth=1)

        if legend:
            ax.legend()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    """Private Methods"""

    def _calculate_betas(self, 
                         _beta: ArrayLike,
                         method: Literal['svd', 'qr', 'qr2', 'normal'] = 'svd',
                         XTX: ArrayLike = None, 
                         XTX_x1: ArrayLike = None, 
                         XTX_x2: ArrayLike = None) -> ArrayLike:
        """Calculates final beta estimates.

        This method takes data calculated in fit() or fit_regularized(), forms the linear equation for the lambda 
        vector (STEP 1), solves this equation (STEP 2), and then calculates the final beta estimates (STEP 3).

        Args:
            _beta (ArrayLike): standard linear regression beta estimates for given data.
            method (Literal['svd', 'qr', 'qr2', 'normal'], optional): method for fitting model. Defaults to 'svd'.
            XTX (ArrayLike, optional): inverse of (X[i].T @ X[i]) where X[i] refers to the data from the i-th segment. 
            Required for 'svd', 'qr2', and 'normal' methods. Defaults to None.
            XTX_x1 (ArrayLike, optional): (X[i].T @ X[i])^(-1) * wexog_init[i+1] where X[i] refers to the data from the 
            i-th segment. Only required if method is 'qr'. Defaults to None.
            XTX_x2 (ArrayLike, optional): (X[i+1].T @ X[i+1])^(-1) * wexog_init[i+1] where X[i] refers to the data from 
            the i-th segment. Only required if method is 'qr'. Defaults to None.

        Returns:
            ArrayLike: Calculated beta estimates.
        """        
        m = self.nsegs
        
        # Will store the matrix A for calculating the lambdas.
        A = np.zeros((m - 1, m - 1))
        # Will store the vector c for calculating the lambdas.
        c = np.zeros(m - 1)

        if method == 'qr':
            if XTX_x1 is None or XTX_x2 is None:
                raise ValueError(('"qr" has not been used to fit but is now attempting to be used for calculating') 
                                 ('betas. Ensure "qr" is selected in fit() if it\'s desired for _calculate_betas().'))
            # STEP 1: form linear equation, i.e. calculate matrix A and vector c. See documentation for more details.

            A[0][0] = np.dot(self.wexog_init[1], XTX_x1[0] + XTX_x2[0])
            A[0][1] = -np.dot(self.wexog_init[1], XTX_x1[1])
            
            c[0] = np.dot(self.wexog_init[1], _beta[0] - _beta[1]).item()

            for j in range(1, m - 2):
                # A is self-transpose so don't need to recalculate values
                A[j][j-1] = A[j-1][j]
                A[j][j] = np.dot(self.wexog_init[j+1], XTX_x1[j] + XTX_x2[j])
                A[j][j+1] = -np.dot(self.wexog_init[j+1], XTX_x1[j+1])

                c[j] = np.dot(self.wexog_init[j+1], _beta[j] - _beta[j+1]).item()

            A[m-2][m-3] = A[m-3][m-2]
            A[m-2][m-2] = np.dot(self.wexog_init[m-1], XTX_x1[m-2] + XTX_x2[m-2])

            c[m-2] = np.dot(self.wexog_init[m-1], _beta[m-2] - _beta[m-1]).item()

            # STEP 2: calculate lambda values

            # A is converted into a 3 x m-1 matrix since it is tri-diagonal
            ab = np.zeros((3, m-1))
            for i in range(m-1):
                for j in range(max(0, i-1), min([2+i, m-1])):
                    ab[1+i-j, j] = A[i, j]
            # Solve linear equation with tri-diagonal matrix
            L = scipy.linalg.solve_banded((1,1), ab, c)

            # STEP 3: calculate final beta estimates
            betas = []
            betas.append(_beta[0] - L[0] * XTX_x1[0])
            for j in range(1, m-1):
                betas.append(_beta[j] + L[j-1] * XTX_x2[j-1] - L[j] * XTX_x1[j])
            betas.append(_beta[m-1] + L[m-2] * XTX_x2[m-2])
        else:
            if XTX is None:
                raise ValueError(('"qr" has been used to fit but is now not being used for calculating betas. ') 
                                 ('Ensure "qr" is selected in _calculate_betas() if it\'s desired for fit().'))
            # STEP 1: form linear equation, i.e. calculate matrix A and vector c. See documentation for more details.

            A[0][0] = np.linalg.multi_dot([self.wexog_init[1], XTX[0] + XTX[1], self.wexog_init[1]])
            A[0][1] = -np.linalg.multi_dot([self.wexog_init[1], XTX[1], self.wexog_init[2]])
            c[0] = np.dot(self.wexog_init[1], _beta[0] - _beta[1]).item()
            
            for j in range(1, m - 2):
                # A is self-transpose so don't need to recalculate values
                A[j][j-1] = A[j-1][j]
                A[j][j] = np.linalg.multi_dot([self.wexog_init[j+1], XTX[j] + XTX[j+1], self.wexog_init[j+1]])
                A[j][j+1] = -np.linalg.multi_dot([self.wexog_init[j+1], XTX[j+1], self.wexog_init[j+2]])

                c[j] = np.dot(self.wexog_init[j+1], _beta[j] - _beta[j+1]).item()

            A[m-2][m-3] = A[m-3][m-2]
            A[m-2][m-2] = np.linalg.multi_dot([self.wexog_init[m-1], XTX[m-2] + XTX[m-1], self.wexog_init[m-1]])

            c[m-2] = np.dot(self.wexog_init[m-1], _beta[m-2] - _beta[m-1]).item()

            # STEP 2: calculate lambda values

            # A is converted into a 3 x m-1 matrix since it is tri-diagonal
            ab = np.zeros((3, m-1))
            for i in range(m-1):
                for j in range(max(0, i-1), min([2+i, m-1])):
                    ab[1+i-j, j] = A[i, j]
            # Solve linear equation with tri-diagonal matrix
            L = scipy.linalg.solve_banded((1,1), ab, c)

            # STEP 3: calculate final beta estimates
            betas = []
            betas.append(_beta[0]-L[0] * XTX[0].dot(self.wexog_init[1]))
            for j in range(1, m-1):
                betas.append(_beta[j]+ XTX[j].dot(L[j-1] * self.wexog_init[j] - L[j] * self.wexog_init[j+1]))
            betas.append(_beta[m-1]+L[m-2] * XTX[m-1].dot(self.wexog_init[m-1]))

        return betas
    
def is_pos_def(x):
    return np.linalg.eigvals(x)