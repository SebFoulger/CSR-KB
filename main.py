# pylint: disable=C0114, C0103, W0105, E1131, C0411, E0611
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

    def __init__(
        self,
        endog: ArrayLike,
        breakpoints: ArrayLike,
        exog: ArrayLike = None,
        obs_weights: ArrayLike = None,
        seg_weights: ArrayLike = None,
        hasconst: bool = False,
    ) -> None:
        """Initializer function for continuous segmented regression with known breakpoints class.

        Parameters
        ----------
        endog : ArrayLike
            Endogenous variable.
        breakpoints : ArrayLike
            Indices of breakpoints in exogenous data.
        exog : ArrayLike, optional
            Exogenous variable. If set to None, exog will be 0, 1,..., nobs. By default None.
        obs_weights : ArrayLike, optional
            Weights for observations. Length of obs_weights must be equal to the number of
            observations. Set to None if no observation weights are desired. By default None.
        seg_weights : ArrayLike, optional
            Weights for segments. Length of seg_weights must be equal to the number of segments. Set
            to None if no segment weights desired. By default None.
        hasconst : bool, optional
            Indicates whether exogenous variable contains a constant. By default False.
        """

        self.endog = endog
        # Number of observations
        self.nobs = len(endog)
        if exog.ndim == 1:
            self.ndim = 1
        else:
            self.ndim = exog.shape[1]
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
                warnings.warn(
                    (
                        "No exogenous variable provided but hasconst is set to True. Setting"
                        " hasconst to False and proceeding."
                    ),
                    RuntimeWarning,
                )
                hasconst = False
            self.exog = np.arange(len(endog))
        else:
            if len(exog) != self.nobs:
                raise ValueError("Length of exog must be equal to length of endog")
            self.exog = exog
        # Check that the length of the observation weights is equal to the number of observations
        if obs_weights is not None:
            if len(obs_weights) != self.nobs:
                raise ValueError(
                    "Length of obs_weights must be equal to length of endog"
                )
        # exog_no_const is exogenous variable with no constant
        self.exog_no_const = self.exog
        if not hasconst:
            self.exog = sm.add_constant(self.exog)
        # Defining in-sample exogenous values
        in_sample = []
        for i in range(self.nsegs):
            # This includes the first x value of the next segment since the sample space should
            # extend from one segment to the next, i.e. there is no gaps between segments for the
            # sample space. See the mathematical documentation for more details.
            X = self.exog[self.breakpoints[i] : (self.breakpoints[i + 1] + 1)]
            if self.ndim == 1:
                X_min = np.min(X)
                X_max = np.max(X)
                rec = np.array([[X_min, X_max]])
            else:
                X_min = np.min(X, axis=0)
                X_max = np.max(X, axis=0)
                rec = np.array(list(zip(X_min, X_max)))
            in_sample.append(rec)
        # Warning if segments overlap
        overlap_warnings = []
        for i, rec1 in enumerate(in_sample):
            for j, rec2 in enumerate(in_sample[i + 1 :]):
                overlap = (rec1[:, 0] > rec2[:, 0]).all() and (
                    rec1[:, 0] < rec2[:, 1]
                ).all()
                overlap = overlap or (
                    (rec1[:, 1] > rec2[:, 0]).all() and (rec1[:, 1] < rec2[:, 1]).all()
                )
                if overlap:
                    overlap_warnings.append((i, j))
        if len(overlap_warnings) > 0:
            warnings.warn(
                (
                    "Some segments overlap, meaning model will be one-to-many. These "
                    "overlapping segments are",
                    overlap_warnings,
                ),
                RuntimeWarning,
            )
        self.in_sample = in_sample
        self.overlap_warnings = overlap_warnings

        self.obs_weights = obs_weights
        # weights combines the segment and observation weights
        self.weights = obs_weights

        if seg_weights is not None:
            if len(seg_weights) != self.nsegs:
                raise ValueError(
                    "Length of seg_weights must be equal to number of segments"
                )
            # Segment weights are converted so they can be treated as observation weights
            combined_seg_weights = np.empty(0)
            for i in range(self.nsegs):
                cur_seg_weight = np.repeat(seg_weights[i], self.seg_sizes[i])
                combined_seg_weights = np.append(combined_seg_weights, cur_seg_weight)

            if self.weights is not None:
                # If both segment weights and observation weights have been supplied, combine them
                self.weights = self.weights * combined_seg_weights
            else:
                # If only segment weights have been supplied, use these as the combined weights
                self.weights = combined_seg_weights
        self.seg_weights = seg_weights
        # Whiten exogenous and endogenous variables
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)
        # wexog_init stores the first observation of each segment
        self.wexog_init = self.wexog[self.breakpoints[:-1]]
        self.rank = np.linalg.matrix_rank(self.exog)
        self.df_model = float(self.rank - 1)
        self.df_resid = self.nobs - self.rank
        # Defining variables to be added in future functions
        self.betas = None
        self.y_hat = None
        self._llf = None
        self._aic = None
        self._bic = None

    # pylint: disable=W0105
    """Public Methods"""

    def whiten(self, x: ArrayLike) -> ArrayLike:
        """Whitening method for model data.

        Parameters
        ----------
        x : ArrayLike
            Data to whiten.

        Returns
        -------
        ArrayLike
            Whitened data (according to segment weights and observation weights).
        """
        if self.weights is not None:
            if x.ndim == 1:
                return np.sqrt(self.weights) * x
            else:
                return np.sqrt(self.weights)[:, None] * x
        else:
            return x

    def fit(self, method: Literal["svd", "qr", "qr2", "normal"] = "svd") -> ArrayLike:
        """Fitting method.

        This method calculates (X.T @ X)^(-1) and (X.T @ X)^(-1) @ X.T @ y for exogenous data X and
        endogenous data y of each segment, passing these on to _calculate_betas which does the rest
        of the calculation.

        Parameters
        ----------
        method : Literal['svd', 'qr', 'qr2', 'normal'], optional
            Method to use for fitting. By default 'svd'

        Returns
        -------
        ArrayLike
            Beta coefficients for the model.
        """
        m = self.nsegs
        # _beta will store (X.T @ X)^(-1) @ X.T @ y for exogenous X and endogenous y of each segment
        _beta = []

        if method == "svd":
            # XTX will store (X.T @ X)^(-1) for exogenous data X of each segment
            XTX = []
            for i in range(m):
                # Data from current segment
                X = self.wexog[self.breakpoints[i] : self.breakpoints[i + 1]]
                y = self.wendog[self.breakpoints[i] : self.breakpoints[i + 1]]

                u, s, vt = np.linalg.svd(X, 0)
                v = vt.T

                sd = s * s
                vs = v / sd

                XTX.append(np.dot(vs, vt))
                _beta.append(np.dot(vs * s, np.dot(u.T, y)))

            XTX = np.array(XTX)
            kwargs = {"XTX": XTX}

        elif method == "qr":
            # Q and R will store the components of QR decompositions of exogenous data in segments
            Q = []
            R = []

            for i in range(m):
                # Data from current segment
                X = self.wexog[self.breakpoints[i] : self.breakpoints[i + 1]]
                y = self.wendog[self.breakpoints[i] : self.breakpoints[i + 1]]
                Q_, R_ = np.linalg.qr(X)
                Q.append(Q_)
                R.append(R_)
                _beta.append(
                    scipy.linalg.solve_triangular(
                        R_, np.dot(Q_.T, y), check_finite=False
                    )
                )

            # XTX_x1 will store (X[i].T @ X[i])^(-1) x[i+1] for each segment.
            XTX_x1 = []
            # XTX_x2 will store (X[i+1].T @ X[i+1])^(-1) x[i+1] for each segment.
            XTX_x2 = []

            for i in range(m - 1):
                x = self.wexog_init[i + 1]

                _XTX_x1 = scipy.linalg.solve_triangular(
                    R[i].T, x, lower=True, check_finite=False
                )
                XTX_x1.append(
                    scipy.linalg.solve_triangular(R[i], _XTX_x1, check_finite=False)
                )

                _XTX_x2 = scipy.linalg.solve_triangular(
                    R[i + 1].T, x, lower=True, check_finite=False
                )
                XTX_x2.append(
                    scipy.linalg.solve_triangular(R[i + 1], _XTX_x2, check_finite=False)
                )

            XTX_x1 = np.array(XTX_x1)
            XTX_x2 = np.array(XTX_x2)

            kwargs = {"XTX_x1": XTX_x1, "XTX_x2": XTX_x2}

        elif method == "qr2":
            # XTX will store (X.T @ X)^(-1) for exogenous data X of each segment
            XTX = []
            for i in range(m):
                # Data from current segment
                X = self.wexog[self.breakpoints[i] : self.breakpoints[i + 1]]
                y = self.wendog[self.breakpoints[i] : self.breakpoints[i + 1]]

                Q, R = np.linalg.qr(X)
                # Invert upper triangular matrix R
                R_i = dtrtri(R)[0]

                XTX.append(np.dot(R_i, R_i.T))
                _beta.append(np.dot(XTX[i], np.dot(X.T, y)))

            XTX = np.array(XTX)
            kwargs = {"XTX": XTX}

        elif method == "normal":
            # XTX will store (X.T @ X)^(-1) for exogenous data X of each segment
            XTX = []
            for i in range(m):
                # Data from current segment
                X = self.wexog[self.breakpoints[i] : self.breakpoints[i + 1]]
                y = self.wendog[self.breakpoints[i] : self.breakpoints[i + 1]]

                XTX.append(np.linalg.inv(np.dot(X.T, X)))
                _beta.append(np.linalg.multi_dot([XTX[i], X.T, y]))

            XTX = np.array(XTX)
            kwargs = {"XTX": XTX}

        else:
            raise ValueError('method has to be "svd", "normal", "qr", or "qr2"')

        _beta = np.array(_beta)
        # Finish calculating betas from current calculated data
        betas = self._calculate_betas(method=method, _beta=_beta, **kwargs)
        self.betas = betas
        # Calculate model in-sample predictions
        y_hat = np.empty(0)
        for i, (prev_break, next_break) in enumerate(
            zip(self.breakpoints[:-1], self.breakpoints[1:])
        ):
            y_hat = np.append(
                y_hat, self.exog[prev_break:next_break, :].dot(self.betas[i])
            )
        self.y_hat = y_hat
        return betas

    def fit_regularized(
        self, mu: float, method: Literal["svd", "normal"] = "svd"
    ) -> ArrayLike:
        """Fits L2 regularised model.

        This method calculates (X.T @ X + mu * I)^(-1) and (X.T @ X + mu * I)^(-1) @ X.T @ y for
        exogenous data X and endogenous data y of each segment, passing these on to _calculate_betas
        which does the rest of the calculation.

        Parameters
        ----------
        mu : float
            L2 penalty parameter.
        method : Literal['svd', 'normal'], optional
            Method for fitting regularized model. By default 'svd'.

        Returns
        -------
        ArrayLike
            Beta coefficients for the model.
        """
        m = self.nsegs
        # XTX will store (X.T @ X)^(-1) for exogenous data X of each segment
        XTX = []
        # _beta will store (X.T @ X)^(-1) @ X.T @ y for exogenous data X and endogenous data y of
        # each segment
        _beta = []

        if method == "svd":

            for i in range(m):
                # Data from current segment
                X = self.wexog[self.breakpoints[i] : self.breakpoints[i + 1]]
                y = self.wendog[self.breakpoints[i] : self.breakpoints[i + 1]]

                u, s, vt = np.linalg.svd(X, 0)
                v = vt.T

                sd = s * s + mu[i]
                vs = v / sd

                XTX.append(np.dot(vs, vt))
                _beta.append(np.dot(vs * s, np.dot(u.T, y)))

        elif method == "normal":

            for i in range(m):
                # Data from current segment
                X = self.wexog[self.breakpoints[i] : self.breakpoints[i + 1]]
                y = self.wendog[self.breakpoints[i] : self.breakpoints[i + 1]]

                XTX.append(
                    np.linalg.inv(np.dot(X.T, X) + mu[i] * np.identity(X.shape[1]))
                )
                _beta.append(np.linalg.multi_dot([XTX[i], X.T, y]))

        else:
            raise ValueError('method has to be "svd" or "normal"')

        XTX = np.array(XTX)
        _beta = np.array(_beta)
        # Finish calculating betas from current calculated data
        betas = self._calculate_betas(method=method, _beta=_beta, XTX=XTX)
        self.betas = betas
        # Calculate model in-sample predictions
        y_hat = np.empty(0)
        for i, (prev_break, next_break) in enumerate(
            zip(self.breakpoints[:-1], self.breakpoints[1:])
        ):
            y_hat = np.append(
                y_hat, self.exog[prev_break:next_break, :].dot(self.betas[i])
            )
        self.y_hat = y_hat
        return betas

    def in_predict(self, exog: ArrayLike = None) -> ArrayLike:
        """In-sample predictions for fit model.

        Parameters
        ----------
        exog : ArrayLike, optional
            Input values for prediction. Set to None if predictions for fitting exogenous variables
            is desired. By default None.

        Returns
        -------
        ArrayLike
            Predictions for given exogenous values. If any of the segments overlap, a
            multi-dimensional output with potentially multiple
        """

        if self.betas is None:
            warnings.warn("Model has not been run yet.")
            return None

        if exog is None:
            return self.y_hat
        elif exog.shape != self.exog.shape:
            raise ValueError(
                (
                    "Input for in_predict must have the same shape as exogenous variable"
                    " used for fitting"
                )
            )
        else:
            # If model is one-to-one
            if not self.overlap_warnings:
                preds = []
                for x in exog:
                    for i, rec in enumerate(self.in_sample):
                        # Check if x in each hyper-rectangle
                        in_rec = (rec[:, 0] < x).all() and (rec[:, 1] > x).all()
                        if in_rec:
                            preds.append(self.betas[i].dot(x))
                            break
                    preds = np.array(preds)
            # If model is one-to-many
            else:
                preds = []
                for x in exog:
                    x_preds = []
                    for i, rec in enumerate(self.in_sample):
                        # Check if x in each hyper-rectangle
                        in_rec = (rec[:, 0] < x).all() and (rec[:, 1] > x).all()
                        if in_rec:
                            # Append the prediction for each hyper-rectangle that x belongs to
                            x_preds.append(self.betas[i].dot(x))
                    preds.append(x_preds)

        return preds

    def plot(
        self,
        plot_model: bool = True,
        plot_data: bool = True,
        model_color: str = None,
        data_color: str = None,
        title: str = "CSR-KB plot",
        xlabel: str = None,
        ylabel: str = None,
        legend: bool = True,
        model_label: str = "Model",
        data_label: str = "Data",
        plot_breakpoints: bool = True,
    ) -> None:
        """Plot model and/or original data.

        Parameters
        ----------
        plot_model : bool, optional
            Set to True to plot model data. By default True.
        plot_data : bool, optional
            Set to True to plot original data. By default True.
        model_color : str, optional
            Color of model data line. Set to None for default color. By default None.
        data_color : str, optional
            Color of original data line. Set to None for default color. By default None.
        title : str, optional
            Title for plot. Defaults to 'CSR-KB plot'. By default 'CSR-KB plot'.
        xlabel : str, optional
            x-axis label for plot. Set to None for no label. By default None.
        ylabel : str, optional
            y-axis label for plot. Set to None for no label. By default None.
        legend : bool, optional
            Set to True to include legend in plot. By default True.
        model_label : str, optional
            Model data line label. By default 'Model'.
        data_label : str, optional
            Original data line label. By default 'Data'.
        plot_breakpoints : bool, optional
            Set to True to plot breakpoints. By default True.
        """

        if self.exog_no_const.ndim != 1:
            raise ValueError("Can only plot data with univariate exogenous variable.")

        if not plot_model and not plot_data:
            raise ValueError("Either plot_model or plot_data should be set to True.")

        _, ax = plt.subplots()

        # Plot original data
        if plot_data:
            ax.plot(self.exog_no_const, self.endog, color=data_color, label=data_label)
        # Check if betas have been calculated
        if self.betas is None:
            if not plot_data:
                raise ValueError(
                    "No data to plot. Neither fit nor fit_regularised have been called."
                )
        else:
            ax.plot(
                self.exog_no_const, self.y_hat, color=model_color, label=model_label
            )

        if plot_breakpoints:
            ax.axvline(
                x=self.exog_no_const[self.breakpoints[1]],
                color="r",
                linestyle="dashed",
                linewidth=1,
                label="Breakpoints",
            )
            for break_ in self.breakpoints[2:-1]:
                ax.axvline(
                    x=self.exog_no_const[break_],
                    color="r",
                    linestyle="dashed",
                    linewidth=1,
                )

        if legend:
            ax.legend()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    """Properties"""

    @property
    def llf(self):
        """
        Log likelihood function for the fitted model.
        """
        if self.betas is None:
            warnings.warn("Model has not been run yet.")
            return None
        if self._llf is None:
            nobs2 = self.nobs / 2.0
            for i, (prev_break, next_break) in enumerate(
                zip(self.breakpoints[:-1], self.breakpoints[1:])
            ):
                pred = np.dot(self.wexog[prev_break:next_break], self.betas[i].T)
                SSR = np.sum((self.wendog[prev_break:next_break] - pred) ** 2, axis=0)
            llf = -np.log(SSR) * nobs2  # concentrated likelihood
            llf -= (1 + np.log(np.pi / nobs2)) * nobs2  # with constant
            llf += 0.5 * np.sum(np.log(self.weights))
            self._llf = llf
        return self._llf

    @property
    def aic(self):
        """
        AIC for the fitted model.
        """
        if self.betas is None:
            warnings.warn("Model has not been run yet.")
            return None
        if not hasattr(self, "_aic"):
            if not hasattr(self, "llf"):
                self.loglike()
            k_params = self.df_model + 1
            self._aic = -2 * self.llf + 2 * k_params
        return self._aic

    @property
    def bic(self):
        """
        BIC for the fitted model.
        """
        if self.betas is None:
            warnings.warn("Model has not been run yet.")
            return None
        if self._bic is None:
            k_params = self.df_model + 1
            self._bic = -2 * self.llf + np.log(self.nobs) * k_params
        return self._bic

    """Private Methods"""

    def _calculate_betas(
        self,
        _beta: ArrayLike,
        method: Literal["svd", "qr", "qr2", "normal"] = "svd",
        XTX: ArrayLike | None = None,
        XTX_x1: ArrayLike | None = None,
        XTX_x2: ArrayLike | None = None,
    ) -> ArrayLike:
        """Calculates final beta estimates.

        This method takes data calculated in fit() or fit_regularized(), forms the linear equation
        for the lambda vector (STEP 1), solves this equation (STEP 2), and then calculates the final
         beta estimates (STEP 3).

        Parameters
        ----------
        _beta : ArrayLike
            Standard linear regression beta estimates for given data.
        method : Literal['svd', 'qr', 'qr2', 'normal'], optional
            Method for fitting model. By default "svd".
        XTX : ArrayLike, optional
            Inverse of (X[i].T @ X[i]) where X[i] refers to the data from the i-th segment. Required
             for 'svd', 'qr2', and 'normal' methods. By default None.
        XTX_x1 : ArrayLike, optional
            (X[i].T @ X[i])^(-1) * wexog_init[i+1] where X[i] refers to the data from the i-th
            segment. Only required if method is 'qr'. By default None.
        XTX_x2 : ArrayLike, optional
            (X[i+1].T @ X[i+1])^(-1) * wexog_init[i+1] where X[i] refers to the data from the i-th
            segment. Only required if method is 'qr'. By default None.

        Returns
        -------
        ArrayLike
            Calculated beta estimates.
        """
        m = self.nsegs

        # Will store the matrix A for calculating the lambdas.
        A = np.zeros((m - 1, m - 1))
        # Will store the vector c for calculating the lambdas.
        c = np.zeros(m - 1)

        if method == "qr":
            if XTX_x1 is None or XTX_x2 is None:
                raise ValueError(
                    (
                        '"qr" has not been used to fit but is now attempting to be used for '
                        'calculating betas. Ensure "qr" is selected in fit() if it\'s desired for '
                        "_calculate_betas()."
                    )
                )
            # STEP 1: form linear equation, i.e. calculate matrix A and vector c.
            A[0][0] = np.dot(self.wexog_init[1], XTX_x1[0] + XTX_x2[0])
            A[0][1] = -np.dot(self.wexog_init[1], XTX_x1[1])

            c[0] = np.dot(self.wexog_init[1], _beta[0] - _beta[1]).item()

            for j in range(1, m - 2):
                # A is self-transpose so don't need to recalculate values
                A[j][j - 1] = A[j - 1][j]
                A[j][j] = np.dot(self.wexog_init[j + 1], XTX_x1[j] + XTX_x2[j])
                A[j][j + 1] = -np.dot(self.wexog_init[j + 1], XTX_x1[j + 1])

                c[j] = np.dot(self.wexog_init[j + 1], _beta[j] - _beta[j + 1]).item()

            A[m - 2][m - 3] = A[m - 3][m - 2]
            A[m - 2][m - 2] = np.dot(
                self.wexog_init[m - 1], XTX_x1[m - 2] + XTX_x2[m - 2]
            )

            c[m - 2] = np.dot(
                self.wexog_init[m - 1], _beta[m - 2] - _beta[m - 1]
            ).item()

            # STEP 2: calculate lambda values

            # A is converted into a 3 x m-1 matrix since it is tri-diagonal
            ab = np.zeros((3, m - 1))
            for i in range(m - 1):
                for j in range(max(0, i - 1), min([2 + i, m - 1])):
                    ab[1 + i - j, j] = A[i, j]
            # Solve linear equation with tri-diagonal matrix
            L = scipy.linalg.solve_banded((1, 1), ab, c)

            # STEP 3: calculate final beta estimates
            betas = (
                [_beta[0] - L[0] * XTX_x1[0]]
                + [
                    _beta[j] + L[j - 1] * XTX_x2[j - 1] - L[j] * XTX_x1[j]
                    for j in range(1, m - 1)
                ]
                + [_beta[m - 1] + L[m - 2] * XTX_x2[m - 2]]
            )

        else:
            if XTX is None:
                raise ValueError(
                    (
                        '"qr" has been used to fit but is now not being used for calculating betas.'
                        ' Ensure "qr" is selected in _calculate_betas() if it\'s desired for fit().'
                    )
                )
            # STEP 1: form linear equation, i.e. calculate matrix A and vector c.

            A[0][0] = np.linalg.multi_dot(
                [self.wexog_init[1], XTX[0] + XTX[1], self.wexog_init[1]]
            )
            A[0][1] = -np.linalg.multi_dot(
                [self.wexog_init[1], XTX[1], self.wexog_init[2]]
            )
            c[0] = np.dot(self.wexog_init[1], _beta[0] - _beta[1]).item()

            for j in range(1, m - 2):
                # A is self-transpose so don't need to recalculate values
                A[j][j - 1] = A[j - 1][j]
                A[j][j] = np.linalg.multi_dot(
                    [
                        self.wexog_init[j + 1],
                        XTX[j] + XTX[j + 1],
                        self.wexog_init[j + 1],
                    ]
                )
                A[j][j + 1] = -np.linalg.multi_dot(
                    [self.wexog_init[j + 1], XTX[j + 1], self.wexog_init[j + 2]]
                )

                c[j] = np.dot(self.wexog_init[j + 1], _beta[j] - _beta[j + 1]).item()

            A[m - 2][m - 3] = A[m - 3][m - 2]
            A[m - 2][m - 2] = np.linalg.multi_dot(
                [
                    self.wexog_init[m - 1],
                    XTX[m - 2] + XTX[m - 1],
                    self.wexog_init[m - 1],
                ]
            )

            c[m - 2] = np.dot(
                self.wexog_init[m - 1], _beta[m - 2] - _beta[m - 1]
            ).item()

            # STEP 2: calculate lambda values

            # A is converted into a 3 x m-1 matrix since it is tri-diagonal
            ab = np.zeros((3, m - 1))
            for i in range(m - 1):
                for j in range(max(0, i - 1), min([2 + i, m - 1])):
                    ab[1 + i - j, j] = A[i, j]
            # Solve linear equation with tri-diagonal matrix
            L = scipy.linalg.solve_banded((1, 1), ab, c)

            # STEP 3: calculate final beta estimates
            betas = (
                [_beta[0] - L[0] * XTX[0].dot(self.wexog_init[1])]
                + [
                    _beta[j]
                    + XTX[j].dot(
                        L[j - 1] * self.wexog_init[j] - L[j] * self.wexog_init[j + 1]
                    )
                    for j in range(1, m - 1)
                ]
                + [_beta[m - 1] + L[m - 2] * XTX[m - 1].dot(self.wexog_init[m - 1])]
            )

        return np.array(betas)
