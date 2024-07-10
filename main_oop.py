import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
from typing import Literal
import time

class CSR_KB:

    def __init__(self, endog, breakpoints, exog = None, weights = None, seg_weights = None):
        """_summary_

        Args:
            endog (_type_): _description_
            breakpoints (_type_): Include start and end of data. Each breakpoints indicates new segment.
            exog (_type_, optional): If None will be 0,1,...,nobs. Defaults to None.
            weights (_type_, optional): _description_. Defaults to None.
            seg_weights (_type_, optional): _description_. Defaults to None.
        """        
        self.endog = endog
        self.nobs = len(endog)
        self.breakpoints = breakpoints
        self.nsegs = len(breakpoints) - 1
        self.seg_sizes = np.diff(breakpoints)
        
        if exog is None:
            exog = np.arange(len(endog))
        else:
            assert(len(exog) == len(endog))
            self.exog = exog
        
        if weights is not None:
            assert(len(weights)==self.nobs)
        self.weights = weights
        self.combined_weights = weights

        if seg_weights is not None:
            assert(len(seg_weights)==self.nsegs)
            combined_seg_weights = np.array([])
            for i in range(self.nsegs):
                cur_seg_weight = np.repeat(seg_weights[i], self.seg_sizes[i])
                combined_seg_weights = np.append(combined_seg_weights, cur_seg_weight)

            if self.combined_weights is not None:
                self.combined_weights = self.combined_weights * combined_seg_weights
            else:
                self.combined_weights = combined_seg_weights
        self.seg_weights = seg_weights

        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)

        self.wexog_init = self.wexog[self.breakpoints[:-1]]

    def whiten(self, x):
        if self.combined_weights is not None:
            if x.ndim == 1:
                return np.sqrt(self.combined_weights) * x 
            else:
                return np.sqrt(self.combined_weights)[:, None] * x 
        else:
            return x

    def fit(self, 
            method: Literal['svd', 'normal'] = 'svd'):
        XTX = {}
        _beta = {}

        if method == 'svd':

            for i in range(self.nsegs):
                X = self.wexog[self.breakpoints[i]:self.breakpoints[i+1]]
                y = self.wendog[self.breakpoints[i]:self.breakpoints[i+1]]

                u, s, vt = np.linalg.svd(X, 0)
                v = vt.T

                sd = s * s
                vs = v / sd

                XTX[i] = np.dot(vs, vt)
                _beta[i] = np.dot(vs * s, np.dot(u.T, y))

        elif method == 'normal':
            
            for i in range(self.nsegs):
                X = self.wexog[self.breakpoints[i]:self.breakpoints[i+1]]
                y = self.wendog[self.breakpoints[i]:self.breakpoints[i+1]]

                XTX[i] = np.linalg.inv(np.dot(X.T, X))
                _beta[i] = np.linalg.multi_dot([XTX[i], X.T, y])

        betas = self._calculate_betas(XTX, _beta)
        self.betas = betas
        return betas

    def fit_regularized(self, 
                        mu, 
                        method: Literal['svd', 'normal'] = 'svd'):

        XTX = {}
        _beta = {}

        if method == 'svd':

            for i in range(self.nsegs):
                X = self.wexog[self.breakpoints[i]:self.breakpoints[i+1]]
                y = self.wendog[self.breakpoints[i]:self.breakpoints[i+1]]

                u, s, vt = np.linalg.svd(X, 0)
                v = vt.T

                sd = s * s + mu[i]
                vs = v / sd

                XTX[i] = np.dot(vs, vt)
                _beta[i] = np.dot(vs * s, np.dot(u.T, y))

        elif method == 'normal':
            
            for i in range(self.nsegs):
                X = self.wexog[self.breakpoints[i]:self.breakpoints[i+1]]
                y = self.wendog[self.breakpoints[i]:self.breakpoints[i+1]]

                XTX[i] = np.linalg.inv(np.dot(X.T, X) + mu[i]*np.identity(X.shape[1]))
                _beta[i] = np.linalg.multi_dot([XTX[i], X.T, y])

        betas = self._calculate_betas(XTX, _beta)
        self.betas = betas
        return betas


    def _calculate_betas(self, XTX, _beta):
        m = self.nsegs

        A = np.zeros((m - 1, m - 1))
        c = np.zeros(m - 1)

        A[0][0] = np.linalg.multi_dot([self.wexog_init[1], XTX[0] + XTX[1], self.wexog_init[1]])
        A[0][1] = -np.linalg.multi_dot([self.wexog_init[1], XTX[1], self.wexog_init[2]])
        
        c[0] = np.dot(self.wexog_init[1], _beta[0] - _beta[1]).item()
        
        for j in range(1, m - 2):

            A[j][j-1] = A[j-1][j]
            A[j][j] = np.linalg.multi_dot([self.wexog_init[j+1], XTX[j] + XTX[j+1], self.wexog_init[j+1]])
            A[j][j+1] = -np.linalg.multi_dot([self.wexog_init[j+1], XTX[j+1], self.wexog_init[j+2]])

            c[j] = np.dot(self.wexog_init[j+1], _beta[j] - _beta[j+1]).item()

        A[m-2][m-3] = A[m-3][m-2]
        A[m-2][m-2] = np.linalg.multi_dot([self.wexog_init[m-1], XTX[m-2] + XTX[m-1], self.wexog_init[m-1]])

        c[m-2] = np.dot(self.wexog_init[m-1], _beta[m-2] - _beta[m-1]).item()

        print(is_pos_def(A))

        ab = np.zeros((3, m-1))

        for i in range(m-1):
            for j in range(max(0, i-1), min([2+i, m-1])):
                ab[1+i-j, j] = A[i, j]

        L = scipy.linalg.solve_banded((1,1), ab, c)

        betas = []
        betas.append(_beta[0]-L[0] * XTX[0].dot(self.wexog_init[1]))
        for j in range(1, m-1):
            betas.append(_beta[j]+ XTX[j].dot(L[j-1] * self.wexog_init[j] - L[j] * self.wexog_init[j+1]))
        betas.append(_beta[m-1]+L[m-2] * XTX[m-1].dot(self.wexog_init[m-1]))

        return betas
    
def is_pos_def(x):
    return np.linalg.eigvals(x)

for k in range(100):
    df = pd.read_csv('test1.csv')[k*1000:(k+1)*1000].reset_index(drop=True)

    breakpoints = [0, 450, 500, 550, 610, 666, 710, 827, 941, 1000]

    Xs = np.array([])
    ys = np.array([])
    prev_break = 0
    for _break in breakpoints:
        Xs = np.append(Xs, np.array(list(map(lambda x: [x], df[' timeExp'][prev_break:_break]))))
        ys = np.append(ys, np.array(list(map(lambda x: [x], df[' controller_x'][prev_break:_break]))))
        prev_break = _break
    mu = np.repeat(1, len(Xs))

    Xs = sm.add_constant(Xs)
    weights =  np.arange(1, len(Xs)+1)

    model = CSR_KB(ys, breakpoints, Xs, weights = weights, seg_weights = np.arange(1, len(breakpoints)))
    betas = model.fit_regularized(mu = np.repeat(0, len(breakpoints)-1))

Xs = []
ys = []
prev_break = 0
for _break in breakpoints[1:]:
    Xs.append(sm.add_constant(np.array(list(map(lambda x: [x], df[' timeExp'][prev_break:_break])))))
    ys.append(np.array(list(map(lambda x: [x], df[' controller_x'][prev_break:_break]))))
    prev_break = _break
plt.plot(df[' timeExp'], df[' controller_x'])
for i, X in enumerate(Xs):
    plt.plot(X[:, 1], X.dot(betas[i]))

plt.show()