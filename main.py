import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
import time

def whiten(Xs, ys, w: list = None, Ws: list = None):
    # Assertions?
    if w is None or Ws is None:
        # Change this
        return "error"
    # Can do this more efficient probably
    Xs_white = [np.sqrt(w[i])*np.dot(Ws[i], Xs[i]) for i in range(len(Xs))]
    ys_white = [np.sqrt(w[i])*np.dot(Ws[i], ys[i]) for i in range(len(ys))]
    return Xs_white, ys_white

def fit(Xs, ys, weight_segments: bool = False, w : list = None, Ws: list = None, mu: np.array = None):
    assert(len(Xs)==len(ys))
    # More assertions?

    m = len(Xs)

    if w is None:
        w = np.repeat(1, m)
    else:
        assert(len(w)==m)

    if Ws is None:
        Ws = list(map(lambda x:np.identity(len(x)), Xs))

    if mu is None:
        mu = np.repeat(0, m)

    if weight_segments:
        seg_lengths = list(map(lambda x:len(x), Xs))

        total_length = sum(seg_lengths)

        w = np.array(list(map(lambda x:total_length/x, seg_lengths)))
        
        w = w/min(w)

    Xs, ys = whiten(Xs, ys, w, Ws)
    temp_x = np.array([])
    temp_y = np.array([])
    for X, y in zip(Xs, ys):
        temp_x = np.append(temp_x, X)
        temp_y = np.append(temp_y, y)

    A = np.zeros((m-1,m-1))
    c = np.zeros(m-1)

    Xs_i = {}
    Xs_iy = {}

    Xs_i[0] = np.linalg.inv(np.dot(Xs[0].T, Xs[0]) + mu[0]*np.identity(Xs[0].shape[1]))
    Xs_i[1] = np.linalg.inv(np.dot(Xs[1].T, Xs[1]) + mu[1]*np.identity(Xs[1].shape[1]))

    Xs_iy[0] = np.linalg.multi_dot([Xs_i[0], Xs[0].T, ys[0]])
    Xs_iy[1] = np.linalg.multi_dot([Xs_i[1], Xs[1].T, ys[1]])

    A[0][0] = np.linalg.multi_dot([Xs[1][0, :], Xs_i[0] + Xs_i[1], Xs[1][0, :]])
    A[0][1] = -np.linalg.multi_dot([Xs[1][0, :], Xs_i[1], Xs[2][0, :]])
    
    c[0] = np.dot(Xs[1][0, :], Xs_iy[0] - Xs_iy[1]).item()
    
    for j in range(1, m-2):
        Xs_i[j+1] = np.linalg.inv(np.dot(Xs[j+1].T, Xs[j+1]) + mu[j+1] * np.identity(Xs[j+1].shape[1]))
        Xs_iy[j+1] = np.linalg.multi_dot([Xs_i[j+1], Xs[j+1].T, ys[j+1]])

        A[j][j-1] = A[j-1][j]
        A[j][j] = np.linalg.multi_dot([Xs[j+1][0, :], Xs_i[j] + Xs_i[j+1], Xs[j+1][0, :]])
        A[j][j+1] = -np.linalg.multi_dot([Xs[j+1][0, :], Xs_i[j+1], Xs[j+2][0, :]])

        c[j] = np.dot(Xs[j+1][0, :], Xs_iy[j] - Xs_iy[j+1]).item()

    Xs_i[m-1] = np.linalg.inv(np.dot(Xs[m-1].T, Xs[m-1])+mu[m-1] * np.identity(Xs[m-1].shape[1]))
    Xs_iy[m-1] = np.linalg.multi_dot([Xs_i[m-1], Xs[m-1].T, ys[m-1]])

    A[m-2][m-3] = A[m-3][m-2]
    A[m-2][m-2] = np.linalg.multi_dot([Xs[m-1][0, :], Xs_i[m-2] + Xs_i[m-1], Xs[m-1][0, :]])

    c[m-2] = np.dot(Xs[m-1][0, :], Xs_iy[m-2] - Xs_iy[m-1]).item()

    ab = np.zeros((3, m-1))

    for i in range(m-1):
        for j in range(max(0, i-1), min([2+i, m-1])):
            try:
                ab[1+i-j, j] = A[i, j]
            except:
                print(i, j)

    L = scipy.linalg.solve_banded((1,1), ab, c)

    betas = []
    betas.append(np.dot(Xs_i[0], np.dot(Xs[0].T, ys[0]).reshape((2, )) - L[0] * Xs[1][0, :]))
    for j in range(1, m-1):
        _beta = np.dot(Xs[j].T, ys[j]).reshape((2, )) + L[j-1] * Xs[j][0, :] - L[j] * Xs[j+1][0, :]
        betas.append(np.dot(Xs_i[j], _beta))
    betas.append(np.dot(Xs_i[m-1], np.dot(Xs[m-1].T, ys[m-1]).reshape((2, )) + L[m-2] * Xs[m-1][0, :]))

    return betas

def fit_svd(Xs, ys, weight_segments: bool = False, w : list = None, Ws: list = None, mu: np.array = None):
    assert(len(Xs)==len(ys))
    # More assertions?

    m = len(Xs)

    if w is None:
        w = np.repeat(1, m)
    else:
        assert(len(w)==m)

    if Ws is None:
        Ws = list(map(lambda x:np.identity(len(x)), Xs))

    if mu is None:
        mu = np.repeat(0, m)

    if weight_segments:
        seg_lengths = list(map(lambda x:len(x), Xs))

        total_length = sum(seg_lengths)

        w = np.array(list(map(lambda x:total_length/x, seg_lengths)))
        
        w = w/min(w)

    Xs, ys = whiten(Xs, ys, w, Ws)

    A = np.zeros((m-1,m-1))
    c = np.zeros(m-1)

    Xs_i = {}
    Xs_iy = {}

    for i in range(len(Xs)):
        u, s, vt = np.linalg.svd(Xs[i], 0)
        v = vt.T

        sd = s * s + mu[i]
        vs = v / sd
        Xs_i[i] = np.dot(vs, vt)
        Xs_iy[i] = np.dot(vs * s, np.dot(u.T, ys[i]))

    A[0][0] = np.linalg.multi_dot([Xs[1][0, :], Xs_i[0] + Xs_i[1], Xs[1][0, :]])
    A[0][1] = -np.linalg.multi_dot([Xs[1][0, :], Xs_i[1], Xs[2][0, :]])
    
    c[0] = np.dot(Xs[1][0, :], Xs_iy[0] - Xs_iy[1]).item()
    
    for j in range(1, m-2):
        A[j][j-1] = A[j-1][j]
        A[j][j] = np.linalg.multi_dot([Xs[j+1][0, :], Xs_i[j] + Xs_i[j+1], Xs[j+1][0, :]])
        A[j][j+1] = -np.linalg.multi_dot([Xs[j+1][0, :], Xs_i[j+1], Xs[j+2][0, :]])

        c[j] = np.dot(Xs[j+1][0, :], Xs_iy[j] - Xs_iy[j+1]).item()

    A[m-2][m-3] = A[m-3][m-2]
    A[m-2][m-2] = np.linalg.multi_dot([Xs[m-1][0, :], Xs_i[m-2] + Xs_i[m-1], Xs[m-1][0, :]])

    c[m-2] = np.dot(Xs[m-1][0, :], Xs_iy[m-2] - Xs_iy[m-1]).item()

    ab = np.zeros((3, m-1))

    for i in range(m-1):
        for j in range(max(0, i-1), min([2+i, m-1])):
            try:
                ab[1+i-j, j] = A[i, j]
            except:
                print(i, j)

    L = scipy.linalg.solve_banded((1,1), ab, c)

    betas = []
    betas.append(np.dot(Xs_i[0], np.dot(Xs[0].T, ys[0]).reshape((2, )) - L[0] * Xs[1][0, :]))
    for j in range(1, m-1):
        _beta = np.dot(Xs[j].T, ys[j]).reshape((2, )) + L[j-1] * Xs[j][0, :] - L[j] * Xs[j+1][0, :]
        betas.append(np.dot(Xs_i[j], _beta))
    betas.append(np.dot(Xs_i[m-1], np.dot(Xs[m-1].T, ys[m-1]).reshape((2, )) + L[m-2] * Xs[m-1][0, :]))

    return betas

df = pd.read_csv('test1.csv')[:1000].reset_index(drop=True)

breakpoints = [0, 450, 500, 550, 610, 666, 710, 827, 941, len(df)]

Xs = []
ys = []
prev_break = 0
for _break in breakpoints[1:]:
    Xs.append(sm.add_constant(np.array(list(map(lambda x: [x], df[' timeExp'][prev_break:_break])))))
    ys.append(np.array(list(map(lambda x: [x], df[' controller_x'][prev_break:_break]))))
    prev_break = _break
mu = np.repeat(1, len(Xs))

w = np.arange(1, len(breakpoints))
Ws = np.array([])
for i in range(len(breakpoints)-1):
    Ws = np.append(Ws, np.sqrt(range(breakpoints[i]+1, breakpoints[i+1]+1)))

start = time.time()
for k in range(1):
    betas = fit(Xs, ys, w = w, Ws = w, mu = mu)
    print(betas)
print('Base', time.time()-start)

"""
start = time.time()
for k in range(2000):
    betas = fit_svd(Xs, ys, weight_segments=True, mu = mu)
print('SVD', time.time()-start)

"""
plt.plot(df[' timeExp'], df[' controller_x'])
for i, X in enumerate(Xs):
    plt.plot(X[:, 1], X.dot(betas[i]))

plt.show()
