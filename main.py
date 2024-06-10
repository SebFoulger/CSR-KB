import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm

def temp(Xs, ys, weight_segments: bool = False, w : list = None):
    assert(len(Xs)==len(ys))
    
    m = len(Xs)

    if w is None:
        w = np.repeat(1, m)
    else:
        assert(len(w)==m)

    if weight_segments:
        seg_lengths = list(map(lambda x:len(x), Xs))

        total_length = sum(seg_lengths)

        w = np.array(list(map(lambda x:total_length/x, seg_lengths)))
        
        w = w/min(w)

    A = np.zeros((m-1,m-1))
    c = np.zeros(m-1)

    Xs_i = {}
    Xs_i[0] = np.linalg.inv(Xs[0].T.dot(Xs[0]))
    Xs_i[1] = np.linalg.inv(Xs[1].T.dot(Xs[1]))

    A[0][0] = Xs[1][0, :].dot((Xs_i[0]/w[0] + Xs_i[1]/w[1]).dot(Xs[1][0, :]))
    A[0][1] = -Xs[1][0, :].dot(Xs_i[1].dot(Xs[2][0, :]))/w[1]
    c[0] = 2*Xs[1][0, :].dot(Xs_i[0].dot(Xs[0].T.dot(ys[0])))[0]
    c[0] -= 2*Xs[1][0, :].dot(Xs_i[1].dot(Xs[1].T.dot(ys[1])))[0]
    
    for j in range(1, m-2):
        Xs_i[j+1] = np.linalg.inv(Xs[j+1].T.dot(Xs[j+1]))

        A[j][j-1] = -Xs[j+1][0, :].dot(Xs_i[j].dot(Xs[j][0, :]))/w[j]
        A[j][j] = Xs[j+1][0, :].dot((Xs_i[j]/w[j] + Xs_i[j+1]/w[j+1]).dot(Xs[j+1][0, :]))
        A[j][j+1] = -Xs[j+1][0, :].dot(Xs_i[j+1].dot(Xs[j+2][0, :]))/w[j+1]

        c[j] = 2*Xs[j+1][0, :].dot(Xs_i[j].dot(Xs[j].T.dot(ys[j])))[0]
        c[j] -= 2*Xs[j+1][0, :].dot(Xs_i[j+1].dot(Xs[j+1].T.dot(ys[j+1])))[0]

    Xs_i[m-1] = np.linalg.inv(Xs[m-1].T.dot(Xs[m-1]))

    A[m-2][m-3] = -Xs[m-1][0, :].dot(Xs_i[m-2].dot(Xs[m-2][0, :]))/w[m-2]
    A[m-2][m-2] = Xs[m-1][0, :].dot((Xs_i[m-2]/w[m-2] + Xs_i[m-1]/w[m-1]).dot(Xs[m-1][0, :]))
    c[m-2] = 2*Xs[m-1][0, :].dot(Xs_i[m-2].dot(Xs[m-2].T.dot(ys[m-2])))[0]
    c[m-2] -= 2*Xs[m-1][0, :].dot(Xs_i[m-1].dot(Xs[m-1].T.dot(ys[m-1])))[0]
    
    ab = np.zeros((3, m-1))

    for i in range(m-1):
        for j in range(max(0, i-1), min([2+i, m-1])):
            try:
                ab[1+i-j, j] = A[i, j]
            except:
                print(i, j)

    L = scipy.linalg.solve_banded((1,1), ab, c)

    betas = []

    betas.append(Xs_i[0].dot(Xs[0].T.dot(ys[0]).T[0] - L[0] * Xs[1][0, :] / (2 * w[0])))
    for j in range(1, m-1):
        betas.append(Xs_i[j].dot(Xs[j].T.dot(ys[j]).T[0] + L[j-1] * Xs[j][0, :] / (2 * w[j]) - L[j] * Xs[j+1][0, :] / (2 * w[j])))
    betas.append(Xs_i[m-1].dot(Xs[m-1].T.dot(ys[m-1]).T[0] + L[m-2] * Xs[m-1][0, :] / (2 * w[m-1])))

    return betas


df = pd.read_csv('test1.csv')[:1000].reset_index(drop=True)


breakpoints = [450, 500, 550, 610, 666, 710, 827, 941, len(df)]

Xs = []
ys = []
prev_break = 0

for _break in breakpoints:
    Xs.append(sm.add_constant(np.array(list(map(lambda x: [x], df[' timeExp'][prev_break:_break])))))
    ys.append(np.array(list(map(lambda x: [x], df[' controller_x'][prev_break:_break]))))
    prev_break = _break

betas = temp(Xs, ys, weight_segments=True)
plt.plot(df[' timeExp'], df[' controller_x'])

prev_break = 0

for i, X in enumerate(Xs):
    plt.plot(X[:, 1], X.dot(betas[i]))

    """
    if i>=1:
        print(X[0, :].dot(betas[i-1]))
        print(X[0, :].dot(betas[i]))
        print('\n')
    """
plt.show()
