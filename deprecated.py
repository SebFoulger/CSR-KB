def temp_w(Xs, ys, weight_segments: bool = False, w : list = None, Ws: list = None, mu: np.array = None):
    assert(len(Xs)==len(ys))
    # More assertions?

    m = len(Xs)

    if w is None:
        w = np.repeat(1, m)
    else:
        assert(len(w)==m)

    if Ws is None:
        Ws = list(map(lambda x:np.identity(len(x)), Xs))

    w[5] = 100

    if mu is None:
        mu = np.repeat(0, m)

    if weight_segments:
        seg_lengths = list(map(lambda x:len(x), Xs))

        total_length = sum(seg_lengths)

        w = np.array(list(map(lambda x:total_length/x, seg_lengths)))
        
        w = w/min(w)

    w[5]=100

    A = np.zeros((m-1,m-1))
    c = np.zeros(m-1)

    Xs_i = {}
    Xs_i[0] = np.linalg.inv(Xs[0].T.dot(Ws[0]).dot(Xs[0])+(mu[0]/w[0]) * np.identity(len(Xs[0][0, :])))
    Xs_i[1] = np.linalg.inv(Xs[1].T.dot(Ws[1]).dot(Xs[1])+(mu[1]/w[1]) * np.identity(len(Xs[1][0, :])))

    A[0][0] = Xs[1][0, :].dot((Xs_i[0]/w[0] + Xs_i[1]/w[1]).dot(Xs[1][0, :]))
    A[0][1] = -Xs[1][0, :].dot(Xs_i[1].dot(Xs[2][0, :]))/w[1]
    c[0] = 2*Xs[1][0, :].dot(Xs_i[0].dot(Xs[0].T.dot(Ws[0]).dot(ys[0])))[0]
    c[0] -= 2*Xs[1][0, :].dot(Xs_i[1].dot(Xs[1].T.dot(Ws[1]).dot(ys[1])))[0]
    
    for j in range(1, m-2):
        Xs_i[j+1] = np.linalg.inv(Xs[j+1].T.dot(Ws[j+1]).dot(Xs[j+1])+(mu[j+1]/w[j+1]) * np.identity(len(Xs[j+1][0, :])))

        A[j][j-1] = -Xs[j+1][0, :].dot(Xs_i[j].dot(Xs[j][0, :]))/w[j]
        A[j][j] = Xs[j+1][0, :].dot((Xs_i[j]/w[j] + Xs_i[j+1]/w[j+1]).dot(Xs[j+1][0, :]))
        A[j][j+1] = -Xs[j+1][0, :].dot(Xs_i[j+1].dot(Xs[j+2][0, :]))/w[j+1]

        c[j] = 2*Xs[j+1][0, :].dot(Xs_i[j].dot(Xs[j].T.dot(Ws[j]).dot(ys[j])))[0]
        c[j] -= 2*Xs[j+1][0, :].dot(Xs_i[j+1].dot(Xs[j+1].T.dot(Ws[j+1]).dot(ys[j+1])))[0]

    Xs_i[m-1] = np.linalg.inv(Xs[m-1].T.dot(Ws[m-1]).dot(Xs[m-1])+(mu[m-1]/w[m-1]) * np.identity(len(Xs[m-1][0, :])))

    A[m-2][m-3] = -Xs[m-1][0, :].dot(Xs_i[m-2].dot(Xs[m-2][0, :]))/w[m-2]
    A[m-2][m-2] = Xs[m-1][0, :].dot((Xs_i[m-2]/w[m-2] + Xs_i[m-1]/w[m-1]).dot(Xs[m-1][0, :]))
    c[m-2] = 2*Xs[m-1][0, :].dot(Xs_i[m-2].dot(Xs[m-2].T.dot(Ws[m-2]).dot(ys[m-2])))[0]
    c[m-2] -= 2*Xs[m-1][0, :].dot(Xs_i[m-1].dot(Xs[m-1].T.dot(Ws[m-1]).dot(ys[m-1])))[0]
    
    ab = np.zeros((3, m-1))

    for i in range(m-1):
        for j in range(max(0, i-1), min([2+i, m-1])):
            try:
                ab[1+i-j, j] = A[i, j]
            except:
                print(i, j)

    L = scipy.linalg.solve_banded((1,1), ab, c)

    betas = []

    betas.append(Xs_i[0].dot(Xs[0].T.dot(Ws[0]).dot(ys[0]).T[0] - L[0] * Xs[1][0, :] / (2 * w[0])))
    for j in range(1, m-1):
        betas.append(Xs_i[j].dot(Xs[j].T.dot(Ws[j]).dot(ys[j]).T[0] + L[j-1] * Xs[j][0, :] / (2 * w[j]) - L[j] * Xs[j+1][0, :] / (2 * w[j])))
    betas.append(Xs_i[m-1].dot(Xs[m-1].T.dot(Ws[m-1]).dot(ys[m-1]).T[0] + L[m-2] * Xs[m-1][0, :] / (2 * w[m-1])))

    return betas

def fit_svd2(Xs, ys, weight_segments: bool = False, w : list = None, Ws: list = None, mu: np.array = None):
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

    Us = []
    Ss = []
    Vs = []

    Xs_i = {}
    Xs_iy = {}
    Xs_v = {}

    for i in range(len(Xs)):
        u, s, vt = np.linalg.svd(Xs[i], 0)
        v = vt.T
        Us.append(u)
        Ss.append(s)
        Vs.append(v)

        sd = s * s + mu[i]
        vs = v / sd
        Xs_i[i] = np.dot(vs, vt)
        Xs_iy[i] = np.dot(vs * s, np.dot(u.T, ys[i]))
        Xs_v[i] = vs

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

    betas.append(np.dot(Xs_v[0], np.dot(Us[0].T, ys[0]).T[0] * Ss[0] - L[0] * np.dot(Vs[0].T, Xs[1][0, :])))
    for j in range(1, m-1):
        _beta = np.dot(Us[j].T, ys[j]).T[0] * Ss[j] + np.dot(Vs[j].T, L[j-1] * Xs[j][0, :] - L[j] * Xs[j+1][0, :])
        betas.append(np.dot(Xs_v[j], _beta))
    betas.append(np.dot(Xs_v[m-1], np.dot(Us[m-1].T, ys[m-1]).T[0] * Ss[m-1] + L[m-2] * np.dot(Vs[m-1].T, Xs[m-1][0, :])))

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

    Us = []
    Ss = []
    Vs = []

    Xs_i = {}
    Xs_iy = {}

    for i in range(len(Xs)):
        u, s, vt = np.linalg.svd(Xs[i], 0)
        v = vt.T
        Us.append(u)
        Ss.append(s)
        Vs.append(v)

        sd = s * s + mu[i]
        Xs_i[i] = (v / sd).dot(vt)

    Xs_iy[0] = np.linalg.multi_dot([Xs_i[0], Xs[0].T, ys[0]])
    Xs_iy[1] = np.linalg.multi_dot([Xs_i[1], Xs[1].T, ys[1]])

    A[0][0] = np.linalg.multi_dot([Xs[1][0, :], Xs_i[0] + Xs_i[1], Xs[1][0, :]])
    A[0][1] = -np.linalg.multi_dot([Xs[1][0, :], Xs_i[1], Xs[2][0, :]])
    
    c[0] = np.dot(Xs[1][0, :], Xs_iy[0] - Xs_iy[1]).item()
    
    for j in range(1, m-2):
        Xs_iy[j+1] = np.linalg.multi_dot([Xs_i[j+1], Xs[j+1].T, ys[j+1]])

        A[j][j-1] = A[j-1][j]
        A[j][j] = np.linalg.multi_dot([Xs[j+1][0, :], Xs_i[j] + Xs_i[j+1], Xs[j+1][0, :]])
        A[j][j+1] = -np.linalg.multi_dot([Xs[j+1][0, :], Xs_i[j+1], Xs[j+2][0, :]])

        c[j] = np.dot(Xs[j+1][0, :], Xs_iy[j] - Xs_iy[j+1]).item()

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