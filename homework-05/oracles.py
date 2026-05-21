import numpy as np
import scipy
from scipy.special import expit

class BaseSmoothOracle(object):
    def func(self, x): raise NotImplementedError('Func oracle is not implemented.')
    def grad(self, x): raise NotImplementedError('Grad oracle is not implemented.')
    def hess(self, x): raise NotImplementedError('Hessian oracle is not implemented.')
    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))
    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))

class QuadraticOracle(BaseSmoothOracle):
    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b
    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)
    def grad(self, x):
        return self.A.dot(x) - self.b
    def hess(self, x):
        return self.A 

class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = len(b)

    def func(self, x):
        z = self.matvec_Ax(x)
        loss = np.sum(np.logaddexp(0, -self.b * z)) / self.m
        reg = 0.5 * self.regcoef * np.dot(x, x)
        return loss + reg

    def grad(self, x):
        z = self.matvec_Ax(x)
        s = expit(-self.b * z)
        v = (-self.b * s) / self.m
        return self.matvec_ATx(v) + self.regcoef * x

    def hess(self, x):
        z = self.matvec_Ax(x)
        s = expit(-self.b * z)
        w = s * (1 - s) / self.m
        n = x.size
        return self.matmat_ATsA(w) + self.regcoef * np.eye(n)

class LogRegL2OptimizedOracle(LogRegL2Oracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self._x_cache = None; self._Ax_cache = None
        self._d_cache = None; self._Ad_cache = None
        self._x_hat_cache = None; self._Ax_hat_cache = None

    def _match(self, a, b):
        return a is not None and b is not None and np.array_equal(a, b)

    def _get_Ax(self, x):
        if self._match(x, self._x_cache):
            return self._Ax_cache
        if self._match(x, self._x_hat_cache):
            self._x_cache = x.copy()
            self._Ax_cache = self._Ax_hat_cache
            return self._Ax_cache
        self._x_cache = x.copy()
        self._Ax_cache = self.matvec_Ax(x)
        self._x_hat_cache = None
        self._Ad_cache = None
        return self._Ax_cache

    def _get_Ad(self, d):
        if self._match(d, self._d_cache):
            return self._Ad_cache
        self._d_cache = d.copy()
        self._Ad_cache = self.matvec_Ax(d)
        return self._Ad_cache

    def func(self, x):
        z = self._get_Ax(x)
        return np.sum(np.logaddexp(0, -self.b * z)) / self.m + 0.5 * self.regcoef * np.dot(x, x)

    def grad(self, x):
        z = self._get_Ax(x)
        s = expit(-self.b * z)
        v = (-self.b * s) / self.m
        return self.matvec_ATx(v) + self.regcoef * x

    def hess(self, x):
        z = self._get_Ax(x)
        s = expit(-self.b * z)
        w = s * (1 - s) / self.m
        return self.matmat_ATsA(w) + self.regcoef * np.eye(x.size)

    def _prepare_directional(self, x, d, alpha):
        Ax = self._get_Ax(x)
        Ad = self._get_Ad(d)
        z = Ax + alpha * Ad
        x_hat = x + alpha * d
        self._x_hat_cache = x_hat.copy()
        self._Ax_hat_cache = z.copy()
        return z

    def func_directional(self, x, d, alpha):
        z = self._prepare_directional(x, d, alpha)
        x_new = x + alpha * d
        return np.sum(np.logaddexp(0, -self.b * z)) / self.m + 0.5 * self.regcoef * np.dot(x_new, x_new)

    def grad_directional(self, x, d, alpha):
        z = self._prepare_directional(x, d, alpha)
        x_new = x + alpha * d
        s = expit(-self.b * z)
        v = (-self.b * s) / self.m
        return np.dot(v, self._Ad_cache) + self.regcoef * np.dot(x_new, d)

def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)
    def matmat_ATsA(s):
        s_col = s.reshape(-1, 1)
        return A.T.dot(A.multiply(s_col)) if scipy.sparse.issparse(A) else A.T.dot(A * s_col)

    if oracle_type == 'usual': oracle = LogRegL2Oracle
    elif oracle_type == 'optimized': oracle = LogRegL2OptimizedOracle
    else: raise ValueError('Unknown oracle_type=%s' % oracle_type)
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

def grad_finite_diff(func, x, eps=1e-8):
    n = len(x); g = np.zeros(n); f0 = func(x)
    for i in range(n):
        x_eps = x.copy(); x_eps[i] += eps
        g[i] = (func(x_eps) - f0) / eps
    return g

def hess_finite_diff(func, x, eps=1e-5):
    n = len(x); H = np.zeros((n, n)); f0 = func(x)
    for i in range(n):
        x_i = x.copy(); x_i[i] += eps; f_i = func(x_i)
        for j in range(n):
            x_j = x.copy(); x_j[j] += eps; f_j = func(x_j)
            x_ij = x.copy(); x_ij[i] += eps; x_ij[j] += eps
            H[i, j] = (func(x_ij) - f_i - f_j + f0) / (eps**2)
    return H