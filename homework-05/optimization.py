import numpy as np
from numpy.linalg import LinAlgError
import time
from collections import defaultdict

try:
    from scipy.optimize.linesearch import scalar_search_wolfe2
except ImportError:
    from scipy.optimize import line_search as _scipy_line_search
    def scalar_search_wolfe2(phi, derphi, c1=1e-4, c2=0.9, amax=50.0, amin=0, xtol=1e-14):
        res = _scipy_line_search(lambda a: phi(a[0]), lambda a: np.array([derphi(a[0])]), 
                                 np.array([0.0]), np.array([1.0]), c1=c1, c2=c2, amax=amax)
        if res[0] is not None:
            return res[0], phi(res[0]), derphi(res[0])
        return None, None, None

class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == 'Constant':
            return self.c

        phi0 = oracle.func_directional(x_k, d_k, 0)
        derphi0 = oracle.grad_directional(x_k, d_k, 0)
        if np.isnan(phi0) or np.isinf(phi0) or np.isnan(derphi0) or np.isinf(derphi0):
            return None

        start_alpha = previous_alpha if previous_alpha is not None else self.alpha_0

        if self._method == 'Armijo':
            alpha = start_alpha
            while oracle.func_directional(x_k, d_k, alpha) > phi0 + self.c1 * alpha * derphi0:
                alpha /= 2.0
            return alpha

        if self._method == 'Wolfe':
            def phi(a): return oracle.func_directional(x_k, d_k, a)
            def derphi(a): return oracle.grad_directional(x_k, d_k, a)
            res = scalar_search_wolfe2(phi, derphi, c1=self.c1, c2=self.c2, amax=None)
            if res[0] is not None:
                return res[0]
            alpha_fb = self.alpha_0
            while phi(alpha_fb) > phi0 + self.c1 * alpha_fb * derphi0:
                alpha_fb /= 2.0
            return alpha_fb

def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()

def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    g_k = oracle.grad(x_k)
    grad_norm_sq = np.dot(g_k, g_k)
    initial_grad_norm_sq = grad_norm_sq

    if np.isinf(grad_norm_sq) or np.any(np.isnan(g_k)) or np.any(np.isinf(g_k)) or np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)):
        if display: print("Iter 0: computational_error")
        return x_k, 'computational_error', history
    if initial_grad_norm_sq == 0:
        if trace:
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(0.0)
            history['time'].append(0.0)
            if x_k.size <= 2: history['x'].append(x_k.copy())
        if display: print("Iter 0: success")
        return x_k, 'success', history

    start_time = time.time()
    if trace:
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.sqrt(grad_norm_sq))
        history['time'].append(0.0)
        if x_k.size <= 2: history['x'].append(x_k.copy())
    if display:
        print(f"Iter 0: func={oracle.func(x_k):.6f}, grad_norm={np.sqrt(grad_norm_sq):.6f}")

    for k in range(max_iter):
        if np.any(np.isinf(x_k)) or np.any(np.isnan(x_k)) or np.any(np.isinf(g_k)) or np.any(np.isnan(g_k)):
            return x_k, 'computational_error', history
        if grad_norm_sq <= tolerance * initial_grad_norm_sq:
            break
            
        d_k = -g_k
        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        if alpha is None:
            return x_k, 'computational_error', history
            
        x_k = x_k + alpha * d_k
        g_k = oracle.grad(x_k)
        grad_norm_sq = np.dot(g_k, g_k)
        
        if np.isinf(grad_norm_sq) or np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)) or np.any(np.isnan(g_k)) or np.any(np.isinf(g_k)):
            return x_k, 'computational_error', history
            
        if trace:
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_norm_sq))
            history['time'].append(time.time() - start_time)
            if x_k.size <= 2: history['x'].append(x_k.copy())
        if display:
            print(f"Iter {k+1}: func={oracle.func(x_k):.6f}, grad_norm={np.sqrt(grad_norm_sq):.6f}")

    msg = 'success' if grad_norm_sq <= tolerance * initial_grad_norm_sq else 'iterations_exceeded'
    return x_k, msg, history

def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    from scipy.linalg import cho_factor, cho_solve
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    g_k = oracle.grad(x_k)
    grad_norm_sq = np.dot(g_k, g_k)
    initial_grad_norm_sq = grad_norm_sq

    if np.isinf(grad_norm_sq) or np.any(np.isnan(g_k)) or np.any(np.isinf(g_k)) or np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)):
        if display: print("Iter 0: computational_error")
        return x_k, 'computational_error', history
    if initial_grad_norm_sq == 0:
        if trace:
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(0.0)
            history['time'].append(0.0)
            if x_k.size <= 2: history['x'].append(x_k.copy())
        if display: print("Iter 0: success")
        return x_k, 'success', history

    start_time = time.time()
    if trace:
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.sqrt(grad_norm_sq))
        history['time'].append(0.0)
        if x_k.size <= 2: history['x'].append(x_k.copy())
    if display:
        print(f"Iter 0: func={oracle.func(x_k):.6f}, grad_norm={np.sqrt(grad_norm_sq):.6f}")

    for k in range(max_iter):
        if np.any(np.isinf(x_k)) or np.any(np.isnan(x_k)) or np.any(np.isinf(g_k)) or np.any(np.isnan(g_k)):
            return x_k, 'computational_error', history
        if grad_norm_sq <= tolerance * initial_grad_norm_sq:
            break
            
        try:
            H_k = oracle.hess(x_k)
            cho = cho_factor(H_k)
            d_k = -cho_solve(cho, g_k)
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        if alpha is None:
            return x_k, 'computational_error', history
            
        x_k = x_k + alpha * d_k
        g_k = oracle.grad(x_k)
        grad_norm_sq = np.dot(g_k, g_k)
        
        if np.isinf(grad_norm_sq) or np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)) or np.any(np.isnan(g_k)) or np.any(np.isinf(g_k)):
            return x_k, 'computational_error', history

        if trace:
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_norm_sq))
            history['time'].append(time.time() - start_time)
            if x_k.size <= 2: history['x'].append(x_k.copy())
        if display:
            print(f"Iter {k+1}: func={oracle.func(x_k):.6f}, grad_norm={np.sqrt(grad_norm_sq):.6f}")

    msg = 'success' if grad_norm_sq <= tolerance * initial_grad_norm_sq else 'iterations_exceeded'
    return x_k, msg, history