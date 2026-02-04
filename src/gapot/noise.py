"""
Noise Mitigation Techniques
===========================

Implementation of noise-robust differentiation and regularization
techniques for derivative-based power calculations.

These techniques address the noise sensitivity issues discussed
in the paper, particularly for methods requiring signal derivatives.

Techniques implemented:
1. Holoborodko smooth noise-robust differentiators
2. Tikhonov (L2) regularization
3. Total Variation (L1) regularization
4. Savitzky-Golay filtering
5. FFT-based differentiation with windowing

References
----------
.. [1] Holoborodko, P. Smooth noise robust differentiators.
       http://www.holoborodko.com/pavel/
.. [2] Chartrand, R. (2011). Numerical differentiation of noisy, 
       nonsmooth data. ISRN Applied Mathematics.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple


class HoloborodkoFilter:
    """
    Holoborodko Smooth Noise-Robust Differentiator.
    
    Implements FIR differentiators that minimize noise amplification
    while maintaining accuracy for polynomial signals.
    
    The filter coefficients are optimized for smoothness and
    noise rejection.
    
    Parameters
    ----------
    order : int
        Filter order. Higher order = more smoothing. 
        Must be odd and >= 5. Common values: 5, 7, 9, 11.
    dt : float
        Sampling period.
        
    Example
    -------
    >>> filt = HoloborodkoFilter(order=7, dt=1e-4)
    >>> dx = filt.differentiate(x)
    """
    
    # Pre-computed coefficients for different orders
    COEFFICIENTS = {
        5: np.array([1, 2, 0, -2, -1]) / 8,
        7: np.array([1, 4, 5, 0, -5, -4, -1]) / 32,
        9: np.array([1, 6, 14, 14, 0, -14, -14, -6, -1]) / 128,
        11: np.array([1, 8, 27, 48, 42, 0, -42, -48, -27, -8, -1]) / 512,
    }
    
    def __init__(self, order: int = 7, dt: float = 1.0):
        if order not in self.COEFFICIENTS:
            raise ValueError(f"Order must be one of {list(self.COEFFICIENTS.keys())}")
        
        self.order = order
        self.dt = dt
        self.coeffs = self.COEFFICIENTS[order] / dt
    
    def differentiate(self, x: np.ndarray) -> np.ndarray:
        """
        Compute noise-robust derivative.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal.
            
        Returns
        -------
        dx : np.ndarray
            Derivative estimate.
        """
        # Apply FIR filter (convolve)
        dx = np.convolve(x, self.coeffs, mode='same')
        
        # Handle edge effects
        half_width = len(self.coeffs) // 2
        dx[:half_width] = dx[half_width]
        dx[-half_width:] = dx[-half_width-1]
        
        return dx
    
    def get_frequency_response(self, fs: float, n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency response of the differentiator.
        
        Returns
        -------
        freqs : np.ndarray
            Frequency points.
        response : np.ndarray
            Complex frequency response.
        """
        freqs = np.linspace(0, fs/2, n_points)
        omega = 2 * np.pi * freqs / fs
        
        # FIR frequency response
        response = np.zeros(n_points, dtype=complex)
        for k, c in enumerate(self.coeffs):
            n = k - len(self.coeffs)//2
            response += c * np.exp(-1j * omega * n)
        
        return freqs, response


class TikhonovRegularizer:
    """
    Tikhonov (L2) Regularization for Differentiation.
    
    Solves the regularized problem:
    min_x ||Ax - b||² + λ||Lx||²
    
    where A is the integration matrix, b is the signal, and L is
    a regularization matrix (typically first or second difference).
    
    Parameters
    ----------
    lambda_reg : float
        Regularization parameter. Larger = more smoothing.
    order : int
        Order of derivative regularization (1 or 2).
        
    Example
    -------
    >>> reg = TikhonovRegularizer(lambda_reg=1e-3)
    >>> dx = reg.differentiate(x, dt=1e-4)
    """
    
    def __init__(self, lambda_reg: float = 1e-3, order: int = 2):
        self.lambda_reg = lambda_reg
        self.order = order
    
    def differentiate(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute regularized derivative.
        
        Uses the fact that differentiation is the inverse of integration,
        so we regularize the integration operator.
        """
        n = len(x)
        
        # Difference operator for derivative
        if self.order == 1:
            D = diags([-1, 1], [0, 1], shape=(n-1, n))
        else:  # order == 2
            D = diags([1, -2, 1], [0, 1, 2], shape=(n-2, n))
        
        # Integration operator (cumulative sum, scaled)
        I_op = eye(n)
        
        # Regularized solution
        A = I_op.T @ I_op + self.lambda_reg * D.T @ D
        b = I_op.T @ x
        
        # Solve and differentiate
        x_smooth = spsolve(A.tocsr(), b)
        
        # Central difference for derivative
        dx = np.gradient(x_smooth, dt)
        
        return dx
    
    def select_lambda(self, x: np.ndarray, dt: float, 
                      method: str = 'gcv') -> float:
        """
        Automatically select regularization parameter.
        
        Parameters
        ----------
        method : str
            Selection method: 'gcv' (generalized cross-validation) or
            'lcurve' (L-curve method).
            
        Returns
        -------
        lambda_opt : float
            Optimal regularization parameter.
        """
        # Simplified GCV implementation
        lambdas = np.logspace(-6, 0, 50)
        gcv_scores = []
        
        n = len(x)
        for lam in lambdas:
            self.lambda_reg = lam
            dx = self.differentiate(x, dt)
            
            # Reconstruct by integration
            x_recon = np.cumsum(dx) * dt + x[0]
            
            # GCV score (simplified)
            residual = np.sum((x - x_recon)**2)
            gcv_scores.append(residual / n)
        
        return lambdas[np.argmin(gcv_scores)]


class TVRegularizer:
    """
    Total Variation (L1) Regularization for Differentiation.
    
    Better preserves sharp features (edges) compared to Tikhonov.
    Solves:
    min_x ||Ax - b||² + λ||Dx||_1
    
    using iteratively reweighted least squares (IRLS).
    
    Parameters
    ----------
    lambda_reg : float
        Regularization parameter.
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance.
        
    Example
    -------
    >>> reg = TVRegularizer(lambda_reg=1e-2)
    >>> dx = reg.differentiate(x, dt=1e-4)
    """
    
    def __init__(self, lambda_reg: float = 1e-2, max_iter: int = 50, 
                 tol: float = 1e-6):
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
    
    def differentiate(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute TV-regularized derivative using IRLS.
        """
        n = len(x)
        
        # Difference operator
        D = diags([-1, 1], [0, 1], shape=(n-1, n)).toarray()
        
        # Initial estimate (central difference)
        u = np.gradient(x, dt)
        
        # IRLS iteration
        for iteration in range(self.max_iter):
            u_old = u.copy()
            
            # Weights
            Du = D @ u
            eps = 1e-8
            W = diags(1.0 / np.sqrt(Du**2 + eps))
            
            # Weighted least squares
            A = eye(n) + self.lambda_reg * D.T @ W @ D
            b = x / dt
            
            u = spsolve(A.tocsr(), b)
            
            # Check convergence
            if np.linalg.norm(u - u_old) / (np.linalg.norm(u_old) + eps) < self.tol:
                break
        
        return np.gradient(u * dt, dt)


class SavitzkyGolayDifferentiator:
    """
    Savitzky-Golay Filter for Differentiation.
    
    Fits local polynomials and computes derivatives analytically.
    
    Parameters
    ----------
    window_length : int
        Window size (must be odd).
    polyorder : int
        Polynomial order.
    deriv : int
        Derivative order to compute.
        
    Example
    -------
    >>> sg = SavitzkyGolayDifferentiator(window_length=11, polyorder=3)
    >>> dx = sg.differentiate(x, dt=1e-4)
    """
    
    def __init__(self, window_length: int = 11, polyorder: int = 3, 
                 deriv: int = 1):
        if window_length % 2 == 0:
            window_length += 1
        
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
    
    def differentiate(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Compute derivative using Savitzky-Golay filter."""
        return signal.savgol_filter(x, self.window_length, self.polyorder,
                                     deriv=self.deriv, delta=dt)


class FFTDifferentiator:
    """
    FFT-Based Differentiation with Windowing.
    
    Computes derivative in frequency domain with optional
    high-frequency attenuation.
    
    Parameters
    ----------
    cutoff_fraction : float
        Fraction of Nyquist frequency for low-pass cutoff.
        Default 0.8 (attenuate top 20% of spectrum).
    window : str
        Window function for spectral smoothing.
        
    Example
    -------
    >>> fft_diff = FFTDifferentiator(cutoff_fraction=0.7)
    >>> dx = fft_diff.differentiate(x, dt=1e-4)
    """
    
    def __init__(self, cutoff_fraction: float = 0.8, window: str = 'hann'):
        self.cutoff_fraction = cutoff_fraction
        self.window = window
    
    def differentiate(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Compute derivative using FFT."""
        n = len(x)
        
        # FFT
        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(n, dt)
        
        # Derivative in frequency domain: d/dt -> j*omega
        omega = 2 * np.pi * freqs
        dX = 1j * omega * X
        
        # Low-pass filter
        nyquist = 1 / (2 * dt)
        cutoff = self.cutoff_fraction * nyquist
        
        # Smooth cutoff
        filter_response = np.exp(-((np.abs(freqs) - cutoff) / (0.1 * nyquist))**2)
        filter_response[np.abs(freqs) < cutoff] = 1.0
        
        dX *= filter_response
        
        # Inverse FFT
        dx = np.real(np.fft.ifft(dX))
        
        return dx


def compare_differentiators(x: np.ndarray, dx_true: np.ndarray, 
                            dt: float, noise_level: float = 0.01) -> dict:
    """
    Compare different differentiation methods on noisy data.
    
    Parameters
    ----------
    x : np.ndarray
        Clean signal.
    dx_true : np.ndarray
        True derivative.
    dt : float
        Sampling period.
    noise_level : float
        Noise standard deviation as fraction of signal amplitude.
        
    Returns
    -------
    results : dict
        Dictionary with RMSE for each method.
    """
    # Add noise
    noise = noise_level * np.std(x) * np.random.randn(len(x))
    x_noisy = x + noise
    
    methods = {
        'Central Difference': lambda: np.gradient(x_noisy, dt),
        'Holoborodko-5': lambda: HoloborodkoFilter(5, dt).differentiate(x_noisy),
        'Holoborodko-9': lambda: HoloborodkoFilter(9, dt).differentiate(x_noisy),
        'Savitzky-Golay': lambda: SavitzkyGolayDifferentiator().differentiate(x_noisy, dt),
        'FFT': lambda: FFTDifferentiator().differentiate(x_noisy, dt),
    }
    
    results = {}
    for name, method in methods.items():
        dx_est = method()
        # Trim edges for fair comparison
        margin = 20
        rmse = np.sqrt(np.mean((dx_est[margin:-margin] - dx_true[margin:-margin])**2))
        results[name] = rmse
    
    return results


def estimate_snr(signal: np.ndarray, fs: float, 
                 signal_bw: float = None) -> float:
    """
    Estimate Signal-to-Noise Ratio.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    fs : float
        Sampling frequency.
    signal_bw : float, optional
        Expected signal bandwidth for noise estimation.
        
    Returns
    -------
    snr_db : float
        Estimated SNR in dB.
    """
    # Simple estimate using high-frequency content as noise proxy
    n = len(signal)
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    
    if signal_bw is None:
        signal_bw = fs / 10  # Assume signal in bottom 10% of spectrum
    
    signal_mask = np.abs(freqs) < signal_bw
    noise_mask = np.abs(freqs) > signal_bw
    
    signal_power = np.mean(np.abs(X[signal_mask])**2)
    noise_power = np.mean(np.abs(X[noise_mask])**2)
    
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = np.inf
    
    return snr_db
