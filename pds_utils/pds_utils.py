"""
pds_utils.py

Módulo com funções desenvolvidas ao longo do projeto.

Autor:
Jezrael Filgueiras

Data:
2025-09-19

Seções:
1. Métricas de Validação
    - rel_rmse: Raiz do erro médio quadrático (RMS) relativo
    - rel_rmse_cupy: Equivalente rel_rmse para GPU
    - phase_metrics: RMSE absoluto entre fases de um sinal complexo
    - num_validate: Erro médio quadrático (MSE) e correlação de Pearson

2. Sinais e sistemas no tempo discreto
    - Soma de Convolução: 
        Conv_sum; 
        Conv_sum_vectorized;
        Conv_sum_numba_parallel; 
        Conv_sum_cupy.
    - pyfilter: Resolução de equações de diferenças

3. Transformadas discretas
    - Algoritmos DFT e IDFT:
        DFT;
        IDFT;
        DFT_numba;
        IDFT_numba;
        DFT_cupy;
        IDFT_cup.
    - Algoritmos FFT e IFFT com decimação na frequência:
        FFT_freq;
        IFFT_freq;
        FFT_freq_numba;
        IFFT_freq_numba.


"""
import numpy as np
from numba import njit, prange
import cupy as cp

#-----------------------------------------------------------------------------
# 1. Métricas de Validação
#-----------------------------------------------------------------------------

# RMSE relativo
def rel_rmse(y, y_ref, eps=1e-15):
    """
    Calcula o erro quadrático médio relativo (RMSE normalizado).

    Parâmetros:
    y : array-like (estimado)
    y_ref : array-like (referência)
    eps   : float, evita divisão por zero

    Retorna:
    float : RMSE relativo
    """
    return np.linalg.norm(y - y_ref) / (np.linalg.norm(y_ref) + eps)

# Equivalente rel_rmse para GPU
def rel_rmse_cupy(y, y_ref, eps=1e-15):
    """
    Calcula o RMSE relativo diretamente na GPU com CuPy.
    """
    return cp.linalg.norm(y - y_ref) / (cp.linalg.norm(y_ref) + eps) 

# Métrica de fase utilizando RMSE absoluto
def phase_metrics(X, X_ref, mag_thresh_ratio=1e-6, eps=1e-15):
    '''Calcula métricas de fase entre dois sinais. Ignora componentes com magnitude baixa.
    
    Parâmetros:
    X : array-like (estimado)
    X_ref : array-like (referência)
    '''
    mag = np.abs(X)
    mask = mag > (mag_thresh_ratio * mag.max() + eps)
    if not np.any(mask):
        return {'phase_RMSE': np.nan}
    
    phase_err = np.angle(X[mask]) - np.angle(X_ref[mask])
    phase_err = (phase_err + np.pi) % (2*np.pi) - np.pi
    
    return {'phase_RMSE': np.sqrt(np.mean(phase_err**2))}

# Cálculo de RMSE relativo e correlação
def Num_validate(ref, estimado, function):
    """
    Calcula o RMSE relativo e a correlação de dois vetores, e exibe os valores.

    Parâmetros
    ----------
    ref : array_like
        Vetor com os valores de referência.
    estimado : array_like
        Vetor com os valores estimados.
    function : str
        Nome da função utilizada para estimar o vetor.
    """
    rmse_rel = rel_rmse(estimado, ref)  # usa sua função já definida
    corr = np.corrcoef(ref, estimado)[0, 1]

    print(f" {function}:")
    print(f"   ➤ Erro Médio Quadrático Relativo (RMSE): {rmse_rel:.6e}")
    print(f"   ➤ Correlação: {corr:.6f}")
    print()

#-----------------------------------------------------------------------------
# 2. Sinais e sistemas no domínio discreto
#-----------------------------------------------------------------------------

def Conv_sum(x, h, mode='full'):
    """
    Function that computes the linear convolution of two signals x and h.

    The convolution returns a signal of length (M+N-1). This occurs because the convolution shifts the 𝑁-sample signal across the M-sample signal, 
    meaning the index of the resulting signal starts at 0 and goes up to (N-1) + (M-1), totaling  M+n-1 indices

    mode 'full': returns the complete result of the convolution.
    mode'same': returns only the central part of the result, with the same length as the input signal x.

    args:
        x (np.ndarray): Input signal.
        h (np.ndarray): Impulse response.
        mode (str): Convolution mode, can be 'full' or 'same'.
        
    returns:
        np.ndarray: Resulting convolved signal.
    """
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)

    for n in range(N + M - 1):
            for k in range(N):
                  y[n] += x[k] * h[n-k] if 0 <= n - k < M else 0
    
    if mode == 'full':
        return y                                #y = np.convolve(x_n, h_n, mode='full')
    elif mode == 'same':
        begin = (M - 1) // 2
        return y[begin:begin + N]         #y = np.convolve(x_n, h_n, mode='same')

def Conv_sum_vectorized(x, h, mode='full'):
    '''
    Function that computes the convolution sum by generating the Toeplitz matrix, formed from the impulse response ℎ[𝑛], shifted and zero‑padded.

    args:
        x (np.ndarray): Input signal.
        h (np.ndarray): Impulse response.
        mode (str): Convolution mode, can be 'full' or 'same'.
        
    returns:
        np.ndarray: Resulting convolved signal.
    '''

    N = len(x)
    M = len(h)
    L = N + M - 1

    H = np.zeros((L, N))
    for n in range(L):
        for k in range(N):
            if 0 <= n - k < M:
                H[n, k] = h[n - k]

    y = H @ x  

    if mode == 'full':
        return y
    elif mode == 'same':
        begin = (M - 1) // 2
        return y[begin:begin + N]

@njit(parallel=True)
def Conv_sum_numba_parallel(x, h, mode='full'):
    '''
    Computes linear convolution between two vectors x and h, optimized with Numba
    args:
        x (np.ndarray): Input signal.
        h (np.ndarray): Impulse response.
        mode (str): Convolution mode, can be: 'full' or 'same'.
    returns:
        np.ndarray: Convolution's resulting signal.
    
    '''
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)

    for n in prange(N + M - 1):
        for k in range(N):
            if 0 <= n - k < M:
                y[n] += x[k] * h[n-k]

    if mode == 'full':
        return y
    else:  
        begin = (M-1)//2
        return y[begin:begin + N]

# Função utilizando cp.convolve 
def Conv_sum_cupy(x, h, mode='full'):
    '''
    Computes the convolution between tho vectors using cp.convolve.

    args:
        x (np.ndarray): Input signal.
        h (np.ndarray): Impulse response.
        mode (str): Convolution mode, can be: 'full' or 'same'.
    returns:
        np.ndarray: Convolution's resulting signal.
        
    '''
    x_gpu = cp.array(x)
    h_gpu = cp.array(h)
    y_gpu = cp.convolve(x_gpu, h_gpu, mode=mode)
    return cp.asnumpy(y_gpu)

# Função para resolução de equações de diferenças
def pyfilter(b, a, x):
    """
    Function that computes the output of a system represented by a difference equation 
    with coefficients b and a, and input x, for a system initially at rest.

    Args:
        b (list): Numerator coefficients of the difference equation.
        a (list): Denominator coefficients of the difference equation.
        x (list): Input signal.

    Returns:
        y: Output signal.
"""
    N = len(x)
    M = len(b)
    L = len(a)
    y = np.zeros(N)  # Inicializa o vetor de saída com zeros

    for n in range(N):

        x_pond = sum(b[k] * x[n-k] for k in range(M) if n-k >= 0)
        
        y_pond = sum(a[k] * y[n-k] for k in range(1, L) if n-k >= 0)
        y[n] = x_pond - y_pond

    return y # y = lfilter(b, a, x, z_i = NULL)

#-----------------------------------------------------------------------------
# 3. Transformadas discretas
#-----------------------------------------------------------------------------

def DFT(x, N):
    '''Compute the N-point Discrete Fourier Transform of a 1D array x with length L.
    If N > L, zero-pad x to length N.
    
    Parameters:
    x : 1D array
    N : int        Length of the DFT.

    Returns:
    X : 1D array   DFT of x.
    '''

    W_N = np.exp(-1j * 2 * np.pi / N)
    X = np.zeros(N, dtype=np.complex128)
    if len(x) < N:
        x = np.pad(x, (0, N - len(x)), 'constant') # Zero-padding
    else:
        x = x[:N]  # Truncate if longer than N
    for k in range(N):
        for n in range(len(x)):
            X[k] += x[n] * W_N**(k*n)

    return X

def IDFT(X, N):
    '''Compute the N-point Inverse Discrete Fourier Transform of a 1D array X with length N. Eq.(1.8)

    Parameters:
    X : 1D array
    N : int        Length of the IDFT.

    Returns:
    x : 1D array   IDFT of X.
    '''

    return (1/N)*np.conj(DFT(np.conj(X), N))

# DFT e IDFT com Numba

@njit(parallel=True)
def DFT_numba(x, N):
    '''Compute the N-point Discrete Fourier Transform of a 1D array x with length L using Numba for acceleration.
    If N > L, zero-pad x to length N.

    Parameters:
    x : 1D array
    N : int        Length of the DFT.

    Returns:
    X : 1D array   DFT of x.
    '''

    L = len(x)
    x_padded = np.zeros(N, dtype=np.complex128)
    x = np.asarray(x, dtype=np.complex128)
    if L < N:
        x_padded[:L] = x
    else:
        x_padded = x[:N]

    X = np.zeros(N, dtype=np.complex128)
    W_N = np.exp(-2j * np.pi / N)

    for k in prange(N):
        s = 0.0 + 0.0j
        for n in range(N):
            s += x_padded[n] * W_N**(k * n)
        X[k] = s
    return X

@njit(parallel=True)
def IDFT_numba(X, N):
    '''Compute the N-point Inverse Discrete Fourier Transform of a 1D array X with length N using Numba for acceleration. Eq.(1.8)

    Parameters:
    X : 1D array
    N : int        Length of the IDFT.

    Returns:
    x : 1D array   IDFT of X.
    '''

    return (1/N)*np.conj(DFT_numba(np.conj(X), N))

# DFT e IDFT com CuPy

def DFT_cupy(x, N):
    '''Compute the N-point Discrete Fourier Transform of a 1D array x with length L using CuPy for acceleration.
    If N > L, zero-pad x to length N.

    Parameters:
    x : 1D array
    N : int        Length of the DFT.

    Returns:
    X : 1D array   DFT of x.
    '''

    x = cp.asarray(x, dtype=cp.complex128)
    L = x.shape[0]
    if L < N:
        x = cp.pad(x, (0, N - L), 'constant')
    n = cp.arange(N)
    k = n.reshape((N, 1))

    W_N = cp.exp(-2j * cp.pi / N)
    M = W_N ** (k * n)  # Matriz de exponenciais
    return M @ x  # Produto matricial

def IDFT_cupy(X, N):
    '''Compute the N-point Inverse Discrete Fourier Transform of a 1D array X with length N using CuPy for acceleration.

    Parameters:
    X : 1D array
    N : int        Length of the IDFT.

    Returns:
    x : 1D array   IDFT of X.
    '''

    X = cp.asarray(X, dtype=cp.complex128)
    n = cp.arange(N)
    k = n.reshape((N, 1))
    W_N = cp.exp(-2j * cp.pi / N)
    M = W_N ** (-k * n)
    return (M @ X) / N

# FFT e IFFT com decimação na frequência

def FFT_freq(x, N):
    '''Compute the FFT of a 1D array x using frequency domain decomposition.

    Parameters:
    x : 1D array
        Input signal.

    Returns:
    X : 1D array
        FFT of the input signal.
    '''

    # Verifica se N é potência de 2
    # Caso não seja, ajusta para a próxima potência de 2
    exp = int(np.ceil(np.log2(N)))
    if((N > 0) and (N & (N-1)) != 0):
        N = 2**exp

    x_padded = np.zeros(N, dtype=np.complex128)
    x_padded[:len(x)] = x

    for i in range(exp):
        L = N//(2**i)
        ramos = N//(2**(exp-i))
        W_L = np.exp(-2j * np.pi / L)
        for ramo in range(ramos):
            t = 1
            for n in range(L//2):
                idx = ramo*L + n
                x_temp = x_padded[idx]
                x_padded[idx] = x_temp + x_padded[idx + L//2]
                x_padded[idx+L//2] = (x_temp - x_padded[idx + L//2]) * t
                t *= W_L
    
    indices = np.zeros(N, dtype=np.int64)
    for idx_out in range(N):
        rev = 0
        i_temp = idx_out
        for _ in range(exp):
            rev = (rev << 1) | (i_temp & 1)
            i_temp = i_temp >> 1
        indices[idx_out] = rev
    return x_padded[indices]

@njit(parallel=True)
def FFT_freq_numba(x, N):
    '''Compute the FFT of a 1D array x using frequency domain decomposition, optimized with Numba.

    Parameters:
    x : 1D array
        Input signal.

    Returns:
    X : 1D array
        FFT of the input signal.
    '''

    exp = int(np.ceil(np.log2(N)))
    if((N > 0) and (N & (N-1)) != 0):
        N = 2**exp
    

    x_padded = np.zeros(N, dtype=np.complex128)
    x_padded[:len(x)] = x

    
    for i in range(exp):
        L = N//(2**i)
        ramos = N//(2**(exp-i))
        W_L = np.exp(-2j * np.pi / L)
        for ramo in prange(ramos):
            t = 1
            for n in range(L//2):
                idx = ramo*L + n
                x_temp = x_padded[idx]
                x_padded[idx] = x_temp + x_padded[idx + L//2]
                x_padded[idx+L//2] = (x_temp - x_padded[idx + L//2]) * t
                t *= W_L
    
    indices = np.zeros(N, dtype=np.int64)
    for idx_out in range(N):
        rev = 0
        i_temp = idx_out
        for _ in range(exp):
            rev = (rev << 1) | (i_temp & 1)
            i_temp = i_temp >> 1
        indices[idx_out] = rev
    return x_padded[indices]

def IFFT(X, N):
    '''Compute the N-point Inverse Fast Fourier Transform of a 1D array X with length N. Eq.(1.8)

    Parameters:
    X : 1D array
    N : int        Length of the IDFT.

    Returns:
    x : 1D array   IDFT of X.
    '''

    return (1/N)*np.conj(FFT_freq(np.conj(X), N))

@njit(parallel=True)
def IFFT_numba(X, N):
    '''Compute the N-point Inverse Fast Fourier Transform of a 1D array X with length N using Numba for acceleration. Eq.(1.8)

    Parameters:
    X : 1D array
    N : int        Length of the IDFT.

    Returns:
    x : 1D array   IDFT of X.
    '''

    return (1/N)*np.conj(FFT_freq_numba(np.conj(X), N))