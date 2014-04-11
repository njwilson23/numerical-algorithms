
import numpy as np
from numpy import cos, sin, exp, pi

# Not positive that these are correct - seems okay testing with random
# sequences of numbers

def dft(x):
    n = len(x)
    xhat = np.zeros(n, dtype=np.complex64)
    fs = np.arange(n)
    for k in range(len(xhat)):
        xhat[k] = np.sum(x * exp(-2j*pi*k*fs/n))
    return xhat

def idft(xhat):
    n = len(xhat)
    x = np.zeros(n, dtype=np.complex64)
    ts = np.arange(n)
    xhat_ = xhat / n
    for t in range(n):
        x[t] = sum(xhat_ * exp(2j*pi*t*ts/n))
    return x

