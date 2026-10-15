"""Historical energy function bodies; optional acceleration only."""
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]): return args[0]
        return lambda fn: fn
def genParamsDH(temp, ionic):
    """ Generate Debye-Huckel parameters. """
    temp = float(temp)
    kT = 8.3145 * temp * 1e-3
    fepsw = lambda T: 5321 / T + 233.76 - 0.9297 * T + 0.1417 * 1e-2 * T**2 - 0.8292 * 1e-6 * T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2 / (4 * np.pi * 8.854188 * epsw) * 6.022 * 1000 / kT
    eps_yu = lB * kT
    k_yu = np.sqrt(8 * np.pi * lB * ionic * 6.022 / 10)
    return eps_yu, k_yu

@njit(nopython=True)
def ah_energy(eps, sigma, lambda_param, r, rc):
    """ Calculate Ashbaugh-Hatch energy. """
    if r <= 2**(1/6) * sigma:
        return eps * (1 - lambda_param)
    elif r <= rc:
        shift = (sigma / rc)**12 - (sigma / rc)**6
        return lambda_param * (4 * eps * ((sigma/r)**12 - (sigma/r)**6 - shift))
    else:
        return 0.0

@njit(nopython=True)
def lj_potential(r, sigma, epsilon):
    ulj = 4.0 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    return ulj

@njit(nopython=True)
def yukawa_potential(r, q, kappa_yu, eps_yu, rc_yu=4.0):
    #print(r, rc)
    if r <= rc_yu:
        shift = np.exp(-kappa_yu * rc_yu) / rc_yu
        u = eps_yu * q * (np.exp(-kappa_yu * r) / r - shift)
        #print(u)
        return u  # Returns float
    else:
        return 0.0

@njit(nopython=True)
def ah_potential(r, sigma, epsilon, lam, rc=4.0):
    #print(r, rc)
    if r <= 2**(1./6.) * sigma:
        ah = lj_potential(r, sigma, epsilon) - lam * lj_potential(rc, sigma, epsilon) + epsilon * (1 - lam)
    elif r <= rc:

        ah = lam * (lj_potential(r, sigma, epsilon) - lj_potential(rc, sigma, epsilon))
    else:
        ah = 0.0  # Ensure this is a float
    return ah

def ah_scaled(r,sig,eps,l,rc):
    ah = ah_potential(r,sig,eps,l,rc)
    ahs = ah*4*np.pi*r**2
    return ahs
