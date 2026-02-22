import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from functools import partial

def lacey_mixing_curve(time, k, tau, m0):
    """Curve for the Lacey mixing model.

    The Lacey mixing model is a exponential model that describes the 
    mixing of a binary system. The model is classically defined as [1] 
    [2]:

    .. math:: 
    
        M(t) = 1 - (1 - M_0) e^{-kt}

    where :math:`M(t)` is the Lacey mixing index at time :math:`t`, 
    :math:`M_0` is the initial Lacey Mixing index, :math:`k` is the 
    mixing rate constant, and :math:`t` is the time.

    An extension of the Lacey mixing model to include a time constant
    :math:`\\tau` to allow for delayed onset of exponential mixing 
    mixing behavior was proposed by Ratnayake et al. [2]:

    .. math::

        M(t) = max(1 - (1 - M_0) e^{-k(t - \\tau)})

    This mixing model is used to fit the mixing data as with no delayed
    onset of exponential mixing Ratnayake et al.'s model reduces to the
    classical Lacey mixing model.

    References
    ----------

    [1] Lacey PM. Developments in the theory of particle mixing. 
        Journal of applied chemistry. 1954 May;4(5):257-68.

    [2] Ratnayake P, Chandratilleke R, Bao J, Shen Y. A soft-sensor 
        approach to mixing rate determination in powder mixers. Powder 
        Technology. 2018 Aug 1;336:493-505.

    
    Parameters
    ----------
    time : array-like
        The time data for the lacey mixing index.
    k : float
        The rate constant with units of inverse time.
    tau : float
        The time constant. Allows for delayed onset of exponential 
        mixing.
    m0 : float
        The initial value of the Lacey mixing index.

    Returns
    -------
    array-like
        The predicted values for the Lacey mixing curve.

    Raises
    ------
    ValueError
        If time is not an array-like object.
    ValueError
        If k is not an integer or float.
    ValueError
        If tau is not an integer or float.
    ValueError
        If m0 is not an integer or float.
    """
    if not isinstance(time, np.ndarray):
        time = np.array(time)
        raise ValueError("time must be an array-like object")
    
    if not isinstance(k, (int, float)):
        raise ValueError("k must be an integer or float")
    
    if not isinstance(tau, (int, float)):
        raise ValueError("tau must be an integer or float")
    
    if not isinstance(m0, (int, float)):
        raise ValueError("m0 must be an integer or float")
    
    return [max((1 - (1 - m0) * np.exp(-k*(t - tau))), m0) for t in time]

def lacey_mixing_curve_fit(time, m, t0=0, tend=None):
    """Fit the Lacey mixing curve to the data.

    The Lacey mixing model is a exponential model that describes the 
    mixing of a binary system. The model is classically defined as [1] 
    [2]:

    .. math:: 
    
        M(t) = 1 - (1 - M_0) e^{-kt}

    where :math:`M(t)` is the Lacey mixing index at time :math:`t`, 
    :math:`M_0` is the initial Lacey Mixing index, :math:`k` is the 
    mixing rate constant, and :math:`t` is the time.

    An extension of the Lacey mixing model to include a time constant
    :math:`\\tau` to allow for delayed onset of exponential mixing 
    mixing behavior was proposed by Ratnayake et al. [2]:

    .. math::

        M(t) = max(1 - (1 - M_0) e^{-k(t - \\tau)})

    This mixing model is used to fit the mixing data as with no delayed
    onset of exponential mixing Ratnayake et al.'s model reduces to the
    classical Lacey mixing model.

    References
    ----------

    [1] Lacey PM. Developments in the theory of particle mixing. 
        Journal of applied chemistry. 1954 May;4(5):257-68.

    [2] Ratnayake P, Chandratilleke R, Bao J, Shen Y. A soft-sensor 
        approach to mixing rate determination in powder mixers. Powder 
        Technology. 2018 Aug 1;336:493-505.

    
    Parameters
    ----------
    time : array-like
        The time data for the lacey mixing index.
    m : array-like
        The lacey mixing index data.
    t0 : int or float
        The time at which mixing begins by default 0.
    tend : int or float, optional
        The time at which mixing ends, by default None. If None, then
        all the time data from the start time will be used.

    Returns
    -------
    popt : array-like
        The optimal values for the parameters k and tau.
    pcov : 2D array
        The estimated covariance of popt as returned by curve_fit.
    time_mixing : array-like
        The time data for the mixing period used in the fit.
    m_mixing : array-like
        The lacey mixing index data for the mixing period used in the
        fit.
    m_fit : array-like
        The predicted lacey mixing index values for the mixing period 
        calculated using the optimal parameters.

    Raises
    ------
    ValueError
        If time is not an array-like object.
    ValueError
        If m is not an array-like object.
    ValueError
        If t0 is not an integer or float.
    ValueError
        If tend is not an integer or float.
    ValueError
        If time and m are not the same length.
    """
    if not isinstance(time, np.ndarray):
        time = np.array(time)
        raise ValueError("time must be an array-like object")
    
    if not isinstance(m, np.ndarray):
        m = np.array(m)
        raise ValueError("m must be an array-like object")
    
    if not isinstance(t0, (int, float)):
        raise ValueError("t0 must be an integer or float")
    
    if tend is None:
        tend = time[-1]
    elif not isinstance(tend, (int, float)):
        raise ValueError("tend must be an integer or float")
    
    if len(time) != len(m):
        raise ValueError("time and m must be the same length")
    
    mixing_indices = (time >= t0) & (time <= tend)
    time_mixing = time[mixing_indices]
    m_mixing = m[mixing_indices]

    m0 = m_mixing[0]
    t0 = time_mixing[0]

    partial_lacey_mixing_curve = partial(lacey_mixing_curve, m0=m0)
    popt, pcov = curve_fit(partial_lacey_mixing_curve, 
                        time_mixing, 
                        m_mixing,
                        p0=(0, t0), 
                        bounds=([0, t0], [np.inf, np.inf]),
                        maxfev=10000,
                        )
    
    m_fit = partial_lacey_mixing_curve(time_mixing, *popt)

    return popt, pcov, time_mixing, m_mixing, m_fit