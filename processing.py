'''
Functions for processing data
'''
import numpy as np
from scipy.interpolate import interp1d


# Interpolates curvature from given positions (s) and corresponding curvature vectors (u)
# Returns an array of n curvature vectors that span over length
def interpolate(positions, curvatures, n, length, method = 'linear'):
    f1 = interp1d(positions, curvatures[:,0], kind=method, fill_value="extrapolate")
    f2 = interp1d(positions, curvatures[:, 1], kind=method, fill_value="extrapolate")
    f3 = interp1d(positions, curvatures[:, 2], kind=method, fill_value="extrapolate")
    x = np.linspace(0, length, num=n, endpoint=True)
    interpolated = np.vstack([f1(x), f2(x), f3(x)]).T
    return x, interpolated

