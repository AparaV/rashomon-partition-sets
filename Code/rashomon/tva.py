import itertools
import numpy as np


#
# Enumerate policies
#


def __enumerate_policies_classic__(M, R):
    # All arms have the same intensities
    intensities = np.arange(R - 1) + 1
    policies = []
    for pol in itertools.product(intensities, repeat=M):
        policies.append(pol)
    return policies


def __enumerate_policies_complex__(R):
    # Each arm may have different intensities
    intensities = []
    for Ri in R:
        intensities.append(np.arange(Ri - 1) + 1)
    policies = []
    for pol in itertools.product(*intensities, repeat=1):
        policies.append(pol)
    return policies


def enumerate_policies(M, R):
    if isinstance(R, int) or len(R) == 1:
        if not isinstance(R, int):
            R = R[0]
        policies = __enumerate_policies_classic__(M, R)
    else:
        policies = __enumerate_policies_complex__(R)
    return policies
