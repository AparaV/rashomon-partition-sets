import numpy as np

class RashomonCache:
    """
    Caching object to keep track of fixed sigma matrices
    """

    def __init__(self):
        self.C = set()

    def insert(self, sigma, i, j):
        sigma_ij = self.__process__(sigma, i, j)
        self.C.add(sigma_ij)

    def seen(self, sigma, i, j):
        sigma_ij = self.__process__(sigma, i, j)
        return sigma_ij in self.C

    def __process__(self, sigma, i, j):
        sigma_ij = np.copy(sigma)
        sigma_ij[i, j:] = np.nan
        # Some discussion on how to hash a numpy array
        # https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
        return sigma_ij.tobytes()



if __name__ == "__init__":
    sigma = np.array([[1, 1, 0],
                      [0, 1, 1]], dtype='float64')
    i = 0
    j = 0
    
    seen_sigma = RashomonCache()
    seen_sigma.insert(sigma, i, j)

    # Should be True
    assert(seen_sigma.seen(sigma, i, j) == True)

    # Should be False
    sigma2 = np.copy(sigma)
    sigma2[1, 1] = 1 - sigma2[1, 1]
    assert(seen_sigma.seen(sigma2, i, j) == False)

    # Should be True
    sigma3 = np.copy(sigma)
    sigma3[0, 2] = 1 - sigma3[0, 2]
    assert(seen_sigma.seen(sigma3, i, j) == True)

