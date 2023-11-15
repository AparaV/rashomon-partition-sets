import numpy as np


class RashomonSubproblemCache:
    """
    Caching object to keep track of subproblems
    Some discussion on how to hash a numpy array
    https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        self.C = set()

    def insert(self, sigma: np.ndarray, i: int, j: int) -> None:
        self.__verify_shape__(sigma)
        sigma_ij = self.__process__(sigma, i, j)
        self.C.add(sigma_ij)

    def seen(self, sigma: np.ndarray, i: int, j: int) -> bool:
        self.__verify_shape__(sigma)
        sigma_ij = self.__process__(sigma, i, j)
        return sigma_ij in self.C

    @property
    def size(self):
        return len(self.C)

    def __process__(self, sigma: np.ndarray, i: int, j: int):
        sigma_ij = np.copy(sigma)
        sigma_ij[i, j:] = np.nan
        byte_rep = sigma_ij.tobytes()
        return hash(byte_rep)

    def __verify_shape__(self, sigma: np.ndarray):
        # Distinct arrays of different shapes may have same byte representation
        if self.shape != sigma.shape:
            raise RuntimeError(
                f"Expected array of dimensions {self.shape}. Received {sigma.shape}"
            )


if __name__ == "__init__":
    sigma = np.array([[1, 1, 0], [0, 1, 1]], dtype="float64")
    i = 0
    j = 0

    seen_sigma = RashomonSubproblemCache(shape=(2, 3))
    seen_sigma.insert(sigma, i, j)

    # Should be True
    test1 = seen_sigma.seen(sigma, i, j)
    assert test1

    # Should be False
    sigma2 = np.copy(sigma)
    sigma2[1, 1] = 1 - sigma2[1, 1]
    test2 = ~seen_sigma.seen(sigma2, i, j)
    assert test2

    # Should be True
    sigma3 = np.copy(sigma)
    sigma3[0, 2] = 1 - sigma3[0, 2]
    test3 = seen_sigma.seen(sigma3, i, j)
    assert test3
