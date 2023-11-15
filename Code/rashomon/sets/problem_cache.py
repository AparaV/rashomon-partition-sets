import numpy as np


class RashomonProblemCache:
    """
    Caching object to keep track of full problems
    Some discussion on how to hash a numpy array
    https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        self.C = set()

    def insert(self, sigma: np.ndarray) -> None:
        self.__verify_shape__(sigma)
        sigma_hash = self.__process__(sigma)
        self.C.add(sigma_hash)

    def seen(self, sigma: np.ndarray) -> bool:
        self.__verify_shape__(sigma)
        sigma_hash = self.__process__(sigma)
        return sigma_hash in self.C

    @property
    def size(self):
        return len(self.C)

    def __process__(self, sigma: np.ndarray):
        byte_rep = sigma.tobytes()
        return hash(byte_rep)

    def __verify_shape__(self, sigma: np.ndarray):
        # Distinct arrays of different shapes may have same byte representation
        if self.shape != sigma.shape:
            raise RuntimeError(
                f"Expected array of dimensions {self.shape}. Received {sigma.shape}"
            )
