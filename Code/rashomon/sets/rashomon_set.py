import numpy as np


class RashomonSet:
    """
    Caching object to keep track of poolings in Rashomon set
    """

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        self.P_hash = set()
        self.P_qe = []

    def insert(self, sigma: np.ndarray) -> None:
        if self.seen(sigma):
            return
        sigma_hash = self.__process__(sigma)
        self.P_hash.add(sigma_hash)
        self.P_qe.append(sigma)

    def seen(self, sigma: np.ndarray) -> bool:
        self.__verify_shape__(sigma)
        sigma_hash = self.__process__(sigma)
        return sigma_hash in self.P_hash

    @property
    def size(self):
        return len(self.P_qe)

    def __process__(self, sigma: np.ndarray):
        byte_rep = sigma.tobytes()
        return hash(byte_rep)

    def __verify_shape__(self, sigma: np.ndarray):
        # Distinct arrays of different shapes may have same byte representation
        if self.shape != sigma.shape:
            raise RuntimeError(
                f"Expected array of dimensions {self.shape}. Received {sigma.shape}"
            )

    def __iter__(self):
        return iter(self.P_qe)

    def __repr__(self):
        return repr(self.P_qe)
