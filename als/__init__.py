import csv
from abc import ABC
import numpy as np
from scipy import sparse
from typing import Tuple


def load_data(path, headers=True):
    max_item = 0
    max_user = 0
    data = []
    pairs = []
    with open(path, 'rt') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        if headers:
            next(reader, None)
        for row in reader:
            item_id = int(row[0])
            user_id = int(row[1])
            rating = float(row[2])
            max_item = max(item_id, max_item)
            max_user = max(user_id, max_user)
            data.append((item_id, user_id, rating,))
            pairs.append((item_id, user_id,))

    ratings = sparse.lil_matrix((max_item, max_user))
    for d in data:
        ratings[d[0] - 1, d[1] - 1] = d[2]
    return ratings


class ALS(ABC):
    def __init__(self, ratings: sparse.lil_matrix, rank: int):
        self._ratings = ratings
        self._rank = rank


DEFAULT_ALPHA = 0.0001
DEFAULT_BETA = 0.01


class BatchALS(ALS):
    def __init__(self, ratings: sparse.lil_matrix,
                 rank: int,
                 alpha: float = DEFAULT_ALPHA,
                 beta: float = DEFAULT_BETA) -> None:
        super().__init__(ratings, rank)
        self._alpha = alpha
        self._beta = beta

    def run(self,
            item_factors: np.ndarray,
            user_factors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        user_factors = user_factors.T
        indices = self._ratings.nonzero()
        for i in indices[0]:
            for j in indices[1]:
                rating = self._ratings[i, j]
                error: float = rating - \
                               self.predict(item_factors, user_factors, i, j)
                for k in range(self._rank):
                    item_factors[i][k] = item_factors[i][k] + self._alpha * \
                                         (2 * error * user_factors[k][j] - self._beta * item_factors[i][k])
                    user_factors[k][j] = user_factors[k][j] + self._alpha * \
                                         (2 * error * item_factors[i][k] - self._beta * user_factors[k][j])
        e = 0.0
        for i in indices[0]:
            for j in indices[1]:
                rating = self._ratings[i, j]
                e = e + pow(rating -
                            np.dot(item_factors[i, :], user_factors[:, j]), 2)
                for k in range(self._rank):
                    e = e + (self._beta / 2.0) * \
                        (item_factors[i][k] ** 2.0 + user_factors[k][j] ** 2.0)
        return item_factors, user_factors.T

    @staticmethod
    def predict(item_factors: np.ndarray,
                user_factors: np.ndarray,
                i: int,
                j: int) -> np.ndarray:
        return np.dot(item_factors[i, :], user_factors[:, j])

    @staticmethod
    def random_factors(nUsers: int,
                       nRatings: int,
                       rank: int) -> Tuple[np.ndarray, np.ndarray]:
        item_factors: np.ndarray = np.random.rand(nRatings, rank)
        user_factors: np.ndarray = np.random.rand(nUsers, rank)

        return item_factors, user_factors
