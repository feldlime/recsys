import time

from collections import Counter
from typing import Dict, TypedDict

import numpy as np
import pandas as pd
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender


class UserKnn:
    """Class for fit-predict UserKNN model
       based on ItemKNN model from implicit.nearest_neighbours
    """

    def __init__(self, model: ItemItemRecommender, N_users: int = 50):
        self.N_users = N_users
        self.model = model
        self.is_fitted = False

    @staticmethod
    def get_matrix(df: pd.DataFrame,
            user_col: str = 'user_id',
            item_col: str = 'item_id',
            weight_col: str = None,
            users_mapping: Dict[int, int] = None,
            items_mapping: Dict[int, int] = None):

        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        interaction_matrix = sp.sparse.csr_matrix((
            weights,
            (
                df[user_col].map(users_mapping.get),
                df[item_col].map(items_mapping.get)
            )
        ))
        return interaction_matrix

    def get_mappings(self, train):
        self.users_inv_mapping = dict(enumerate(train['user_id'].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(enumerate(train['item_id'].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    @staticmethod
    def idf(n: int, x: float):
        return np.log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, df: pd.DataFrame):
        item_cnt = Counter(df['item_id'].values)
        item_idf = pd.DataFrame.from_dict(item_cnt, orient='index', columns=['doc_freq']).reset_index()
        item_idf['idf'] = item_idf['doc_freq'].apply(lambda x: self.idf(self.n, x))
        self.item_idf = item_idf

    def fit(self, train: pd.DataFrame):
        self.user_knn = self.model
        self.get_mappings(train)
        self.weights_matrix = self.get_matrix(train, users_mapping=self.users_mapping,
                                              items_mapping=self.items_mapping)

        self.n = train.shape[0]
        self._count_item_idf(train)

        self.user_knn.fit(self.weights_matrix)
        self.is_fitted = True

    @staticmethod
    def _generate_recs_mapper(model: ItemItemRecommender, user_mapping: Dict[int, int],
            user_inv_mapping: Dict[int, int], N: int):
        def _recs_mapper(user):
            user_id = user_mapping[user]
            recs = model.similar_items(user_id, N=N)
            return [user_inv_mapping[user] for user, _ in recs], [sim for _, sim in recs]

        return _recs_mapper

    @staticmethod
    def get_viewed_item_ids(user_items: sp.sparse.csr_matrix, user_id: TypedDict) -> list[int]:
        """
        Return indices of items that user has interacted with.
        Parameters
        ----------
        user_items : csr_matrix
            Matrix of interactions.
        user_id : int
            Internal user id.
        Returns
        -------
        np.ndarray
            Internal item indices that user has interacted with.
        """
        return [user_items.indices[user_items.indptr[i]: user_items.indptr[i + 1]] for i in user_id]

    def timeit(func):
        """
        Decorator for measuring function's running time.
        """

        def measure_time(*args, **kw):
            start_time = time.time()
            result = func(*args, **kw)
            print("Processing time of %s(): %.2f seconds."
                  % (func.__qualname__, time.time() - start_time))
            return result

        return measure_time

    @timeit
    def predict(self, test: pd.DataFrame, interactions: pd.DataFrame, N_recs: int = 10) -> pd.DataFrame:
        """
        Function for predict recommendation for user
        :param test: users for predict
        :param interactions: past interactions
        :param N_recs: number of recommendations
        :return: pd.DataFrame with user_id, recs
        """
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        # create recs dataset with unique test id
        recs = pd.DataFrame({'user_id': test['user_id'].unique()})
        # find cold users
        cold_users = np.array(recs[~recs['user_id'].isin(self.users_inv_mapping.values())]['user_id'])
        # calculate popular items
        popular_recs = np.array(interactions.groupby('item_id').count().sort_values(
            by='user_id', ascending=False)[:N_recs].index)
        # add popular items for cold users
        recs_for_cold_users = pd.DataFrame({'user_id': np.repeat(cold_users, N_recs),
                                            'item_id': np.tile(popular_recs, cold_users.shape[0])})
        # drop cold users from recs
        recs = recs[~recs['user_id'].isin(cold_users)]

        mapper = self._generate_recs_mapper(
            model=self.user_knn,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N=self.N_users
        )

        # looking for watched
        watched = pd.DataFrame(
            {'user_id': self.users_inv_mapping.values(),
             'item_id': self.get_viewed_item_ids(user_items=self.weights_matrix,
                                                 user_id=self.users_mapping.values())}).set_index('user_id')

        recs['sim_user_id'], recs['sim'] = zip(*recs['user_id'].map(mapper))
        recs = recs.set_index('user_id').apply(pd.Series.explode).reset_index()

        recs = recs[~(recs['sim'] >= 1)] \
            .merge(watched, left_on=['sim_user_id'], right_on=['user_id'], how='left') \
            .explode('item_id') \
            .sort_values(['user_id', 'sim'], ascending=False) \
            .drop_duplicates(['user_id', 'item_id'], keep='first') \
            .merge(self.item_idf, left_on='item_id', right_on='index', how='left')

        recs['score'] = recs['sim'] * recs['idf']
        recs = recs.sort_values(['user_id', 'score'], ascending=False)
        recs['rank'] = recs.groupby('user_id').cumcount() + 1
        return pd.concat([recs[recs['rank'] <= N_recs][['user_id', 'item_id']], recs_for_cold_users])


    @timeit
    def predict_one(self, test: int, interactions: pd.DataFrame, N_recs: int = 10) -> np.array:
        """
        Function for predict recommendation for user
        :param test: users for predict
        :param interactions: past interactions
        :param N_recs: number of recommendations
        :return: pd.DataFrame with user_id, recs
        """
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        # create recs dataset with unique test id
        recs = pd.DataFrame({'user_id': [test]})

        # calculate popular items
        popular_recs = np.array(interactions.groupby('item_id').count().sort_values(
            by='user_id', ascending=False)[:N_recs].index)

        # check cold users
        if test not in self.users_inv_mapping.values():
            return popular_recs

        mapper = self._generate_recs_mapper(
            model=self.user_knn,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N=self.N_users
        )

        recs['sim_user_id'], recs['sim'] = zip(*recs['user_id'].map(mapper))
        recs = recs.set_index('user_id').apply(pd.Series.explode).reset_index()
        sim_user_map = recs['sim_user_id'].map(self.users_mapping.get)

        # looking for watched
        watched = pd.DataFrame(
            {'user_id': recs['sim_user_id'],
             'item_id': self.get_viewed_item_ids(user_items=self.weights_matrix,
                                                 user_id=sim_user_map)}).set_index('user_id')

        recs = recs[~(recs['sim'] >= 1)] \
            .merge(watched, left_on=['sim_user_id'], right_on=['user_id'], how='left') \
            .explode('item_id') \
            .sort_values(['user_id', 'sim'], ascending=False) \
            .drop_duplicates(['user_id', 'item_id'], keep='first') \
            .merge(self.item_idf, left_on='item_id', right_on='index', how='left')

        recs['score'] = recs['sim'] * recs['idf']
        recs = recs.sort_values(['user_id', 'score'], ascending=False)
        recs['rank'] = recs.groupby('user_id').cumcount() + 1
        return recs[recs['rank'] <= N_recs][['user_id', 'item_id']]


