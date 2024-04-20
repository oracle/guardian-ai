import numpy as np
import pandas as pd
from abc import abstractmethod
from recommenders.utils.constants import SEED
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import (
    map, ndcg_at_k, precision_at_k, recall_at_k
)


class CFModel:
    """
    Wrapper for the target and shadow recommender models.
    For now, we're only supporting microsoft recommenders.
    
    """
    
    def __init__(self, top_k, n_factors, layers, epochs, batch_size, lr):
        """
        Create the target model that is being attacked.
        Parameters
        ----------
        df: dataframe with three columns: userID, itemID, rating.
            The original dataset X, Y where X is a 1-d array of users and
            Y is a ndarray of shape (n_users, ) needs to be transformed
            into a pandas dataframe.
        data: Recommenders data-object
            Data object that could be utilized to train the model.
        index2user: A dictionary having the indices as keys and user ids as values.
            Used to convert the indexed user id into the original user id.
        user2index: A dictionary having the user ids as keys and indices as values.
            Used to convert the original user id into an indexed user id.
        item2index: A dictionary having the indices as keys and item ids as values.
            Used to convert the indexed item id into the original item id.
        index2item: A dictionary having the item ids as keys and indices as values.
            Used to convert the original item id into an indexed item id.
        top_k: int
            Specifies the number of recommendations.
        model: str
            Model name. Three models are supported: NCF, GMF and MLP
        """
        self.df = None
        self.data = None
        self.index2user = None
        self.user2index = None
        self.item2index = None
        self.index2item = None
        self.model = None
        self.n_factors = n_factors
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.top_k = top_k
        self.model_name = self.get_model_name()
    
    @abstractmethod
    def get_model_name(self):
        """Get default model name."""
        pass
    
    def train_test_split(self, df_reindex):
        """
        Split the dataframe into train and test datasets.
        Return
        ----------
        tuple: A tuple containing two pandas.DataFrame objects
            - the first dataframe contains the training dataset with columns userId, itemId, rating
            - the second dataframe contains the test dataset with columns userId, itemId, rating
        """
        train, test = python_stratified_split(df_reindex, 0.8)
        assert df_reindex.userID.nunique() == train.userID.nunique() == test.userID.nunique()
        return train, test
    
    def data_object(self, train, test):
        """
        Creates validation set using the leave out once cross validation approach and a NCF data object
        Parameters
        ----------
        train: pandas.DataFrame containing the training dataset with three columns userId, itemID, rating
        test: pandas.DataFrame containing the test dataset with three columns userId, itemID, rating
        Return
        ----------
        None
        """
        leave_one_out_test = test.groupby("userID").last().reset_index()
        train_file = "./train.csv"
        test_file = "./test.csv"
        leave_one_out_test_file = "./leave_one_out_test.csv"
        train.to_csv(train_file, index=False)
        test.to_csv(test_file, index=False)
        print (test)
        leave_one_out_test.to_csv(leave_one_out_test_file, index=False)
        self.data = NCFDataset(train_file=train_file, test_file=leave_one_out_test_file, seed=SEED,
                               overwrite_test_file_full=True)
    
    @abstractmethod
    def get_model(self):
        """
        Create the target model that is being attacked.

        Returns
        -------
        Model that is not yet trained.
        """
        pass
    
    def reindex(self, X, y):
        """
        Transforms the dataset represented by X which is a 1-d array of user ids and Y which
        is a ndarray of (n_users, n_items)

        Returns
        -------
        train: Transformed pandas.dataframe object containing three columns userId, itemID and rating.
        """
        user_ids = []
        item_ids = []
        ratings = []
        temp_df = pd.concat([X, y], axis=1)
        for index, row in temp_df.iterrows():
            user_id = row['userID']
            interactions = row['interactions']
            for item_id, rating in enumerate(interactions, start=1):
                if rating != 0:
                    user_ids.append(user_id)
                    item_ids.append(item_id)
                    ratings.append(rating)
        
        self.df = pd.DataFrame(
            {
                'userID': user_ids,
                'itemID': item_ids,
                'rating': ratings
            }
        )
        user_list = list(self.df['userID'].drop_duplicates())
        item_list = list(self.df['itemID'].drop_duplicates())
        self.item2index = {w: i for i, w in enumerate(item_list)}
        self.index2item = {i: w for i, w in enumerate(item_list)}
        self.user2index = {w: i for i, w in enumerate(user_list)}
        self.index2user = {i: w for i, w in enumerate(user_list)}
        df_reindex = self.df.copy()
        df_reindex['userID'] = self.df['userID'].apply(lambda x: self.user2index[x])
        df_reindex['itemID'] = self.df['itemID'].apply(lambda x: self.item2index[x])
        return df_reindex
    
    def train_model(self):
        """
        Train the model that is being attacked.

        Parameters
        ----------
        data: data object
        
        Returns
        -------
        Trained model

        """
        self.model = NCF(
            n_users=self.data.n_users,
            n_items=self.data.n_items,
            model_type=self.model_name,
            n_factors=self.n_factors,
            layer_sizes=self.layers,
            n_epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=10)
        return self.model.fit(self.data)
    
    def test_model(self, test, predictions):
        """
        Test the model that is being attacked.

        Parameters
        ----------
        test: {array-like, sparse matrix} of shape (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
            Input variables of the test set for the target model.
        predictions: ndarray of shape (n_samples,)
            Output labels of the test set for the target model.
        """
        eval_map = map(test, predictions, k=self.top_k)
        eval_ndcg = ndcg_at_k(test, predictions, k=self.top_k)
        eval_precision = precision_at_k(test, predictions, k=self.top_k)
        eval_recall = recall_at_k(test, predictions, k=self.top_k)
        return eval_map, eval_ndcg, eval_precision, eval_recall
    
    def get_hit_rate(self):
        """
        Gets hit rate.
        """
        k = self.top_k
        
        ndcg = []
        hit_ratio = []
        
        for test in self.data.test_loader():
            user_input, item_input, labels = test
            output = self.model.predict(user_input, item_input, is_list=True)
            output = np.squeeze(output)
            rank = sum(output >= output[0])
            if rank <= k:
                ndcg.append(1 / np.log(rank + 1))
                hit_ratio.append(1)
            else:
                ndcg.append(0)
                hit_ratio.append(0)
        
        eval_ndcg = np.mean(ndcg)
        eval_hr = np.mean(hit_ratio)
        return eval_ndcg, eval_hr
    
    def get_predictions_user(self, user_id, items_id):
        """
        Gets model prediction for a single user.

        Parameters
        ----------
        user_id: An integer representing the user id.
        
        Returns
        ----------
        sorted_predictions_reindex: List of items recommended to the user.
        """
        user_indexed = self.user2index[user_id]
        user_indexed_list = [user_indexed] * len(items_id)
        items_not_rated = self.df[~self.df['itemID'].isin(items_id)]
        items_not_rated_indexed = [self.item2index[item] for item in items_not_rated]
        predictions = self.model.predict(user_indexed_list, items_not_rated_indexed, is_list=True)
        sorted_predictions = [x for _, x in sorted(zip(predictions, items_not_rated_indexed), reverse=True)]
        sorted_predictions_reindex = [self.index2item[item] for item in sorted_predictions]
        return sorted_predictions_reindex
    
    def get_predictions(self, train):
        """
        Gets model prediction for number of users.

        Parameters
        ----------
        train: pandas.Dataframe object with three columns userId, itemID and rating.

        Returns
        -------
        all_predictions: pandas.DataFrame object with user, item and the corresponding score from the model.
        """
        users, items, predictions = [], [], []
        item = list(train.itemID.unique())
        for user in train.userID.unique():
            user = [user] * len(item)
            users.extend(user)
            items.extend(item)
            predictions.extend(list(self.model.predict(user, item, is_list=True)))
        
        all_predictions = pd.DataFrame(data={"userID": users, "itemID": items, "prediction": predictions})
        
        merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)
        return all_predictions
    
    def save_model(self, filename):
        """
        Save model.

        Parameters
        ----------
        filename: FileDescriptorOrPath

        """
        self.model.save(filename)
    
    def load_model(self, filename):
        """
        Load model.

        Parameters
        ----------
        filename: FileDescriptorOrPath
        
        Returns
        ----------
        """
        self.model = self.model.load(filename)
    
    def get_most_popular(self, items):
        """
        Recommends the most popular items.

        Parameters
        ----------
        items: Items that the user has interacted with

        Returns
        -------
        top_recommendations: List of top_k most popular recommendations sans the items the user has already interacted
        with.
        """
        
        item_popularity = self.df['itemID'].value_counts().reset_index()
        item_popularity.columns = ['itemID', 'interaction_count']
        # Filter out items the user has already interacted with
        recommended_items = item_popularity[~item_popularity['itemID'].isin(items)]
        # Get the top N popular items
        top_recommendations = recommended_items.head(self.top_k)['itemID'].tolist()
        return top_recommendations


class NeuMF(CFModel):
    def __init__(self, top_k, n_factors, layers, epochs, batch_size, lr):
        super(NeuMF, self).__init__(top_k, n_factors, layers, epochs, batch_size, lr)
    
    # return self.model.fit(x_train, y_train)
    def get_model_name(self):
        return "NeuMF"


class GMF(CFModel):
    def __init__(self, top_k, n_factors, layers, epochs, batch_size, lr):
        super(GMF, self).__init__(top_k, n_factors, layers, epochs, batch_size, lr)
    
    # return self.model.fit(x_train, y_train)
    def get_model_name(self):
        return "GMF"


class MLP(CFModel):
    def __init__(self, top_k, n_factors, layers, epochs, batch_size, lr):
        super(MLP, self).__init__(top_k, n_factors, layers, epochs, batch_size, lr)
    
    def get_model_name(self):
        return "MLP"
