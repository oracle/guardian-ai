import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from abc import abstractmethod
import random
from model_utils import GeneralizedMatrixFactorization, NeuralCollaborativeFiltering, MultiLayerPerceptron


class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, user_list, item_list, rating_list):
        super(RatingDataset, self).__init__()
        self.user_list = user_list
        self.item_list = item_list
        self.rating_list = rating_list
    
    def __len__(self):
        return len(self.user_list)
    
    def __getitem__(self, idx):
        user_ = self.user_list[idx]
        item_ = self.item_list[idx]
        rating = self.rating_list[idx]
        
        return (
            torch.tensor(user_, dtype=torch.long),
            torch.tensor(item_, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float)
        )


class CFModel:
    """
    Wrapper for the target and shadow recommender models.
    For now, we're only supporting microsoft recommenders.
    
    """
    
    def __init__(self, top_k, batch_size, epochs, lr, num_negatives, num_negatives_test):
        """
        Create the target model that is being attacked.
        Parameters
        ----------
        """
        self.model = None
        self.num_ng = num_negatives
        self.num_ng_test = num_negatives_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.top_k = top_k
        self.model_name = self.get_model_name()
    
    @abstractmethod
    def get_model_name(self):
        """Get default model name."""
        pass
    
    def _leave_one_out(dataframe):
        """
        leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
        """
        train_df = pd.DataFrame(columns=['userID', 'itemID'])
        test_df = pd.DataFrame(columns=['userID', 'itemID'])
        for _, group in dataframe.groupby('userID'):
            train_df = pd.concat([train_df, group.iloc[:-1]], ignore_index=True)
            test_df = pd.concat([test_df, group.iloc[-1:]], ignore_index=True)
        return train_df, test_df
    
    def negative_sampling(self, dataframe):
        items = set(dataframe.itemID.tolist())
        interact_status = (
            dataframe.groupby('userID')['itemID']
            .apply(set)
            .reset_index()
            .rename(columns={'itemID': 'interactions'}))
        interact_status['negatives'] = interact_status['interactions'].apply(lambda x: items - x)
        interact_status['sampled_negatives'] = interact_status['interactions'].apply(
            lambda x: random.sample(list(x), self.num_ng_test))
        print(interact_status)
        return interact_status[['userID', 'negatives', 'sampled_negatives']]
    
    def train_test_split(self, df_reindex):
        """
        Split the dataframe into train and test datasets.
        Return
        ----------
        tuple: A tuple containing two pandas.DataFrame objects
            - the first dataframe contains the training dataset with columns userId, itemId, rating
            - the second dataframe contains the test dataset with columns userId, itemId, rating
        """
        train, test = self._leave_one_out(df_reindex)
        assert train.userID.nunique() == test.userID.nunique()
        negatives = self.negative_sampling(df_reindex)
        return train, test, negatives
    
    def get_train_instance(self, train, negatives):
        users, items, ratings = [], [], []
        train_ratings = pd.merge(train, negatives[['userID', 'negatives']], on='userID')
        train_ratings['sampled_negatives'] = train_ratings['negatives'].apply(
            lambda x: random.sample(list(x), self.num_ng))
        for row in train_ratings.itertuples():
            users.append(int(row.userID))
            items.append(int(row.itemID))
            ratings.append(float(1))
            for i in range(self.num_ng):
                users.append(int(row.userID))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = RatingDataset(
            user_list=users,
            item_list=items,
            rating_list=ratings)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def get_test_instance(self, test, negatives):
        users, items, ratings = [], [], []
        test_ratings = pd.merge(test, negatives[['userID', 'sampled_negatives']], on='userID')
        for row in test_ratings.itertuples():
            users.append(int(row.userID))
            items.append(int(row.itemID))
            ratings.append(float(1))
            for i in getattr(row, 'sampled_negatives'):
                users.append(int(row.userID))
                items.append(int(i))
                ratings.append(float(0))
        dataset = RatingDataset(
            user_list=users,
            item_list=items,
            rating_list=ratings)
        return torch.utils.data.DataLoader(dataset, batch_size=self.num_ng_test + 1, shuffle=False, num_workers=2)
    
    @abstractmethod
    def get_model(self, num_users, num_items):
        """
        Create the target model that is being attacked.

        Returns
        -------
        Model that is not yet trained.
        """
        pass
    
    @staticmethod
    def create_index_map(X, y):
        user_ids = X.userID.unique()
        exploded_df = y.explode('itemID')
        item_ids = exploded_df['itemID'].unique()
        user_map = [{original_id: index + 1 for index, original_id in enumerate(user_ids)},
                    {index + 1: original_id for index, original_id in enumerate(user_ids)}]
        item_map = [{original_id: index + 1 for index, original_id in enumerate(item_ids)},
                    {index + 1: original_id for index, original_id in enumerate(item_ids)}]
        
        return user_map, item_map
    
    def reindex(self, X, y):
        """
        Transforms the dataset represented by X which is a 1-d array of user ids and Y which
        is a ndarray of (n_users, n_items)

        Returns
        -------
        train: Transformed pandas.dataframe object containing three columns userId, itemID and rating.
        """
        temp_df = pd.concat([X, y], axis=1)
        df = temp_df.explode('itemID')
        df.reset_index(drop=True)
        user_map, item_map = self.create_index_map(X, y)
        df_reindex = df.copy()
        df_reindex['userID'] = df['userID'].apply(lambda x: user_map[0][x])
        df_reindex['itemID'] = df['itemID'].apply(lambda x: item_map[0][x])
        return df_reindex
    
    def train_model(self, train, negatives):
        """
        Train the model that is being attacked.

        Parameters
        ----------
        train:
        negatives:
        Returns
        -------
        Trained model

        """
        num_users = train.userID.nunique()
        num_items = train.itemID.nunique()
        self.model = self.get_model(num_users, num_items)
        train_loader = self.get_train_instance(train, negatives)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(1, self.epochs):
            for user, item, label in train_loader:
                optimizer.zero_grad()
                prediction = self.model(user, item)
                loss = loss_function(prediction, label)
                loss.backward()
                optimizer.step()
    
    def test_model(self, test, negatives):
        """
        Test the model that is being attacked.

        Parameters
        ----------
        test:
        negatives:
        """
        self.model.eval()
        test_loader = self.get_test_instance(test, negatives)
        with torch.no_grad():
            HR, NDCG = self.metrics(self.model, test_loader, self.top_k)
        return HR, NDCG
    
    @staticmethod
    def hit(ng_item, pred_items):
        if ng_item in pred_items:
            return 1
        return 0
    
    @staticmethod
    def ndcg(ng_item, pred_items):
        if ng_item in pred_items:
            index = pred_items.index(ng_item)
            return np.reciprocal(np.log2(index + 2))
        return 0
    
    def metrics(self, model, test_loader, top_k):
        HR, NDCG = [], []
        for user, item, label in test_loader:
            predictions = model(user, item)
            _, indices = torch.topk(predictions, top_k)
            recommends = torch.take(
                item, indices).cpu().numpy().tolist()
            ng_item = item[0].item()  # leave one-out evaluation has only one item per user
            HR.append(self.hit(ng_item, recommends))
            NDCG.append(self.ndcg(ng_item, recommends))
        
        return np.mean(HR), np.mean(NDCG)
    
    def get_predictions_user(self, all_items_reindexed, user_id, items_id):
        """
        Gets model prediction for a single user.

        Parameters
        ----------
        user_id: An integer representing the user id.
        items_id: List of item ids that the user interacted with.
        Returns
        ----------
        sorted_predictions_reindex: List of items recommended to the user.
        """
        items_not_rated = all_items_reindexed[all_items_reindexed['itemID'].isin(items_id)]
        user_list = [user_id] * len(items_not_rated)
        predictions = self.model.predict(user_list, items_not_rated, is_list=True)
        sorted_predictions = [x for _, x in sorted(zip(predictions, items_not_rated), reverse=True)]
        return sorted_predictions
        
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
    
    def get_most_popular(self, all_items, interactions):
        """
        Recommends the most popular items.

        Parameters
        ----------

        Returns
        -------
        top_recommendations: List of top_k most popular recommendations sans the items the user has already interacted
        with.
        """
        all_items_df = all_items.explode('itemID')
        item_popularity = all_items_df['itemID'].value_counts().reset_index()
        item_popularity.columns = ['itemID', 'interaction_count']
        recommended_items = item_popularity[~item_popularity['itemID'].isin(interactions)]
        top_recommendations = recommended_items.head(self.top_k)['itemID'].tolist()
        return top_recommendations


class NCFTargetModel(CFModel):
    def __init__(self, top_k, layers, latent_dim, epochs, batch_size, lr, num_negatives, num_negatives_test):
        self.top_k = top_k
        self.layers = layers
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_ng = num_negatives
        self.num_ng_test = num_negatives_test
        super(NCFTargetModel, self).__init__(top_k, batch_size, epochs, lr, num_negatives, num_negatives_test)
    
    def get_model(self, num_users, num_items):
        return NeuralCollaborativeFiltering(num_users, num_items, self.layers)
    
    def get_model_name(self):
        return "NCF"


class GMFTargetModel(CFModel):
    def __init__(self, top_k, latent_dim, epochs, batch_size, lr, num_negatives, num_negatives_test):
        self.top_k = top_k
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_ng = num_negatives
        self.num_ng_test = num_negatives_test
        super(GMFTargetModel, self).__init__(top_k, batch_size, epochs, lr, num_negatives, num_negatives_test)
    
    def get_model(self, num_users, num_items):
        return GeneralizedMatrixFactorization(num_users, num_items, self.latent_dim)
    
    def get_model_name(self):
        return "GMF"


class MLPTargetModel(CFModel):
    def __init__(self, top_k, layers, latent_dim, epochs, batch_size, lr, num_negatives, num_negatives_test):
        self.top_k = top_k
        self.layers = layers
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_ng = num_negatives
        self.num_ng_test = num_negatives_test
        super(MLPTargetModel, self).__init__(top_k, batch_size, epochs, lr, num_negatives, num_negatives_test)
    
    def get_model(self, num_users, num_items):
        return MultiLayerPerceptron(num_users, num_items, self.layers)
    
    def get_model_name(self):
        return "MLP"
