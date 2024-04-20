import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from abc import abstractmethod
import random
from random import sample
from guardian_ai.privacy_estimation.model_utils import GeneralizedMatrixFactorization, NeuralCollaborativeFiltering, \
    MultiLayerPerceptron
from torch.utils.data import DataLoader, TensorDataset


class CFModel:
    """
    Wrapper for the target and shadow recommender models.
    For now, we're only supporting PyTorch recommenders.
    
    """
    
    def __init__(self, top_k, batch_size, epochs, lr):
        """
        Create the target model that is being attacked.
        Parameters
        ----------
        top_k: top k recommendations
        batch_size: batch size
        epochs: number of epochs
        lr: learning rate
        """
        self.num_users = None
        self.num_items = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.top_k = top_k
        self.model = None
        self.model_name = self.get_model_name()
    
    @abstractmethod
    def get_model_name(self):
        """Get default model name."""
        pass
    
    @abstractmethod
    def get_model(self, num_users, num_items):
        """
        Get the appropriate model object.
        Parameters
        ----------
        num_users: number of users
        num_items: number of items
        Returns
        -------
        Model that is not yet trained.
        """
        pass
    
    def create_index_map(self, X, y):
        """
        Returns the dictionaries that map the original indices to continuous indices
        Parameters
        ----------
        X: pandas.DataFrame with a single column `userID`.
        y: pandas.DataFrame with a single column `interactions` that contains the list of items the user interacted with
        
        Returns
        -------
        user_map: dictionary that maps the original user ids to continuous user ids
        item_map: dictionary that maps the original item ids to continuous item ids
        """
        user_ids = sorted(X.userID.unique())
        exploded_df = y.explode('interactions').rename(columns={"interactions": "itemID"})
        item_ids = sorted(exploded_df['itemID'].unique())
        user_map = [{original_id: index for index, original_id in enumerate(user_ids)},
                    {index: original_id for index, original_id in enumerate(user_ids)}]
        item_map = [{original_id: index for index, original_id in enumerate(item_ids)},
                    {index: original_id for index, original_id in enumerate(item_ids)}]
        print(item_map[0])
        self.num_users = X.userID.nunique()
        self.num_items = exploded_df['itemID'].nunique()
        print(self.num_users, self.num_items)
        return user_map, item_map
    
    def reindex(self, X, y):
        """
        Transforms the dataset
        Parameters
        ----------
        X: pandas.DataFrame with a single column `userID`.
        y: pandas.DataFrame with a single column `interactions` that contains the list of items the user interacted with

        Returns
        -------
        df_reindex: pandas.DataFrame containing the continuous user ids and the items ids
        """
        user_map, item_map = self.create_index_map(X, y)
        temp_df = pd.concat([X, y], axis=1)
        df_reindex = temp_df.explode('interactions').rename(columns={"interactions": "itemID"})
        df_reindex.reset_index(drop=True, inplace=True)
        df_reindex['userID'] = df_reindex['userID'].apply(lambda x: user_map[0][x])
        df_reindex['itemID'] = df_reindex['itemID'].apply(lambda x: item_map[0][x])
        print(df_reindex)
        return df_reindex
    
    @staticmethod
    def leave_one_out(dataframe):
        """
        leave-one-out evaluation protocol to create the test dataset
        Parameters
        ----------
        dataframe: pandas.DataFrame containing user ids and items ids.
        
        Returns
        -------
        train_df: pandas.DataFrame containing the training dataset
        test_df: pandas.DataFrame containing the test dataset
        """
        train_df = pd.DataFrame(columns=['userID', 'itemID'])
        test_df = pd.DataFrame(columns=['userID', 'itemID'])
        for _, group in dataframe.groupby('userID'):
            train_df = pd.concat([train_df, group.iloc[:-1]], ignore_index=True)
            test_df = pd.concat([test_df, group.iloc[-1:]], ignore_index=True)
        return train_df, test_df
    
    @staticmethod
    def negative_sampling(train, test):
        """
        Adds a column that consists the list of items that the user did not interact with.
        In the test dataset, this list also includes the one item that was removed from the
        training dataset during LOOCV protocol.
        Parameters
        ----------
        train: pandas.DataFrame containing training dataset
        test: pandas.DataFrame containing test dataset
        
        Return
        ----------
        tuple: A tuple containing two pandas.DataFrame objects
            - the first dataframe contains the training dataset with columns userId, interactions, negatives
            - the second dataframe contains the test dataset with columns userId, itemId, sampled_negatives
        """
        items = set(train.itemID.tolist())
        train_negatives = (
            train.groupby('userID')['itemID']
            .apply(set)
            .reset_index()
            .rename(columns={'itemID': 'interactions'}))
        train_negatives = pd.merge(train_negatives, test, on='userID')
        train_negatives['negatives'] = train_negatives.apply(
            lambda row: list(set(items) - set(row['interactions']) - {row['itemID']}), axis=1
        )
        train_negatives.drop(columns=["itemID"], axis=1, inplace=True)
        test_negatives = pd.merge(train_negatives, test, on='userID')
        test_negatives['sampled_negatives'] = test_negatives['negatives'].apply(lambda x: random.sample(list(x), 100))
        test_negatives['sampled_negatives'] = test_negatives.apply(
            lambda row: set(row['sampled_negatives']).union({row['itemID']}),
            axis=1
        )
        return train_negatives[['userID', 'interactions', 'negatives']], test_negatives[
            ['userID', 'itemID', 'sampled_negatives']]
    
    def train_test_split(self, df_reindex):
        """
        Split the dataframe into train and test datasets.
        Return
        ----------
        tuple: A tuple containing two pandas.DataFrame objects
            - the first dataframe contains the training dataset with columns userId, interactions, negatives
            - the second dataframe contains the test dataset with columns userId, itemId, sampled_negatives
        """
        train, test = self.leave_one_out(df_reindex)
        assert train.userID.nunique() == test.userID.nunique()
        # negatives for training dataset
        train_negatives, test_negatives = self.negative_sampling(train, test)
        return train_negatives, test_negatives
    
    def get_train_instance(self, train_negatives):
        """
        Returns a PyTorch Dataloader for the training dataset.
        Parameters
        ----------
        train_negatives: pandas.DataFrame containing the training dataset
        
        Returns
        -------
        torch.utils.data.DataLoader
        """
        users, items, ratings = [], [], []
        for index, row in train_negatives.iterrows():
            for i in range(len(row['interactions'])):
                users.append(int(row.userID))
                items.append(list(row['interactions'])[i])
                ratings.append(float(1))
            negative_samples = list(set(row.negatives))
            sampled_negatives = sample(negative_samples, len(row['interactions']))
            for item in sampled_negatives:
                users.append(int(row.userID))
                items.append(item)
                ratings.append(float(0))
        user_tensor = torch.tensor(users, dtype=torch.long)
        item_tensor = torch.tensor(items, dtype=torch.long)
        rating_tensor = torch.tensor(ratings, dtype=torch.float)
        dataset = TensorDataset(user_tensor, item_tensor, rating_tensor)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def metrics(self, test_negatives):
        """
        Compute metrics on the test dataset.

        Parameters
        ----------
        test_negatives: pandas.DataFrame containing the test dataset
        
        Returns
        -------
        hr_mean: the mean hit rate
        ndcg_mean: the mean normalized discounted cumulative gain

        """
        HR = []
        NDCG = []
        self.model.eval()
        with torch.no_grad():
            for index, row in test_negatives.iterrows():
                user = [row['userID']] * len(row['sampled_negatives'])
                item = row['itemID']
                negatives = list(row['sampled_negatives'])
                user_ = torch.tensor(user, dtype=torch.long)
                item_ = torch.tensor(negatives, dtype=torch.long)
                predictions = self.model(user_, item_)
                _, indices = torch.topk(predictions, self.top_k)
                recommends = torch.take(item_, indices).cpu().numpy().tolist()
                HR.append(int(item in recommends))
                NDCG.append(np.reciprocal(np.log2(recommends.index(item) + 2)) if item in recommends else 0)
        hr_mean = np.mean(HR)
        ndcg_mean = np.mean(NDCG)
        return hr_mean, ndcg_mean
    
    def train_model(self, train_negatives, test_negatives):
        """
        Trains the model that is being attacked.

        Parameters
        ----------
        train_negatives: pandas.DataFrame containing the training dataset
        test_negatives: pandas.DataFrame containing the test dataset
        Returns
        -------
        None

        """
        print(self.num_users, self.num_items)
        self.model = self.get_model(self.num_users + 1, self.num_items + 1)
        train_loader = self.get_train_instance(train_negatives)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(1, self.epochs):
            self.model.train()
            sum_loss = 0
            count = 0
            for user, item, label in train_loader:
                optimizer.zero_grad()
                prediction = self.model(user, item)
                loss = loss_function(prediction, label)
                sum_loss += loss
                count += 1
                loss.backward()
                optimizer.step()
            avg_loss = sum_loss / count
            hr_mean, ndcg_mean = self.metrics(test_negatives)
            print(f"Epoch {epoch},  Loss: {avg_loss}, Test HR: {hr_mean}, Test NDCG: {ndcg_mean}")
    
    def get_predictions_user(self, all_items_mapped, user_id, items_id):
        """
        Gets model prediction for a single user.

        Parameters
        ----------
        all_items_mapped: List of all item ids.
        user_id: An integer representing the user id.
        items_id: List of item ids that the user interacted with.
        
        Returns
        -------
        sorted_predictions_reindex: List of items recommended to the user.
        """
        items_not_rated = [i for i in all_items_mapped if i not in items_id]
        user_list = [user_id] * len(items_not_rated)
        user_ = torch.tensor(user_list, dtype=torch.long)
        item_ = torch.tensor(items_not_rated, dtype=torch.long)
        predictions = self.model(user_, item_)
        _, indices = torch.topk(predictions, self.top_k)
        recommends = torch.take(
            user_, indices).cpu().numpy().tolist()
        return recommends
    
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
        Recommends the most popular items to the user excluding the items that the user
        interacted with.

        Parameters
        ----------
        all_items: pandas.DataFrame containing list of item ids that the user interacted with
        interactions: List of item ids that a user interacted with
        
        Returns
        -------
        top_recommendations: List of top_k most popular recommendations sans the items the user has already interacted
        with.
        """
        all_items_df = all_items.explode('interactions').rename(columns={"interactions": "itemID"})
        item_popularity = all_items_df['itemID'].value_counts().reset_index()
        item_popularity.columns = ['itemID', 'interaction_count']
        recommended_items = item_popularity[~item_popularity['itemID'].isin(interactions)]
        top_recommendations = recommended_items.head(self.top_k)['itemID'].tolist()
        return top_recommendations


class NCFTargetModel(CFModel):
    def __init__(self, top_k, layers, latent_dim, epochs, batch_size, lr):
        self.top_k = top_k
        self.layers = layers
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        super(NCFTargetModel, self).__init__(top_k, batch_size, epochs, lr)
    
    def get_model(self, num_users, num_items):
        return NeuralCollaborativeFiltering(num_users, num_items, self.layers, self.latent_dim)
    
    def get_model_name(self):
        return "NCF"


class GMFTargetModel(CFModel):
    def __init__(self, top_k, latent_dim, epochs, batch_size, lr):
        self.top_k = top_k
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        super(GMFTargetModel, self).__init__(top_k, batch_size, epochs, lr)
    
    def get_model(self, num_users, num_items):
        return GeneralizedMatrixFactorization(num_users, num_items, self.latent_dim)
    
    def get_model_name(self):
        return "GMF"


class MLPTargetModel(CFModel):
    def __init__(self, top_k, layers, epochs, batch_size, lr):
        self.top_k = top_k
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        super(MLPTargetModel, self).__init__(top_k, batch_size, epochs, lr)
    
    def get_model(self, num_users, num_items):
        return MultiLayerPerceptron(num_users, num_items, self.layers)
    
    def get_model_name(self):
        return "MLP"
