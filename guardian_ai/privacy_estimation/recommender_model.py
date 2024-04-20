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
        Retrieves an instance of the model configured for a specified number of users and items.

        Parameters
        ----------
        num_users : int
            The number of users.
        num_items : int
            The number of items.

        Returns
        -------
        object
            An untrained model instance configured for the given numbers of users and items.
        """
        pass
    
    def create_index_map(self, X, y):
        """
        Generates mappings from original indices to continuous indices for users and items.

        Parameters
        ----------
        X : pandas.DataFrame
            A dataframe with a single column `userID` representing user identifiers.
        y : pandas.DataFrame
            A dataframe with a single column `interactions` that contains lists of
            item identifiers each user interacted with.

         Returns
        -------
        user_map : dict
            A dictionary that maps original user IDs to continuous, zero-indexed user IDs.
        item_map : dict
            A dictionary that maps original item IDs to continuous, zero-indexed item IDs.

        Notes
        -----
        - The function assumes that `X` and `y` are aligned such that the i-th row in each corresponds to the same user.
        - This function also updates the instance attributes `num_users` and `num_items` based on the unique counts of
          users and items, respectively.
        """
        user_ids = sorted(X.userID.unique())
        exploded_df = y.explode('interactions').rename(columns={"interactions": "itemID"})
        item_ids = sorted(exploded_df['itemID'].unique())
        user_map = [{original_id: index for index, original_id in enumerate(user_ids)},
                    {index: original_id for index, original_id in enumerate(user_ids)}]
        item_map = [{original_id: index for index, original_id in enumerate(item_ids)},
                    {index: original_id for index, original_id in enumerate(item_ids)}]
        self.num_users = X.userID.nunique()
        self.num_items = exploded_df['itemID'].nunique()
        return user_map, item_map
    
    def reindex(self, X, y):
        """
        Transforms the dataset by mapping original user and item identifiers to continuous indices.

        This function applies the mappings created by `create_index_map` to convert the original
        user and item identifiers in the input dataframes `X` and `y` into continuous, zero-indexed identifiers.

        Parameters
        ----------
        X : pandas.DataFrame
            A dataframe with a single column `userID` representing user identifiers.
        y : pandas.DataFrame
            A dataframe with a single column `interactions` that contains lists of item
            identifiers each user interacted with.

        Returns
        -------
        df_reindex : pandas.DataFrame
            A dataframe with two columns, `userID` and `itemID`, containing the continuous
            indices for users and items, respectively. Each row represents an interaction
            between a user and an item.

        Notes
        -----
        The resulting `df_reindex` dataframe will have a row for each user-item interaction,
        effectively 'exploding' lists found in `y`.
        """
        user_map, item_map = self.create_index_map(X, y)
        temp_df = pd.concat([X, y], axis=1)
        df_reindex = temp_df.explode('interactions').rename(columns={"interactions": "itemID"})
        df_reindex.reset_index(drop=True, inplace=True)
        df_reindex['userID'] = df_reindex['userID'].apply(lambda x: user_map[0][x])
        df_reindex['itemID'] = df_reindex['itemID'].apply(lambda x: item_map[0][x])
        return df_reindex
    
    @staticmethod
    def leave_one_out(dataframe):
        """
        Splits the input dataframe into training and testing datasets using the leave-one-out evaluation protocol.

        For each user, all but one of the interactions are included in the training dataset, and the remaining
        interaction is used for testing. This method ensures that each user is represented in both the training
        and testing datasets.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            A dataframe containing at least two columns: `userID` for user
            identifiers and `itemID` for item identifiers.

        Returns
        -------
        train_df : pandas.DataFrame
            The training dataset, containing all but one interaction per user.
        test_df : pandas.DataFrame
            The testing dataset, containing exactly one interaction per user.
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
        Enhances the training and test datasets with negative sampling.

        For the training dataset, this involves adding a list of items each user has not interacted with.
        For the test dataset, the non-interacted items list also includes the item withheld during the leave-one-out
        cross-validation (LOOCV) process, ensuring it is considered as a potential negative example.

        Parameters
        ----------
        train : pandas.DataFrame
            The training dataset containing columns 'userID' and 'itemID' for interactions.
        test : pandas.DataFrame
            The test dataset containing columns 'userID' and 'itemID', where 'itemID' is the item withheld during LOOCV.

        Returns
        -------
        tuple
            A tuple containing two pandas.DataFrame objects:
            - The first DataFrame includes the training dataset augmented with a 'negatives' column, listing items
            not interacted with by each user.
            - The second DataFrame augments the test dataset with a 'sampled_negatives' column, which includes a
            sample of non-interacted items for each user, ensuring the withheld item is also considered.
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
        Splits the reindexed dataframe into training and testing datasets using a leave-one-out approach.
        The negative sampling process adds non-interacted items to both datasets for model training and
        evaluation purposes.

        This method first applies the leave-one-out strategy to ensure each user is represented in both the
        training and testing datasets. It then performs negative sampling to identify items not interacted with
        by users. The process ensures the training dataset includes user-item interactions and their corresponding
        negative samples, while the test dataset includes a single withheld interaction per user and a sampled
        set of negatives that includes the withheld item.

        Parameters
        ----------
        df_reindex : pandas.DataFrame
            The preprocessed dataframe with continuous user and item IDs, ready for splitting and negative sampling.

        Returns
        -------
        tuple
            A tuple containing two pandas.DataFrame objects:
            - The first DataFrame, representing the training dataset, includes columns 'userID', 'interactions'
              (positive samples), and 'negatives' (negative samples).
            - The second DataFrame, representing the test dataset, includes columns 'userID', 'itemID'
              (the single withheld positive sample per user), and 'sampled_negatives' (a sampled set of negative items
              that includes the withheld positive item, simulating a recommendation scenario).
        """
        train, test = self.leave_one_out(df_reindex)
        assert train.userID.nunique() == test.userID.nunique()
        # negatives for training dataset
        train_negatives, test_negatives = self.negative_sampling(train, test)
        return train_negatives, test_negatives
    
    def get_train_instance(self, train_negatives):
        """
        Prepares and returns a DataLoader for the training dataset suitable for PyTorch models.

        This method processes the input DataFrame to structure the training data in a format compatible
        with PyTorch, facilitating batch processing during model training. It involves converting the
        training dataset, which includes user-item interactions and their corresponding negative samples,
        into a DataLoader object that efficiently handles data loading and batching operations.

        Parameters
        ----------
        train_negatives : pandas.DataFrame
            A DataFrame containing the training dataset. This dataset should include columns for user IDs,
            item IDs for positive interactions, and a column for negative sample IDs (items not interacted with).

        Returns
        -------
        torch.utils.data.DataLoader
            A DataLoader instance that provides iterable access to the dataset, formatted for training a PyTorch model.
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
        Calculates evaluation metrics for the recommendation model on a test dataset. Specifically, it computes
        the mean hit rate (HR) and the mean normalized discounted cumulative gain (NDCG) to assess the model's
        performance. The HR measures the proportion of times the true item (the one user interacted with) is
        among the top-k recommended items. The NDCG accounts for the position of the true item in the ranked list
        of recommendations, providing higher scores for hits at higher ranks.

        Each row in the test_negatives DataFrame should include a userID, the itemID of the withheld item, and
        a list of sampled_negatives, which includes the withheld item along with a sample of items not interacted with
        by the user. The method iterates over each user in the test dataset, predicts the scores for the sampled negative
        items (including the withheld item), and calculates the HR and NDCG based on these predictions.

        Parameters
        ----------
        test_negatives : pandas.DataFrame
            The test dataset containing columns 'userID', 'itemID', and 'sampled_negatives'. The 'sampled_negatives'
            column should include a list of item IDs representing the negative samples and the withheld positive sample.

        Returns
        -------
        hr_mean : float
            The mean hit rate across all users in the test dataset, indicating the fraction of times the withheld item
            was correctly recommended within the top-k items.
        ndcg_mean : float
            The mean normalized discounted cumulative gain across all users, measuring the model's ability to rank
            the withheld item highly among the recommended items.

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
        Trains the recommendation model using the provided training dataset and evaluates its performance
        on the test dataset. After each epoch, the model's performance is evaluated using the test dataset
        to calculate metrics such as hit rate (HR) and normalized discounted cumulative gain (NDCG).

        Parameters
        ----------
        train_negatives : pandas.DataFrame
            A DataFrame containing the training dataset, with user-item interactions and negative samples.
        test_negatives : pandas.DataFrame
            A DataFrame containing the test dataset, with user-item interactions and a set of sampled negatives
            for each user, used for model evaluation.

        Returns
        -------
        None
            This method does not return any value but outputs the training progress and evaluation metrics
            to the console.

        Notes
        -----
        - The training and test DataFrames must include columns for 'userID', 'itemID', and 'label' for the
          training set, where 'label' indicates whether the interaction is positive or negative.
        - The 'test_negatives' DataFrame should include a 'sampled_negatives' column that lists the negative
          item IDs considered for each user in the test set, alongside the actual item ID the user interacted with.
        """
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
        Generates a list of recommended items for a single user based on model predictions. This function
        first identifies items not yet rated by the user, then predicts the user's rating for these items using
        the trained model. It returns the top-k items with the highest predicted ratings, where k is defined
        externally (e.g., as a property of the class).

        The function is designed for use with models that generate predictions for individual user-item pairs.
        The output is a list of item IDs recommended for the user, sorted by the model's prediction scores
        in descending order.

        Parameters
        ----------
        all_items_mapped : List[int]
            A list of all item IDs in the dataset, representing the entire set of items that can be recommended.
        user_id : int
            The ID of the user for whom recommendations are to be generated.
        items_id : List[int]
            A list of item IDs that the user has already interacted with.

        Returns
        -------
        sorted_predictions_reindex : List[int]
            A list of item IDs recommended to the user, sorted by the model's prediction scores in descending order.
            The length of this list is determined by the model's top_k attribute.

        Notes
        -----
        - This method assumes that `all_items_mapped` includes all possible item IDs that could be recommended.
        - `items_id` should include only those item IDs that the user has already interacted with, to exclude them
          from the recommendations.
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
        Identifies and returns the most popular items across all users, excluding those already interacted with
        by a specific user. Popularity is determined by the number of interactions an item has received. This method
        can be used to generate baseline recommendations based on item popularity, ensuring that recommended items
        are not among those the user has previously interacted with.

        Parameters
        ----------
        all_items : pandas.DataFrame
            A DataFrame containing all item interactions in the dataset. It should have at least one column that
            lists item IDs, which this method will explode to count interactions per item. This column must be
            named 'interactions' before calling this method.
        interactions : List[int]
            A list of item IDs that the specific user has interacted with. These items will be excluded from the
            recommendations.

        Returns
        -------
        top_recommendations : List[int]
            A list of the top_k most popular item IDs recommended to the user, excluding those the user has
            already interacted with. The number of recommendations returned is determined by the class attribute
            `top_k`.

        Notes
        -----
        - The `all_items` DataFrame is expected to potentially contain multiple rows per item, where each row
          represents an interaction with the item. The method calculates item popularity based on the count
          of these interactions.
        - The `interactions` parameter should accurately reflect the user's history to ensure meaningful
          recommendations are generated.
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
