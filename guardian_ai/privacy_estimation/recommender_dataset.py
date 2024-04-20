import pandas as pd
from typing import List
from dataset import Dataset
from sklearn.decomposition import NMF
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds



class RecommenderDataset(Dataset):
    def __init__(self, name):
        self.users = None
        self.items = None
        self.ratings = None
        self.target_model_data = None
        self.shadow_model_data = None
        super(RecommenderDataset, self).__init__(name)

    @staticmethod
    def load_csv_or_dat(file, sep='', columns=None, data_encoding=None):
            if file[-4:] == '.csv':
                loaded_df = pd.read_csv(
                    file, sep=",", skiprows=1, header=None, encoding="utf-8"
                )
            elif file[-4:] == '.dat':
                loaded_data = [i.strip().split("::") for i in open(file, 'r', encoding=data_encoding).readlines()]
                loaded_df =  pd.DataFrame(loaded_data, columns=columns)
            else:
                return ValueError("Acceptable file types for users are .csv and .dat")
            return loaded_df

    def load_data(
        self,
        users_file,
        items_file,
        ratings_file,
    ):
       self.items = self.load_csv_or_dat(items_file, data_encoding='latin-1', columns=['MovieID', 'Title', 'Genres'])
       self.items['MovieID'] = self.items['MovieID'].apply(pd.to_numeric)
       self.ratings = self.load_csv_or_dat(ratings_file, columns=['user_id', 'item_id', 'rating', 'timestamp'])
       self.ratings = self.ratings.astype(int)
       self.users = self.load_csv_or_dat(users_file)
       print("User length: " + str(len(pd.unique(self.ratings['user_id']))))

    def perform_matrix_factorization(self, num_components):
        x = self.ratings[['user_id', 'item_id']].values
        min_rating = min(self.ratings['rating'])
        max_rating = max(self.ratings['rating'])
        y = self.ratings["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    
        R_df = self.ratings.pivot(index = 'user_id', columns ='item_id', values = 'rating')
        R_df = R_df.fillna(R_df.mean())

        mtrx = R_df.to_numpy()
        ratings_mean = np.mean(mtrx, axis = 1)
        R_demeaned = mtrx - ratings_mean.reshape(-1, 1)
        user_vector, sigma, item_vector_t = svds(R_demeaned, k = num_components)

        return item_vector_t.transpose()

    
        
        