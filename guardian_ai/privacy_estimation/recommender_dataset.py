import pandas as pd
pd.options.mode.chained_assignment = None
from typing import List
from dataset import Dataset
from sklearn.decomposition import NMF
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.model_selection import GroupShuffleSplit


class RecommenderDataset(Dataset):
    def __init__(self, name):
        self.users = None
        self.items = None
        self.ratings = None
        self.target_model_data = None
        self.shadow_model_data = None
        self.attack_model_data = None
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
        user_columns=None, 
        ratings_columns=None,
        items_columns=None
    ):
       self.items = self.load_csv_or_dat(items_file, data_encoding='latin-1', columns=items_columns)
       self.items['MovieID'] = self.items['MovieID'].apply(pd.to_numeric)
       self.ratings = self.load_csv_or_dat(ratings_file, columns=ratings_columns)
       self.ratings = self.ratings.astype(int)
       self.users = self.load_csv_or_dat(users_file)

    @staticmethod
    def split_helper(data, train_size=0.8, group_id='user_id'):
        gss = GroupShuffleSplit(n_splits=2, train_size=train_size, random_state=42)
        split = (gss.split(data, None, groups=data[group_id]))
        group1_inds, group2_inds = next(split)
        group1 = data.iloc[group1_inds]
        group2 = data.iloc[group2_inds]
        return group1, group2

    def split_dataset(self):
        members, non_members = self.split_helper(self.ratings)
        members['label'] = 0
        non_members['label'] = 1

        target_members, shadow_members = self.split_helper(members, train_size=0.5)
        target_non_members, shadow_non_members = self.split_helper(non_members, train_size=0.5)

        self.target_model_data = pd.concat([target_members, target_non_members])
        self.shadow_model_data = pd.concat([shadow_members, shadow_non_members])
        
        print(self.target_model_data)
        print(self.shadow_model_data)

    
    
    def perform_matrix_factorization(self, num_components):
        min_rating = min(self.ratings['rating'])
        max_rating = max(self.ratings['rating'])
    
        R_df = self.ratings.pivot(index = 'user_id', columns ='item_id', values = 'rating')
        R_df = R_df.fillna(R_df.mean())

        mtrx = R_df.to_numpy()
        ratings_mean = np.mean(mtrx, axis = 1)
        R_demeaned = mtrx - ratings_mean.reshape(-1, 1)
        user_vector, sigma, item_vector_t = svds(R_demeaned, k = num_components)

        return item_vector_t.transpose()

    
        
        