import pandas as pd

# Implementation of matrix factorization for recommender inference attack featurization.
class RecommenderDataSplit(Enum):
    ATTACK_TRAIN_IN = 0
    ATTACK_TRAIN_OUT = 1
    ATTACK_TEST_IN = 2
    ATTACK_TEST_OUT = 3
    TARGET_TRAIN = 4
    TARGET_VALID = 5
    TARGET_TEST = 6
    SHADOW_TRAIN = 7
    SHADOW_VALID = 8
    SHDAOW_TEST = 9

class RecommenderDataset(Dataset):
    def __init__(self, name, df_x=None, df_y=None):
        self.df_x = df_x
        self.df_y = df_y
        self.column_transformer = None
        self.label_encoder = None
        self.target_model_data = None
        self.attack_model_data = None
        super(RecommenderDataset, self).__init__(name)
        print(df_x)
    
    def load_data(
        self,
        source_file,
        contains_header: bool = False,
        target_ix: int = None,
        ignore_ix: List[int] = None,
    ):
  
        df = pd.read_csv(
            source_file, sep=",", skiprows=1, header=None, encoding="utf-8"
        )
        y_ix = target_ix if target_ix is not None else len(df.columns) - 1
        self.df_y = df.iloc[:, y_ix]
        if isinstance(self.df_y[0], bytes):
            self.df_y = self.df_y.str.decode("utf-8")
        self.df_x = df.drop(df.columns[y_ix], axis=1)

        # next remove the ones that need to be ignored.
        if ignore_ix is not None:
            self.df_x = self.df_x.drop(ignore_ix, axis=1)
        print(df)
        
        def matrix_factorization():
            return NotImplementedError

        def get_target_features():
            return NotImplementedError
        
        def get_shadow_features():
            return NotImplementedError
        