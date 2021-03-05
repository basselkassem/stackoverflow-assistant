#%%
import pandas as pd
import pickle

class IOHandler(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def read_txt(path):
        res = []
        with open(path, 'r') as txt_file:
            res = txt_file.readlines()
        return res

    @staticmethod
    def read_txt_as_df(path):
        return pd.read_csv(path, sep = '\t')
    
    @staticmethod
    def serialize(some_obj, path):
        with open(path, 'wb') as f:
            pickle.dump(some_obj, f)

    @staticmethod
    def deserialize(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
#%%
