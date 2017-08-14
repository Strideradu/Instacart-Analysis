import os
import pandas as pd
import pickle
import numpy as np

data_path = "/mnt/home/dunan/Learn/Kaggle/instacart/data/"
file_list = map(lambda x: os.path.join(data_path, x), ["prediction_lgbm.pkl",
                                                       "prediction_lgbm_64Bed.pkl",
                                                       "prediction_arboretum.pkl"])


file_list = list(file_list)


file1 = pd.read_pickle(file_list[0])
file2 = pd.read_pickle(file_list[1])
file3 = pd.read_pickle(file_list[2])
file_res = file1.copy()

file_res[['prediction']] = file2[["prediction"]] *0.5 + file3[["prediction"]] * 0.5
file_res.to_pickle("data/prediction_arb_64bed_cb_55.pkl")