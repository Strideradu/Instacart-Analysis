import numpy as np
import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split

data_dir = "/mnt/home/dunan/Learn/Kaggle/intracart/data/"
feature_dir = "/mnt/home/dunan/Learn/Kaggle/intracart/features/"

########################################################################
### Load data
########################################################################

print('loading prior')
priors = pd.read_csv(os.path.join(data_dir,'order_products__prior.csv'), dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading train')
train = pd.read_csv(os.path.join(data_dir,'order_products__train.csv'), dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders')
orders = pd.read_csv(os.path.join(data_dir,'orders.csv'), dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

########################################################################
### Combinations of user/produyct
########################################################################

priors = priors.merge(orders[['order_id','user_id']],on="order_id",how="left")
priors['user_product_id'] = priors['user_id'].map(str) + "_" + priors['product_id'].map(str)

userXproduct = pd.DataFrame({'user_product_id':priors['user_product_id'].unique()})
userXproduct['user_id'] = userXproduct['user_product_id'].apply(lambda x: int(x.split("_")[0]))
userXproduct['product_id'] = userXproduct['user_product_id'].apply(lambda x: int(x.split("_")[1]))

idx = orders.groupby("user_id")['order_number'].transform(max) == orders['order_number']
last_orders = orders[idx]
userXproduct = userXproduct.merge(last_orders[['user_id','order_id']],on="user_id",how="left")

train = train.merge(orders[['order_id','user_id']],on="order_id",how="left")
last_mb = train.groupby("user_id")['product_id'].apply(set).to_dict()

def product_in_last_order(x):
    try:
        return int(x['product_id'] in last_mb[x['user_id']])
    except:
        return np.nan

userXproduct['labels'] = userXproduct.apply(product_in_last_order,axis=1)
print(userXproduct)
userXproduct.to_csv(os.path.join(feature_dir,'userXproduct.csv'), index=False)
