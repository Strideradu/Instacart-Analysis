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
### Define training/test set customers
########################################################################

# Compute train/test customers
train_customers = pd.merge(train,orders,on="order_id",how="left")['user_id'].unique().tolist()
test_customers  = orders['user_id'][~orders['user_id'].isin(train_customers)].unique().tolist()

train_customers, valid_customers = train_test_split(train_customers,test_size=0.1,random_state=42)

customers = {'train_customers':train_customers,
            'valid_customers':valid_customers,
            'test_customers':test_customers}

print("dump with pickle")
pickle.dump(customers, open( os.path.join(feature_dir,"customers.p"), "wb" ) )
