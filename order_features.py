import numpy as np
import pandas as pd
import os

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

print('loading products')
products = pd.read_csv(os.path.join(data_dir,'products.csv'), dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8})

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
### Compute features
########################################################################

print('generating features')

orders = orders[['order_id',
                'order_number',
                'order_dow',
                'order_hour_of_day',
                'days_since_prior_order']]

print('writing features to csv')
orders.to_csv(os.path.join(feature_dir,'order_features.csv'), index=False)
