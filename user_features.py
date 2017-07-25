import numpy as np
import pandas as pd
import os

data_dir = "/mnt/home/dunan/Learn/Kaggle/instacart/data/"
feature_dir = "/mnt/home/dunan/Learn/Kaggle/instacart/features/"

########################################################################
### Load data
########################################################################

print('loading prior')
priors = pd.read_csv(os.path.join(data_dir, 'order_products__prior.csv'), dtype={
    'order_id': np.int32,
    'product_id': np.uint16,
    'add_to_cart_order': np.int16,
    'reordered': np.int8})

print('loading orders')
orders = pd.read_csv(os.path.join(data_dir, 'orders.csv'), dtype={
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

# Add order info to priors
priors = priors.merge(orders, on="order_id", how="left")

usr = pd.DataFrame()
usr['user_average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['user_std_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].std().astype(np.float32)
usr['user_nb_orders'] = orders.groupby('user_id').size().astype(np.float32)

users = pd.DataFrame()
users['user_total_items'] = priors.groupby('user_id').size().astype(np.float32)
users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
users['user_total_distinct_items'] = (users.all_products.map(len)).astype(np.float32)

users = users.join(usr)
del usr
users['user_average_basket'] = (users.user_total_items / users.user_nb_orders).astype(np.float32)
# users['user_std_basket'] = (users.user_total_items / users.user_nb_orders).astype(np.float32)

users['user_id'] = users.index
users = users[['user_id',
               'user_nb_orders',
               'user_average_days_between_orders',
               'user_std_days_between_orders',
               'user_total_items',
               'user_total_distinct_items',
               'user_average_basket']]

print('writing features to csv')
users.to_csv(os.path.join(feature_dir, 'user_features.csv'), index=False)
