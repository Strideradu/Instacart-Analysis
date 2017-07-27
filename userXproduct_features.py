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

print('loading products')
products = pd.read_csv(os.path.join(data_dir, 'products.csv'), dtype={
    'product_id': np.uint16,
    'order_id': np.int32,
    'aisle_id': np.uint8,
    'department_id': np.uint8})

########################################################################
### Compute features
########################################################################

print('generating features')

# Add order info to priors
priors = priors.merge(orders, on="order_id", how="left")

priors['user_product_id'] = priors['user_id'].map(str) + "_" + priors['product_id'].map(str)

user_product_group = priors.groupby('user_product_id')
userXproduct = user_product_group.size().astype(np.float32).to_frame()
userXproduct.columns = ['UP_orders']
userXproduct['user_product_id'] = userXproduct.index
userXproduct['user_id'] = userXproduct['user_product_id'].apply(lambda x: int(x.split("_")[0]))
userXproduct['product_id'] = userXproduct['user_product_id'].apply(lambda x: int(x.split("_")[1]))
userXproduct['UP_nb_orders'] = user_product_group.size().astype(np.float32)
userXproduct['UP_reorders'] = priors.groupby('user_product_id')['reordered'].sum()
userXproduct['UP_mean_add_to_cart'] = user_product_group['add_to_cart_order'].mean()
userXproduct['UP_std_add_to_cart'] = user_product_group['add_to_cart_order'].std()
userXproduct['UP_last_add_to_cart'] = user_product_group['add_to_cart_order'].apply(lambda x: x.iloc[-1])  # ?

# userXproduct['product_average_days_between_orders'] = priors.groupby('product_id')['days_since_prior_order'].mean().astype(np.float32)
# userXproduct['product_std_days_between_orders'] = priors.groupby('product_id')['days_since_prior_order'].std().astype(np.float32)

# Compute features dependent on user data
userXproduct['UP_order_numbers'] = user_product_group['order_number'].apply(np.array)

user_features = pd.read_csv(os.path.join(feature_dir, 'user_features.csv'))
userXproduct = userXproduct.merge(user_features[['user_id', 'user_nb_orders']], on="user_id", how="left")
userXproduct['UP_order_rate'] = userXproduct['UP_orders'] / userXproduct['user_nb_orders']

userXproduct['UP_orders_since_last_order'] = userXproduct.apply(
    lambda x: np.min(x['user_nb_orders'] - x['UP_order_numbers']), axis=1)
userXproduct['UP_order_rate_since_first_order'] = userXproduct.apply(
    lambda x: x['UP_orders'] / (x['user_nb_orders'] - np.min(x['UP_order_numbers'])), axis=1)

# Other
userXproduct['user_reorder_probability'] = userXproduct.groupby('user_id')['UP_orders'].transform(
    lambda x: np.sum(x > 1) / x.size)

# aisle and department features
priors = priors.merge(products, on="product_id", how="left")

priors['user_aisle_id'] = priors['user_id'].map(str) + "_" + priors['aisle_id'].map(str)
user_aisle_group = priors.groupby('user_aisle_id')

UA = pd.DataFrame()
UA['UA_reorders'] = user_aisle_group['reordered'].sum()
UA['UA_mean_add_to_cart'] = user_aisle_group['add_to_cart_order'].mean()
UA['UA_std_add_to_cart'] = user_aisle_group['add_to_cart_order'].std()
UA['UA_last_add_to_cart'] = user_aisle_group['add_to_cart_order'].apply(lambda x: x.iloc[-1])
UA['UA_all_orders'] = user_aisle_group['order_id'].apply(set)
UA['UA_nb_orders'] = (UA.UA_all_orders.map(len)).astype(np.float32)
UA['UA_order_numbers'] = user_aisle_group['order_number'].apply(np.array)

UA['user_aisle_id'] = UA.index
UA['user_id'] = UA['user_aisle_id'].apply(lambda x: int(x.split("_")[0]))
UA['aisle_id'] = UA['user_aisle_id'].apply(lambda x: int(x.split("_")[1]))

userXproduct = userXproduct.merge(products[['product_id', 'aisle_id', 'department_id']], on="product_id", how="left")
userXproduct = userXproduct.merge(UA, on=["user_id", "aisle_id"], how="left")

priors['user_department_id'] = priors['user_id'].map(str) + "_" + priors['department_id'].map(str)
user_department_group = priors.groupby('user_department_id')

UD = pd.DataFrame()
UD['UD_reorders'] = user_department_group['reordered'].sum()
UD['UD_mean_add_to_cart'] = user_department_group['add_to_cart_order'].mean()
UD['UD_std_add_to_cart'] = user_department_group['add_to_cart_order'].std()
UD['UD_last_add_to_cart'] = user_department_group['add_to_cart_order'].apply(lambda x: x.iloc[-1])
UD['UD_all_orders'] = user_department_group['order_id'].apply(set)
UD['UD_nb_orders'] = (UD.UD_all_orders.map(len)).astype(np.float32)
UD['UD_order_numbers'] = user_department_group['order_number'].apply(np.array)

UD['user_department_id'] = UD.index
UD['user_id'] = UD['user_department_id'].apply(lambda x: int(x.split("_")[0]))
UD['department_id'] = UD['user_department_id'].apply(lambda x: int(x.split("_")[1]))
userXproduct = userXproduct.merge(UD, on=["user_id", "department_id"], how="left")

# drop duplicate features
userXproduct = userXproduct.drop(
    ['user_nb_orders', 'UP_order_numbers', 'UA_all_orders', 'UD_all_orders', 'user_id', 'product_id', 'aisle_id',
     'department_id', 'user_aisle_id', 'user_department_id', 'UA_all_orders','UD_all_orders'], axis=1)

print('writing features to csv')
userXproduct.to_csv(os.path.join(feature_dir, 'userXproduct_features.csv'), index=False)
