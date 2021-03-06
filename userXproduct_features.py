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
userXproduct['UP_mean_add_to_cart'] = user_product_group['add_to_cart_order'].mean().astype(np.float32)
# median add to cart
userXproduct['UP_std_add_to_cart'] = user_product_group['add_to_cart_order'].std().fillna(0.0).astype(np.float32)
# up first order
userXproduct['UP_first_add_to_cart'] = user_product_group['add_to_cart_order'].apply(lambda x: x.iloc[0])
userXproduct['UP_last_add_to_cart'] = user_product_group['add_to_cart_order'].apply(lambda x: x.iloc[-1])  # ?

userXproduct['UP_average_dows'] = user_product_group['order_dow'].mean().astype(np.float32)
userXproduct['UP_std_dows'] = user_product_group['order_dow'].std().fillna(0.0).astype(np.float32)
userXproduct['UP_median_dows'] = user_product_group['order_dow'].median()
# median
userXproduct['UP_average_hour_of_day'] = user_product_group['order_hour_of_day'].mean().astype(np.float32)
userXproduct['UP_std_hour_of_day'] = user_product_group['order_hour_of_day'].std().fillna(0.0).astype(np.float32)
userXproduct['UP_median_hour_of_day'] = user_product_group['order_hour_of_day'].median()
# median
# products how many user odered
# product how many use reordered


order_size  = pd.DataFrame()
order_size['order_size'] = priors.groupby('order_id').size()
order_size['order_id'] = order_size.index
priors = priors.merge(order_size, on="order_id", how="left")
userXproduct['average_order_size'] = priors.groupby('user_product_id')['order_size'].mean().astype(np.float32)
userXproduct['UP_revert_add_to_cart'] = userXproduct.average_order_size - userXproduct.UP_mean_add_to_cart
userXproduct['relative_add_to_cart'] = userXproduct.UP_mean_add_to_cart / userXproduct.average_order_size
# userXproduct['days_since_prior_order'] = priors.order_id.map(orders.days_since_prior_order)




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
"""
userXproduct['UP_delta_hour_vs_last'] = abs(
    userXproduct.order_hour_of_day - userXproduct.UP_last_order_id.map(orders.order_hour_of_day)).map(
    lambda x: min(x, 24 - x)).astype(np.int8)
"""

d = dict()
for row in priors.itertuples():
    z = row.user_product_id
    if z not in d:
        d[z] = (
            (row.order_number, row.order_id),
            row.add_to_cart_order)
    else:
        d[z] = (
            max(d[z][0], (row.order_number, row.order_id)),
            d[z][1] + row.add_to_cart_order)

print('to dataframe (less memory)')
d = pd.DataFrame.from_dict(d, orient='index')
d.columns = ['last_order_id', 'sum_pos_in_cart']
d.last_order_id = d.last_order_id.map(lambda x: x[1]).astype(np.int32)
d.sum_pos_in_cart = d.sum_pos_in_cart.astype(np.int16)
d['user_product_id'] = d.index

userXproduct["UP_last_order_id"] = userXproduct.user_product_id.map(d.last_order_id)
userXproduct["UP_sum_pos_in_cart"] = userXproduct.user_product_id.map(d.sum_pos_in_cart)

del d

userXproduct['UP_average_pos_in_cart'] = (userXproduct["UP_sum_pos_in_cart"] / userXproduct['UP_nb_orders']).astype(
    np.float32)

userXproduct = userXproduct.merge(
    orders[orders['eval_set'] != 'prior'][
        ['order_id', 'user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']],
    on='user_id', how="left")

usr = pd.DataFrame()
usr['user_average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)

usr['user_id'] = usr.index
userXproduct['user_average_days_between_orders'] = userXproduct.user_id.map(usr.user_average_days_between_orders)
userXproduct['days_since_ratio'] = userXproduct.days_since_prior_order / userXproduct.user_average_days_between_orders
del usr
# Other
userXproduct['user_reorder_probability'] = userXproduct.groupby('user_id')['UP_orders'].transform(
    lambda x: np.sum(x > 1) / float(x.size))
# dep reorder ratio
# aisle reordr ratio

userXproduct['order_hour_of_day'] = userXproduct.order_id.map(orders.order_hour_of_day)
orders['cum_days'] = orders.groupby('user_id')['days_since_prior_order'].apply(lambda x: x.cumsum()).fillna(0)
userXproduct['UP_days_since_last_order'] = abs(
    userXproduct.order_id.map(orders.cum_days) - userXproduct.UP_last_order_id.map(orders.cum_days)
)



# aisle and department features
priors = priors.merge(products, on="product_id", how="left")

priors['user_aisle_id'] = priors['user_id'].map(str) + "_" + priors['aisle_id'].map(str)
user_aisle_group = priors.groupby('user_aisle_id')

UA = pd.DataFrame()
UA['UA_reorders'] = user_aisle_group['reordered'].sum()
UA['UA_mean_add_to_cart'] = user_aisle_group['add_to_cart_order'].mean()
UA['UA_std_add_to_cart'] = user_aisle_group['add_to_cart_order'].std().fillna(0.0)
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
UD['UD_std_add_to_cart'] = user_department_group['add_to_cart_order'].std().fillna(0.0)
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
     'department_id', 'user_aisle_id', 'user_department_id', 'UA_order_numbers', 'UD_order_numbers',
     'user_average_days_between_orders', 'order_id', 'UP_last_order_id', 'order_dow', 'order_hour_of_day',
     'days_since_prior_order'], axis=1)

print('writing features to csv')
userXproduct.to_csv(os.path.join(feature_dir, 'userXproduct_features_test.csv'), index=False)
