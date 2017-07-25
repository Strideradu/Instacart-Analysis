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

prods = pd.DataFrame()
prods['product_orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
prods['product_reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.int32)
prods['product_reorder_rate'] = (prods.product_reorders / prods.product_orders).astype(np.float32)

priors = priors.merge(products[['product_id','aisle_id','department_id']],on="product_id",how="left")

aisle = pd.DataFrame()
aisle['aisle_orders'] = priors.groupby(priors.aisle_id).size().astype(np.int32)
aisle['aisle_reorders'] = priors['reordered'].groupby(priors.aisle_id).sum().astype(np.int32)
aisle['aisle_reorder_rate'] = (aisle.aisle_reorders / aisle.aisle_orders).astype(np.float32)

department = pd.DataFrame()
department['department_orders'] = priors.groupby(priors.department_id).size().astype(np.int32)
department['department_reorders'] = priors['reordered'].groupby(priors.department_id).sum().astype(np.int32)
department['department_reorder_rate'] = (department.department_reorders / department.department_orders).astype(np.float32)

products = products.join(prods, on='product_id')
products = products.join(aisle, on='aisle_id')
products = products.join(department, on='department_id')
products = products[['product_id',
                    'product_orders',
                    'product_reorders',
                    'product_reorder_rate',
                    'aisle_orders',
                    'aisle_reorders',
                    'aisle_reorder_rate',
                    'department_orders',
                    'department_reorders',
                    'department_reorder_rate']]

print('writing features to csv')
products.to_csv(os.path.join(feature_dir,'product_features.csv'), index=False)
