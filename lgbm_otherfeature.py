import numpy as np
import pandas as pd
import pickle
import os
import gc

import lightgbm as lgb
import json
import sklearn.metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.sparse import dok_matrix, coo_matrix
from sklearn.utils.multiclass import  type_of_target

feature_dir = "/mnt/home/dunan/Learn/Kaggle/instacart/features/"


def build_features(data, features, feature_dir="/mnt/home/dunan/Learn/Kaggle/instacart/features/"):
    for feature in features:
        path = os.path.join(feature_dir, feature[0])
        merge_on = feature[1]

        print(path, merge_on)

        data = data.merge(pd.read_csv(path), on=merge_on, how="left")

    try:
        labels = data['labels'].values
        return data.drop(['labels'], axis=1), labels
    except:
        return data, []


########################################################################
### Define training/test set and labels
########################################################################

n_train_customers = 10000

features = [('product_features.csv', 'product_id'),
            ('user_features.csv', 'user_id'),
            ('userXproduct_features.csv', 'user_product_id'),
            ('order_features.csv', 'order_id')]

customers = pickle.load(open(os.path.join(feature_dir, "customers.p"), "rb"))
train_customers = customers['train_customers'][:]
valid_customers = customers['valid_customers']
test_customers = customers['test_customers']

userXproduct = pd.read_csv(os.path.join(feature_dir, "userXproduct.csv"))

train = userXproduct[userXproduct['user_id'].isin(train_customers)]
valid = userXproduct[userXproduct['user_id'].isin(valid_customers)]
test = userXproduct[userXproduct['user_id'].isin(test_customers)]
test = test.drop(['labels'], axis=1)

X_train, Y_train = build_features(train, features)
X_valid, Y_valid = build_features(valid, features)

########################################################################
### lightgbm
########################################################################

categories = ['product_id']

f_to_use = X_train.columns
f_to_use = [e for e in f_to_use if e not in ('user_product_id','user_id','order_id')]

lgb_train = lgb.Dataset(X_train[f_to_use], Y_train, categorical_feature=categories)
lgb_valid = lgb.Dataset(X_valid[f_to_use], Y_valid, categorical_feature=categories)


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 256,
    'min_sum_hessian_in_leaf': 20,
    'max_depth': 12,
    'learning_rate': 0.05,
    'feature_fraction': 0.6,
    # 'bagging_fraction': 0.9,
    # 'bagging_freq': 3,
    'verbose': 1
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_valid,
early_stopping_rounds = 100,
                num_boost_round=1500)

prediction = gbm.predict(X_valid[f_to_use])
X_valid['preds'] = prediction

# Set threshold to optimize F1 score
thresholds = np.linspace(0,0.5,50)

best_score = 0.
for t in thresholds:
    X_valid['preds_binary'] = (X_valid['preds']>t).map(int)
    score = f1_score(Y_valid,X_valid['preds_binary'])
    if score>best_score:
        best_score = score
        threshold = t

print("F1 score",best_score,"Threshold:",threshold)

########################################################################
### Predict and submit test set
########################################################################

X_test,_ = build_features(test,features)
prediction = gbm.predict(X_test[f_to_use])

orders = X_test.order_id.values
products = X_test.product_id.values

result = pd.DataFrame({'product_id': products, 'order_id': orders, 'prediction': prediction})
result.to_pickle('/mnt/home/dunan/Learn/Kaggle/instacart/data/prediction_lgbm_other_feature_set.pkl')
