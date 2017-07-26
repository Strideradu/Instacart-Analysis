import numpy as np
import pandas as pd
import pickle
import os

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from xgboostwrapper import XGBoostWrapper

feature_dir = "/mnt/home/dunan/Learn/Kaggle/instacart/features/"

def build_features(data,features,feature_dir="/mnt/home/dunan/Learn/Kaggle/instacart/features/"):
    for feature in features:
        path = os.path.join(feature_dir,feature[0])
        merge_on = feature[1]

        print(path,merge_on)

        data = data.merge(pd.read_csv(path),on=merge_on,how="left")

    try:
        labels = data['labels'].values
        return data.drop(['labels'],axis=1),labels
    except:
        return data, []

########################################################################
### Define training/test set and labels
########################################################################

n_train_customers = 10000

features = [('product_features.csv','product_id'),
            ('user_features.csv','user_id'),
            ('userXproduct_features.csv','user_product_id'),
            ('order_features.csv','order_id')]

customers = pickle.load(open( os.path.join(feature_dir,"customers.p"), "rb" ) )
train_customers = customers['train_customers'][:]
valid_customers = customers['valid_customers']
test_customers  = customers['test_customers']

userXproduct = pd.read_csv(os.path.join(feature_dir,"userXproduct.csv"))

train = userXproduct[userXproduct['user_id'].isin(train_customers)]
valid = userXproduct[userXproduct['user_id'].isin(valid_customers)]
test  = userXproduct[userXproduct['user_id'].isin(test_customers)]
test = test.drop(['labels'],axis=1)


########################################################################
### XGBoost
########################################################################

params = {}
params['eta'] = 0.02
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['max_depth'] = 7
params['subsample'] = 0.8

# Init and train
clf = XGBoostWrapper(params)

X_train,Y_train = build_features(train,features)
X_valid,Y_valid = build_features(valid,features)

f_to_use = X_train.columns
f_to_use = [e for e in f_to_use if e not in ('user_product_id','user_id','product_id','order_id')]

print("Number of features",len(f_to_use))
print(f_to_use)

clf.train(X_train[f_to_use],
        Y_train,
        X_valid[f_to_use],
        Y_valid,
        10000)

########################################################################
### Optimize F1 score on validation set
########################################################################

# Predict
preds_valid = clf.predict(X_valid[f_to_use])
X_valid['preds'] = preds_valid

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

test,_ = build_features(test,features)

preds = clf.predict(test[f_to_use])
test['preds'] = preds

d = dict()
for row in test.itertuples():
    if row.preds > threshold:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in test.order_id:
    if order not in d:
        d[order] = 'None'

sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv('/mnt/home/dunan/Learn/Kaggle/instacart/submission_full_add_more_features_data.csv', index=False)

clf.plot_importance()
