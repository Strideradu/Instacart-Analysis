import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns

class XGBoostWrapper(object):

    def __init__(self,params):
        self.params = params

    def train(self,X_train,Y_train,X_valid,Y_valid,epochs,categorical=[],early_stopping_rounds=100,verbose_eval=10):

        #self.categorical = categorical

        #if len(self.categorical)>0:
        #    self.categorical_max_labels = []

        #    cat_cols = np.zeros((len(X_train),0),dtype=np.float32)
        #    for cat in categorical:



        d_train = xgb.DMatrix(X_train, label=Y_train)
        d_valid = xgb.DMatrix(X_valid, label=Y_valid)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        self.model = xgb.train(self.params,
                                d_train,
                                epochs,
                                watchlist,
                                early_stopping_rounds=early_stopping_rounds,
                                verbose_eval=verbose_eval)

    def predict(self,test):
        d_test = xgb.DMatrix(test)
        return self.model.predict(d_test)

    def plot_importance(self):
        plot_importance(self.model)
        plt.savefig('/mnt/home/dunan/Learn/Kaggle/instacart/feature_importance.pdf', bbox_inches='tight')
