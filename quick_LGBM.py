# %%
import sys
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import japanize_matplotlib
from pprint import pprint as print
import shap
shap.initjs()

plt.rcParams['figure.figsize'] = 12, 3
sns.set_style('whitegrid')


def min_LGBM(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # X_train,X_test=feat(X_train),feat(X_test)
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_boost_round': 100,

        'num_leaves': None,
        'max_depth': 3,
        'learning_rate': 0.5,
        'min_gain_to_split': 0.5,
        # 'feature_fraction': 1.0,
        # 'bagging_fraction': 1.0,
        # 'bagging_freq': 0,
        'seed': seed,
        'verbosity': -1,
        'n_jobs': -1,
        # 'class_weight':'balanced',
    }

    kf = KFold(n_splits=2, shuffle=True, random_state=0)
    ms = []
    vs = []
    for i, (ind_train, ind_test) in tqdm(enumerate(kf.split(y_train))):
        print(f"Fold {i}:")
        m = train_LGBM(X_train.iloc[ind_train], y_train.iloc[ind_train],
                       X_train.iloc[ind_test], y_train.iloc[ind_test],
                       params=param, feval=None)
        m.save_model('model_lgbm'+str(i))
        ms.append(m)
        vs.append(m.predict(X_test))
        # lgb.plot_importance(m,importance_type='gain',max_num_features=10)

    v = pd.DataFrame(vs).mean()
    sns.jointplot(x=v, y=y_test, height=4)
    plt.figure()
    pd.DataFrame(
        [m.feature_importance(importance_type='gain') for m in ms], columns=m.feature_name()
    ).mean().sort_values().tail(10).plot.barh()

    plt.figure()
    explainer = shap.TreeExplainer(m)
    # explainer = shap.Explainer(m)
    shap_values = explainer.shap_values(X_train.iloc[ind_train])
    shap.summary_plot(
        shap_values,
        X_train.iloc[ind_train])
    return ms


def train_LGBM(features, target, eval_f=None, eval_t=None, params={},
               feval=None):

    # categorical_features=features.columns.to_list()
    categorical_features = None
    train_lgb = lgb.Dataset(features, target,
                            categorical_feature=categorical_features)
    eval_lgb = lgb.Dataset(eval_f, eval_t, reference=train_lgb,
                           categorical_feature=categorical_features)
    evals_result = {}

    model = lgb.train(
        params=params,
        train_set=train_lgb,
        valid_sets=[train_lgb, eval_lgb],
        feval=feval,
        callbacks=[
            lgb.record_evaluation(evals_result),
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(50),
        ],
        # verbosity=-1,
        # time_budget=1000,
    )
    print(model.params)
    lgb.plot_metric(evals_result)
    return model


if __name__ == '__main__':
    if 'ipykernel_launcher.py' in sys.argv[0]:
        filepath = '../data/ETFs.csv'
        target_col = 'VTI'
    else:
        filepath = sys.argv[1]
        try:
            target_col = sys.argv[2]
        except IndexError:
            target_col = None

df = pd.read_csv(filepath,
                 # index_col=0,
                 parse_dates=True).select_dtypes(exclude='object')
if target_col in df.columns:
    pass
else:
    target_col = df.columns[-1]

y = df[target_col]
X = df.drop(target_col, axis=1)
ms = min_LGBM(X, y)
plt.show()
# %%
