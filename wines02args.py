import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--quantile', default=99)
parser.add_argument('--scaler_type', default='minmax')
parser.add_argument('--k', default=5)
parser.add_argument('--max_depth', default=6)
parser.add_argument('--min_samples_leaf', default=15)

args = parser.parse_args()


### Get the data

data_path = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(data_path, sep=";")

X = data.drop(columns=['quality'])
y = data.quality

### Preprocessing

# Removing outliers
quantile = int(args.quantile)
for col in X.columns.values:
    min_ = X[col].quantile(1-quantile/100)
    max_ = X[col].quantile(quantile/100)
    X = X[X[col].between(min_, max_)]
y = y.loc[X.index]

# Scaling
scaler_type = args.scaler_type
scalers = {'minmax': MinMaxScaler, 'maxabs': MaxAbsScaler, 'standard': StandardScaler}
scaler = scalers[scaler_type]().fit(X)
X = pd.DataFrame(scaler.transform(X),
                 columns=X.columns,
                 index=X.index)

# Feature selection
k = int(args.k)
skb = SelectKBest(k=k).fit(X, y)
selected_features = X.columns[skb.get_support()]  # Artifact
X = X[selected_features]


### Modelling

# Modelling
max_depth = int(args.max_depth)
min_samples_leaf = int(args.min_samples_leaf)
dt_model = DecisionTreeRegressor(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf)
dt_model.fit(X, y)  # Model
importances = dt_model.feature_importances_  # Artifact

# Scoring
scores = -cross_val_score(dt_model, X, y, cv=3, scoring='neg_mean_squared_error')
RMSE = np.mean(scores)  # Metric

print(RMSE)


