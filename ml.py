import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv(r'C:\Users\sadan\OneDrive\Desktop\chatbot\AQI.csv')
print("info: \n")
print(df.info())

print("Head: \n")
print(df.sample(10))

print("Null: \n")
print(df.isnull().sum())

print("Null percentage: \n")
print((df.isnull().sum()*100)/len(df))

print("Duplicates: \n")
print(df.duplicated().sum())

print("Describe: \n")
print(df.describe())

print("Duplicates removed")
df.dropna(subset=["AQI"],inplace=True)

df.drop(['AQI_Bucket','Date'], axis=1, inplace=True)

print("Shape:\n")
print(df.shape)

print("Duplicates: \n")
print(df.isnull().sum())
'''
for i in df.select_dtypes('number'):
    sns.boxplot(df[i])
    plt.title (f"{i} dist")
    plt.show()

for i in df.select_dtypes('number'):
    df[i]=df[i].fillna(df[i].median())


print(df.isnull().sum())

sns.heatmap(df.corr(),annot = True)
plt.show()
'''

x = df[['PM2.5', 'PM10', 'NO2','CO','NOx','SO2','City']]
y = df['AQI']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_train.info())

transform = ColumnTransformer(
    transformers = [
        ("scaler", RobustScaler(), [0,1,2,3,4,5]),
        ("encoder", OrdinalEncoder(), [6])
    ],
    remainder = "passthrough"
)

pipe = Pipeline(
    steps = [
        ("transform", transform),
        ("model",RandomForestRegressor(random_state = 42))
    ]
)

param_dist = {
    "model__n_estimators" : [50, 100, 150],
    
    "model__max_depth" : [5, 10, 15, None],

    "model__min_samples_split" : [2, 5, 4],

    "model__min_samples_leaf" : [1, 2, 4]
}

start_time = time.time()

random_search = RandomizedSearchCV(
    pipe,
    param_distributions = param_dist,
    cv = 5,
    random_state = 42,
)

random_search.fit(x_train, y_train)

training_time = time.time() - start_time

print("\nBest Parameters :\n")
print(random_search.best_params_)

print("\nBest CV Score :\n")
print(random_search.best_score_)

# predictions
y_pred = random_search.predict(x_test)

# test_score
test_r2 = r2_score(y_test, y_pred)

print("\nTesting R2 Score :\n")
print(test_r2)

print("\nTraining Time :\n")
print(training_time)

# results dataframe
results_df = pd.DataFrame(
    random_search.cv_results_
)

# useful columns
results_df = results_df[
    [
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]
]

# add exta metrics
results_df["test_r2"] = test_r2
results_df["training_time"] = training_time

print(results_df.head())

# save csv
results_df.to_csv("AQI_Model_Results.csv", index = False)

print("\n CSV Saved Successfully!")