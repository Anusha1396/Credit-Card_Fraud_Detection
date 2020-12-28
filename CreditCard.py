import metrics as metrics
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import imblearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


# Load NSL_KDD train dataset
dfkdd_train = pd.read_table("train_data.csv", sep=",")  # change path to where the dataset is located.
#dfkdd_train = dfkdd_train.iloc[:, :-1]  # removes an unwanted extra field

# Load NSL_KDD test dataset
dfkdd_test = pd.read_table("test_data.csv", sep=",")
#dfkdd_test = dfkdd_test.iloc[:, :-1]
print("test data")
print(dfkdd_train)

pd.set_option("display.float", "{:.2f}".format)

LABELS = ["Normal", "Fraud"]
print(f"Train Columns or Feature names :- \n {dfkdd_train.columns}")
print(f"Test Columns or Feature names :- \n {dfkdd_test.columns}")

# determine the number of records in the dataset
print('The dataset contains {0} train and {1} test.'.format(dfkdd_train.shape[0], dfkdd_test.shape[0]))

# ### Explore label class
df = pd.concat([dfkdd_train, dfkdd_test], ignore_index=True)
print("df values")
print(df)
#Convert the fraudulent label to a binary classification problem 0=normal 1=fraudulent
print("df class")
print(df["Class"])
print(f"Unique values of target variable :- \n {df['Class'].unique()}")
print(f"Number of samples under each target value :- \n {df['Class'].value_counts()}")

df["fraud"] = df["Class"].apply(lambda x: 0 if x==0 else 1)
print("normal and fraud")
print(df["fraud"])

df = df.drop(columns=['Amount'])


# resplit the data
dfkdd_train = df.iloc[0:48454, :]
dfkdd_test = df.iloc[48454:69220, :]

print("ccc")
print(dfkdd_train)
print("ddd")
print(dfkdd_test)

# print("class train")
# print(dfkdd_train['Class'])

# Create the training target for each dataset
y_train = np.array(dfkdd_train["fraud"])
y_test = np.array(dfkdd_test["fraud"])


print("y train")
print(y_train)
print("y test")
print(y_test)
# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(dfkdd_train.drop(["fraud", "Class"], axis=1))
X_test = scaler.transform(dfkdd_test.drop(["fraud", "Class"], axis=1))

df = df.drop(columns=['Class'])
print("x train")
print(X_train)
print("x test")
print(X_test)



from imblearn.over_sampling import RandomOverSampler, SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# DTC_Classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth=20, random_state=42,class_weight='balanced')
# y_pred= DTC_Classifier.fit(X_train, y_train).predict(X_test)

RF_Classifier = RandomForestClassifier(random_state= 101)
y_pred= RF_Classifier.fit(X_train, y_train).predict(X_test)

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data,
                       model, param_grid, cv=20, scoring_fit='neg_log_loss',
                       do_probabilities=False):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train_data, y_train_data)

    if do_probabilities:
        pred = fitted_model.predict_proba(X_test_data)
    else:
        pred = fitted_model.predict(X_test_data)

    return fitted_model, pred

clf = XGBClassifier()

param_grid = {
    'random_state':[101],
    'n_jobs': [1],
    'n_estimators': [200, 400],
    'max_depth': [20],
    'learning_rate':[0.1],
    'min_child_weight':[1],
    'gamma':[0],
    'subsample':[0.8],
    'colsample_bytree':[0.8],
    'nthread':[4],
    'scale_pos_weight':[1],
    'seed':[27]
    #'max_leaf_nodes': [40, 50],
}

model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, clf, param_grid, cv=10, scoring_fit='accuracy',)

print(model.best_score_)
print(model.best_params_)


# train model by using fit method
print("Model training starts........")
print("Model training completed")
acc_score = RF_Classifier.score(X_test, y_test)* 100
print(f'Accuracy of model on test dataset :- {acc_score}')

submission = pd.DataFrame({'Id':dfkdd_test['ID'],'Class':y_pred})

#Visualize the first 5 rows
print(submission)
filename = 'CreditCard.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)