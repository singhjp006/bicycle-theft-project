import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
# Construct some pipelines
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("/Users/jashan/Documents/py/BicycleTheft/Bicycle_Thefts-copy.csv")

print("Data size: ", data.size, "\n")
print("Data shape: ", data.shape, "\n")

print("Overall description of data \n", data.describe(include=np.object))

# Statistical assessments about numeric data
print("Description of numeric data", data.describe(), "\n")

# Correlation
from sklearn.preprocessing import OrdinalEncoder
data_corr = data.copy()
data_corr = data_corr[data.Status != 'UNKNOWN']
ordinalEncoder = OrdinalEncoder(categories=[['RECOVERED', 'STOLEN']])
status_encoded = ordinalEncoder.fit_transform(data_corr.loc[:,['Status']])
data_corr['Status_code'] = pd.Series(status_encoded.reshape(len(status_encoded)))
corr_matrix = data_corr.corr()
print("Correlation\n", corr_matrix['Status_code'].sort_values(ascending=False))

print("Missing data\n", data.isnull().sum())
# When it comes to geographical data, missing values are encoded in "NSA"
nsa_dict = {
    "Division": len(data[data['Division'] == "NSA"]), 
    "NeighbourhoodName": len(data[data['NeighbourhoodName'] == "NSA"]),
    "Hood_ID": len(data[data['Hood_ID'] == "NSA"]),
}
nsa_number = pd.Series(nsa_dict)
nsa_number

sns.heatmap(data.isnull(),yticklabels=False,cbar=True,cmap='viridis')

data.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)

data = data[data.Status != 'UNKNOWN']
data.Status.unique()
data_features = data.drop('Status', axis=1)
data_target = data['Status']

# data transformations - date
date = pd.to_datetime(data_features['Occurrence_Date'])
data_features = data_features.drop('Occurrence_Date', axis=1)
data_features['Occurrence_Month'] = date.dt.month
data_features['Occurrence_DayOfWeek'] = date.dt.dayofweek

# data transformations - handling NSA(Not Specified Area)
data_features['Division'] = data_features['Division'].replace('NSA', np.nan)
data_features['NeighbourhoodName'] = data_features['NeighbourhoodName'].replace('NSA', np.nan)
data_features.columns

features_to_keep = ["Occurrence_Year", "Occurrence_Month", "Occurrence_Hour", "Bike_Speed", "Cost_of_Bike", "Longitude", "Latitude", "Primary_Offence", "Division", "Location_Type", "Bike_Make", "Bike_Type", "Bike_Colour","NeighbourhoodName"]
data_features = data_features[features_to_keep]
data_features.columns
data_features.shape


data_target = data_target.replace('STOLEN', 0)
data_target = data_target.replace('RECOVERED', 1)
data_target.unique()
data_target.shape


num_attribs = ["Occurrence_Year", "Occurrence_Month", "Occurrence_Hour", "Bike_Speed", "Cost_of_Bike",
               "Longitude", "Latitude"]
cat_attribs = ["Primary_Offence", "Division", "Location_Type", "Bike_Make", 
               "Bike_Type", "Bike_Colour","NeighbourhoodName"]


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline([
        ('freq_imputer', SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore')),
    ])

preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_attribs),
        ('cat', categorical_transformer, cat_attribs)])

X_transformed = preprocessor.fit_transform(data_features)
X_transformed.shape

X_train, X_test, y_train, y_test = train_test_split(X_transformed, data_target, test_size = .20, random_state = 40)

# Check how imbalanced is the data
target_count = y_train.value_counts()
print('Stolen:', target_count[0])
print('Recovered:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1]), ': 1')
target_count.plot(kind='bar', title='Count (target)')

sm = SMOTE(random_state=40, sampling_strategy = 'minority')
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

target_count = y_train_smote.value_counts()
print('Stolen:', target_count[0])
print('Recovered:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1]), ': 1')
target_count.plot(kind='bar', title='Count (target)')

# logistic regression classifier grid search
clf_lr = LogisticRegression(max_iter=1000)
param_grid_lr={
    'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
    'C':  [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet'],
}
grid_search_lr = RandomizedSearchCV(
    estimator=clf_lr,
    param_distributions=param_grid_lr,
    scoring='accuracy',
    cv=3,
    n_iter=7,
    refit = True,
    verbose = 3
)
grid_search_lr.fit(X_train_smote, y_train_smote)

print(f"Logistic Regression Best Params: \n{grid_search_lr.best_params_}")
clf_lr_best = grid_search_lr.best_estimator_


# random forest classifier grid search
clf_rf = RandomForestClassifier()
param_grid_rf = {
    'n_estimators': range(100, 500, 100),
    'max_depth': range(30, 50, 10),
    'max_features' : ['sqrt', 'log2'],
    'min_samples_leaf': range(1,15,3),
    'min_samples_split': range(2,10,2),
}
grid_search_rf = RandomizedSearchCV(
    estimator=clf_rf,
    param_distributions=param_grid_rf,
    scoring='accuracy',
    cv=2,
    n_iter=7,
    refit = True,
    verbose = 3
)
grid_search_rf.fit(X_train_smote, y_train_smote)

print(f"Random Forest Best Params: \n{grid_search_rf.best_params_}")
clf_rf_best = grid_search_rf.best_estimator_

# SVC grid search
clf_svc = SVC()
param_grid_svc = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.01,0.1, 1, 10, 100],
    'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
}
grid_search_svc = RandomizedSearchCV(
    estimator=clf_svc,
    param_distributions=param_grid_svc,
    scoring='accuracy',
    cv=2,
    n_iter=7,
    refit = True,
    verbose = 3
)
grid_search_svc.fit(X_train_smote, y_train_smote)

print(f"Support Vector Best Params: \n{grid_search_svc.best_params_}")
clf_svc_best = grid_search_svc.best_estimator_

# decision tree classifier grid search
clf_dt = DecisionTreeClassifier()
param_grid_dt = {
    'max_depth': range(10, 100, 10),
    'max_features' : ['sqrt', 'log2'],
    'min_samples_leaf': range(1,15,3),
    'min_samples_split': range(2,10,2),
}
grid_search_dt = RandomizedSearchCV(
    estimator=clf_dt,
    param_distributions=param_grid_dt,
    scoring='accuracy',
    cv=2,
    n_iter=7,
    refit = True,
    verbose = 3
)
grid_search_dt.fit(X_train_smote, y_train_smote)

print(f"Decision Tree Best Params: \n{grid_search_dt.best_params_}")
clf_dt_best = grid_search_dt.best_estimator_

# MLP classifier grid search
clf_mlp = MLPClassifier()
param_grid_mlp = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
    'learning_rate': ['constant', 'adaptive']
}
grid_search_mlp = RandomizedSearchCV(
    estimator=clf_mlp,
    param_distributions=param_grid_mlp,
    scoring='accuracy',
    cv=2,
    n_iter=7,
    refit = True,
    verbose = 3
)
grid_search_mlp.fit(X_train_smote, y_train_smote)

print(f"MLP Classifier Best Params: \n{grid_search_mlp.best_params_}")
clf_mlp_best = grid_search_mlp.best_estimator_

cv = KFold(n_splits=10, shuffle = True, random_state = 76)

y_pred_class_logreg = cross_val_predict(clf_lr, X_train_smote, y_train_smote, cv = cv)
y_pred_class_svc = cross_val_predict(clf_svc, X_train_smote, y_train_smote, cv = cv)
y_pred_class_dt = cross_val_predict(clf_dt, X_train_smote, y_train_smote, cv = cv)
y_pred_class_mlp = cross_val_predict(clf_mlp, X_train_smote, y_train_smote, cv = cv)
y_pred_class_rf = cross_val_predict(clf_rf, X_train_smote, y_train_smote, cv = cv)


report_logreg = metrics.classification_report(y_train_smote, y_pred_class_logreg)
report_svc = metrics.classification_report(y_train_smote, y_pred_class_svc)
report_dt = metrics.classification_report(y_train_smote, y_pred_class_dt)
report_mlp = metrics.classification_report(y_train_smote, y_pred_class_mlp)
report_rf = metrics.classification_report(y_train_smote, y_pred_class_rf)


# # Model scoring and evaluation
# #Print out the score of the model
# y_pred = grid_search_mlp.predict(X_test)

# print(f'''Accuracy of grid_search_mlp:
# {metrics.accuracy_score(y_test, y_pred)}''')

# print(f'''{classification_report( y_test, y_pred)}''')

# y_test_predicted = clf_mlp_best.predict(X_test)
# confusion_matrix(y_test, y_test_predicted)

# metrics.plot_roc_curve(clf_mlp_best, X_test, y_test)  
# plt.show()





# Best params and estimators
print(f"Logistic Regression Best Params: \n{grid_search_lr.best_params_}")
print(f"Random Forest Best Params: \n{grid_search_rf.best_params_}")
print(f"Support Vector Best Params: \n{grid_search_svc.best_params_}")
print(f"Decision Tree Best Params: \n{grid_search_dt.best_params_}")
print(f"MLP Classifier Best Params: \n{grid_search_mlp.best_params_}")
clf_lr_best = grid_search_lr.best_estimator_
clf_rf_best = grid_search_rf.best_estimator_
clf_svc_best = grid_search_svc.best_estimator_
clf_dt_best = grid_search_dt.best_estimator_
clf_mlp_best = grid_search_mlp.best_estimator_

from sklearn import model_selection
outcome = []
model_names = []
models = [('clf_lr', LogisticRegression(max_iter=1000)), 
          ('clf_svc', SVC()), 
          ('clf_dt', DecisionTreeClassifier()),
          ('clf_mlp', MLPClassifier()),
          ('clf_rf', RandomForestClassifier()),
          ]

for model_name, model in models:
    k_fold_validation = model_selection.KFold(n_splits=10)
    results = model_selection.cross_val_score(model,X_train_smote, y_train_smote, cv=k_fold_validation, scoring='accuracy')
    outcome.append(results)
    model_names.append(model_name)
    output_message = "%s| Mean=%f STD=%f" % (model_name, results.mean(), results.std())
    print(output_message)          

full_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', clf_svc_best)])

# Saving model to disk
import pickle
pickle.dump(full_pipeline, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
model.fit(data_features, data_target)

predict_data_from_API = pd.DataFrame([[2020, 10, 5, 14, 200, 900, -8850629.735, 5411195.656, 'THEFT UNDER - BICYCLE', 'D22', 'Ttc Subway Station', 'SPECIALIZED', 'MT', 'BLU' 'Stonegate-Queensway (16)']], columns=features_to_keep)
model.predict(predict_data_from_API)

# model.predict([2020, 10, 5, 14, 200, 900, -8850629.735, 5411195.656, 'THEFT UNDER - BICYCLE', 'D22', 'Ttc Subway Station', 'SPECIALIZED', 'MT', 'BLU' 'Stonegate-Queensway (16)'])