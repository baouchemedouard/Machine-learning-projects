# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:37:55 2020

@author: z011348
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve

# Import the data
heart_failure = pd.read_csv("C:/Users/z011348/Desktop/ML/input/Heart Failure Prediction/datasets_727551_1263738_heart_failure_clinical_records_dataset.csv")
print(heart_failure.head())

# Split the data into X and y
X = heart_failure.drop("DEATH_EVENT", axis=1)
y = heart_failure["DEATH_EVENT"]

# Split into Train and Test sets
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)

# put models in a dictionary 
models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()
    }

def fit_and_score(model, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

model_scores = fit_and_score(models, 
                             X_train, 
                             X_test, 
                             y_train, 
                             y_test)

print(model_scores)

print("==================================================================")
print("Hyperparameter tuning by RandomizedSearchCV for LogisticRegression")
print("==================================================================")

# Create hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-1, 4, 20),
                "solver": ["liblinear"]}

# Tune LogisticRegression
np.random.seed(42)
rs_log_reg = RandomizedSearchCV(LogisticRegression(), 
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)
# Fit the model
rs_log_reg.fit(X_train, y_train)

print("== Best paramerts for Log_reg: ")
print(rs_log_reg.best_params_)
print("== Score for Log_reg:")
print(rs_log_reg.score(X_test, y_test))

print("======================================================================")
print("Hyperparameter tuning by RandomizedSearchCV for RandomForestClassifier")
print("======================================================================")
# Create hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2,20,2),
           "min_samples_leaf": np.arange(1,20,2)}

np.random.seed(42)
rs_rf = RandomizedSearchCV(RandomForestClassifier(), 
                                param_distributions=rf_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)
# Fit the model
rs_rf.fit(X_train, y_train)

print("== Best paramerts for RandomForestClassifier: ")
print(rs_rf.best_params_)
print("== Score for RandomForestClassifier:")
print(rs_rf.score(X_test, y_test))

print("==============================================")
print("Hyperparameter tuning by GridSearchCV")
print("==============================================")
# Hyperparameter tuning by GridSearchCV
# we are going to tune 
#1. LogisticRegression
#2. RandomForestClassifier
# Create hyperparameter grid for LogisticRegression
log_reg_grid2 = {"C": np.logspace(-1, 4, 30),
                "solver": ["liblinear"]}

np.random.seed(42)
gs_log_reg = GridSearchCV(LogisticRegression(), 
                                param_grid=log_reg_grid2,
                                cv=5,
                                verbose=True)
gs_log_reg.fit(X_train, y_train)
print("== Best paramerts for LogisticRegression: ")
print(gs_log_reg.best_params_)
print("== Score for LogisticRegression:")
print(gs_log_reg.score(X_test, y_test))
print("")
# --------------------------------------
# all hyperparametr tuning score is not crossed the inital model score
# --------------------------------------
print("LogisticRegression =======>")
np.random.seed(42)
lr = LogisticRegression()
lr.fit(X_train,y_train)

y_preds = lr.predict(X_test)
# print(y_preds)
print("Model score accuracy:")
print(lr.score(X_train, y_train))

# Confution matrix
print(" == confusion_matrix")
print(confusion_matrix(y_test, y_preds))

# plot the confusion matrix
sns.set(font_scale=1.5)
def plot_conf_mat(y_test, y_pred):
    fig2, ax2 = plt.subplots(figsize=(3,3))
    ax2=sns.heatmap(confusion_matrix(y_test, y_pred),
                    annot=True,
                    cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    
plot_conf_mat(y_test, y_preds)

# Lets get classification report
print(" == classification_report:") 
print(classification_report(y_test, y_preds))
