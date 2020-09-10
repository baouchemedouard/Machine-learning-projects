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
heart_disease = pd.read_csv("C:/Users/Desktop/ML/input/heart-disease.csv")
# print(heart_disease.shape)
# print(heart_disease.isna().sum())
# print(heart_disease.describe()) # to know the numeric details

# just compare the data between gender and target column to understand more 
print(heart_disease.sex.value_counts())
print("")
# compare target column with gender
print(pd.crosstab(heart_disease.target, heart_disease.sex))
pd.crosstab(heart_disease.target, heart_disease.sex).plot(kind='bar',
                                                          figsize=(10,6),
                                                          color=["salmon", "lightblue"])
plt.title("Heart disease frequency for genders")
plt.xlabel("0 - no disease, 1 - disease")
plt.ylabel("number of count")
plt.legend(["Female", "Male"])    # update legend 
plt.xticks(rotation=0)

# create an another figure 
plt.figure(figsize=(10,6))
# Scatter with positive values 
plt.scatter(heart_disease.age[heart_disease.target==1],
            heart_disease.thalach[heart_disease.target==1],
            c="salmon")

# Scatter with -ve values 
plt.scatter(heart_disease.age[heart_disease.target==0],
            heart_disease.thalach[heart_disease.target==0],
            c="lightblue")
# add some useful information 
plt.title("Heart disease in function of age and max heart rate ")
plt.xlabel("Age")
plt.ylabel("Max heart rate")
plt.legend(["Disease", "No disease"])
print("")

# Heart disease frequency per chest pain type
# compare target column with chest pain(cp)
print(pd.crosstab(heart_disease.cp, heart_disease.target))
# make the crosstab more visualize 
pd.crosstab(heart_disease.cp, heart_disease.target).plot(kind="bar",
                                                         figsize=(10,6),
                                                         color=["salmon", "lightblue"])
plt.title("Heart disese frequency per chest pain type")
plt.xlabel("Chest pain type")
plt.ylabel("Count")
plt.legend(["no disease", "disease"])
plt.xticks(rotation=0)

# correlation matrix 
corr_matrix = heart_disease.corr()
# print(corr_matrix)
fig, ax = plt.subplots(figsize=(10,6))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu")

## -------------------------------------
# ML modeling 
## -------------------------------------
# Split the data into X and y
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# split data into train and test sets
np.random.seed(42)
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=(0.2))

# put models in a dictionary 
models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()
    }

# create function to fit score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    # make a dictionary to keep model score
    model_scores = {}
    # Loop thru models
    for name, model in models.items():
        # fit the model to the data
        model.fit(X_train, y_train)
        # evaluate the model score and sppend it to model_scores 
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

model_scores = fit_and_score(models=models, 
                             X_train=X_train, 
                             X_test=X_test, 
                             y_train=y_train, 
                             y_test=y_test)
print(model_scores)
print("")
# model comparison 
model_compare = pd.DataFrame(model_scores, index=["accuracy"])
print(model_compare)
model_compare = model_compare.T
print(model_compare)
model_compare.plot.bar()

#===================================
# Hyperparameter tuning 
#===================================
# Feature importance
# Confustion matrix
# cross validation
# Precision
# Recall
# F1 score 
# classification report
# ROC curve
# Area under the curve (AUC)

# Lets tune KNN
# Hyperparameter tuning by hand
train_score = []
test_score = []
# Create a different list of values for n-neighbors 
neighbors = range(1,21)
# setup KNN instance              
knn = KNeighborsClassifier()
# look thru different neihbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    # fit the model - knn
    knn.fit(X_train, y_train)
    #update the training score list 
    train_score.append(knn.score(X_test, y_test))
    #update the test score list
    test_score.append(knn.score(X_test, y_test))

print("== train_score ==:")
print(train_score)
print("== test_score ==:")
print(test_score)

'''
fig1, ax1 = plt.subplots(figsize=(10,6))
ax1.plot(neighbors, train_score)
ax1.plot(neighbors, test_score)
ax1.set(title='Plot with no. of neighbors and model score',
        xlabel='Number of Neighbors',
        ylabel='Model score')
'''
print(f"Max KNN score on the test data {max(test_score)*100:.2f}%")

print("==============================================")
print("Hyperparameter tuning by RandomizedSearchCV")
print("==============================================")
# Hyperparameter tuning by RandomizedSearchCV
# we are going to tune 
#1. LogisticRegression
#2. RandomForestClassifier

# Create hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-1, 4, 20),
                "solver": ["liblinear"]}

# Create hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2,20,2),
           "min_samples_leaf": np.arange(1,20,2)}

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

# Tune RandomForestClassifier
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

# Evaluate ======================
# Confustion matrix
# cross validation
# Precision
# Recall
# F1 score 
# classification report
# ROC curve
# Area under the curve (AUC)

# Make predictions
y_pred = gs_log_reg.predict(X_test)
print(y_pred)

# plot ROC curve and calculate AUC metric
plot_roc_curve(gs_log_reg, X_test, y_test)

# Confution matrix
print(" == confusion_matrix")
print(confusion_matrix(y_test, y_pred))
# plot the confusion matrix
sns.set(font_scale=1.5)
def plot_conf_mat(y_test, y_pred):
    fig2, ax2 = plt.subplots(figsize=(3,3))
    ax2=sns.heatmap(confusion_matrix(y_test, y_pred),
                    annot=True,
                    cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    
plot_conf_mat(y_test, y_pred)

# Lets get classification report
print(" == classification_report:") 
print(classification_report(y_test, y_pred))

# Calculate evaluation metrics using cross validation 
# using cross_val_score()

# Create a new classifier with best parameters which were identified before
clf = LogisticRegression(C=0.2212216291070449,
                         solver='liblinear')

# cross validated accuracy
print("cross validated accuracy ===") 
cv_acc = cross_val_score(clf,
                      X,
                      y,
                      cv=5,
                      scoring="accuracy")
print(cv_acc)
print(np.mean(cv_acc))

# cross validated precesion
print("cross validated precesion ===")
cv_precision = cross_val_score(clf,
                      X,
                      y,
                      cv=5,
                      scoring="precision")
print(cv_precision)
print(np.mean(cv_precision))

# cross validated recall
print("cross validated recall ===")
cv_recall = cross_val_score(clf,
                      X,
                      y,
                      cv=5,
                      scoring="recall") 
print(cv_recall)
print(np.mean(cv_recall))

# cross validated f1
print("cross validated f1 ===")
cv_f1 = cross_val_score(clf,
                      X,
                      y,
                      cv=5,
                      scoring="f1")
print(cv_f1)
print(np.mean(cv_f1))

# Visualize cross validation metrics
cv_metrics = pd.DataFrame({"Accuracy": np.mean(cv_acc),
                           "Precision": np.mean(cv_precision),
                           "Recall": np.mean(cv_recall),
                           "F1": np.mean(cv_f1)},
                          index=[0])
print(cv_metrics.T)
