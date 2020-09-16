"""
task : https://www.kaggle.com/c/titanic
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LogisticRegression

# Evaluation
#from sklearn.model_selection import train_test_split,cross_val_score
#from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
#from sklearn.metrics import confusion_matrix, classification_report
#from sklearn.metrics import precision_score, recall_score, f1_score
#from sklearn.metrics import plot_roc_curve


#pd.set_option('display.max_rows', 100)
#pd.set_option('display.max_columns', None)

# Import the data
df = pd.read_csv("C:/Users/z011348/Desktop/ML/input/titanic/train.csv")
print(df.head())

# Make a copy of original dataframe - for future reference
df_tmp = df.copy()

# check for missing data
print(df_tmp.isna().sum())
print("")

# check for non-numeric data
print(df_tmp.dtypes)

print("=== Convert Strings to Pandas categories ===")
# find columns which contains strings
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
       # print(label)
        df_tmp[label] = content.astype("category").cat.as_ordered()
print(df_tmp.info())
print("")
#print(df_tmp.Embarked.cat.codes)
#print(df_tmp.Name[:1])

def preprocess_date(df):
    # Fill numeric rows with median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
        #x = pd.isnull(content).sum()
        #print(x)
            if pd.isnull(content).sum():
            # add binary column which tells us if the data was missing or not
                df[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with meadian
                df[label] = content.fillna(content.median())
            
        # Fill catogortical missing data and turn into numbers
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            # we add +1 to category code
            df[label] = pd.Categorical(content).codes+1
    
    return df

# Process train data
df_train = preprocess_date(df_tmp)
print(df_train.head())
print("")

# Check for nulls again
print(df_train.isna().sum())
#print(df_train.shape)
#print(df_train.info())
print("")
# Split into X and y (on train set)
X_train = df_train.drop("Survived", axis=1)
y_train = df_train["Survived"]

# -------------------------------------------------
# Process test data set
# -------------------------------------------------
# import test set
df_test = pd.read_csv("C:/Users/z011348/Desktop/ML/input/titanic/test.csv")
print(df_test.head())
print(df_test.dtypes)
print(X_train.shape, df_test.shape)
print(df_test.isna().sum()) 

# Ptr-Process test data
df_test = preprocess_date(df_test)
print(df_test.shape)
print("")

# Find the column differences b/w training and test data sets
print("Find the column differences b/w training and test data sets:")
print(set(df_test.columns) - set(X_train.columns)) 
print("")

# manually adjust X_train to have Fare_is_missing column 
print("manually adjust X_train to have Fare_is_missing column:")
X_train["Fare_is_missing"] = False
print(X_train.shape, y_train.shape)
print("")

# -----------------------------------------
# Model RandomForestClassifier
# -----------------------------------------
model = RandomForestClassifier(random_state = 42)
model.fit(X_train, y_train)
print(f"Model(RandomForestClassifier) score: ", model.score(X_train, y_train))
print("")

# -----------------------------------------
# Model RandomForestClassifier
# -----------------------------------------
#model = LogisticRegression()
#model.fit(X_train, y_train)
#print(f"Model(LogisticRegression) score: ", model.score(X_train, y_train))
#print("")

# Now we have same columns at df_test and X_train. Lets do prediction
print("========= Predicted Survived ============ ")
test_pred = model.predict(df_test)
#print(len(test_pred))
print(test_pred)
print("")

# We have made some predictions but they are not in the same as Kaggle
# Now we need to format as per the Kaggle format
df_prediction_survived = pd.DataFrame()
df_prediction_survived["PassengerId"] = df_test["PassengerId"]
df_prediction_survived["Survived"] = test_pred
print(df_prediction_survived)

# Save the prediction results in the csv file
df_prediction_survived.to_csv("C:/Users/z011348/Desktop/ML/output/titanic/Survived_classification_prediction.csv",
                               index=False)
#X_train.to_csv("C:/Users/z011348/Desktop/ML/output/titanic/X_train.csv",
#                               index=False)
