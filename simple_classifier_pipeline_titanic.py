import pandas as pd
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

df = pd.read_csv(
    'https://gist.githubusercontent.com/michhar/2dfd2de0d4f8727f873422c5d959fff5/raw/'
    'fa71405126017e6a37bea592440b4bee94bf7b9e/titanic.csv'
)

df_y = df['Survived']
df_X = df.drop('Survived', axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=42)


#
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_cols = ['Pclass', 'Embarked', 'Sex']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


column_trans = ColumnTransformer(
    [
        ('cat', categorical_transformer, categorical_cols),
        ('num', numeric_transformer, numerical_cols)
    ],
    remainder='drop'
)

# clf
clf = Pipeline(steps=[('preprocessor', column_trans),
                      ('classifier', LogisticRegression())])

clf.fit(X_train, y_train)

score_clf = clf.score(X_test, y_test)


# clf_2
clf_2 = Pipeline(steps=[('preprocessor', column_trans),
                        ('classifier', SVC())])

clf_2.fit(X_train, y_train)

score_clf_2 = clf_2.score(X_test, y_test)


# model export
dump(clf_2, 'clf_2.joblib')

