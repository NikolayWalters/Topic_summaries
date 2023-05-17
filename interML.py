"""
Kaggle intermediate ML course notes
"""

# === On missing values ===

# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns with missing vals in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# SimpleImputer from scikit
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))  # fits the data; default strategy is mean
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid)) # only applies the transform, i.e. no fitting
# not fitting validation is important to avoid data leakage

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# a different strategy to simple imputation that also creates an additional column that indicates whether
# an imputation took place. this column is then also fed to the model to help it make a more informed choice

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

# === Categorical variables
# three approaches: 1 drop; 2 ordinal encoding; 3 one-hot

# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

# drop cat vars
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

# ordinal encodings
from sklearn.preprocessing import OrdinalEncoder

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

# One hot
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
# two things to note
# handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
# setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) 
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

# === Pipes ===
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)

# Create and evaluate the pipe
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

# === Cross-validation ===
# folds bby

# much easier to do cross-validation with pipes
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# define pipe
my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5, # number of folds
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)

# can also do without a pipe
# Train and score baseline model
baseline = RandomForestRegressor(criterion="absolute_error", random_state=0)
baseline_score = cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
)
baseline_score = -1 * baseline_score.mean()

# === XGBoost ===
#1) Initial model creation: XGBoost starts with a simple decision tree as the first model. This model predicts 
#the target variable based on the available input features.

#2) Residual calculation: After the first model is created, the residuals (the differences between the actual 
#target values and the predicted values) are calculated for each data point in the training set.

#3) Model improvement: The next step is to create a new decision tree that predicts the residuals of the previous 
#model. This new model is then added to the previous model, and the combined model now predicts the target 
#variable more accurately than the first model alone.

#4) Iterative improvement: Steps 2 and 3 are repeated until the model accuracy reaches a satisfactory level. 
#Each new model is trained to predict the residuals of the combined previous models.

#5) Regularization: XGBoost uses regularization techniques to avoid overfitting. Regularization techniques 
#penalize large model weights and simplify the model to reduce its complexity.

#6) Prediction: Once the XGBoost model is trained, it can be used to make predictions on new data.

from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

# useful params
# N trees; typical 100-1000
my_model = XGBRegressor(n_estimators=500)

# stopping rounds; terminates after that many iterations if not validation improvement
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)

# learning rate
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

# parallelisation
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)


# more contained example of xgboost
from sklearn.model_selection import StratifiedKFold
kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=42)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    # Convert our data into XGBoost format
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(test.values)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
    # and the custom metric (maximize=True tells xgb that higher metric is better)
    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=gini_xgb, maximize=True, verbose_eval=100)

    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
    # Predict on our test data
    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
    sub['target'] += p_test/kfold
