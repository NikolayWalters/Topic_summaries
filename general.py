"""
Some general strating steps
"""


.head()
.tail()
.shape
.info() # useful to show datatypes
.columns # might want to separate into num and cat here
# something like this:
nums_columns = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_columns = train_data.select_dtypes(include=['object']).columns.tolist()
features = nums_columns + cat_columns

# tabled summary
df.describe().T\
        .style.bar(subset=['mean'], color=px.colors.qualitative.G10[2])\
        .background_gradient(subset=['std'], cmap='Blues')\
        .background_gradient(subset=['50%'], cmap='BuGn')
# look out for low variance features (consider dropping or transforming)

# if data is noisy can be helpful to bin continuous features into discrete values
# startegies:
# >Uniform
# Each bin has same width in span of possible values for variable.
# >Quantile
# Each bin has same number of values, split based on percentiles.
# >Clustered
# Clusters are identified & examples are assigned to each group

# discretize into 10 equal sized bins
from sklearn.preprocessing import KBinsDiscretizer
#Use a numpy array of the features
X = data.data
#Dfine the discretization strategy, number of bins and encoding of
#the categorical features that will be created from the continuous ones:
disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
#Learn transformation:
disc.fit(X)
#Apply transformation:
X = disc.transform(X)
#Create a dataframe using the discretized feature values & add the target column:
df = pd.DataFrame(X,columns=data.feature_names)
df['target'] = pd.Series(data.target) # name the label column as 'target' for clarity



# check for dups
.drop_duplicates()
.shape

# just plots everything
sns.set(style="ticks")
sns.pairplot(train_data)
plt.show()

# look out for multimodal distributions - suggest subpopulations or systematic errors
# i.e. multiple units of measurements used


# check for class imbalance, can plot
plt.figure(figsize=(6,6))
# Pie plot
train['Target'].value_counts().plot.pie(explode=[0.1,0.1], autopct='%1.1f%%',
 shadow=True, textprops={'fontsize':16}).set_title("Target distribution")
 # or if want a fancier pie plot
 import plotly.express as px
class_map = {0: 'Class 0', 1: 'Class 1'}
train_data['ClassMap'] = train_data['Machine failure'].map(class_map)
fig = px.pie(train_data, names='ClassMap', height=540, width=840, hole=0.45,
             title='Target Overview - Machine failure',
             color_discrete_sequence=["#00008B", "#ADD8E6"])
fig.update_layout(font_color='#000000',
                  title_font_size=18,
                  showlegend=False)
fig.add_annotation(x=0.5, y=0.5, align='center', xref='paper', yref='paper',
                   showarrow=False, font_size=22, text='Class<br>Imbalance')
fig.show()

# if yes, under/over-sampling or more advanced like tomek/smote (need to make some scatter plots like pca maybe to decide
# if appropriate)
# be careful with smote though. not always a good idea, usually don't use when:
# 1) Small minority class, i.e. 99/1 split
# 2) noisy data, i.e. uncertainties
# 3) class overlap
# 4) sequential data/time series (smote assumes points independent)
# smote
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
train = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['Machine failure'])], axis=1)

# sometimes can reframe the problem into outlier detection

#missingno - a python library allowing for quick inspection of missing data and missingness correlation / mechanism.
# plot of missing vals, note sample size 250
import missingno as msno
#Get & visualize missingness matrix
msno.matrix(data.sample(250))

# or as a bar plot
msno.bar(data.sample(1000))

# missing vals
train.isnull().values.any()
#or
print(train.isna().sum())
# don't forget to think about why data is missing
# and whether there's any correlation with other features

# missing vals corr map
msno.heatmap(data)
# for deeper interpretation can use dendrogram
msno.dendrogram(data)


# gives a table with missing data percentage
from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ['Column Name', 'Data Type', 'Train Missing %', 'Test Missing %']
for column in train.columns:
    data_type = str(train[column].dtype)
    non_null_count_train= 100-train[column].count()/train.shape[0]*100
    if column!='Age':
        non_null_count_test = 100-test[column].count()/test.shape[0]*100
    else:
        non_null_count_test="NA"
    table.add_row([column, data_type, non_null_count_train,non_null_count_test])
print(table)


# or if it's some specific val:
# could also read it in with replacement to nans: train=pd.read_csv('../input/train.csv', na_values=-1)
vars_with_missing = []

for f in train.columns:
    missings = train[train[f] == -1][f].count()  # in this case missing val is = -1
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings/train.shape[0]
        
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
        
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))
# above 20% missing -> drop
# also consider dropping very low variance cols (if have a lot of cols)
# below 20% replace with mean/median/mode etc
# sometimes good to keep missing vals as separate (i.e. is_missing col = True)


# consider plotting a heatmap of missing values and dropping rows with many missing vals
# Heatmap of missing values
plt.figure(figsize=(12,6))
sns.heatmap(train[na_cols].isna().T, cmap='summer')
plt.title('Heatmap of missing values')


# plot to see if there's any pattern between target and missing values
# Countplot of number of missing values by passenger
train['na_count']=train.isna().sum(axis=1)
plt.figure(figsize=(10,4))
sns.countplot(data=train, x='na_count', hue='Transported')
plt.title('Number of missing entries by passenger')
train.drop('na_count', axis=1, inplace=True)



# if imputing missing features might be worth for example to split data into groups and then impute based
# on group's statistic. for example, for missing expenditure create age groups and fill expenditure
# based on age group's mean
# e.g. 
train['Age_group']=np.nan
train.loc[train['Age']<=12,'Age_group']='Age_0-12'
train.loc[(train['Age']>12) & (train['Age']<18),'Age_group']='Age_13-17'
# etc
# or
# Missing values before
A_bef=data[exp_feats].isna().sum().sum()
# Fill missing values using the median
na_rows_A=data.loc[data['Age'].isna(),'Age'].index
data.loc[data['Age'].isna(),'Age']=data.groupby(['HomePlanet','No_spending','Solo','Cabin_deck'])['Age'].transform(lambda x: x.fillna(x.median()))[na_rows_A]


# when imputing think if some inference can be made, i.e. if people are from the same group
# then makes sense to impute their destination/origin to be the same (based on present values
# in the group, i.e. if one is from origin A then the whole group is likely from A)
# another example are people filled in into the cabins based on the origin?
# make heatmaps and look for patterns


# KNN imputation
from sklearn.impute import KNNImputer
#Create numpy arrays with the values of feature 'LoanAmount' on the train & test set
LoanAmount_train = X_train['LoanAmount'].values
LoanAmount_test = X_test['LoanAmount'].values
#Create a KNNImputer object, specifying the number of nearest neighbors
imp = KNNImputer(n_neighbors=3)#, add_indicator=True)
#Learn the imputation value (here: mean) on the training set
imp.fit(LoanAmount_train.reshape(-1, 1))
#Impute missing training data with training set mean value:
LoanAmount_train = imp.transform(LoanAmount_train.reshape(-1, 1))
#Impute missing test set data with training set mean value:
LoanAmount_test = imp.transform(LoanAmount_test.reshape(-1, 1))


#MICE (Multiple Imputation by Chained Equations) 
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
#Create numpy arrays with the values of feature 'LoanAmount' on the train & test set
LoanAmount_train = X_train['LoanAmount'].values
LoanAmount_test = X_test['LoanAmount'].values
#Create a KNNImputer object, specifying the number of nearest neighbors
imp = IterativeImputer()#, add_indicator=True)
#Learn the imputation value (here: mean) on the training set
imp.fit(LoanAmount_train.reshape(-1, 1))
#Impute missing training data with training set mean value:
LoanAmount_train = imp.transform(LoanAmount_train.reshape(-1, 1))
#Impute missing test set data with training set mean value:
LoanAmount_test = imp.transform(LoanAmount_test.reshape(-1, 1))


# display join distribution tables
# Joint distribution
data.groupby(['HomePlanet','Destination','Solo','Cabin_deck'])['Cabin_deck'].size().unstack().fillna(0)

# is cabin number proportional to group number and can na be filled with regression?
# Extrapolate linear relationship on a deck by deck basis
for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    # Features and labels
    X_CN=data.loc[~(data['Cabin_number'].isna()) & (data['Cabin_deck']==deck),'Group']
    y_CN=data.loc[~(data['Cabin_number'].isna()) & (data['Cabin_deck']==deck),'Cabin_number']
    X_test_CN=data.loc[(data['Cabin_number'].isna()) & (data['Cabin_deck']==deck),'Group']
    # Linear regression
    model_CN=LinearRegression()
    model_CN.fit(X_CN.values.reshape(-1, 1), y_CN)
    preds_CN=model_CN.predict(X_test_CN.values.reshape(-1, 1))
    # Fill missing values with predictions
    data.loc[(data['Cabin_number'].isna()) & (data['Cabin_deck']==deck),'Cabin_number']=preds_CN.astype(int)


# consider adding a column that tracks if an imputation was made


# before imputing combine train and test
# Labels and features
y=train['Target'].copy().astype(int)
X=train.drop('Target', axis=1).copy()
# Concatenate dataframes
data=pd.concat([X, test], axis=0).reset_index(drop=True)
# can then split back up using:
X=data[data['PassengerId'].isin(train['PassengerId'].values)].copy()
X_test=data[data['PassengerId'].isin(test['PassengerId'].values)].copy()


# good way to describe data stats
# Check basic statistics for numerical columns
numerical_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
# Calculate basic statistics
statistics = data[numerical_columns].describe().transpose()
# Remove the "count" row from the statistics table
statistics = statistics.drop('count', axis=1)
# Plot the statistics as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(statistics, annot=True, cmap='YlGnBu', fmt=".2f", cbar=False)
plt.title("Basic Statistics Heatmap")
plt.xlabel("Statistics")
plt.ylabel("Numerical Columns")
plt.xticks(rotation=45)
plt.show()


# cool looking radar chart
# Radar Chart: Comparison of average values of different variables for each failure type.
avg_values = data.groupby('Machine failure')[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']].mean().reset_index()
labels = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, avg_values.iloc[0, 1:].tolist() + avg_values.iloc[0, 1:2].tolist(), alpha=0.25, label='No Failure')
ax.fill(angles, avg_values.iloc[1, 1:].tolist() + avg_values.iloc[1, 1:2].tolist(), alpha=0.25, label='Failure')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylim(0, 100)
ax.set_title('Radar Chart: Comparison of average values for each failure type')

ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()


# consider IDs, especially if they are composite
# can something till be gotten out of it
# for example number of people in a group
# or get family size from second names 


# check if any columns contain bools and cast to binary:
dataset_df['bool_col'] = dataset_df['bool_col'].astype(int)


# check cardinality of cat
train.nunique()
# consider encoding; one-hot or cont (apply one hot after all the imputations)
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='if_binary', 
	handle_unknown='ignore',sparse=False))])
ct = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols)],
        remainder='passthrough')
# Apply preprocessing
X = ct.fit_transform(X)
X_test = ct.transform(X_test)
# there's also dummy encoding when one hot creates too many cols 



# check if any cats can be split up like title+name, address, cabin number
dataset_df[["Deck", "Cabin_num", "Side"]] = dataset_df["Cabin"].str.split("/", expand=True) # splits on /
# original example: B/0/P


# correlation heatmaps
sns.set(style="white")
# Compute the correlation matrix
corr = train.corr()
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
# consider dropping or combining high correlation features


# corr matrix
correlation_matrix = processed_df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title("Correlation Matrix Heatmap")
plt.show()


# or just correlation with the target
corr = processed_df.corr()
target_corr = corr['Machine failure'].drop('Machine failure')
# Sort correlation values in descending order
target_corr_sorted = target_corr.sort_values(ascending=False)
# Create a heatmap of the correlations with the target column
sns.set(font_scale=0.8)
sns.set_style("white")
sns.set_palette("PuBuGn_d")
sns.heatmap(target_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
plt.title('Correlation with Medication cost')
plt.show()


# cat plots
# Categorical features
cat_feats=['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
# Plot categorical features
fig=plt.figure(figsize=(10,16))
for i, var_name in enumerate(cat_feats):
    ax=fig.add_subplot(4,1,i+1)
    sns.countplot(data=train, x=var_name, axes=ax, hue='Transported')
    ax.set_title(var_name)
fig.tight_layout()  # Improves appearance a bit
plt.show()

# consider dropping if target split 50/50 for a feature (to reduce overfitting)


# can also do violin plots for cats to compare with target feature
original['Sex'] = pd.Categorical(original['Sex'], categories = ['I', 'M', 'F'], ordered = True)
fig, axes = plt.subplots(1, 2, figsize = (15, 6))
sns.boxplot(ax = axes[0], data = train, x = 'Sex', y = 'Age').set_title('Competition Dataset')
sns.boxplot(ax = axes[1], data = original, x = 'Sex', y = 'Age').set_title('Original Dataset');


# continious features; should plot
# Figure size
plt.figure(figsize=(10,4))
# Histogram
sns.histplot(data=train, x='Age', hue='Transported', binwidth=1, kde=True)
# Aesthetics
plt.title('Age distribution')
plt.xlabel('Age (years)')


# or another set of plots
# Expenditure features
exp_feats=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# Plot expenditure features
fig=plt.figure(figsize=(10,20))
for i, var_name in enumerate(exp_feats):
    # Left plot
    ax=fig.add_subplot(5,2,2*i+1)
    sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=False, hue='Transported')
    ax.set_title(var_name)
    # Right plot (truncated)
    ax=fig.add_subplot(5,2,2*i+2)
    sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
    plt.ylim([0,100])
    ax.set_title(var_name)
fig.tight_layout()  # Improves appearance a bit
plt.show()



# scatterplots of target and some feature
fig, axes = plt.subplots(1, 2, figsize = (15, 6))
sns.scatterplot(ax = axes[0], data = train, x = 'Shell Weight', y = 'Age', color = 'steelblue').set_title('Competition Dataset')
sns.scatterplot(ax = axes[1], data = original, x = 'Shell Weight', y = 'Age', color = 'orange').set_title('Original Dataset');


# here think about feature creation based on any discrepancies between the two target classes
# i.e. are certain ages produce different kde? Create a new feature that indicates whether 
# the passanger is a child, adolescent or adult. 
# multiple features can be combined, e.g. money spent on food, shopping -> tot money spent
# create a binary feature if there is a lot of zeros in one col, e.g. if a person spent nothing (0)
# or something (1) instead of lsiting expenditures


# consider transforms, like log transform if a distribution is exponential to reduce skew

# or scale:
# Scale numerical data to have mean=0 and variance=1
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
ct = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols)],
        remainder='passthrough')
# Apply preprocessing
X = ct.fit_transform(X)
X_test = ct.transform(X_test)

# feature engineering

# interaction variables: think domain knowledge, physics equations, etc
# basically a new column that combines somehow two or more cols


# generate a simple forest and get+plot feature importance
X_train = train.drop(['id', 'target'], axis=1)
y_train = train['target']

feat_labels = X_train.columns

rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

rf.fit(X_train, y_train)
importances = rf.feature_importances_

indices = np.argsort(rf.feature_importances_)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))


# consider dropping non-important features
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(rf, threshold='median', prefit=True)
print('Number of features before selection: {}'.format(X_train.shape[1]))
n_features = sfm.transform(X_train).shape[1]
print('Number of features after selection: {}'.format(n_features))
selected_vars = list(feat_labels[sfm.get_support()])



# consider applying scaling/normalisation before classification
# doesn't matter for RF
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(train.drop(['target'], axis=1))



# PCA want about 99% of var explained by ~15 components
from sklearn.decomposition import PCA
pca = PCA(n_components=20, svd_solver='full', random_state=1001)
X_pca = pca.fit_transform(X)
print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())
print('Individual variance contributions:')
for j in range(n_comp):
    print(pca.explained_variance_ratio_[j])

# or tSNE


# if good accuracy check for data leaks


# if cats consider
from catboost import CatBoostClassifier
# or 
import lightgbm as lgb #which will also do one hot encodings automatically
# slap xgboost
# NB defaults
XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train_sub, y_train)
y_pred = model.predict(X_test_sub)


# for regression
xgb_md = XGBRegressor(objective = 'reg:pseudohubererror',
                          tree_method = 'gpu_hist',
                          colsample_bytree = 0.9, 
                          gamma = 0.65, 
                          learning_rate = 0.01, 
                          max_depth = 7, 
                          min_child_weight = 20, 
                          n_estimators = 1000, 
                          subsample = 0.7).fit(X_train, Y_train)



# or xgboost parameter grid search
# with cross validation
from sklearn.model_selection import GridSearchCV
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 500, 1000]
}
xgb_model = xgb.XGBClassifier()
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train_sub, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test_sub, y_test)
print("Best Parameters: ", best_params)
print("Best Score: ", best_score)
print("Test Accuracy: ", test_accuracy)



# random grid search is better for hyperparameter search
# can also look into Bayesian optimization:
# + reliable results quicker
# - not parallelizable



# Bayesian optimization for logistic regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
from timeit import default_timer as timer
# Define the learning algorithm
model = LogisticRegression()
# Define hyperparameter search space
space = dict()
#space['solver'] = ['newton-cg', 'lbfgs']
#space['penalty'] = ['l1', 'l2', 'elasticnet']
space['C'] = (1e-6, 100.0, 'log-uniform')
# Define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# Define hyperparameter search
search = BayesSearchCV(estimator=model, search_spaces=space, n_jobs=-1, cv=cv)
# Execute hyperparameter search -- keep track of time
# Note: we use the training data for hyperparameter optimization
start_search = timer()
result = search.fit(X_train, y_train) 
end_search = timer()
# Report the best result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
print('Hyperparameter optimization time (sec): '+str(end_search - start_search))




# post result analysis

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()


# roc/auc
# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc_score), color='b')
plt.plot([0, 1], [0, 1], 'k--', color='r', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# Calculate Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve (AP = {:.2f})'.format(average_precision), color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()


#LIME/SHAP