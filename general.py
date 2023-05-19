"""
Some general strating steps
"""


.head()
.tail()
.shape
.info()

# check for dups
.drop_duplicates()
.shape


# check for class imbalance, can plot
plt.figure(figsize=(6,6))
# Pie plot
train['Target'].value_counts().plot.pie(explode=[0.1,0.1], autopct='%1.1f%%',
 shadow=True, textprops={'fontsize':16}).set_title("Target distribution")

# if yes, under/over-sampling or more advanced like smote (need to make some scatter plots like pca maybe to decide
# if appropriate)


# missing vals
train.isnull().values.any()
#or
print(train.isna().sum())

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


# if imputing missing features might be worth for example to split data into groups and then impute based
# on group's statistic. for example, for missing expenditure create age groups and fill expenditure
# based on age group's mean
# e.g. 
train['Age_group']=np.nan
train.loc[train['Age']<=12,'Age_group']='Age_0-12'
train.loc[(train['Age']>12) & (train['Age']<18),'Age_group']='Age_13-17'
#etc

# consider adding a column that tracks if an imputation was made


# consider IDs, especially if they are composite
# can something till be gotten out of it
# for example number of people in a group
# or get family size from second names 


# check if any columns contain bools and cast to binary:
dataset_df['bool_col'] = dataset_df['bool_col'].astype(int)


# check cardinality of cat
train.nunique()
# consider encoding; one-hot or cont



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



# here think about feature creation based on any discrepancies between the two target classes
# i.e. are certain ages produce different kde? Create a new feature that indicates whether 
# the passanger is a child, adolescent or adult. 
# multiple features can be combined, e.g. money spent on food, shopping -> tot money spent
# create a binary feature if there is a lot of zeros in one col, e.g. if a person spent nothing (0)
# or something (1) instead of lsiting expenditures


# consider transforms, like log transform if a distribution is exponential to reduce skew



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