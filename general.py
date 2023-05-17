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


# missing vals
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


# check cardinality of cat
# consider encoding; one-hot or cont


# correlation heatmaps

# plots


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