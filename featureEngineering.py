"""
Summary of the course on feature engineering from kaggle
"""

# === Mutual information ===

from sklearn.feature_selection import mutual_info_regression

#Feature utility metric = a function measuring associations between a feature and the target

# The mutual information (MI) between two quantities is a measure of the extent to which 
# knowledge of one quantity reduces uncertainty about the other

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

# === Creating features ===

# math transforms

# think of interesting/useful formulas

# log scale data to disperse it; use np.log1p (log(x+1)) if there are 0s

# consider summing/counting features such as risk conditions for disaease or 
# roadway features for accidents instead of feeding columns with true/false; present or not
# e.g. 
roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)
# another example, count concrete ingredients
components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1) # gt = greater than

# break up and build down features, e.g. phone numbers/addresses/dates
# for area codes, towns/regions, day of week etc
# e.g. 
customer[["Type", "Level"]] = (  # Create two new features
    customer["Policy"]           # from the Policy feature
    .str                         # through the string accessor
    .split(" ", expand=True)     # by splitting on " "
                                 # and expanding the result into separate columns
)
#>> 	Policy			Type		Level
#>> 0	Corporate L3	Corporate	L3
#>> 1	Personal L3		Personal	L3
# or
X_4["MSClass"] = X["MSSubClass"].str.split("_", n=1, expand=True)[0]
# takes the first substring before _

# can sometimes replace categorical variable by something more meaningful, 
# e.g coverage type (basic, premium, ...) to average claim (num values)

#Ratios seem to be difficult for most models to learn. Ratio combinations often 
#lead to some easy performance gains.

#Counts are especially helpful for tree models, since these models don't 
#have a natural way of aggregating information across many features at once.

# Consider using unsupervised clustering and then using cluster number as
# an additional column (but one-hot encode clusters rather than giving 1,2,3,...)
# or even fit a different model to each cluster
# Create cluster feature
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

# === PCA ===
# standardize data before applying PCA
# Consider removing or constraining outliers, since they can have an undue influence on the results.

# can be used for anomaly detection
# Anomaly detection: Unusual variation, not apparent from the original features, will often show up 
# in the low-variance components. These components could be highly informative in an anomaly or outlier detection task.

# as well as decorrelation
# Decorrelation: Some ML algorithms struggle with highly-correlated features. PCA transforms correlated features 
# into uncorrelated components, which could be easier for your algorithm to work with.

features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]

X = df.copy()
y = X.pop('price')
X = X.loc[:, features]

# Standardize
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

from sklearn.decomposition import PCA

# Create principal components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

# and to show loadings
loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)

# Show dataframe sorted by PC3
idx = X_pca["PC3"].sort_values(ascending=False).index
cols = ["make", "body_style", "horsepower", "curb_weight"]
df.loc[idx, cols]

# Show outliers based on PCA components
component = "PC1"

idx = X_pca[component].sort_values(ascending=False).index
df.loc[idx, ["SalePrice", "Neighborhood", "SaleCondition"] + features]

# === Target encoding ===

# simple mean encoding
# out
# index make 		price 	make_encoded
#>> 1	alfa-romero	16500	15498.333333
#>> 2	alfa-romero	16500	15498.333333
#>> 3	audi		13950	17859.166667
#>> 4	audi		17450	17859.166667

# Smoothing: blend the in-category average with the overall average (protects against rare extremes)
encoding = weight * in_category + (1 - weight) * overall
# weight based on category frequency, e.g.:
weight = n / (n + m)
# where m is the smoothing factor

# Target encoding is great for:
# High-cardinality features: A feature with a large number of categories can be troublesome to encode: 
# a one-hot encoding would generate too many features and alternatives, like a label encoding, might 
# not be appropriate for that feature. A target encoding derives numbers for the categories using the 
# feature's most important property: its relationship with the target.

# Domain-motivated features: From prior experience, you might suspect that a categorical feature 
# should be important even if it scored poorly with a feature metric. A target encoding can help reveal 
# a feature's true informativeness.

# 25% train split to train the target encoder
X = df.copy()
y = X.pop('Rating')

X_encode = X.sample(frac=0.25)
y_encode = y[X_encode.index]
X_pretrain = X.drop(X_encode.index)
y_train = y[X_pretrain.index]

from category_encoders import MEstimateEncoder

# Create the encoder instance. Choose m to control noise.
encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)

# Fit the encoder on the encoding split.
encoder.fit(X_encode, y_encode)

# Encode the Zipcode column to create the final training data
X_train = encoder.transform(X_pretrain)

# usually good idea to encode high cardinality cats. can use this to check
df.select_dtypes(["object"]).nunique()

# to prevent overfitting need to split before encoding
# Encoding split
X_encode = df.sample(frac=0.20, random_state=0)
y_encode = X_encode.pop("SalePrice")

# Training split
X_pretrain = df.drop(X_encode.index)
y_train = X_pretrain.pop("SalePrice")



# notes on transformations
#1) Log Transformation: This transformation involves taking the logarithm of each data point. It is useful when the data is highly skewed and the variance increases with the mean.
#         y = log(x)
#2) Square Root Transformation: This transformation involves taking the square root of each data point. It is useful when the data is highly skewed and the variance increases with the mean.
#         y = sqrt(x)
#3) Box-Cox Transformation: This transformation is a family of power transformations that includes the log and square root transformations as special cases. It is useful when the data is highly skewed and the variance increases with the mean.
#         y = [(x^lambda) - 1] / lambda if lambda != 0
#         y = log(x) if lambda = 0
#4) Yeo-Johnson Transformation: This transformation is similar to the Box-Cox transformation, but it can be applied to both positive and negative values. It is useful when the data is highly skewed and the variance increases with the mean.
#         y = [(|x|^lambda) - 1] / lambda if x >= 0, lambda != 0
#         y = log(|x|) if x >= 0, lambda = 0
#         y = -[(|x|^lambda) - 1] / lambda if x < 0, lambda != 2
#         y = -log(|x|) if x < 0, lambda = 2
#5) Power Transformation: This transformation involves raising each data point to a power. It is useful when the data is highly skewed and the variance increases with the mean. The power can be any value, and is often determined using statistical methods such as the Box-Cox or Yeo-Johnson transformations.
#         y = [(x^lambda) - 1] / lambda if method = "box-cox" and lambda != 0
#         y = log(x) if method = "box-cox" and lambda = 0
#         y = [(x + 1)^lambda - 1] / lambda if method = "yeo-johnson" and x >= 0, lambda != 0
#         y = log(x + 1) if method = "yeo-johnson" and x >= 0, lambda = 0
#         y = [-(|x| + 1)^lambda - 1] / lambda if method = "yeo-johnson" and x < 0, lambda != 2
#         y = -log(|x| + 1) if method = "yeo-johnson" and x < 0, lambda = 2

# implementation
# get continious data cols
cont_cols=[f for f in train.columns if train[f].dtype!="O" and f not in ['Age','Height','original'] and train[f].nunique()>2]

sc=MinMaxScaler()
dt_params={'criterion': 'absolute_error'}
table = PrettyTable()
unimportant_features=[]
overall_best_score=100
overall_best_col='none'
table.field_names = ['Feature', 'Original MAE', 'Transformation', 'Tranformed MAE']
for col in cont_cols:
    
    # Log Transformation after MinMax Scaling(keeps data between 0 and 1)
    train["log_"+col]=np.log1p(sc.fit_transform(train[[col]]))
    test["log_"+col]=np.log1p(sc.transform(test[[col]]))
    
    # Square Root Transformation
    train["sqrt_"+col]=np.sqrt(sc.fit_transform(train[[col]]))
    test["sqrt_"+col]=np.sqrt(sc.transform(test[[col]]))
    
    # Box-Cox transformation
    combined_data = pd.concat([train[[col]], test[[col]]], axis=0)
    transformer = PowerTransformer(method='box-cox')
    # Apply scaling and transformation on the combined data
    scaled_data = sc.fit_transform(combined_data)+1
    transformed_data = transformer.fit_transform(scaled_data)

    # Assign the transformed values back to train and test data
    train["bx_cx_" + col] = transformed_data[:train.shape[0]]
    test["bx_cx_" + col] = transformed_data[train.shape[0]:]
    
    # Yeo-Johnson transformation
    transformer = PowerTransformer(method='yeo-johnson')
    train["y_J_"+col] = transformer.fit_transform(train[[col]])
    test["y_J_"+col] = transformer.transform(test[[col]])
    
    # Power transformation, 0.25
    power_transform = lambda x: np.power(x + 1 - np.min(x), 0.25)
    transformer = FunctionTransformer(power_transform)
    train["pow_"+col] = transformer.fit_transform(sc.fit_transform(train[[col]]))
    test["pow_"+col] = transformer.transform(sc.transform(test[[col]]))
    
    # Power transformation, 0.1
    power_transform = lambda x: np.power(x + 1 - np.min(x), 0.1)
    transformer = FunctionTransformer(power_transform)
    train["pow2_"+col] = transformer.fit_transform(sc.fit_transform(train[[col]]))
    test["pow2_"+col] = transformer.transform(sc.transform(test[[col]]))
    
    # log to power transformation
    train["log_sqrt"+col]=np.log1p(train["sqrt_"+col])
    test["log_sqrt"+col]=np.log1p(test["sqrt_"+col])
    
    temp_cols=[col,"log_"+col,"sqrt_"+col, "bx_cx_"+col,"y_J_"+col ,"pow_"+col,"pow2_"+col,"log_sqrt"+col ]
    
    # Fill na becaue, it would be Nan if the vaues are negative and a transformation applied on it
    train[temp_cols]=train[temp_cols].fillna(0)
    test[temp_cols]=test[temp_cols].fillna(0)

    #Apply PCA on  the features and compute an additional column
    pca=TruncatedSVD(n_components=1)
    x_pca_train=pca.fit_transform(train[temp_cols])
    x_pca_test=pca.transform(test[temp_cols])
    x_pca_train=pd.DataFrame(x_pca_train, columns=[col+"_pca_comb"])
    x_pca_test=pd.DataFrame(x_pca_test, columns=[col+"_pca_comb"])
    temp_cols.append(col+"_pca_comb")
    test=test.reset_index(drop=True) # to combine with pca feature
    
    train=pd.concat([train,x_pca_train],axis='columns')
    test=pd.concat([test,x_pca_test],axis='columns')
    
    # See which transformation along with the original is giving you the best univariate fit with target
    kf=KFold(n_splits=5, shuffle=True, random_state=42)
    
    MAE=[]
    
    for f in temp_cols:
        X=train[[f]].values
        y=train["Age"].values
        
        mae=[]
        for train_idx, val_idx in kf.split(X,y):
            X_train,y_train=X[train_idx],y[train_idx]
            x_val,y_val=X[val_idx],y[val_idx]
            
            model=LinearRegression()
            model.fit(X_train,y_train)
            y_pred=model.predict(x_val)
            mae.append(mean_absolute_error(y_val,y_pred))
        MAE.append((f,np.mean(mae)))
        if overall_best_score>np.mean(mae):
            overall_best_score=np.mean(mae)
            overall_best_col=f
        if f==col:
            orig_mae=np.mean(mae)
    best_col, best_acc=sorted(MAE, key=lambda x:x[1], reverse=False)[0]
    
    cols_to_drop = [f for f in temp_cols if  f!= best_col and f not in col]
    final_selection=[f for f in temp_cols if f not in cols_to_drop]
    if cols_to_drop:
        unimportant_features=unimportant_features+cols_to_drop
    table.add_row([col,orig_mae,best_col ,best_acc])
print(table)  
print("overall best CV score: ",overall_best_score)