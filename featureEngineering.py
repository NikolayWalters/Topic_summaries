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