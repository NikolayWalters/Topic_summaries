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