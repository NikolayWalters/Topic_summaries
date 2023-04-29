"""
Kagle Machine Learning Explainability course notes
"""

# === Permutation importance ===
# basically randomly re-ordering a single column and re-computing
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

# out: feature weight for each feature

# === Partial Plots ===
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
# Create and plot the data
disp1 = PartialDependenceDisplay.from_estimator(model, val_X, ['Some col'])
plt.show()

# 2D plot
fig, ax = plt.subplots(figsize=(8, 6))
f_names = [('feature1', 'feature2')]
# Similar to previous PDP plot except we use tuple of features instead of single feature
disp4 = PartialDependenceDisplay.from_estimator(tree_model, val_X, f_names, ax=ax)
plt.show()

# === SHAP ===
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)
# or shap.DeepExplainer for Deep Learning models
# or KernelExplainer works with all models, though it is slower than other Explainers and only approximates

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

# Plot
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

# Example
# Use SHAP values to show the effect of each feature of a given patient

sample_data_for_prediction = val_X.iloc[0].astype(float)  # to test function

def patient_risk_factors(model, patient_data):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_data)

# SHAP summary plot for all features
# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(val_X)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[1], val_X)

# dependence contribution plot
# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
shap_values = explainer.shap_values(X)

# make plot.
shap.dependence_plot('Ball Possession %', shap_values[1], X, interaction_index="Goal Scored")