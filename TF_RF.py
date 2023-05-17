"""
Example implementation of a random forest via tensor flow
"""

import tensorflow as tf
import tensorflow_decision_forests as tfdf

# need to convert pandas df to TF Datasets
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label='Target_label_col')
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label='Target_label_col')

rf = tfdf.keras.RandomForestModel() 
rf.compile(metrics=["accuracy"]) # Optional, you can use this to include a list of eval metrics

rf.fit(x=train_ds)

# can plot branched out tree of a given index
tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)

# can plot metric vs number of trees
import matplotlib.pyplot as plt
logs = rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])

# print variable importance
for importance in inspector.variable_importances().keys():
    print("\t", importance)