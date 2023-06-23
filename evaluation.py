"""
Notes on model evaluation (i.e. metrics)
Plus calibration
"""

# Simple metrics
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
#Max Error
reg_test_max_error = max_error(y_test, y_test_pred)

#Mean Absolute Error (MAE)
reg_test_mean_absolute_error = mean_absolute_error(y_test, y_test_pred)

#Mean Squared Error (MSE)
reg_test_mean_squared_error = mean_squared_error(y_test, y_test_pred)

#Mean Squared Log Error
reg_test_mean_squared_log_error = mean_squared_log_error(y_test, y_test_pred)

#Median Absolute Error
reg_test_median_absolute_error = median_absolute_error(y_test, y_test_pred)

#Mean Absolute Percentage Error (MAPE) #Note: Actually a fraction! Need to convert to %
reg_test_mean_absolute_percentage_error = mean_absolute_percentage_error(y_test, 
                                                                         y_test_pred)
reg_train_mean_absolute_percentage_error = mean_absolute_percentage_error(y_train, 
                                                                          y_train_pred)
print("Mean Absolute Percentage Error (MAPE) [Lower Better]: ")    
print("Training: "+str(100*reg_train_mean_absolute_percentage_error)+"% -- Test: "
      +str(100*reg_test_mean_absolute_percentage_error)+"%")

#Coefficient of Determination (R^2)
reg_test_r2_score = r2_score(y_test, y_test_pred)


# classification metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
#Accuracy
clf_test_accuracy = accuracy_score(y_test, y_test_pred)

#Balanced Accuracy
clf_test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)

#Precision
clf_test_precision = precision_score(y_test, y_test_pred)

#Recall
clf_test_recall = recall_score(y_test, y_test_pred)

#F1-measure
clf_test_f1_score = f1_score(y_test, y_test_pred)

#Cohen's Kappa
clf_test_cohen_kappa = cohen_kappa_score(y_test, y_test_pred)

#Matthews Correlation Coefficient
clf_test_matthews_corrcoef = matthews_corrcoef(y_test, y_test_pred)

#ROC AUC
clf_test_roc_auc = roc_auc_score(y_test, y_test_pred)


# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_test_pred)
# can just print the output
# or to get individual
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
print('TPs:'+str(tp))
print('TNs:'+str(tn))
print('FPs:'+str(fp))
print('FNs:'+str(fn))
print('Accuracy='+str((tn+tp)/(tn+tp+fn+fp)))


# ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
#Get the predicted scores (probability estimates) of the model on the test data
#VERY IMPORTANT: USE SCORES -i.e. .predict_proba(),
#DON'T USE PREDICTED CLASSES -i.e. .predict()!
y_test_scores = clf.predict_proba(X_test)
#Compute TPR & FPR value for each possible score threshold:
#Remember to use scores for positive class (class 1)
fpr, tpr, _ = roc_curve(y_test, y_test_scores[:,1])#CORRECT WAY
#fpr, tpr, _ = roc_curve(y_test, y_test_pred)#LESS CORRECT WAY

#Based on the above, calculate AUC:
roc_auc = auc(fpr, tpr)
#Now plot TPR vs FPR value for each possible score threshold (i.e. ROC):
plt.figure(figsize = (12,9))
plt.plot( fpr, tpr, color="darkorange",
         label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()



# Calibration
# Start by plotting model scores/probabilities

# then calibration plots
from sklearn.calibration import calibration_curve

def plot_calibration_curve(name, fig_index, probs, color = 'b'):
    """Plot calibration curve for model w/o and with calibration. """

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    frac_of_pos, mean_pred_value = calibration_curve(y_test, probs, n_bins=10)

    ax1.plot(mean_pred_value, frac_of_pos, "s-", label=f'{name}', color = color)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration plot ({name})')
    
    ax2.hist(probs, range=(0, 1), bins=10, label=name, histtype="step", lw=2, color = color)
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")

plot_calibration_curve("Logistic regression", 1, probs_logreg)


#Calibration
from sklearn.calibration import CalibratedClassifierCV

logreg = LogisticRegression(C=1, solver='lbfgs')
svm = SVC(max_iter=10000, probability=False) # Set to True for Platt
rf = RandomForestClassifier(max_depth=10)

#Logistic Regression
calibrated_logreg = CalibratedClassifierCV(logreg, cv=5, method='sigmoid')
calibrated_logreg.fit(X_train, y_train)
probs_calibrated_logreg = calibrated_logreg.predict_proba(X_test)[:,1]

#SVM
calibrated_svm = CalibratedClassifierCV(svm, cv=5, method='sigmoid')
calibrated_svm.fit(X_train, y_train)
probs_calibrated_svm = calibrated_svm.predict_proba(X_test)[:,1]

#Random Forest
calibrated_rf = CalibratedClassifierCV(rf, cv=5, method='sigmoid')
calibrated_rf.fit(X_train, y_train)
probs_calibrated_rf = calibrated_rf.predict_proba(X_test)[:,1]
