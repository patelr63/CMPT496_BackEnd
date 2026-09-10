# random forest 
# need to clean and organize data as well 
# split/ check that data is split into correct feutures and targets (X = independent vars, Y = dependent/ target) 


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# load csv 
data = pd.read_csv('crypto_scam_transaction_dataset.csv'
)

# define features 
X = data.drop(columns=['is_scam'])
y = data['is_scam']

# Convert categorical columns into numeric values
X = pd.get_dummies(X, drop_first=True)

# Handle any remaining missing values
X = X.fillna(X.median(numeric_only=True))

# If categorical columns still have NaN after get_dummies
X = X.fillna(0)

#  Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# make random forest 
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

#  Train Random Forest
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# get prob that each transaction is a scam 
# need for ROC curve 
y_prob = rf_model.predict_proba(X_test)[:, 1]

#  Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# confusion matrix 

cm = confusion_matrix(y_test, y_pred) 
print("\nConfusion Matrix:")
print(cm) 

# get vals from matrix
'''
tn = true neg; legit = predicted legit 
fp = false pos; legit = predicted scam 
fn = false neg; scam = predicted legit
tp = true pos; scam = predicted scam 
'''
tn, fp, fn, tp = cm.ravel() 

print("\nTrue Neg:", tn) 
print("\nFalse Pos:", fp)
print("\nFalse Neg:", fn)
print("\nTrue Pos:", tp)

# plot confusion matrix as a heatmap 
plt.figure(figsize=(7,5)) 
sns.heatmap(
    cm, 
    annot = True,
    fmt = 'd',
    cmap = 'Blues',
    xticklabels=['Legit', 'Scam'],
    yticklabels=['Legit', 'Scam']
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Random Forest Confusion Matrix")

plt.show()

# ROC Curve 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob) # auc = area under curve 
print("\nROC AUC Score:")

# diag line = random guessing 
plt.plot(
    [0,1],
    [0,1],
    linestyle = '--',
    label = 'Random Classfier'
)

# label curve graph and set up 
plt.xlabel("False Pos Rate")
plt.ylabel("True Pos Rate")
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.grid(alpha=0.3)
plt.show() 

