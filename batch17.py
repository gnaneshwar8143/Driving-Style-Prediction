# %% 
# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# %%
# Read the dataset
dataset = pd.read_csv(r"C:\Users\rendl\OneDrive\Desktop\RTP\creditcard.csv")

# Read the first 5 and last 5 rows of the data
pd.concat([dataset.head(), dataset.tail()])

# %%
# Check for relative proportion
print("Fraudulent Cases: " + str(len(dataset[dataset["Class"] == 1])))
print("Valid Transactions: " + str(len(dataset[dataset["Class"] == 0])))
print("Proportion of Fraudulent Cases: " + str(len(dataset[dataset["Class"] == 1]) / dataset.shape[0]))

# To see how small are the number of Fraud transactions
data_p = dataset.copy()
data_p["Label"] = np.where(data_p["Class"] == 1, "Fraud", "Genuine")

# Plot a pie chart
data_p["Label"].value_counts().plot(kind="pie", autopct='%1.1f%%')
plt.ylabel('')  # Cleaner look
plt.title("Fraud vs Genuine Transactions")
plt.show()

# %%
# Plot the named features
f, axes = plt.subplots(1, 2, figsize=(18, 4), sharex=True)

amount_value = dataset['Amount'].values
time_value = dataset['Time'].values

sns.kdeplot(amount_value, shade=True, color="m", ax=axes[0])
axes[0].set_title('Distribution of Amount')

sns.kdeplot(time_value, shade=True, color="m", ax=axes[1])
axes[1].set_title('Distribution of Time')

plt.show()

# %%
print("Average Amount in a Fraudulent Transaction: " + str(dataset[dataset["Class"] == 1]["Amount"].mean()))
print("Average Amount in a Valid Transaction: " + str(dataset[dataset["Class"] == 0]["Amount"].mean()))

# %%
print("Summary of the feature - Amount\n-------------------------------")
print(dataset["Amount"].describe())

# %%
# Reorder the columns Amount, Time then the rest
data_plot = dataset.copy()
amount = data_plot['Amount']
data_plot.drop(labels=['Amount'], axis=1, inplace=True)
data_plot.insert(0, 'Amount', amount)

# Plot the distributions of the features
columns = data_plot.iloc[:, 0:30].columns
plt.figure(figsize=(12, 30 * 4))
grids = gridspec.GridSpec(30, 1)
for grid, index in enumerate(data_plot[columns]):
    ax = plt.subplot(grids[grid])
    sns.kdeplot(data_plot[index][data_plot.Class == 1], shade=True, label='Fraud')
    sns.kdeplot(data_plot[index][data_plot.Class == 0], shade=True, label='Genuine')
    ax.set_title("Distribution of Column: " + str(index))
    ax.legend()
plt.show()

# %%
# Check for null values
print("Non-missing values: " + str(dataset.shape[0] - dataset.isnull().sum().sum()))
print("Missing values: " + str(dataset.isnull().sum().sum()))

# %%
# Scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler().fit(dataset[["Time", "Amount"]])
dataset[["Time", "Amount"]] = scaler.transform(dataset[["Time", "Amount"]])

pd.concat([dataset.head(), dataset.tail()])

# %%
# Separate response and features
y = dataset["Class"]
X = dataset.iloc[:, 0:30]

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %%
# Cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV

kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# %%
# Imbalanced-learn and metrics
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# %%
# Random Forest
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print("The accuracy is", accuracy_score(y_test, y_pred))
print("The precision is", precision_score(y_test, y_pred))
print("The recall is", recall_score(y_test, y_pred))
print("The F1 score is", f1_score(y_test, y_pred))

# %%
# Function to get best estimator and metrics
def get_model_best_estimator_and_metrics(estimator, params, kf=kf, X_train=X_train,
                                         y_train=y_train, X_test=X_test,
                                         y_test=y_test, is_grid_search=True,
                                         sampling=None, scoring="f1",
                                         n_jobs=2):
    if sampling is None:
        pipeline = make_pipeline(estimator)
    else:
        pipeline = make_pipeline(sampling, estimator)
    estimator_name = estimator.__class__.__name__.lower()
    new_params = {f'{estimator_name}__{key}': params[key] for key in params}
    if is_grid_search:
        search = GridSearchCV(pipeline, param_grid=new_params, cv=kf, scoring=scoring, return_train_score=True, n_jobs=n_jobs, verbose=1)
    else:
        search = RandomizedSearchCV(pipeline, param_distributions=new_params,
                                    cv=kf, scoring=scoring, return_train_score=True,
                                    n_jobs=n_jobs, verbose=1)
    search.fit(X_train, y_train)
    cv_score = cross_val_score(search, X_train, y_train, scoring=scoring, cv=kf)
    y_pred = search.best_estimator_.predict(X_test)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    return {
        "best_estimator": search.best_estimator_,
        "estimator_name": estimator_name,
        "cv_score": cv_score,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
    }

# %%
# ROC table
res_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

# Random Forest Example
rfc_results = get_model_best_estimator_and_metrics(
    estimator=RandomForestClassifier(),
    params={
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 6, 10, 12],
        'random_state': [13]
    },
    sampling=None,
    n_jobs=3,
)

# Append results
res_table = pd.concat([
    res_table,
    pd.DataFrame([{'classifiers': rfc_results["estimator_name"],
                   'fpr': rfc_results["fpr"],
                   'tpr': rfc_results["tpr"],
                   'auc': rfc_results["auc"]}])
], ignore_index=True)

# Print metrics
print(f"==={rfc_results['estimator_name']}===")
print("Model:", rfc_results['best_estimator'])
print("Accuracy:", rfc_results['accuracy'])
print("Recall:", rfc_results['recall'])
print("F1 Score:", rfc_results['f1_score'])
