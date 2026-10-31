# ============================================================
# PROJECT: EDUCATIONAL INEQUALITY IN INDIA (2015-16)
# Dataset: District-wise education data
# Algorithms:
# 1. Linear Regression
# 2. Decision Tree Regression
# 3. Random Forest Regression
# 4. Logistic Regression
# 5. Decision Tree Classifier
# 6. Random Forest Classifier
# 7. K-Means Clustering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("2015_16_Districtwise.csv")

columns = [
    "STATNAME", "DISTNAME", "TOTPOPULAT", "P_URB_POP",
    "GROWTHRATE", "SEXRATIO", "P_SC_POP", "P_ST_POP",
    "AREA_SQKM", "OVERALL_LI", "FEMALE_LIT", "MALE_LIT",
    "SCHTOT", "SCHGIRTOT", "SCHBOYTOT",
    "ENRTOT", "ENRTOTG", "ENRTOTM",
    "TCHTOT", "TCHTOTG", "TCHTOTM",
    "SWATTOT", "SGTOILTOT", "SBTOILTOT",
    "SELETOT", "SCOMPTOT"
]

df = df[columns].copy()

print("First five rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ------------------------------------------------------------
# 2. DATA CLEANING AND FEATURE ENGINEERING
# ------------------------------------------------------------

df = df.drop_duplicates()

# Gender literacy gap: male literacy minus female literacy
df["LITERACY_GAP"] = df["MALE_LIT"] - df["FEMALE_LIT"]
df = df.dropna(subset=["LITERACY_GAP"]).copy()

# Create additional education indicators
df["FEMALE_ENROLMENT_SHARE"] = (
    df["ENRTOTG"] / (df["ENRTOTG"] + df["ENRTOTM"]).replace(0, np.nan)
)

df["SCHOOLS_PER_1000_POP"] = (
    df["SCHTOT"] / df["TOTPOPULAT"].replace(0, np.nan) * 1000
)

print("\nRows after cleaning:", len(df))
print("Average gender literacy gap:", round(df["LITERACY_GAP"].mean(), 2))

# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(df["LITERACY_GAP"], bins=20, edgecolor="black")
plt.title("Distribution of Gender Literacy Gap")
plt.xlabel("Male Literacy - Female Literacy")
plt.ylabel("Number of Districts")
plt.tight_layout()
plt.show()

state_gap = df.groupby("STATNAME")["LITERACY_GAP"].mean().sort_values()

plt.figure(figsize=(10, 7))
state_gap.plot(kind="barh")
plt.title("Average Gender Literacy Gap by State/UT")
plt.xlabel("Average Literacy Gap")
plt.ylabel("State/UT")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["ENRTOTG"], df["ENRTOTM"], alpha=0.6)
plt.title("Female vs Male Student Enrolment")
plt.xlabel("Female Enrolment")
plt.ylabel("Male Enrolment")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["SEXRATIO"], df["LITERACY_GAP"], alpha=0.6)
plt.title("Sex Ratio vs Gender Literacy Gap")
plt.xlabel("Sex Ratio")
plt.ylabel("Gender Literacy Gap")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. MODEL 1: LINEAR REGRESSION
# Predict the gender literacy gap
# ------------------------------------------------------------

print("\n========== MODEL 1: LINEAR REGRESSION ==========")

features = [
    "TOTPOPULAT", "P_URB_POP", "GROWTHRATE", "SEXRATIO",
    "P_SC_POP", "P_ST_POP", "AREA_SQKM",
    "SCHTOT", "SCHGIRTOT", "SCHBOYTOT",
    "ENRTOT", "ENRTOTG", "ENRTOTM",
    "TCHTOT", "TCHTOTG", "TCHTOTM",
    "SWATTOT", "SGTOILTOT", "SBTOILTOT",
    "SELETOT", "SCOMPTOT"
]

X = df[features].apply(pd.to_numeric, errors="coerce")
y = df["LITERACY_GAP"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, linear_pred))
print("MSE:", mean_squared_error(y_test, linear_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, linear_pred)))
print("R2 Score:", r2_score(y_test, linear_pred))

plt.figure(figsize=(8, 5))
plt.scatter(y_test, linear_pred, alpha=0.6)
plt.xlabel("Actual Literacy Gap")
plt.ylabel("Predicted Literacy Gap")
plt.title("Linear Regression - Actual vs Predicted")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. MODEL 2: DECISION TREE REGRESSION
# Predict the gender literacy gap
# ------------------------------------------------------------

print("\\n========== MODEL 2: DECISION TREE REGRESSION ==========")

tree_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_regressor.fit(X_train, y_train)
tree_reg_pred = tree_regressor.predict(X_test)

print("MAE:", mean_absolute_error(y_test, tree_reg_pred))
print("MSE:", mean_squared_error(y_test, tree_reg_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, tree_reg_pred)))
print("R2 Score:", r2_score(y_test, tree_reg_pred))

plt.figure(figsize=(8, 5))
plt.scatter(y_test, tree_reg_pred, alpha=0.6)
plt.xlabel("Actual Literacy Gap")
plt.ylabel("Predicted Literacy Gap")
plt.title("Decision Tree Regression - Actual vs Predicted")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. MODEL 3: RANDOM FOREST REGRESSION
# Predict the gender literacy gap
# ------------------------------------------------------------

print("\n========== MODEL 3: RANDOM FOREST REGRESSION ==========")

rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=8, random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, rf_pred))
print("MSE:", mean_squared_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R2 Score:", r2_score(y_test, rf_pred))

plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_pred, alpha=0.6)
plt.xlabel("Actual Literacy Gap")
plt.ylabel("Predicted Literacy Gap")
plt.title("Random Forest - Actual vs Predicted")
plt.tight_layout()
plt.show()

importance = pd.Series(rf_model.feature_importances_, index=features)
importance = importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
importance.sort_values().plot(kind="barh")
plt.title("Top 10 Features in Random Forest Regression")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. REGRESSION MODEL COMPARISON
# Select the best model using R2, MAE and RMSE

regression_results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree Regression", "Random Forest Regression"],
    "R2 Score": [
        r2_score(y_test, linear_pred),
        r2_score(y_test, tree_reg_pred),
        r2_score(y_test, rf_pred)
    ],
    "MAE": [
        mean_absolute_error(y_test, linear_pred),
        mean_absolute_error(y_test, tree_reg_pred),
        mean_absolute_error(y_test, rf_pred)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, linear_pred)),
        np.sqrt(mean_squared_error(y_test, tree_reg_pred)),
        np.sqrt(mean_squared_error(y_test, rf_pred))
    ]
})

print("\\n========== REGRESSION MODEL COMPARISON ==========")
print(regression_results.round(4).to_string(index=False))

best_model = regression_results.sort_values(
    by=["R2 Score", "RMSE", "MAE"],
    ascending=[False, True, True]
).iloc[0]

print("\\nBest-fit regression model:", best_model["Model"])
print("A higher R2 and lower MAE/RMSE indicate better predictive performance.")

plt.figure(figsize=(8, 5))
plt.bar(regression_results["Model"], regression_results["R2 Score"])
plt.title("Regression Model Comparison - R2 Score")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 8. MODEL 4: LOGISTIC REGRESSION

# Classify districts by literacy level
# ------------------------------------------------------------

print("\n========== MODEL 4: LOGISTIC REGRESSION ==========")

literacy_data = df.dropna(subset=["OVERALL_LI"]).copy()
low_limit = literacy_data["OVERALL_LI"].quantile(1/3)
high_limit = literacy_data["OVERALL_LI"].quantile(2/3)

literacy_data["Literacy_Class"] = pd.cut(
    literacy_data["OVERALL_LI"],
    bins=[-np.inf, low_limit, high_limit, np.inf],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

literacy_features = [
    "TOTPOPULAT", "P_URB_POP", "GROWTHRATE", "SEXRATIO",
    "P_SC_POP", "P_ST_POP", "AREA_SQKM",
    "SCHTOT", "ENRTOT", "TCHTOT", "SWATTOT", "SELETOT"
]

X = literacy_data[literacy_features].apply(pd.to_numeric, errors="coerce")
y = literacy_data["Literacy_Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=2000, class_weight="balanced")
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, log_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_pred, labels=["Low", "Medium", "High"]))
print(classification_report(y_test, log_pred, zero_division=0))

# ------------------------------------------------------------
# 9. MODEL 5: DECISION TREE CLASSIFIER
# Classify female enrolment share
# ------------------------------------------------------------

print("\n========== MODEL 5: DECISION TREE CLASSIFIER ==========")

enrol_data = df.dropna(subset=["FEMALE_ENROLMENT_SHARE"]).copy()
q1 = enrol_data["FEMALE_ENROLMENT_SHARE"].quantile(1/3)
q2 = enrol_data["FEMALE_ENROLMENT_SHARE"].quantile(2/3)

enrol_data["Enrolment_Class"] = pd.cut(
    enrol_data["FEMALE_ENROLMENT_SHARE"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

enrol_features = [
    "TOTPOPULAT", "P_URB_POP", "GROWTHRATE", "SEXRATIO",
    "P_SC_POP", "P_ST_POP", "AREA_SQKM", "SCHTOT",
    "SCHGIRTOT", "SCHBOYTOT", "TCHTOT", "TCHTOTG",
    "TCHTOTM", "SWATTOT", "SGTOILTOT", "SBTOILTOT"
]

X = enrol_data[enrol_features].apply(pd.to_numeric, errors="coerce")
y = enrol_data["Enrolment_Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

tree_model = DecisionTreeClassifier(
    criterion="entropy", max_depth=5, random_state=42, class_weight="balanced"
)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, tree_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, tree_pred, labels=["Low", "Medium", "High"]))
print(classification_report(y_test, tree_pred, zero_division=0))

# ------------------------------------------------------------
# 10. MODEL 6: RANDOM FOREST CLASSIFIER
# Classify school availability per 1,000 population
# ------------------------------------------------------------

print("\n========== MODEL 6: RANDOM FOREST CLASSIFIER ==========")

school_data = df.dropna(subset=["SCHOOLS_PER_1000_POP"]).copy()
q1 = school_data["SCHOOLS_PER_1000_POP"].quantile(1/3)
q2 = school_data["SCHOOLS_PER_1000_POP"].quantile(2/3)

school_data["School_Class"] = pd.cut(
    school_data["SCHOOLS_PER_1000_POP"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

school_features = [
    "P_URB_POP", "GROWTHRATE", "SEXRATIO", "P_SC_POP",
    "P_ST_POP", "AREA_SQKM", "SCHGIRTOT", "SCHBOYTOT",
    "ENRTOT", "ENRTOTG", "ENRTOTM", "TCHTOT",
    "TCHTOTG", "TCHTOTM", "SWATTOT", "SGTOILTOT",
    "SBTOILTOT", "SELETOT", "SCOMPTOT"
]

X = school_data[school_features].apply(pd.to_numeric, errors="coerce")
y = school_data["School_Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

rf_classifier = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
)
rf_classifier.fit(X_train, y_train)
school_pred = rf_classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, school_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, school_pred, labels=["Low", "Medium", "High"]))
print(classification_report(y_test, school_pred, zero_division=0))

# ------------------------------------------------------------
# 11. MODEL 7: K-MEANS CLUSTERING
# Group districts with similar education indicators
# ------------------------------------------------------------

print("\n========== MODEL 7: K-MEANS CLUSTERING ==========")

cluster_features = [
    "LITERACY_GAP", "ENRTOTG", "ENRTOTM",
    "TCHTOTG", "TCHTOTM", "SCHGIRTOT", "SCHBOYTOT",
    "SWATTOT", "SGTOILTOT", "SBTOILTOT", "SELETOT", "SCOMPTOT"
]

X = df[cluster_features].apply(pd.to_numeric, errors="coerce")

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

print("Districts in each cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nAverage values in each cluster:")
print(df.groupby("Cluster")[cluster_features].mean())

plt.figure(figsize=(8, 5))
plt.scatter(
    df["ENRTOTG"], df["ENRTOTM"],
    c=df["Cluster"], cmap="Pastel1", alpha=0.7
)
plt.xlabel("Female Enrolment")
plt.ylabel("Male Enrolment")
plt.title("K-Means - Education District Clusters")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("\n========== PROJECT SUMMARY ==========")
print("Regression models: Linear Regression, Decision Tree Regression and Random Forest Regression")
print("Best-fit regression model:", best_model["Model"])
print("Classification models: Logistic Regression, Decision Tree and Random Forest")
print("Clustering model: K-Means")
print("A positive literacy gap means male literacy is higher.")
print("Feature importance and model relationships do not prove causation.")
print("PROJECT COMPLETED SUCCESSFULLY.")
