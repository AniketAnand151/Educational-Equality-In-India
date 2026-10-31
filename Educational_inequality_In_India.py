import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("2015_16_Districtwise.csv")
columns = [
    "STATNAME", "DISTNAME", "TOTPOPULAT", "P_URB_POP", "GROWTHRATE",
    "SEXRATIO", "P_SC_POP", "P_ST_POP", "AREA_SQKM", "OVERALL_LI",
    "FEMALE_LIT", "MALE_LIT", "SCHTOT", "SCHGIRTOT", "SCHBOYTOT",
    "ENRTOT", "ENRTOTG", "ENRTOTM", "TCHTOT", "TCHTOTG", "TCHTOTM",
    "SWATTOT", "SGTOILTOT", "SBTOILTOT", "SELETOT", "SCOMPTOT"
]
df = df[columns].copy()
numeric_cols = df.select_dtypes(exclude="object").columns
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

print("First five rows:\n", df.head())
print("\nDataset shape:", df.shape)
print("\nDataset information:")
df.info()
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nNumerical summary:\n", df.describe().round(2))
print("\nNumber of states/UTs:", df["STATNAME"].nunique())
print("Number of districts:", df["DISTNAME"].nunique())
df = df.drop_duplicates()
df["LITERACY_GAP"] = df["MALE_LIT"] - df["FEMALE_LIT"]
df = df.dropna(subset=["LITERACY_GAP"]).copy()

# 6. FEATURE ENGINEERING
df["FEMALE_ENROLMENT_SHARE"] = (
    df["ENRTOTG"] / (df["ENRTOTG"] + df["ENRTOTM"]).replace(0, np.nan)
)
df["SCHOOLS_PER_1000_POP"] = (
    df["SCHTOT"] / df["TOTPOPULAT"].replace(0, np.nan) * 1000
)
print("\nRows after cleaning:", len(df))
print("Average gender literacy gap:",
      round(df["LITERACY_GAP"].mean(), 2), "percentage points")

# 7. PROJECT OBJECTIVES
objectives = [
    "Analyse the distribution of gender literacy gap across districts.",
    "Compare average gender literacy gaps across states and union territories.",
    "Examine male and female student enrolment patterns.",
    "Investigate the relationship between sex ratio and gender literacy gap.",
    "Predict gender literacy gap using Linear Regression and Random Forest Regression.",
    "Classify districts by literacy level using Logistic Regression.",
    "Classify districts by female enrolment share using a Decision Tree Classifier.",
    "Group similar districts using K-Means clustering."
]
print("\nPROJECT OBJECTIVES")
for i, item in enumerate(objectives, 1):
    print(f"{i}. {item}")

# 8. DETAILED EDA AND VISUALIZATIONS
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

# HELPER FUNCTIONS (defined after loading, cleaning and EDA)
def fill_missing(train, test):
    medians = train.median()
    return train.fillna(medians), test.fillna(medians)

def regression_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mean_squared_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2 Score": r2_score(actual, predicted)
    }

def actual_vs_predicted(actual, predicted, title):
    plt.figure(figsize=(7, 5))
    plt.scatter(actual, predicted, alpha=0.6)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    plt.plot(limits, limits, "r--")
    plt.xlabel("Actual Literacy Gap")
    plt.ylabel("Predicted Literacy Gap")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def classification_results(actual, predicted, title):
    labels = ["Low", "Medium", "High"]
    matrix = confusion_matrix(actual, predicted, labels=labels)
    print("\n", title)
    print("Accuracy:", round(accuracy_score(actual, predicted), 4))
    print("Confusion Matrix:\n", matrix)
    print(classification_report(actual, predicted, labels=labels,
                                zero_division=0))

    # Confusion matrix graph is retained for each classifier.
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Pastel1")
    plt.title(title + " - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(3), labels)
    plt.yticks(range(3), labels)
    for row in range(3):
        for col in range(3):
            plt.text(col, row, matrix[row, col], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# 9. APPLY THE 5 ML ALGORITHMS

# MODEL 1: LINEAR REGRESSION
features = [
    "TOTPOPULAT", "P_URB_POP", "GROWTHRATE", "SEXRATIO", "P_SC_POP",
    "P_ST_POP", "AREA_SQKM", "SCHTOT", "SCHGIRTOT", "SCHBOYTOT",
    "ENRTOT", "ENRTOTG", "ENRTOTM", "TCHTOT", "TCHTOTG", "TCHTOTM",
    "SWATTOT", "SGTOILTOT", "SBTOILTOT", "SELETOT", "SCOMPTOT"
]
X, y = df[features], df["LITERACY_GAP"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_test = fill_missing(X_train, X_test)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_model = LinearRegression().fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_metrics = regression_metrics(y_test, linear_pred)
print("\nLINEAR REGRESSION:", linear_metrics)
actual_vs_predicted(y_test, linear_pred, "Linear Regression - Actual vs Predicted")

# MODEL 2: RANDOM FOREST REGRESSION
rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=8, random_state=42
).fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_metrics = regression_metrics(y_test, rf_pred)
print("\nRANDOM FOREST REGRESSION:", rf_metrics)
actual_vs_predicted(y_test, rf_pred, "Random Forest - Actual vs Predicted")

importance = pd.Series(rf_model.feature_importances_, index=features)
plt.figure(figsize=(8, 5))
importance.nlargest(10).sort_values().plot(kind="barh")
plt.title("Top 10 Features in Random Forest Regression")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# 10. EVALUATE AND COMPARE REGRESSION MODELS
regression_results = pd.DataFrame([
    {"Model": "Linear Regression", **linear_metrics},
    {"Model": "Random Forest Regression", **rf_metrics}
])
print("\nREGRESSION MODEL COMPARISON\n",
      regression_results.round(4).to_string(index=False))

# 11. SELECT THE BEST-FIT REGRESSION MODEL
best_model = regression_results.sort_values(
    ["R2 Score", "RMSE", "MAE"], ascending=[False, True, True]
).iloc[0]
print("\nBest-fit regression model:", best_model["Model"])
print("Best regression R2 score:", round(best_model["R2 Score"], 4))

plt.figure(figsize=(7, 5))
plt.bar(regression_results["Model"], regression_results["R2 Score"])
plt.title("Regression Model Comparison - R2 Score")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# MODEL 3: LOGISTIC REGRESSION
literacy_data = df.dropna(subset=["OVERALL_LI"]).copy()
low, high = literacy_data["OVERALL_LI"].quantile([1/3, 2/3])
literacy_data["Literacy_Class"] = pd.cut(
    literacy_data["OVERALL_LI"], [-np.inf, low, high, np.inf],
    labels=["Low", "Medium", "High"], include_lowest=True
)
literacy_features = [
    "TOTPOPULAT", "P_URB_POP", "GROWTHRATE", "SEXRATIO", "P_SC_POP",
    "P_ST_POP", "AREA_SQKM", "SCHTOT", "ENRTOT", "TCHTOT",
    "SWATTOT", "SELETOT"
]
X, y = literacy_data[literacy_features], literacy_data["Literacy_Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_test = fill_missing(X_train, X_test)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
log_model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
    X_train, y_train
)
log_pred = log_model.predict(X_test)
classification_results(y_test, log_pred, "Logistic Regression")

# MODEL 4: DECISION TREE CLASSIFIER
enrol_data = df.dropna(subset=["FEMALE_ENROLMENT_SHARE"]).copy()
q1, q2 = enrol_data["FEMALE_ENROLMENT_SHARE"].quantile([1/3, 2/3])
enrol_data["Enrolment_Class"] = pd.cut(
    enrol_data["FEMALE_ENROLMENT_SHARE"], [-np.inf, q1, q2, np.inf],
    labels=["Low", "Medium", "High"], include_lowest=True
)
enrol_features = [
    "TOTPOPULAT", "P_URB_POP", "GROWTHRATE", "SEXRATIO", "P_SC_POP",
    "P_ST_POP", "AREA_SQKM", "SCHTOT", "SCHGIRTOT", "SCHBOYTOT",
    "TCHTOT", "TCHTOTG", "TCHTOTM", "SWATTOT", "SGTOILTOT", "SBTOILTOT"
]
X, y = enrol_data[enrol_features], enrol_data["Enrolment_Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_test = fill_missing(X_train, X_test)
tree_model = DecisionTreeClassifier(
    criterion="entropy", max_depth=5, random_state=42,
    class_weight="balanced"
).fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
classification_results(y_test, tree_pred, "Decision Tree Classifier")

# 12. K-MEANS CLUSTERING
cluster_features = [
    "LITERACY_GAP", "ENRTOTG", "ENRTOTM", "TCHTOTG", "TCHTOTM",
    "SCHGIRTOT", "SCHBOYTOT", "SWATTOT", "SGTOILTOT", "SBTOILTOT",
    "SELETOT", "SCOMPTOT"
]
X = df[cluster_features].copy()
X = X.fillna(X.median())
X = MinMaxScaler().fit_transform(X)
df["Cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)

print("\nDISTRICTS IN EACH CLUSTER:\n", df["Cluster"].value_counts().sort_index())
print("\nAVERAGE VALUES IN EACH CLUSTER:\n",
      df.groupby("Cluster")[cluster_features].mean().round(2))

plt.figure(figsize=(8, 5))
plt.scatter(df["ENRTOTG"], df["ENRTOTM"], c=df["Cluster"],
            cmap="Pastel1", alpha=0.7)
plt.xlabel("Female Enrolment")
plt.ylabel("Male Enrolment")
plt.title("K-Means - Education District Clusters")
plt.tight_layout()
plt.show()

# 13. FINDINGS, CONCLUSION AND RECOMMENDATIONS
print("\nKEY FINDINGS")
print("Cleaned district records:", len(df))
print("Average gender literacy gap:",
      round(df["LITERACY_GAP"].mean(), 2), "percentage points")
print("Best-fit regression model:", best_model["Model"])
print("Best regression R2 score:", round(best_model["R2 Score"], 4))
print("K-Means cluster sizes:\n", df["Cluster"].value_counts().sort_index())