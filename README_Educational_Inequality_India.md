# Educational Inequality in India (2015–16)
## Data Science and Machine Learning Project

This project analyses district-wise education data from India for the academic year 2015–16. It focuses on gender literacy differences, student enrolment, school availability, teacher distribution, and similarities between districts.

The project uses Python for data cleaning, exploratory data analysis (EDA), feature engineering, machine learning, model evaluation, and visualisation.

---

## 1. Project Objectives

The project has seven objectives, aligned with the seven machine learning algorithms implemented.

1. **Predict the Gender Literacy Gap using Linear Regression**  
   Investigate the relationship between demographic and educational indicators and the gender literacy gap, and predict the difference between male and female literacy rates using Linear Regression.

2. **Predict the Gender Literacy Gap using Decision Tree Regression**  
   Identify non-linear relationships between district-level demographic and educational indicators and the gender literacy gap using Decision Tree Regression.

3. **Predict the Gender Literacy Gap using Random Forest Regression**  
   Predict the gender literacy gap using an ensemble of decision trees and evaluate whether Random Forest provides better predictive performance than Linear Regression and Decision Tree Regression.

4. **Classify Districts by Literacy Level using Logistic Regression**  
   Classify districts into Low, Medium, and High literacy categories based on demographic and educational indicators using Logistic Regression.

5. **Classify Districts by Female Enrolment Share using Decision Tree Classification**  
   Categorise districts into Low, Medium, and High female enrolment share groups and identify factors associated with differences in female enrolment.

6. **Classify Districts by School Availability using Random Forest Classification**  
   Classify districts according to Low, Medium, and High school availability per 1,000 population using demographic and educational indicators.

7. **Group Similar Districts using K-Means Clustering**  
   Group districts into three clusters based on gender literacy gaps, student enrolment, teacher availability, and educational infrastructure indicators to identify similarities and differences between districts.

---

## 2. Dataset

- **Dataset name:** District-wise education data, 2015–16
- **Input file expected:** `2015_16_Districtwise.csv`
- **Unit of analysis:** District
- **Main theme:** Educational indicators and gender literacy gap

The dataset file must be placed in the same folder as the Python script.

> **Dataset source:** Add the exact Kaggle, data.gov.in, Mendeley, or AI Kosh source page and download URL used to obtain your CSV before submission. The source URL is not embedded in the project code, so it should not be guessed.

---

## 3. Project Workflow

### A. Data loading
- Load the CSV dataset using Pandas.
- Display the first five rows, dataset dimensions, missing-value counts, and duplicate count.
- Select the demographic and education-related columns required for the analysis.

### B. Data preprocessing and cleaning
- Remove duplicate records.
- Convert selected feature columns to numeric values where required.
- Remove rows where the target gender literacy gap cannot be calculated.
- Handle missing predictor values using median imputation.
- Avoid using male and female literacy columns as predictors when predicting the literacy gap, because they are used to calculate the target.

### C. Feature engineering

The project creates the following derived features:

| Feature | Calculation | Purpose |
|---|---|---|
| `LITERACY_GAP` | `MALE_LIT - FEMALE_LIT` | Difference between male and female literacy rates |
| `FEMALE_ENROLMENT_SHARE` | `ENRTOTG / (ENRTOTG + ENRTOTM)` | Female share of total student enrolment |
| `SCHOOLS_PER_1000_POP` | `(SCHTOT / TOTPOPULAT) × 1000` | Number of schools per 1,000 people |

A positive literacy gap indicates that the male literacy rate is higher than the female literacy rate.

### D. Exploratory Data Analysis (EDA)
The script visualises:
- Distribution of the gender literacy gap across districts.
- Average gender literacy gap by state/UT.
- Female versus male student enrolment.
- Sex ratio versus gender literacy gap.

### E. Machine learning

| No. | Algorithm | Task | Target |
|---:|---|---|---|
| 1 | Linear Regression | Regression | Gender literacy gap |
| 2 | Decision Tree Regressor | Regression | Gender literacy gap |
| 3 | Random Forest Regressor | Regression | Gender literacy gap |
| 4 | Logistic Regression | Classification | Literacy level |
| 5 | Decision Tree Classifier | Classification | Female enrolment share category |
| 6 | Random Forest Classifier | Classification | School availability category |
| 7 | K-Means | Clustering | Groups of similar districts |

For the classification tasks, Low, Medium, and High categories are created using the one-third and two-third quantiles of the relevant target indicator. K-Means uses three clusters.

---

## 4. Model Evaluation

### Regression evaluation
The three regression algorithms are compared using:

- **R² score:** Measures the proportion of variation in the target explained by the model. Higher is better.
- **MAE (Mean Absolute Error):** Measures the average absolute prediction error. Lower is better.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalising larger errors more strongly. Lower is better.

The script prints a comparison table and identifies the best-fit regression model by prioritising the highest R² score, with RMSE and MAE as tie-breakers.

### Classification evaluation
The classification models are evaluated using:
- Accuracy
- Confusion matrix
- Precision
- Recall
- F1-score

### Clustering evaluation
K-Means groups districts into three clusters. The script reports the number of districts in each cluster and the average values of the selected indicators for each cluster.

---

## 5. Regression Results and Best-Fit Model

The following results were obtained in the previous run of the project. Re-run the final script to regenerate and verify the comparison for your current environment.

| Regression model | R² score | MAE | RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.4063 | 3.8259 | 4.6863 |
| Decision Tree Regression | 0.1788 | 4.2522 | 5.5115 |
| Random Forest Regression | **0.4632** | **3.5236** | **4.4560** |

**Best-fit regression model: Random Forest Regression**

It achieved the highest R² score and the lowest MAE and RMSE among the three regression models in that run. Its R² score of 0.4632 means it explained approximately 46.32% of the variation in the gender literacy gap in the test data. Its MAE was approximately 3.52 literacy-gap percentage points.

This is the best-performing model among the three regression models tested; it does not establish that it is the best possible model for every dataset or split.

---

## 6. Key Findings to Report

Use the outputs generated by the latest run to write the final findings. The earlier analysis reported:

- The average gender literacy gap was approximately **17.06 percentage points**.
- The cleaned analysis dataset contained **634 district records**.
- The Random Forest Regression model performed better than Linear Regression and Decision Tree Regression on the reported test split.
- K-Means produced three district clusters, with earlier cluster sizes of **281, 266, and 87**. Cluster labels are arbitrary and should be interpreted using their average feature values, not by their numeric label alone.

Re-run the final code before submission and replace these figures if the current output differs.

---

## 7. Conclusions and Recommendations

### Conclusion
The project uses district-level education indicators to study differences in literacy, enrolment, and education availability across India. It combines regression, classification, and clustering to examine the dataset from multiple perspectives. In the earlier model run, Random Forest Regression was the best of the three regression models for predicting the gender literacy gap.

### Recommendations
1. Prioritise districts with larger gender literacy gaps for further investigation and targeted literacy programmes.
2. Examine barriers to female enrolment in districts with lower female enrolment shares.
3. Use school availability and teacher-distribution indicators to identify districts that may require additional education resources.
4. Use the K-Means cluster profiles to identify groups of districts with similar education characteristics and tailor interventions accordingly.
5. Treat model findings as patterns and associations, not proof that one indicator causes another. Validate recommendations with local and more recent data before implementing policy.

---

## 8. Requirements

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Python 3.9 or later is recommended.

---

## 9. How to Run the Project

Keep these two files in the same directory:

```text
project-folder/
│
├── Educational_Inequality_India_Final_7_Models_with_Best_Model.py
└── 2015_16_Districtwise.csv
```

Run the script:

```bash
python Educational_Inequality_India_Final_7_Models_with_Best_Model.py
```

The script prints preprocessing information, model evaluation metrics, regression comparison, and cluster summaries. Matplotlib windows display the generated charts.

---

## 10. Important Notes and Limitations

- The dataset represents the 2015–16 period and should not be treated as a description of current education conditions.
- The school availability feature measures the number of schools per 1,000 population; it does not measure school quality or infrastructure condition.
- Female enrolment share is an enrolment indicator and does not directly measure attendance, retention, or learning outcomes.
- Teacher counts and school counts do not by themselves establish teacher quality or student–teacher ratios.
- Classification categories are based on quantiles within this dataset; Low, Medium, and High are relative categories, not official government classifications.
- Feature importance and statistical relationships indicate association, not causation.
- The best-fit model conclusion is based on the reported train/test split and metrics. Further validation, such as cross-validation and testing on newer data, would strengthen the conclusion.

---

## 11. Project Structure

```text
Educational-Inequality-India/
│
├── Educational_Inequality_India_Final_7_Models_with_Best_Model.py
├── 2015_16_Districtwise.csv
└── README.md
```

## Author

Add your name, university, programme, course name, faculty name, and submission date here before submitting.
