# Analysis of Educational Inequality in India (2015–16)

A Data Science and Machine Learning project exploring district-wise gender literacy gaps and educational indicators across India using the 2015–16 district-level education dataset.

## Project Overview

This project examines differences in male and female literacy, student enrolment, and educational infrastructure across Indian districts using Python, exploratory data analysis, feature engineering, supervised machine learning, and unsupervised clustering.

**Dataset period:** 2015–16  
**Unit of analysis:** Indian districts  
**Regression target:** `LITERACY_GAP = MALE_LIT - FEMALE_LIT`

A positive literacy gap indicates that male literacy is higher than female literacy.

## Five Objectives and Algorithms

1. **Linear Regression:** Predict the gender literacy gap using demographic, enrolment, and educational infrastructure indicators.
2. **Random Forest Regression:** Predict the gender literacy gap using an ensemble of decision trees and examine predictive feature importance.
3. **Logistic Regression:** Classify districts into Low, Medium, and High literacy categories.
4. **Decision Tree Classifier:** Classify districts into Low, Medium, and High female-enrolment-share categories.
5. **K-Means Clustering:** Group districts using literacy gap, enrolment, teacher, and school-infrastructure indicators.

The Low/Medium/High categories are created using quantile-based thresholds; they are not official government classifications.

## Workflow

1. Import libraries and load the dataset.
2. Inspect dataset structure, missing values, and duplicates.
3. Perform basic exploratory analysis.
4. Clean the data and engineer features.
5. Define project objectives.
6. Perform detailed EDA and visualizations.
7. Train five machine learning algorithms.
8. Evaluate regression and classification performance.
9. Compare regression models and identify the better-performing model.
10. Use K-Means to explore groups of similar districts.
11. Summarize findings and recommendations.

## Feature Engineering

- `LITERACY_GAP`: Male literacy minus female literacy.
- `FEMALE_ENROLMENT_SHARE`: Female enrolment divided by total male and female enrolment.
- `SCHOOLS_PER_1000_POP`: Total schools per 1,000 population.

The regression predictors exclude `MALE_LIT` and `FEMALE_LIT` to avoid directly reconstructing the target from its components.

## Key Findings

- Average gender literacy gap in the cleaned dataset: approximately **17.06 percentage points**.
- Random Forest Regression performed better than Linear Regression in the recorded comparison.
- Random Forest Regression R² score: approximately **0.4632** on the test data.

An R² score of 0.4632 means the model explains approximately 46.32% of the variation in the target values in the test set. It does **not** mean 46.32% prediction accuracy.

## Visualizations

The project includes:
- Distribution of gender literacy gaps.
- State/UT-wise average literacy gap.
- Female versus male student enrolment.
- Sex ratio versus literacy gap.
- Actual versus predicted plots for regression models.
- Random Forest feature-importance plot.
- Regression model comparison.
- Confusion-matrix plots for both classifiers.
- K-Means district-cluster visualization.

## Requirements

Python 3.9 or later, pandas, numpy, matplotlib, and scikit-learn.

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone or download this repository.
2. Place `2015_16_Districtwise.csv` in the same folder as the Python script.
3. Install the required packages.
4. Run:

```bash
python Educational_Inequality_India_Compact_5_Models.py
```

The script displays EDA charts, regression comparisons, classification reports and confusion matrices, and K-Means results.

## Repository Structure

```text
Educational-Inequality-India/
├── Educational_Inequality_India_Compact_5_Models.py
├── 2015_16_Districtwise.csv
├── requirements.txt
└── README.md
```

## Limitations

- The dataset represents 2015–16 and does not describe current educational conditions.
- Missing and inconsistent values are handled during preprocessing.
- School and teacher counts indicate availability, not education quality.
- Feature importance and statistical relationships indicate associations, not causation.
- K-Means cluster labels are group identifiers, not rankings.
- Classification labels are quantile-based categories created for this analysis.

## Dataset Source

**Add the exact original dataset URL and source/attribution here before publishing.** Check the dataset licence and redistribution conditions. If redistribution is not permitted, provide the source link and instructions for obtaining the data instead of committing the CSV.

## Author

**Aniket Anand**  
B.Tech, Computer Science Engineering (Data Science)

- GitHub: [AniketAnand151](https://github.com/AniketAnand151)
- LinkedIn: [Aniket Anand](https://www.linkedin.com/in/aniket-anand-270805d)
