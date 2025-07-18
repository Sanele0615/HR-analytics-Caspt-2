# HR-analytics-Caspt-2
HR analytics ofemployees
# HR Analytics Capstone Project

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 2. Load Dataset
df = pd.read_csv('HR_dataset.csv')  # Replace with actual path

# 3. Initial Data Exploration
print("Dataset Shape:", df.shape)
df.info()
df.describe()
df.head()

# 4. Data Cleaning and Preprocessing
## Check for missing values
print("Missing Values:\n", df.isnull().sum())

## Handle missing values (example)
df.fillna(method='ffill', inplace=True)

## Remove duplicates
df.drop_duplicates(inplace=True)

## Check data types and convert if necessary
# Example: df['Age'] = df['Age'].astype(int)

## Resolve inconsistencies (e.g. casing, typos)
df['Department'] = df['Department'].str.title()

## Detect and handle outliers
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

# 5. Statistical Data Analysis
## Descriptive statistics
print(df.describe())

## Distributions
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

## Correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

## Hypothesis Testing Example: Does department affect attrition?
freq_table = pd.crosstab(df['Department'], df['Attrition'])
chi2, p, dof, ex = stats.chi2_contingency(freq_table)
print(f"Chi2 Test: p-value = {p}")

# 6. Exploratory Data Analysis (EDA)
## Attrition by Department
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Department', hue='Attrition')
plt.title('Attrition Count by Department')
plt.xticks(rotation=45)
plt.show()

## Age vs Monthly Income
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='MonthlyIncome', hue='Attrition')
plt.title('Age vs Monthly Income by Attrition')
plt.show()

## Boxplot of JobSatisfaction by Department
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Department', y='JobSatisfaction')
plt.title('Job Satisfaction by Department')
plt.xticks(rotation=45)
plt.show()

# 7. Insights and Conclusions
## Use markdown or print statements to explain
print("""
Insights:
- Departments with high attrition rates can be identified for HR intervention.
- Certain age and income ranges correlate with higher attrition.
- Job satisfaction varies significantly by department.

Conclusion:
- The organization should focus on improving work conditions in departments with high attrition.
- Consider implementing retention programs for employees in at-risk income/age brackets.

Future Work:
- Build predictive models for attrition.
- Incorporate more advanced analytics such as clustering or time series if applicable.
""")
