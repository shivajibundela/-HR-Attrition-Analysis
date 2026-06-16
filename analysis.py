import pandas as pd
import numpy as np

# CSV load karo
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Shape check karo
print("Rows, Columns:", df.shape)

# First 5 rows print karo
print(df.head())

# Column names dekho
print(df.columns.tolist())
# Missing values check karo
print("\nMissing values:")
print(df.isnull().sum().sum())  # total missing values

# Data types check karo
print("\nData types:")
print(df.dtypes)

# Attrition column ke values count karo
print("\nAttrition counts:")
print(df['Attrition'].value_counts())
# Attrition rate calculate karo
attrition_rate = (df['Attrition'] == 'Yes').sum() / len(df) * 100
print(f"\nOverall Attrition Rate: {attrition_rate:.2f}%")

# Two groups banao
left = df[df['Attrition'] == 'Yes']
stayed = df[df['Attrition'] == 'No']

# NumPy se MonthlyIncome compare karo
print("\n--- Monthly Income Comparison ---")
print("Avg Income (Left):", np.mean(left['MonthlyIncome']))
print("Avg Income (Stayed):", np.mean(stayed['MonthlyIncome']))

# Age comparison
print("\n--- Age Comparison ---")
print("Avg Age (Left):", np.mean(left['Age']))
print("Avg Age (Stayed):", np.mean(stayed['Age']))

# Correlation - JobSatisfaction vs Attrition (numeric encode karna padega)
df['Attrition_num'] = df['Attrition'].map({'Yes': 1, 'No': 0})
correlation = np.corrcoef(df['JobSatisfaction'], df['Attrition_num'])[0, 1]
print(f"\nCorrelation (JobSatisfaction vs Attrition): {correlation:.3f}")
# Department-wise attrition rate
print("\n--- Department-wise Attrition ---")
dept_attrition = df.groupby('Department')['Attrition_num'].mean() * 100
print(dept_attrition)

# OverTime vs Attrition
print("\n--- OverTime vs Attrition ---")
overtime_attrition = df.groupby('OverTime')['Attrition_num'].mean() * 100
print(overtime_attrition)

# JobRole-wise attrition (top contributors)
print("\n--- JobRole-wise Attrition ---")
role_attrition = df.groupby('JobRole')['Attrition_num'].mean().sort_values(ascending=False) * 100
print(role_attrition)
# Age groups banao
df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 60], 
                          labels=['18-25', '26-35', '36-45', '46-60'])

# Years at company groups
df['TenureGroup'] = pd.cut(df['YearsAtCompany'], bins=[-1, 2, 5, 10, 50],
                             labels=['0-2 yrs', '3-5 yrs', '6-10 yrs', '10+ yrs'])

# Income bands (NumPy percentile use karke)
income_25 = np.percentile(df['MonthlyIncome'], 25)
income_50 = np.percentile(df['MonthlyIncome'], 50)
income_75 = np.percentile(df['MonthlyIncome'], 75)
print(f"\nIncome Percentiles - 25th: {income_25}, 50th: {income_50}, 75th: {income_75}")

df['IncomeBand'] = pd.cut(df['MonthlyIncome'], 
                            bins=[0, income_25, income_50, income_75, df['MonthlyIncome'].max()],
                            labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# Processed data ko CSV mein save karo - yeh Power BI mein jayega
df.to_csv("data/processed_hr_data.csv", index=False)
print("\nProcessed file saved successfully!")