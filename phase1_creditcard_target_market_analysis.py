"""
Project: ABC Bank Credit Card Launch – Phase 1
Purpose: Implements Phase 1 of ABC Bank's credit-card launch project — target-market identification through  
         data loading, cleaning, and exploratory preprocessing, and segmentation preparation.
Note:    CSV files are used instead of a database. Paths are placeholders. CSV files are not included.

Data files (placeholders):
    data/customers.csv
    data/credit_profiles.csv
    data/transactions.csv


# Project Structure
ABC_CreditCard_Analysis/
│
├── README.md
├── phase1_creditcard_target_market_analysis.py   ← this script
├── requirements.txt
└── data/
    ├── customers.csv                ← placeholder (user-supplied)
    ├── credit_profiles.csv          ← placeholder (user-supplied)
    └── transactions.csv             ← placeholder (user-supplied)

"""


# ================================
# 1. Import Required Libraries
# ================================
import pandas as pd                # Primary library for data manipulation and analysis
import numpy as np                 # Numerical computing and array operations
import seaborn as sns              # Statistical data visualization built on matplotlib
import matplotlib.pyplot as plt    # Core plotting library for creating visualizations
import warnings
warnings.filterwarnings('ignore')  # Suppress warning messages for cleaner console output

# MySQL connector commented out since CSV files are used instead of database connection
# import mysql.connector


# ================================
# 2. Load CSV Data
# ================================
# Load three primary datasets from CSV files into pandas DataFrames
# These datasets contain customer demographics, credit profiles, and transaction history
df_cust = pd.read_csv("data/customers.csv")          # Customer demographic information
df_cs = pd.read_csv("data/credit_profiles.csv")      # Credit scores and financial profiles
df_trans = pd.read_csv("data/transactions.csv")      # Transaction history and purchase behavior

# Display dataset dimensions (rows, columns) to verify successful data loading
print("Customers:", df_cust.shape)
print("Credit Profiles:", df_cs.shape)
print("Transactions:", df_trans.shape)


# ================================
# 3. Explore Customers Table
# ================================
# Display random sample of 3 records to understand data structure and contents
print(df_cust.sample(3))

# Generate descriptive statistics for all numerical columns
# Includes count, mean, std, min, 25%, 50%, 75%, max
print(df_cust.describe())

# Identify missing values across all columns to assess data quality
# Returns count of null/NaN values per column
print(df_cust.isnull().sum())


# ================================
# 4. Handle Missing Values: Annual Income
# ================================
# Calculate median annual income for each occupation category
# This will be used to impute missing income values based on occupation
occupation_median_income = df_cust.groupby('occupation')['annual_income'].median()

def fill_income(row):
    """
    Impute missing annual_income values using occupation-based median income.
    
    Strategy: Replace null income values with the median income of customers
    with the same occupation, preserving data distribution within occupations.
    
    Args:
        row: A pandas Series representing a single customer record
        
    Returns:
        float: Original annual_income if present, otherwise occupation median
    """
    if pd.isnull(row['annual_income']):
        return occupation_median_income[row['occupation']]
    else:
        return row['annual_income']

# Apply imputation function to all rows in the dataset
df_cust['annual_income'] = df_cust.apply(fill_income, axis=1)

# Verify that all missing income values have been successfully filled
print(df_cust.isnull().sum())

# Create histogram with kernel density estimate to visualize income distribution
# This helps identify skewness, multimodality, and potential remaining outliers
plt.figure()
sns.histplot(df_cust['annual_income'], kde=True, color='green')
plt.title('Distribution of Annual Income')
plt.xlabel('Annual Income ($)')
plt.ylabel('Frequency')
plt.show()


# ================================
# 5. Handle Outliers: Annual Income
# ================================
# Identify records with unrealistically low income (<$100)
# These are likely data entry errors or invalid records
low_income = df_cust[df_cust['annual_income'] < 100]

# Replace outlier values with occupation-specific median income
# This maintains data integrity while removing unrealistic values
for idx, row in low_income.iterrows():
    df_cust.at[idx, 'annual_income'] = occupation_median_income[row['occupation']]

# Confirm all low-income outliers have been successfully corrected
# Should return an empty DataFrame
print(df_cust[df_cust['annual_income'] < 100])


# ================================
# 6. Visualize Average Income by Categories
# ================================
# Calculate mean annual income for each occupation type
# Helps identify high-income vs low-income occupation segments
occupation_mean_income = df_cust.groupby('occupation')['annual_income'].mean()

# Create bar chart showing average income by occupation
plt.figure(figsize=(10,5))
sns.barplot(x=occupation_mean_income.index, y=occupation_mean_income.values, palette='tab10')
plt.xticks(rotation=45)
plt.title("Average Annual Income per Occupation")
plt.xlabel("Occupation")
plt.ylabel("Average Income ($)")
plt.show()

# Analyze income distribution across multiple demographic categories
# This provides insights into which customer segments have higher purchasing power
cat_columns = ['gender', 'location', 'occupation', 'marital_status']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Create a 2x2 grid of bar charts, one for each categorical variable
for i, col in enumerate(cat_columns):
    avg_income = df_cust.groupby(col)['annual_income'].mean()
    sns.barplot(x=avg_income.index, y=avg_income.values, palette='tab10', ax=axes[i])
    axes[i].set_title(f"Average Annual Income by {col.capitalize()}")
    axes[i].set_xlabel(col.capitalize())
    axes[i].set_ylabel("Average Income ($)")
    axes[i].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()


# ================================
# 7. Handle Age Outliers
# ================================
# Calculate median age for each occupation to use for outlier replacement
# Age patterns may vary by occupation (e.g., students vs retirees)
median_age_per_occupation = df_cust.groupby('occupation')['age'].median()

# Identify age values outside reasonable bounds (<15 or >85)
# Ages below 15 are unlikely to have credit cards; ages above 85 may indicate data errors
age_outliers = df_cust[(df_cust.age < 15) | (df_cust.age > 85)]

# Replace outlier ages with occupation-specific median values
# Preserves age distribution patterns within each occupation group
for idx, row in age_outliers.iterrows():
    df_cust.at[idx, 'age'] = median_age_per_occupation[row['occupation']]

# Verify no age outliers remain in the cleaned dataset
# Should return an empty DataFrame
print(df_cust[(df_cust.age < 15) | (df_cust.age > 85)])


# ================================
# 8. Create Age Groups
# ================================
# Define bin edges for age segmentation based on life stages
# 18-25: Young adults, 26-48: Prime working age, 49-65: Pre-retirement
bin_edges = [17, 25, 48, 65]
bin_labels = ['18-25', '26-48', '49-65']

# Create new categorical column for age groups using pd.cut
# This enables age-based segmentation analysis for targeted marketing
df_cust['age_group'] = pd.cut(df_cust['age'], bins=bin_edges, labels=bin_labels)

# Calculate percentage distribution of customers across age groups
# normalize=True converts counts to proportions, then multiply by 100 for percentages
age_group_counts = df_cust['age_group'].value_counts(normalize=True) * 100

# Create pie chart to visualize age group distribution
# explode parameter emphasizes the first two segments
plt.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', shadow=True, explode=(0.1, 0.1, 0))
plt.title("Distribution of Age Groups")
plt.show()


# ================================
# 9. Customer Distribution by Location and Gender
# ================================
# Create cross-tabulation of location and gender to understand demographic composition
# unstack() pivots gender values into columns for side-by-side comparison
customer_location_gender = df_cust.groupby(['location','gender']).size().unstack()

# Create stacked bar chart showing gender distribution within each location
# Helps identify gender imbalances across different geographic markets
customer_location_gender.plot(kind='bar', stacked=True, figsize=(6,4))
plt.title("Customer Distribution by Location and Gender")
plt.xlabel("Location")
plt.ylabel("Count")
plt.legend(title="Gender")
plt.show()


# ================================
# 10. Clean Credit Profile Table
# ================================
# Remove duplicate customer records, keeping only the most recent entry
# keep='last' assumes the most recent record is the most accurate
df_cs_clean = df_cs.drop_duplicates(subset='cust_id', keep='last')

# Create credit score ranges for grouping customers into risk tiers
# Ranges are created in 50-point increments from minimum to maximum score
cs_min, cs_max = df_cs_clean.credit_score.min(), df_cs_clean.credit_score.max()
bin_ranges = list(range(cs_min, cs_max + 50, 50))
bin_labels = [f'{start}-{end-1}' for start, end in zip(bin_ranges, bin_ranges[1:])]

# Assign each customer to a credit score range category
df_cs_clean['credit_score_range'] = pd.cut(df_cs_clean['credit_score'], bins=bin_ranges, labels=bin_labels)

# Impute missing credit_limit values using mode (most common value) within each credit score range
# Assumption: Customers with similar credit scores typically receive similar credit limits
mode_df = df_cs_clean.groupby('credit_score_range')['credit_limit'].agg(lambda x: x.mode().iloc[0]).reset_index()
df_cs_clean = pd.merge(df_cs_clean, mode_df, on="credit_score_range", suffixes=("", "_mode"))
df_cs_clean['credit_limit'].fillna(df_cs_clean['credit_limit_mode'], inplace=True)

# Correct data integrity issue where outstanding_debt exceeds credit_limit
# Cap outstanding_debt at credit_limit value (debt cannot exceed available credit)
df_cs_clean.loc[df_cs_clean.outstanding_debt > df_cs_clean.credit_limit, 'outstanding_debt'] = df_cs_clean['credit_limit']


# ================================
# 11. Merge Customer and Credit Profile Data
# ================================
# Perform inner join to combine customer demographics with credit profiles
# Only customers present in both datasets are retained
df_merged = df_cust.merge(df_cs_clean, on="cust_id", how="inner")

# Select numerical columns for correlation analysis
# Understanding relationships between these variables helps identify key predictors
numerical_cols = ['credit_score', 'credit_utilisation', 'outstanding_debt', 'credit_limit', 'age']
correlation_matrix = df_merged[numerical_cols].corr()

# Create heatmap to visualize correlation coefficients between numerical variables
# Strong correlations (near ±1) indicate potential multicollinearity or predictive relationships
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.8)
plt.title("Correlation Matrix")
plt.show()


# ================================
# 12. Clean Transactions Table
# ================================
# Fill missing platform values with the most frequently occurring platform (mode)
# Assumes most common platform is the best guess for missing values
df_trans['platform'].fillna(df_trans['platform'].mode()[0], inplace=True)

# Replace zero transaction amounts with category-specific median values
# Zero amounts are likely data entry errors; median is robust to outliers
for category in df_trans['product_category'].unique():
    # Calculate median from non-zero transactions in this category
    median_amt = df_trans[(df_trans.tran_amount > 0) & (df_trans.product_category == category)].tran_amount.median()
    # Create boolean mask for zero-amount transactions in this category
    mask = (df_trans.tran_amount == 0) & (df_trans.product_category == category)
    # Replace zeros with category median
    df_trans.loc[mask, 'tran_amount'] = median_amt

# Detect and handle transaction amount outliers using Interquartile Range (IQR) method
# IQR method is robust and identifies extreme values in the upper tail
Q1, Q3 = df_trans['tran_amount'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 2*IQR, Q3 + 2*IQR  # Using 2*IQR for more conservative outlier detection

# Separate outliers from normal transactions for targeted replacement
outliers = df_trans[df_trans.tran_amount >= upper]
normal_trans = df_trans[df_trans.tran_amount < upper]

# Calculate mean transaction amount per category from non-outlier data only
# Replace outliers with category-specific mean to maintain realistic values
mean_per_category = normal_trans.groupby('product_category')['tran_amount'].mean()
df_trans.loc[outliers.index, 'tran_amount'] = outliers['product_category'].map(mean_per_category)


# ================================
# 13. Merge All Data and Visualize
# ================================
# Combine customer demographics, credit profiles, and transaction history
# This creates a comprehensive dataset for analyzing customer behavior patterns
df_merged_2 = pd.merge(df_merged, df_trans, on='cust_id', how='inner')

# Analyze payment method preferences across different age groups
# Helps understand how payment behavior varies by customer age segment
sns.countplot(x='age_group', hue='payment_type', data=df_merged_2)
plt.title("Distribution of Payment Type Across Age Groups")
plt.show()

# Examine platform usage patterns (mobile, web, in-store) by age group
# Younger customers may prefer digital channels while older prefer traditional
sns.countplot(x='age_group', hue='platform', data=df_merged_2)
plt.title("Distribution of Platform Type Across Age Groups")
plt.show()

# Analyze product category preferences across age groups
# Different age segments may have distinct purchasing preferences
sns.countplot(x='age_group', hue='product_category', data=df_merged_2)
plt.title("Distribution of Product Category Across Age Groups")
plt.show()


# ================================
# 14. Average Transaction Amount by Various Categories
# ================================

# 14.1 Average transaction amount by payment type
# Compare spending levels across cash, credit, debit, etc.
# Group transactions by payment method and calculate mean transaction value
avg_tran_by_payment = df_merged_2.groupby('payment_type')['tran_amount'].mean().reset_index()

# Visualize which payment methods are associated with higher spending
sns.barplot(x='payment_type', y='tran_amount', data=avg_tran_by_payment)
plt.title("Average Transaction Amount by Payment Type")
plt.xlabel("Payment Type")
plt.ylabel("Average Transaction Amount ($)")
plt.show()

# 14.2 Average transaction amount by platform
# Identify which channels (mobile, web, store) generate higher transaction values
sns.barplot(x='platform', y='tran_amount', data=df_merged_2, estimator='mean')
plt.title("Average Transaction Amount by Platform")
plt.xlabel("Platform")
plt.ylabel("Average Transaction Amount ($)")
plt.show()

# 14.3 Average transaction amount by product category
# Understand which product categories have highest average transaction values
# This informs credit limit decisions and marketing strategies
sns.barplot(x='product_category', y='tran_amount', data=df_merged_2, estimator='mean')
plt.xticks(rotation=90)  # Rotate labels for better readability
plt.title("Average Transaction Amount by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Average Transaction Amount ($)")
plt.show()

# 14.4 Average transaction amount by marital status
# Married vs single customers may have different spending patterns
# Useful for targeted credit card offers and limit settings
sns.barplot(x='marital_status', y='tran_amount', data=df_merged_2, estimator='mean')
plt.title("Average Transaction Amount by Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Average Transaction Amount ($)")
plt.show()

# 14.5 Average transaction amount by age group
# Identify which age segments spend more per transaction
# Helps prioritize target markets for credit card launch
sns.barplot(x='age_group', y='tran_amount', data=df_merged_2, estimator='mean', palette='tab10')
plt.title("Average Transaction Amount by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Transaction Amount ($)")
plt.show()
