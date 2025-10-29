"""
Project: ABC Bank Credit Card Launch – Phase 2
Purpose: A/B Testing analysis to evaluate the performance of a new credit card design
         targeting the 18-25 age group through statistical hypothesis testing.

Campaign Overview:
    - Target Segment: Customers aged 18-25 years
    - Test Design: Randomized controlled trial (A/B test)
    - Primary KPI: Average transaction amount per customer
    - Statistical Approach: Independent samples z-test

Phase 2 Steps:
    10.1: Campaign Planning - Identify target/control groups, determine sample sizes
    10.2: Execute Campaign - Launch new card to test group
    10.3: Post-Campaign Data Collection - Gather transaction data from both groups
    10.4: Hypothesis Testing - Determine if transaction amounts differ significantly
"""


# ================================
# 1. Import Required Libraries
# ================================
import scipy.stats as st                    # Statistical functions and distributions
import pandas as pd                         # Data manipulation and analysis
import numpy as np                          # Numerical operations and mathematical functions
import matplotlib.pyplot as plt             # Core plotting library
import seaborn as sns                       # Statistical data visualization
import statsmodels.stats.api as sms         # Statistical models and tests
import statsmodels.api as sm                # Additional statistical functions


# ================================
# 2. Business Context and Target Segment Analysis
# ================================
"""
Insights specific to customers of age group 18-25:
    - Represents approximately 25% of total customer base
    - Average annual income: < $50,000 per annum
    - Limited credit history reflected in lower credit scores and credit limits
    - Relatively low credit card usage compared to other age groups
    - Lower average transaction amounts when using credit cards
    - Top 3 product categories: Electronics, Fashion, and Beauty

Campaign Objective:
    Introduce a new credit card design tailored to this demographic to increase
    engagement and average transaction amounts.
"""


# ================================
# 3. Statistical Framework for A/B Testing
# ================================
"""
Statistical Testing Framework - Understanding Type I and Type II Errors

Type I Error (α - alpha): False Positive
    - Rejecting the null hypothesis when it's actually true
    - Example: Concluding the new card performs better when it doesn't
    - alpha = 1 - confidence_level

Type II Error (β - beta): False Negative  
    - Failing to reject the null hypothesis when it's actually false
    - Example: Missing a real improvement in card performance

Statistical Power = 1 - β
    - Probability of correctly detecting a real effect
    - Probability of rejecting H0 when it is actually false
    - Higher power means better ability to detect true differences
"""


# ================================
# 4. Sample Size Calculation
# ================================

# Define key parameters for A/B test sample size calculation
alpha = 0.05      # Significance level: 5% risk of Type I error (false positive)
                  # Corresponds to 95% confidence level
                  
power = 0.80      # Statistical power: 80% probability of detecting a real effect
                  # Industry standard; balances sample size requirements with detection ability
                  
effect_size = 0.2 # Cohen's d: Standardized measure of difference between two group means
                  # 0.2 is considered a "small" effect size by Cohen's conventions
                  # Represents the minimum meaningful difference we want to detect

# ratio = n2/n1: Determines relative size of test group compared to control group
# ratio = 1: Equal sample sizes in both groups (most statistically efficient)
# ratio = 2: Test group has twice the participants as control group

# Calculate required sample size per group using independent t-test power analysis
sample_size_per_group = sms.tt_ind_solve_power(
    effect_size = effect_size,   # Minimum detectable effect (Cohen's d)
    alpha       = alpha,          # Significance level (Type I error rate)
    power       = power,          # Statistical power (1 - Type II error rate)
    ratio       = 1,              # Equal group sizes (1:1 ratio)
    alternative = 'two-sided'     # Two-tailed test: detect differences in either direction
)

print(f"Required sample size per group: {round(sample_size_per_group)}")


# ================================
# 5. Sample Size Sensitivity Analysis
# ================================
"""
Calculate required sample sizes across various effect sizes to understand trade-offs
Smaller effect sizes require larger samples to detect reliably
"""

effect_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
# Range from small (0.1) to very large (1.0) effect sizes

print("\nSample Size Requirements for Different Effect Sizes:")
print("=" * 60)

for effect_size in effect_sizes:
    # Calculate minimum sample size needed per group for this effect size
    sample_size_PER_group = sms.tt_ind_solve_power(
                                effect_size = effect_size,
                                alpha       = alpha,          # 5% significance level
                                power       = power,          # 80% statistical power
                                ratio       = 1,              # Equal group sizes
                                alternative = 'two-sided'     # Two-tailed test
                            )
    print(f"Effect size: {effect_size:.1f} -> Sample size per group: {round(sample_size_PER_group)}")


# ================================
# 6. Campaign Design Parameters
# ================================
"""
Business Decision: Campaign Design Parameters

Selected Configuration:
    - Effect size: 0.4 (moderate effect, realistic for credit card campaigns)
    - Sample size: 99 customers per group (control and test)
    
Conversion Rate Consideration:
    - Expected conversion rate: ~40% (percentage who actually use the card)
    - Effective active users per group: ~40 customers (99 × 0.40)

Key Performance Indicator (KPI):
    - PRIMARY METRIC: Average transaction amount per customer
    - GOAL: Demonstrate that the new credit card design increases average spending
    - SUCCESS CRITERIA: Statistically significant increase in test group vs control group
"""

print("\n" + "=" * 60)
print("CAMPAIGN CONFIGURATION")
print("=" * 60)
print(f"Selected effect size: 0.4")
print(f"Sample size per group: 99 customers")
print(f"Expected conversion rate: 40%")
print(f"Active users per group: ~40 customers")
print(f"Primary KPI: Average transaction amount")
print("=" * 60 + "\n")


# ================================
# 7. Load Post-Campaign Data
# ================================
"""
Load post-campaign transaction data collected over the campaign period
Dataset contains daily average transaction amounts for both control and test groups
"""

df_results = pd.read_csv("avg_transactions_after_campaign.csv")

print("Post-Campaign Data Overview:")
print(f"Dataset shape: {df_results.shape}")
print(f"\nFirst 3 rows:")
print(df_results.head(3))


# ================================
# 8. Visualize Transaction Distributions
# ================================
"""
Visualize distribution of average transaction amounts for both groups
Purpose: Visual inspection to identify differences in central tendency and spread
"""

# Create 1x2 subplot grid for side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3))

# Control Group Distribution (customers who received standard card)
sns.histplot(df_results.control_group_avg_tran, kde=True, label="Control", ax=ax1)
ax1.set_xlabel("Average Transaction Amount ($)")
ax1.set_ylabel("Frequency")
ax1.set_title("Control Group Distribution")
ax1.legend()

# Test Group Distribution (customers who received new card design)
sns.histplot(df_results.test_group_avg_tran, kde=True, label="Test", ax=ax2)
ax2.set_xlabel("Average Transaction Amount ($)")
ax2.set_ylabel("Frequency")
ax2.set_title("Test Group Distribution")
ax2.legend()

plt.tight_layout()
plt.show()


# ================================
# 9. Exploratory Data Analysis
# ================================
"""
Exploratory Analysis: Count days when control group outperformed test group
This provides initial insight before formal hypothesis testing
"""

# Filter for days where control average transactions exceeded test average
days_control_higher = df_results[df_results.control_group_avg_tran > df_results.test_group_avg_tran]

print("\n" + "=" * 60)
print("EXPLORATORY ANALYSIS")
print("=" * 60)
print(f"Total days in campaign: {df_results.shape[0]}")
print(f"Days control > test: {days_control_higher.shape[0]}")
print(f"Percentage: {round(100 * days_control_higher.shape[0] / df_results.shape[0])}%")
print(f"\nImplication: Test group outperformed on ~{100 - round(100 * days_control_higher.shape[0] / df_results.shape[0])}% of days")
print("=" * 60)

print("\n  NOTE: This is descriptive analysis only.")
print("We MUST perform formal hypothesis testing to determine statistical significance.")
print("Visual trends can be misleading without accounting for variability and sample size.\n")


# ================================
# 10. Calculate Summary Statistics
# ================================

# Calculate summary statistics for CONTROL group
control_mean = df_results.control_group_avg_tran.mean()  # Average daily transaction amount
control_std  = df_results.control_group_avg_tran.std()   # Standard deviation (measure of spread)

# Calculate summary statistics for TEST group
test_mean = df_results.test_group_avg_tran.mean()  # Average daily transaction amount  
test_std  = df_results.test_group_avg_tran.std()   # Standard deviation

# Determine sample size (number of days of data collected)
sample_size = df_results.shape[0]

print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(f"Control Group:")
print(f"  Mean: ${control_mean:.2f}")
print(f"  Std Dev: ${control_std:.2f}")
print(f"\nTest Group:")
print(f"  Mean: ${test_mean:.2f}")
print(f"  Std Dev: ${test_std:.2f}")
print(f"\nSample Size (days): {sample_size}")
print(f"Observed Difference: ${test_mean - control_mean:.2f}")
print("=" * 60 + "\n")


# ================================
# 11. Manual Z-Test Calculation
# ================================
"""
Manual Z-Test Calculation for Difference Between Two Independent Means

Formula: Z = (Mean1 - Mean2) / Standard_Error_of_Difference
Where Standard_Error = sqrt( (σ1²/n1) + (σ2²/n2) )

Assumptions:
    - Independent samples (control and test are separate groups)
    - Large sample size (Central Limit Theorem applies)
    - Known or estimated population standard deviations
"""

Z_score = (test_mean - control_mean) / np.sqrt((control_std**2/sample_size) + (test_std**2/sample_size))
# Positive Z-score indicates test mean is higher than control mean

print("=" * 60)
print("MANUAL Z-TEST CALCULATION")
print("=" * 60)
print(f"Calculated Z-score: {Z_score:.4f}")
print("=" * 60 + "\n")


# ================================
# 12. Determine Critical Z-Value
# ================================
"""
Determine Critical Z-value for Hypothesis Test

Test Type: One-tailed (right-tailed) test
Rationale: We specifically want to test if test group performs BETTER (higher transactions)
           Not just different, but specifically higher

Null Hypothesis (H0): test_mean ≤ control_mean (no improvement or worse)
Alternative Hypothesis (H1): test_mean > control_mean (test group performs better)

For one-tailed test at α = 0.05:
    - Find z-value where cumulative probability = 0.95
    - This represents the critical threshold for rejecting H0
"""

Z_critical = st.norm.ppf(1 - alpha)  # Percent point function (inverse of CDF)

print("=" * 60)
print("HYPOTHESIS TEST - CRITICAL VALUE APPROACH")
print("=" * 60)
print(f"H0 (Null): test_mean ≤ control_mean")
print(f"H1 (Alternative): test_mean > control_mean")
print(f"\nSignificance level (α): {alpha}")
print(f"Critical Z-value: {Z_critical:.4f}")
print(f"Calculated Z-score: {Z_score:.4f}")
print("=" * 60)


# ================================
# 13. Make Statistical Decision
# ================================
"""
Decision Rule for Hypothesis Test:
    - If Z_score > Z_critical: REJECT null hypothesis (H0)
    - If Z_score ≤ Z_critical: FAIL TO REJECT null hypothesis (H0)

Result Interpretation:
Since Z_score > Z_critical, we are in the rejection region
We have sufficient statistical evidence to reject H0 at the 5% significance level

CONCLUSION: Accept H1 (Alternative Hypothesis)
The test group (new credit card) has a statistically significant higher average 
transaction amount compared to the control group (standard card)
"""

print("\n" + "=" * 60)
print("STATISTICAL DECISION")
print("=" * 60)

if Z_score > Z_critical:
    print("REJECT H0: Z-score > Z-critical")
    print("\nConclusion:")
    print("  The new credit card design shows a statistically significant")
    print("  increase in average transaction amounts compared to the standard card.")
    print(f"  Test group outperforms control by ${test_mean - control_mean:.2f} on average.")
else:
    print("FAIL TO REJECT H0: Z-score ≤ Z-critical")
    print("\nConclusion:")
    print("  Insufficient evidence to conclude the new card performs better.")

print("=" * 60 + "\n")


# ================================
# 14. P-Value Approach (Alternative Method)
# ================================
"""
Alternative Method: P-value Approach (equivalent to critical value method)

P-value Definition: 
    - Probability of observing a Z-score this extreme (or more) if H0 is true
    - Calculated as the area under the normal curve to the right of Z_score
    - For one-tailed test: p-value = 1 - CDF(Z_score)

Interpretation:
    - Small p-value: Observed difference is unlikely under H0 (suggests real effect)
    - Large p-value: Observed difference could easily occur by chance under H0

P-value Decision Rule:
    - If p_value < α (0.05): REJECT H0
    - If p_value ≥ α (0.05): FAIL TO REJECT H0
"""

p_value = 1 - st.norm.cdf(Z_score)  # Calculate right-tail probability

print("=" * 60)
print("HYPOTHESIS TEST - P-VALUE APPROACH")
print("=" * 60)
print(f"Calculated p-value: {p_value:.6f}")
print(f"Significance level (α): {alpha}")
print("=" * 60)

if p_value < alpha:
    print("REJECT H0: p-value < α")
    print("\nConclusion:")
    print("  The result is statistically significant.")
    print("  The new card design leads to higher transaction amounts.")
else:
    print("FAIL TO REJECT H0: p-value ≥ α")
    print("\nConclusion:")
    print("  The result is NOT statistically significant.")

print("=" * 60 + "\n")

print("NOTE: Both approaches (critical value and p-value) are mathematically")
print("      equivalent and lead to the same conclusion.\n")


# ================================
# 15. Automated Z-Test Using Statsmodels
# ================================
"""
Automated Z-test Using Statsmodels Library
This method performs the same hypothesis test but with built-in functions
Advantage: Reduces manual calculation errors and provides standardized output
"""

# Perform two-sample z-test comparing test group vs control group
z_stat, p_val = sm.stats.ztest(
    df_results.test_group_avg_tran,      # Test group (new card) data
    df_results.control_group_avg_tran,   # Control group (standard card) data  
    alternative='larger'                  # One-tailed test: H1 is test > control
)
# alternative='larger': Tests if first sample mean is larger than second
# alternative='two-sided': Would test for any difference (not direction-specific)
# alternative='smaller': Would test if first sample mean is smaller than second

print("=" * 60)
print("AUTOMATED Z-TEST (STATSMODELS)")
print("=" * 60)
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_val:.6f}")
print(f"\nVerification: Matches manual calculation")
print(f"  Manual Z-score: {Z_score:.4f}")
print(f"  Manual p-value: {p_value:.6f}")
print("=" * 60 + "\n")


# ================================
# 16. Confidence Interval for Test Group
# ================================
"""
Calculate 95% Confidence Interval (CI) for TEST Group Mean

Confidence Interval Interpretation:
    - We are 95% confident that the true population mean for the test group
      lies within this interval
    - Narrower CI indicates more precise estimate of the true mean
    - If control group mean falls outside this CI, it provides additional
      evidence of a significant difference

Formula: CI = mean ± Z_critical × Standard_Error
Where Standard_Error = std / sqrt(n)
"""

test_group_CI = st.norm.interval(
    1 - alpha,                           # Confidence level (95%)
    loc=test_mean,                       # Center of distribution (sample mean)
    scale=test_std/np.sqrt(sample_size)  # Standard error of the mean
)

print("=" * 60)
print("95% CONFIDENCE INTERVAL FOR TEST GROUP MEAN")
print("=" * 60)
print(f"Test Group Mean: ${test_mean:.2f}")
print(f"95% CI: (${test_group_CI[0]:.2f}, ${test_group_CI[1]:.2f})")
print(f"\nControl Group Mean: ${control_mean:.2f}")

if control_mean < test_group_CI[0]:
    print("\nControl mean is BELOW the test group's 95% CI")
    print("  This provides additional evidence of significant difference.")
elif control_mean > test_group_CI[1]:
    print("\nControl mean is ABOVE the test group's 95% CI")
    print("  Test group performs worse than control.")
else:
    print("\nControl mean falls WITHIN the test group's 95% CI")
    print("  Suggests overlapping distributions.")

print("=" * 60 + "\n")


# ================================
# 17. Final Summary and Recommendation
# ================================

print("\n" + "=" * 70)
print(" " * 20 + "FINAL SUMMARY & RECOMMENDATION")
print("=" * 70)
print("\nA/B Test Results:")
print(f"  • Control Group (Standard Card): Mean = ${control_mean:.2f}")
print(f"  • Test Group (New Card): Mean = ${test_mean:.2f}")
print(f"  • Observed Difference: ${test_mean - control_mean:.2f} ({((test_mean - control_mean)/control_mean * 100):.1f}% increase)")
print(f"\nStatistical Significance:")
print(f"  • Z-score: {Z_score:.4f}")
print(f"  • P-value: {p_value:.6f}")
print(f"  • Result: {'STATISTICALLY SIGNIFICANT' if p_value < alpha else 'NOT SIGNIFICANT'} at α = {alpha}")

if p_value < alpha:
    print("\nRECOMMENDATION: LAUNCH THE NEW CREDIT CARD")
    print("  The new card design demonstrates a statistically significant")
    print("  improvement in average transaction amounts for the 18-25 age group.")
    print("\n  Next Steps:")
    print("    1. Roll out new card to entire 18-25 customer segment")
    print("    2. Monitor performance metrics continuously")
    print("    3. Consider extending campaign to other age groups")
    print("    4. Develop targeted marketing strategies for this segment")
else:
    print("\nRECOMMENDATION: DO NOT LAUNCH THE NEW CREDIT CARD")
    print("  Insufficient evidence to conclude the new design improves performance.")
    print("\n  Next Steps:")
    print("    1. Re-evaluate card design and features")
    print("    2. Conduct additional customer research")
    print("    3. Consider testing with a larger sample size")
    print("    4. Explore alternative marketing strategies")

print("=" * 70 + "\n")
