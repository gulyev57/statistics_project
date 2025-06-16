import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
df = pd.read_csv("La_Liga_2016_2020_Data.csv")
df = df[(df['League'].str.strip() == 'La Liga') & (df['Year'].between(2016, 2020))]

sns.set(style="whitegrid")

# ---- HISTOGRAM ----
plt.figure(figsize=(8, 5))
sns.histplot(df['Goals'], bins=10, kde=True, color="skyblue", edgecolor="black")
plt.title('Histogram of Goals Scored')
plt.xlabel('Goals')
plt.ylabel('Number of Players')
plt.tight_layout()
plt.show()

# ---- BOXPLOT ----
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Goals'], color="salmon")
plt.title('Boxplot of Goals Scored')
plt.xlabel('Goals')
plt.tight_layout()
plt.show()

# --- CONFIDENCE INTERVALS ---
import scipy.stats as stats
import numpy as np

# Extract Goals column
goals = df['Goals']
n = len(goals)

# --- DESCRIPTIVE STATISTICS ---
mean_goals = goals.mean()
median_goals = goals.median()
variance_goals = goals.var(ddof=1)
std_dev_goals = goals.std(ddof=1)
standard_error_goals = std_dev_goals / np.sqrt(n)

print("Descriptive Statistics:")
print("Mean:", round(mean_goals, 2))
print("Median:", round(median_goals, 2))
print("Variance:", round(variance_goals, 2))
print("Standard Deviation:", round(std_dev_goals, 2))
print("Standard Error:", round(standard_error_goals, 3))

# --- CONFIDENCE INTERVALS ---
confidence_level = 0.95
alpha = 1 - confidence_level
t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
margin_of_error = t_critical * standard_error_goals
ci_mean = (mean_goals - margin_of_error, mean_goals + margin_of_error)

chi2_lower = stats.chi2.ppf(alpha / 2, df=n - 1)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
ci_variance = ((n - 1) * variance_goals / chi2_upper, (n - 1) * variance_goals / chi2_lower)

print("95% Confidence Interval for Mean: ({:.2f}, {:.2f})".format(ci_mean[0], ci_mean[1]))
print("95% Confidence Interval for Variance: ({:.2f}, {:.2f})".format(ci_variance[0], ci_variance[1]))

# --- SAMPLE SIZE ESTIMATION ---
desired_margin_of_error = 0.1
confidence_level_90 = 0.90
z_score_90 = stats.norm.ppf(1 - (1 - confidence_level_90) / 2)
required_n = int(np.ceil((z_score_90 * std_dev_goals / desired_margin_of_error) ** 2))

print("Required sample size for margin of error 0.1 at 90% confidence:", required_n)

# --- HYPOTHESIS TESTING ---
# Null Hypothesis H0: mean <= 10
# Alternative Hypothesis H1: mean > 10

hypothesized_mean = 10
t_statistic = (mean_goals - hypothesized_mean) / (std_dev_goals / np.sqrt(n))
p_value = 1 - stats.t.cdf(t_statistic, df=n - 1)

print("Hypothesis Test:")
print("Test Statistic (t): {:.3f}".format(t_statistic))
print("P-value: {:.4f}".format(p_value))

alpha_htest = 0.05
if p_value < alpha_htest:
    print("Result: Reject the null hypothesis (H0). The mean number of goals is significantly greater than 10.")
else:
    print("Result: Fail to reject the null hypothesis (H0). Not enough evidence to say the mean is greater than 10.")