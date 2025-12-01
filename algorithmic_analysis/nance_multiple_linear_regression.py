# ==========================================================
# Grocery Price Pandemic Impact Analysis (Optimized for Large Datasets)
# ==========================================================

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

# -----------------------------
# 0️⃣ Setup
# -----------------------------
output_folder = "Linear Regression Graphs"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# 1️⃣ Load and Clean Data
# -----------------------------
df = pd.read_csv("cleaned_data/cleaned_bls_data.csv")  # Replace with your CSV path

# Keep only main regions
regions = ["Northeast", "Midwest", "South", "West"]
df = df[df["area_name"].isin(regions)].copy()

# Convert numeric columns
df["year"] = pd.to_numeric(df["year"], errors='coerce')
df["value"] = pd.to_numeric(df["value"], errors='coerce')

# Drop rows with missing or non-numeric values
df = df.dropna(subset=["year", "value"])

# Create post-pandemic indicator
df["post_pandemic"] = (df["year"] >= 2020).astype(int)

# Reset index
df = df.reset_index(drop=True)

# -----------------------------
# 2️⃣ Fit Multiple Linear Regression
# -----------------------------
# Model: value ~ post_pandemic * area_name + item_name controls
model = smf.ols(
    formula="value ~ post_pandemic * C(area_name) + C(item_name)",
    data=df
).fit()

print(model.summary())

# -----------------------------
# 3️⃣ Check Regression Assumptions
# -----------------------------
residuals = model.resid

# 3a. Residual Normality
plt.figure(figsize=(8,4))
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.savefig(f"{output_folder}/residuals_distribution.png")
plt.close()

plt.figure(figsize=(6,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-plot of Residuals")
plt.savefig(f"{output_folder}/qqplot_residuals.png")
plt.close()

# 3b. Homoscedasticity
plt.figure(figsize=(8,4))
sns.scatterplot(x=model.fittedvalues, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.savefig(f"{output_folder}/residuals_vs_fitted.png")
plt.close()

# 3c. Multicollinearity (VIF)
# ⚠️ Full VIF is skipped for large datasets to prevent freezing.
# If you want, you can calculate VIF on a small subset of items instead.

# -----------------------------
# 4️⃣ Compute Regional Pandemic Impacts
# -----------------------------
coef = model.params
baseline = coef["post_pandemic"]

impact = {
    "Northeast": baseline,  # reference region
    "Midwest": baseline + coef.get("post_pandemic:C(area_name)[T.Midwest]", 0),
    "South": baseline + coef.get("post_pandemic:C(area_name)[T.South]", 0),
    "West": baseline + coef.get("post_pandemic:C(area_name)[T.West]", 0)
}

impact_df = pd.DataFrame(list(impact.items()), columns=["Region", "Pandemic_Impact"])

# -----------------------------
# 5️⃣ Visualize Regional Impacts
# -----------------------------
plt.figure(figsize=(8,5))
sns.barplot(data=impact_df, x="Region", y="Pandemic_Impact", palette="viridis")
plt.title("Estimated Extra Grocery Price Increase by Region Post-Pandemic")
plt.ylabel("Estimated Price Increase ($)")
plt.savefig(f"{output_folder}/regional_pandemic_impact.png")
plt.close()

# -----------------------------
# 6️⃣ Timeline of Average Price per Region
# -----------------------------
plt.figure(figsize=(10,6))
sns.lineplot(
    data=df,
    x="year",
    y="value",
    hue="area_name",
    estimator="mean",
    ci=None,
    marker="o"
)
plt.title("Average Grocery Prices by Region Over Time")
plt.ylabel("Average Price ($)")
plt.xlabel("Year")
plt.axvline(2020, color='red', linestyle='--', label='Pandemic Start (2020)')
plt.legend(title="Region")
plt.savefig(f"{output_folder}/average_price_timeline.png")
plt.close()

# -----------------------------
# ✅ Workflow Complete
# -----------------------------
print(f"All charts saved in '{output_folder}' folder. Workflow complete.")
