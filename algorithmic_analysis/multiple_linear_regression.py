import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

output_folder = "Linear Regression Graphs"
os.makedirs(output_folder, exist_ok=True)

# loading data
df = pd.read_csv("cleaned_data/cleaned_bls_data.csv")

# restricting to regions we care about
regions = ["Northeast", "Midwest", "South", "West"]
df = df[df["area_name"].isin(regions)].copy()

# cleaning data to make sure the year and value are appropriate types
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["year", "value"])

# definiing a post pandemic feature to simplify training
df["post_pandemic"] = (df["year"] >= 2020).astype(int)

# specific grocery items we care to look into
items = [
    "All uncooked ground beef, per lb. (453.6 gm)",
    "Bananas, per lb. (453.6 gm)",
    "Milk, fresh, low-fat, reduced fat, skim, per gal. (3.8 lit)",
    "Potato chips, per 16 oz.",
    "Bread, white, pan, per lb. (453.6 gm)"
]

df = df[df["item_name"].isin(items)]

models = {}
impacts = {}

# running regression per item and saving the graphs
for item in items:
    item_df = df[df["item_name"] == item]

    # tit model 
    model = smf.ols(
        formula="value ~ post_pandemic * C(area_name) + year",
        data=item_df
    ).fit(cov_type="HC3")

    models[item] = model

    coef = model.params
    baseline = coef.get("post_pandemic", 0)

    impacts[item] = {
        "Northeast": baseline,
        "Midwest": baseline + coef.get("post_pandemic:C(area_name)[T.Midwest]", 0),
        "South": baseline + coef.get("post_pandemic:C(area_name)[T.South]", 0),
        "West": baseline + coef.get("post_pandemic:C(area_name)[T.West]", 0),
    }

    file_item = item.replace("/", "-")

    # residuals vs. fitted graphs
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=model.fittedvalues, y=model.resid, s=25)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"Residuals vs Fitted — {item}")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{file_item}_residuals_vs_fitted.png")
    plt.close()

    # qq plot
    plt.figure(figsize=(6, 6))
    sm.qqplot(model.resid, line="45", fit=True)
    plt.title(f"QQ Plot — {item}")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{file_item}_qqplot.png")
    plt.close()

    # timeline of average prices over time
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=item_df,
        x="year",
        y="value",
        hue="area_name",
        marker="o",
        ci=None,
        estimator="mean"
    )
    plt.axvline(2020, color="red", linestyle="--", label="Pandemic Start (2020)")
    plt.title(f"Price Over Time — {item}")
    plt.xlabel("Year")
    plt.ylabel("Price ($)")
    plt.legend(title="Region")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{file_item}_timeline.png")
    plt.close()

# regional impact charts
for item, effect in impacts.items():
    print("\n=====================================")
    print(f"Pandemic Impact for: {item}")
    print("=====================================")
    for region, value in effect.items():
        print(f"{region}: {value:.3f}")
