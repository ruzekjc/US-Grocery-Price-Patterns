import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

plt.ion()

df = pd.read_csv("cleaned_data/cleaned_bls_data.csv")

regions = df['area_name'].unique()

df['year'] = pd.to_datetime(df['year'], format='%Y')

foods = [
    "Rice, white, long grain, uncooked, per lb. (453.6 gm)",
    "Apples, Red Delicious, per lb. (453.6 gm)",
    "Ground chuck, 100% beef, per lb. (453.6 gm)",
    "Cookies, chocolate chip, per lb. (453.6 gm)",
    "Milk, fresh, low-fat, reduced fat, skim, per gal. (3.8 lit)",
    "Eggs, grade AA, large, per doz.",
    "Potatoes, white (cost per pound/453.6 grams)",
    "Spaghetti and macaroni, per lb. (453.6 gm)",
    "Cola, nondiet, per 2 liters (67.6 oz)",
    "Coffee, 100%, ground roast, all sizes, per lb. (453.6 gm)"
]

results = {}

for food in foods:
    print(f"Food: {food}")
    for region in regions:
        print(f"Region: {region}")
        subset = df[(df['item_name'] == food) & (df['area_name'] == region)].copy()
        if subset.empty:
            print("subset empty")
            continue

        series = subset.groupby('year')['value'].mean().sort_index()

        pre = series[(series.index.year >= 2010) & (series.index.year <= 2019)]
        post = series[series.index.year >= 2020]

        if len(pre) < 3 or len(post) == 0:
            continue

        try:
            model = ARIMA(pre, order=(1,1,0))
            model_fit = model.fit()
        except:
            model = ARIMA(pre, order=(0,1,1))
            model_fit = model.fit()

        n_years = len(post)
        forecast = model_fit.get_forecast(steps=n_years)
        predicted = forecast.predicted_mean
        predicted.index = post.index  # Align index to actual post-pandemic years

        comparison = pd.DataFrame({
            'actual': post.values,
            'predicted': predicted.values
        }, index=post.index)
        comparison['difference'] = comparison['actual'] - comparison['predicted']
        comparison['pct_change'] = (comparison['difference'] / comparison['predicted']) * 100

        results[(food, region)] = comparison

        print("reaching plotting code...")

        plt.figure(figsize=(10,5))
        plt.plot(pre.index, pre, label='Pre-pandemic Actual', marker='o')
        plt.plot(post.index, post, label='Post-pandemic Actual', marker='o')
        plt.plot(predicted.index, predicted, label='Predicted (Counterfactual)', linestyle='--', marker='o')
        plt.title(f"{food} — {region}")
        plt.xlabel("Year")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        try:
            plt.savefig(f"AR{food}_{region}.png")
            print("Plot saved successfully")
        except Exception as e:
            print("Error saving plot:", e)
