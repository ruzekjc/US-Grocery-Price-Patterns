import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

import warnings

# Ignore ValueWarning messages
warnings.filterwarnings("ignore", category=UserWarning)  # catch general UserWarnings


plt.ion()

df = pd.read_csv("cleaned_data/cleaned_bls_data.csv")

regions = ["Midwest", "Northeast", "South", "West"]

df['year'] = pd.to_datetime(df['year'], format='%Y')

foods = [
    'All Ham (Excluding Canned Ham and Luncheon Slices), per lb. (453.6 gm)', 'All Other Pork (Excluding Canned Ham and Luncheon Slices), per lb. (453.6 gm)', 'All Pork Chops, per lb. (453.6 gm)', 'All Uncooked Beef Roasts, per lb. (453.6 gm)', 'All Uncooked Beef Steaks, per lb. (453.6 gm)', 'All Uncooked Other Beef (Excluding Veal), per lb. (453.6 gm)', 'All soft drinks, per 2 liters (67.6 oz)', 'All uncooked ground beef, per lb. (453.6 gm)', 'American processed cheese, per lb. (453.6 gm)', 'Bacon, sliced, per lb. (453.6 gm)', 'Bananas, per lb. (453.6 gm)', 'Bread, white, pan, per lb. (453.6 gm)', 'Cheddar cheese, natural, per lb. (453.6 gm)', 'Chicken breast, boneless, per lb. (453.6 gm)', 'Chicken legs, bone-in, per lb. (453.6 gm)', 'Chicken, fresh, whole, per lb. (453.6 gm)', 'Chops, boneless, per lb. (453.6 gm)', 'Chuck roast, USDA Choice, boneless, per lb. (453.6 gm)', 'Grapefruit, per lb. (453.6 gm)', 'Ground beef, 100% beef, per lb. (453.6 gm)', 'Ham, boneless, excluding canned, per lb. (453.6 gm)', 'Ice cream, prepackaged, bulk, regular, per 1/2 gal. (1.9 lit)', 'Lemons, per lb. (453.6 gm)', 'Malt beverages, all types, all sizes, any origin, per 16 oz. (473.2 ml)', 'Milk, fresh, low-fat, reduced fat, skim, per gal. (3.8 lit)', 'Milk, fresh, whole, fortified, per gal. (3.8 lit)', 'Oranges, Navel, per lb. (453.6 gm)', 'Potato chips, per 16 oz.', 'Potatoes, white, per lb. (453.6 gm)', 'Round roast, USDA Choice, boneless, per lb. (453.6 gm)', 'Spaghetti and macaroni, per lb. (453.6 gm)', 'Steak, round, USDA Choice, boneless, per lb. (453.6 gm)', 'Steak, sirloin, USDA Choice, boneless, per lb. (453.6 gm)', 'Tomatoes, field grown, per lb. (453.6 gm)', 'Wine, red and white table, all sizes, any origin, per 1 liter (33.8 oz)', 'Yogurt, per 8 oz. (226.8 gm)'
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

        series = subset.groupby('year')['value'].mean().sort_index().dropna().squeeze()

        # Pre/post split
        pre = series[(series.index.year >= 2010) & (series.index.year <= 2019)]
        post = series[series.index.year >= 2020]

        # Check enough points and variation
        if len(pre) < 2:
            print("Not enough pre-pandemic data")
            continue
        if pre.nunique() <= 1:
            print("Pre-pandemic series has no variation")
            continue
        if len(post) == 0:
            print("Not enough post-pandemic data")
            continue

        # Fit ARIMA robustly
        try:
            model = ARIMA(pre, order=(1,1,0))
            model_fit = model.fit()
        except Exception as e:
            print(f"ARIMA failed with (1,1,0): {e}")
            try:
                model = ARIMA(pre, order=(0,1,1))
                model_fit = model.fit()
            except Exception as e2:
                print(f"ARIMA failed with (0,1,1): {e2}")
                continue


        # to see years from 2020-2025
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
        plt.plot(predicted.index, predicted, label='Predicted', linestyle='--', marker='o')
        plt.title(f"{food} — {region}")
        plt.xlabel("Year")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        try:
            plt.savefig(f"ARIMA graphs/{food}_{region}.png")
            print("Plot saved successfully")
        except Exception as e:
            print("Error saving plot:", e)
