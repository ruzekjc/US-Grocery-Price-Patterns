import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("cleaned_data/cleaned_bls_data.csv")

# Convert year to datetime index
df['year'] = pd.to_datetime(df['year'], format='%Y')
df.set_index('year', inplace=True)

df = df.sort_index()

pre_pandemic = df['2010':'2019']
post_pandemic = df['2020':]

model = ARIMA(pre_pandemic['value'], order=(1,1,0))
model_fit = model.fit()

# forecast post-pandemic years

n_years = len(post_pandemic)
forecast = model_fit.get_forecast(steps=n_years)
predicted = forecast.predicted_mean


# compare actual vs. predicted
comparison = post_pandemic.copy()
comparison['predicted'] = predicted.values
comparison['difference'] = comparison['value'] - comparison['predicted']
comparison['pct_change'] = (comparison['difference'] / comparison['predicted']) * 100