from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

df =pd.read_csv("E:\\house_prices.csv")
x=df[["size_sqft","bedrooms","bathrooms","age_years"]]
y=df["price_thousands"]

poly=PolynomialFeatures(degree=2)

poly_x=poly.fit_transform(x)

model=Ridge(alpha=1)

model.fit(poly_x, y)

predictions=model.predict(poly_x)
print(predictions)