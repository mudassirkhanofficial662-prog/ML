from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

df=pd.read_csv("E:\\house_prices.csv")
x=df[["size_sqft","bedrooms","bathrooms","age_years"]]
y=df["price_thousands"]
from sklearn.linear_model import Ridge
model=Ridge(alpha=1)
model.fit(x,y)
po=model.predict(x)
print(po)