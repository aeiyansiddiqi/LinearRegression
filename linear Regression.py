import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pandas.read_csv(‘Housin_Price_Data.csv’)
x = data[['area', 'bedrooms', 'bathrooms', 'stories']] # independent variables
y = data['price'] # dependant variables

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred, color='blue')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

plt.title('Housing Price Predictions')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.ticklabel_format(style='plain', axis='both')
plt.show()