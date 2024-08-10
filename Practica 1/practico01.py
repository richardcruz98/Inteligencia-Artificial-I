
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generating random data for height (m) and weight (kg)
np.random.seed(0)
heights = np.random.uniform(1.4, 2.0, 100)  # Height between 1.4m and 2.0m
weights = []

# Generating controlled random weights based on height
for h in heights:
    weight = np.random.uniform(18.5 * h ** 2, 25 * h ** 2)  # BMI between 18.5 and 25
    weights.append(weight)

# Create a DataFrame to store the data
data = pd.DataFrame({
    'Height (m)': heights,
    'Weight (kg)': weights
})

# Define a function for the model (linear in this case)
def linear_model(x, a, b):
    return a * x + b

# Fit the curve to the data
popt, pcov = curve_fit(linear_model, data['Height (m)'], data['Weight (kg)'])

# Get the parameters of the fitted line
a, b = popt

# Generate the predicted weights
predicted_weights = linear_model(data['Height (m)'], a, b)

# Plotting the data and the fitted line
plt.scatter(data['Height (m)'], data['Weight (kg)'], label='Data')
plt.plot(data['Height (m)'], predicted_weights, color='red', label=f'Fitted line: y = {{a:.2f}}x + {{b:.2f}}')
plt.xlabel('Height (m)')
plt.ylabel('Weight (kg)')
plt.title('Height vs. Weight with Fitted Line')
plt.legend()
plt.show()
