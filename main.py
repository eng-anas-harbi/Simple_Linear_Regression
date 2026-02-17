# Import required libraries
# pandas: for data manipulation and loading CSV files
# matplotlib: for data visualization
# numpy: for numerical operations
# train_test_split: to split dataset into training and testing sets
# LinearRegression: the regression model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load the dataset (Salary vs Years of Experience)
dataset = pd.read_csv('Salary_Data.csv')

# Extract independent variable (Years of Experience)
# Select all rows and all columns except the last column
X = dataset.iloc[:,:-1].values

# Extract dependent variable (Salary)
# Select all rows and only the last column
y = dataset.iloc[:,-1].values


# Split the dataset into training (80%) and testing (20%) sets
# random_state ensures reproducibility of results
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# Create Linear Regression model instance
regressor = LinearRegression()

# Train (fit) the model using training data
regressor.fit(X_train, y_train)


# Predict salaries for the test set
y_pred = regressor.predict(X_test)



# -----------------------------
# Visualization - Training Set
# -----------------------------
# Scatter plot of actual training data
# Plot regression line based on training data
# (Currently commented out)

# plt.scatter(X_train, y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()




# -----------------------------
# Visualization - Test Set
# -----------------------------
# Scatter plot of actual test data
plt.scatter(X_test, y_test, color='red')

# Plot regression line (based on training model)
plt.plot(X_train, regressor.predict(X_train), color='blue')

# Add chart labels and title
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# Display the plot
plt.show()
