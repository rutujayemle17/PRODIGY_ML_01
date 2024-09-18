# House Price Prediction

## Project Overview
This project aims to predict house prices using a linear regression model. The dataset used contains various features related to the properties, and the goal is to understand how different attributes, such as square footage, number of bedrooms, and bathrooms, impact the sale price of the houses.

## Purpose
The purpose of this project is to:
- Develop a predictive model to estimate house prices based on selected features.
- Evaluate the model's performance using metrics like Mean Absolute Error (MAE) and R-squared (R²).
- Visualize the results to better understand the relationship between actual and predicted prices.

## Dataset
The dataset used for this project can be found in the `Dataset` folder. It contains the following relevant features:
- `GrLivArea`: Above ground living area in square feet
- `BedroomAbvGr`: Number of bedrooms above ground
- `FullBath`: Number of full bathrooms
- `SalePrice`: The sale price of the house (target variable)

## Requirements
To run this code, ensure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn


How to Run the Code
Clone the Repository: Clone the repository to your local machine:

git clone https://github.com/yourusername/PRODIGY_ML_01.git
Navigate to the Project Directory: Change into the project directory:


Load the Dataset: Make sure the train.csv file is located in the Dataset folder as specified in the code.

Run the Code: You can run the code using a Python environment or Jupyter Notebook. If you are using a Jupyter Notebook, ensure that you have installed Jupyter:

View Results: After running the code, the output will display the Mean Absolute Error and R-squared value. Additionally, a scatter plot will visualize the actual vs. predicted house prices.

Conclusion
This project demonstrates a basic implementation of linear regression for predictive modeling in Python. Further improvements could involve feature engineering, model tuning, and exploring other machine learning algorithms for better accuracy.
