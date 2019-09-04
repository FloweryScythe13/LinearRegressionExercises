'''
This script perfoms the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)


The data is downloaded from URL, which is defined below. As is normal
for machine learning problems, the nature of the source data affects
the entire solution. When you change URL to refer to your own data, you
will need to review the data processing steps to ensure they remain
correct.

============
Example Data
============
The example is from http://mldata.org/repository/data/viewslug/stockvalues/
It contains stock prices and the values of three indices for each day
over a five year period. See the linked page for more details about
this data set.

This script uses regression learners to predict the stock price for
the second half of this period based on the values of the indices. This
is a naive approach, and a more robust method would use each prediction
as an input for the next, and would predict relative rather than
absolute values.
'''

# Remember to update the script for the new data when you change this URL
#URL = "http://mldata.org/repository/data/download/csv/stockvalues/"

# This is the column of the sample data to predict.
# Try changing it to other integers between 1 and 155.
TARGET_COLUMN = 32

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# =====================================================================

def get_split_data(ozone):
    np.random.seed(27)
    ozone.is_train = np.random.uniform(0, 1, len(ozone)) <= 0.7
    ozone_train = ozone[ozone.is_train]
    ozone_test = ozone[ozone.is_train == False]
    return ozone_train, ozone_test


# =====================================================================


def evaluate_learner(ozone_train, ozone_test):
    '''
    Train and evaluate the regression model 
    '''
    model_ln_regression = LinearRegression(normalize=True)
    # The below uses all columns except for Ozone as the X data, and the Ozone column as Y data
    model_ln_regression = model_ln_regression.fit(ozone_train.drop('ozone', axis=1), ozone_train['ozone'])
    print(model_ln_regression.coef_)

    # Now let's run the model on test data to generate predictions
    ozone_ln_pred = model_ln_regression.predict(ozone_test.drop('ozone', axis=1))
    ozone_ln_residuals = ozone_ln_pred - ozone_test['ozone']
    
    ozone_ln_mae = metrics.mean_absolute_error(ozone_test['ozone'], ozone_ln_pred)
    ozone_ln_rmse = sqrt(metrics.mean_squared_error(ozone_test['ozone'], ozone_ln_pred))
    ozone_ln_r2 = metrics.r2_score(ozone_test['ozone'], ozone_ln_pred)
    print("MAE: " + str(ozone_ln_mae) + "\nRMSE: " + str(ozone_ln_rmse) 
      + "\nCoefficient of Determination: " + str(ozone_ln_r2))


    plt.figure(2)
    plt.scatter(ozone_test['ozone'], ozone_ln_residuals)
    plt.ylabel('Residuals')
    plt.xlabel('True Values')
    plt.figure(3)
    plt.scatter(ozone_ln_pred, ozone_ln_residuals)
    plt.ylabel('Residuals')
    plt.xlabel('Predicted Values')
    

    # ==================================
    # Display the plot in interactive UI
    plt.show()

    


# =====================================================================




# =====================================================================


if __name__ == '__main__':
    # Read the dataset from local file
    ozone = pd.read_csv('Datasets/ozone.data', delimiter='\t')

    # Visualization
    ozone.describe()
    pd.plotting.scatter_matrix(ozone, figsize=[20,20])
    plt.show()

    ozone_train, ozone_test = get_split_data(ozone)

    # Evaluate multiple regression learners on the data
    print("Evaluating regression learner")
    evaluate_learner(ozone_train, ozone_test)

  
