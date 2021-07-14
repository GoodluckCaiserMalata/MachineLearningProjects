import pandas as pd 
import numpy as np 

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)

features = data.drop(['INDUS', 'AGE'], axis=1)
log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=['PRICE'])

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PT_RATIO = 8

property_stats = np.ndarray(shape= (1, 11))
property_stats[0][CRIME_IDX] = features['CRIM'].mean()
property_stats[0][ZN_IDX] = features['ZN'].mean()
property_stats[0][CHAS_IDX] = features['CHAS'].mean()
property_stats = features.mean().values.reshape(1,11)

regr = LinearRegression().fit(features,target)
fitted_vals = regr.predict(features)

MSE = mean_squared_error(target, fitted_vals)
RSME = np.sqrt(MSE)

def get_log_price(nr_rooms, 
                  students_per_class,
                 next_to_river=False, 
                 high_confidence=True):
    
    # Configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PT_RATIO] = students_per_class
    
    # make a prediction 
    log_estimate = regr.predict(property_stats)[0][0]
    
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
        
    # calc range
    if high_confidence:
        lower_bound = log_estimate - 2*RSME
        upper_bound = log_estimate + 2*RSME
        interval = 95
    else: 
        lower_bound = log_estimate - RSME
        upper_bound = log_estimate + RSME
        interval = 68
        
    
    return log_estimate, upper_bound, lower_bound, interval


np.median(boston_dataset.target)
zillow_median_price = 583.3
scale_factor = zillow_median_price / np.median(boston_dataset.target)

log_est, upper, lower, conf = get_log_price(9, 
                                           students_per_class=15,
                                           next_to_river=False, 
                                           high_confidence=False)



dollar_estimate = np.e**log_est*1000*scale_factor
dollar_hi = np.e**upper*1000*scale_factor
dollar_lo = np.e**lower*1000*scale_factor

# round the estimate to the next thousands

rounded_dollar_value = round(dollar_estimate, -3)
rounded_upper_value = round(dollar_hi, -3)
rounded_lower_value = round(dollar_lo, -3)

# rounded_value
# rounded_lower_value
# rounded_upper_value

print('The estimate dollar price:', rounded_dollar_value)
print(f'with a confidence of {conf}: the range is as below')
print(f'{rounded_lower_value}, to {rounded_upper_value}')