from reusable import data_cleaning
from reusable import date_cnvrsn
from reusable import feature_engineering
from reusable import dropper
from reusable import lower_bound
from reusable import bd_outlier
from reusable import categorical_conversion
from reusable import transformation



# Function to preprocess input data for prediction
def preprocess(data):
    data['gender'] = data['gender'].map({'Male': 'male', 'Female': 'female'})
    data['is_auto_renew'] = data['is_auto_renew'].map({'Yes': 1, 'No': 0})
    data['is_cancel'] = data['is_cancel'].map({'Yes': 1, 'No': 0})
    # Mapping cities explicitly
    city_mapping = {
        'city1': 1, 'city2': 2, 'city3': 3, 'city4': 4, 'city5': 5,
        'city6': 6, 'city7': 7, 'city8': 8, 'city9': 9, 'city10': 10,
        'city11': 11, 'city12': 12, 'city13': 13, 'city14': 14, 'city15': 15,
        'city16': 16, 'city17': 17, 'city18': 18, 'city19': 19, 'city20': 20, 'city21': 21
    }
    data['city'] = data['city'].map(city_mapping)


    # Mapping registered_via explicitly
    registered_via_mapping = {
        'Registered Via 3': 3, 'Registered Via 4': 4, 'Registered Via 7': 7, 'Registered Via 9': 9, 'Registered Via 13': 13
    }
    data['registered_via'] = data['registered_via'].map(registered_via_mapping)


    # Mapping payment_method_id explicitly
    payment_method_id_mapping = {
        'Payment Method 3': 3, 'Payment Method 6': 6, 'Payment Method 8': 8, 'Payment Method 10': 10, 'Payment Method 11': 11,
        'Payment Method 12': 12, 'Payment Method 13': 13, 'Payment Method 14': 14, 'Payment Method 15': 15, 'Payment Method 16': 16,
        'Payment Method 17': 17, 'Payment Method 18': 18, 'Payment Method 19': 19, 'Payment Method 20': 20, 'Payment Method 21': 21,
        'Payment Method 22': 22, 'Payment Method 23': 23, 'Payment Method 26': 26, 'Payment Method 27': 27, 'Payment Method 28': 28,
        'Payment Method 29': 29, 'Payment Method 30': 30, 'Payment Method 31': 31, 'Payment Method 32': 32, 'Payment Method 33': 33,
        'Payment Method 34': 34, 'Payment Method 35': 35, 'Payment Method 36': 36, 'Payment Method 37': 37, 'Payment Method 38': 38,
        'Payment Method 39': 39, 'Payment Method 40': 40, 'Payment Method 41': 41
    }
    data['payment_method_id'] = data['payment_method_id'].map(payment_method_id_mapping)

    data_cleaning(data)
    date_cnvrsn(data)
    feature_engineering(data)
    dropper(data)
    data = lower_bound(data)
    data = bd_outlier(data)
    data = categorical_conversion(data)
    transformation(data)


    return data

    
    






    
