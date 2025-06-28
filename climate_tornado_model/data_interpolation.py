import pandas as pd

from scipy.interpolate import LinearNDInterpolator

def linear_feature_interpolator(data,lat_feature,lon_feature,interp_feature:str):
    """
        Given a dataframe data, performs linear interpolation of the feature named interp_feature (a string). 
            lat_feature and lon_feature are strings giving the name of the features describing latitude and longitude.

            Values of interp_feature which are not NaN are associated with the corresponding latitude and longitude, and interpolation connects these values.
    
        Outputs a dataframe with the same size and contents as data, except in the feature interp_feature, 
        any values which are presently NaN are replaced with the value obtained from the interpolation procedure.
    """
    non_NaN_data = data[[lat_feature,lon_feature,interp_feature]][~data[interp_feature].isna()]
    interp = LinearNDInterpolator(non_NaN_data[[lat_feature,lon_feature]],non_NaN_data[interp_feature])

    data.loc[:,interp_feature] = interp(data[lat_feature],data[lon_feature])

    return data

def linear_multiple_feature_interpolator(data,lat_feature,lon_feature,interp_features:list):
    '''
         Interpolates the values separately for each feature in interp_features, one at a time.
    '''
    for feature in interp_features:
        data = linear_feature_interpolator(data,lat_feature,lon_feature,feature)

    return data

def multiyear_linear_feature_interpolator(data,lat_feature,lon_feature,interp_features:list,year_feature, years:list):
    '''
        Interpolates the values for each feature in interp_features, for each year in years.
        year_feature is a string giving the label of the column of data in which the year is stored.
    '''

    for year in years:
        year_df = data[data[year_feature] == year]
        year_df = linear_multiple_feature_interpolator(year_df,lat_feature,lon_feature,interp_features)
    
        data.loc[year_df.index] = year_df

        #print(str(year) + ' completed')
    
    return data


def remove_nans_interpolator(data,lat_feature,lon_feature,interp_features,year_feature,years:list):
    '''
        Given the dataset, removes all NaN entries for the given features using interpolation, 
        then subsequently removing any values still missing.
    '''
    data = multiyear_linear_feature_interpolator(data,lat_feature,lon_feature,interp_features,year_feature,years)

    for feature in interp_features:
        data = data[~data[feature].isnull()]

    return data

def data_per_year_count(data):
    '''
        Outputs a list of tuple pairs of integers. In each tuple, the first entry is the year, and the second the number of datapoints present in data for that year.
        Sorts the output list by the second value in each tuple.
    '''
    data_per_year = [(year,data[data['DATE'] == year].shape[0]) for year in data['DATE'].unique()]
    data_per_year = sorted(data_per_year, key = lambda x: x[1])

    return data_per_year
