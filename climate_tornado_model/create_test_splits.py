import pandas as pd

def create_testing_year_data_split(data, test_years, features_to_interpolate):
    '''
        Isolates all data from the specified list of the years in the given climate dataframe. 
        That data is set as a test set, with the remaining data set as a corresponding training set.
        (currently removed) Following that, missing values for the specified features are filled in by interpolation.
    '''
    test_year_df = data[data['DATE'].isin(test_years)]
    #test_year_df = remove_nans_interpolator(data, 'LATITUDE','LONGITUDE', features_to_interpolate,'DATE',test_years)

    train_year_df = data[~data['DATE'].isin(test_years)]
    #train_year_df = remove_nans_interpolator(train_year_df, 'LATITUDE','LONGITUDE', features_to_interpolate,'DATE',train_year_df['DATE'].unique())

    return (train_year_df,test_year_df)

from sklearn.model_selection import LeaveOneGroupOut

def create_all_test_train_splits(data,test_years,features_to_interpolate):
    '''
        Given the input dataframe, outputs training and testing splits of each of the desired types.
        Output is of the form
            (train_df,test_df,remaining_df,cv_splits)
           with the first two containing dataframes for the main training and test split, 
           the third containing all the non-test data, and the last containing data splits for cross validation.
    '''
    
    train_df,test_df = create_testing_year_data_split(data,test_years,features_to_interpolate)
    remaining_df = data[~data['DATE'].isin(test_years)].copy()

    remaining_df['decade'] = (remaining_df['DATE'] // 10) * 10
    cv_splits = LeaveOneGroupOut().split(remaining_df,groups=remaining_df['decade'])

    remaining_df = remaining_df.drop('decade',axis=1)
    return (train_df,test_df,remaining_df,cv_splits)
