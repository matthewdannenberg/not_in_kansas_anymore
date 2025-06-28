import pandas as pd
import numpy as np

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

def to_lat_lon_pred_df(lats,lons,preds):
    '''
        Takes in 3 one-dimensional arrays and outputs a dataframe containing the data.
    '''
    df = pd.DataFrame()
    df['preds'] = preds
    df['lats'] = lats
    df['lons'] = lons
    return df

def predictions_to_grid(predict_df,gridpts,years,column_label):
    '''
        predict_df - a dataframe containing LATITUDE, LONGITUDE, DATE, and predictions columns
        gridpts - a numpy array with shape (n,2) with each row consisting of a (longitude,latitude) pair
        years - a list of integers

        Takes the data in predict_df and interpolates it to the grid defined by gridpts, handling each year in years separately.
        The interpolation is done linearly, then any points outside the convex hull are filled by nearest neighbor interpolation.
    '''
    result_list = []
    for year in years:
        year_df = predict_df[predict_df['DATE']==year].copy()

        grid_pred = griddata(year_df[['LONGITUDE','LATITUDE']],year_df[column_label],gridpts,method='linear')
        grid_pred_df = to_lat_lon_pred_df(gridpts[:,1],gridpts[:,0],grid_pred)
        grid_pred_df = grid_pred_df[~grid_pred_df['preds'].isna()]

        filled_grid_pred = griddata(grid_pred_df[['lons','lats']],grid_pred_df['preds'],gridpts,method='nearest')
        filled_grid_pred_df = to_lat_lon_pred_df(gridpts[:,1],gridpts[:,0],filled_grid_pred)
        filled_grid_pred_df['DATE'] = year
        result_list.append(filled_grid_pred_df.copy())
    
    result = pd.concat(result_list,axis=0)
    result = result.reset_index(drop=True)
    return result

def average_predicts(grid_predict_df,bandwidth):
    '''
        grid_predict_df - a dataframe containing LATITUDE, LONGITUDE, DATE, and predictions columns. The LATITUDE, LONGITUDE entries must form a regular grid.
            Represents data from a single year.
        years - a list of integers, the years in the dataframe to be averaged
        bandwidth - a positive float, the bandwidth of the gaussian to convolve with

        Outputs an averaged version of the predictions at each datapoint, by convolving with a gaussian
    '''
    longitudes = sorted(list(grid_predict_df['lons'].unique()))
    x_step = longitudes[1] - longitudes[0]
    latitudes = sorted(list(grid_predict_df['lats'].unique()))
    y_step = latitudes[1] - latitudes[0]
    sigmas = [bandwidth/x_step, bandwidth/y_step]

    grid_predict_df = grid_predict_df.sort_values(by=['lons','lats'])
    predict_grid = np.array(grid_predict_df['preds']).reshape([len(longitudes),len(latitudes)])
    averaged_predicts = gaussian_filter(predict_grid,sigmas,mode='constant',cval=0)
    averaged_predicts = averaged_predicts.reshape(grid_predict_df['preds'].shape[0])
    
    grid_predict_df['preds'] = averaged_predicts
    return grid_predict_df
