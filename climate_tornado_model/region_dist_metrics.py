import numpy as np
import pandas as pd
from interpolate_to_grid import *

def gridpt_dist_sq(coords,pt1,pt2):
    ''' 
        coords - a 3-dimensional numpy array giving x and y coordinates for each point in the grid. Dimensions n*m*2
        pt1,pt2 - tuple pairs of integer indices, the first between 0 and n, the second between 0 and m

        Returns the squared Euclidean distance between the coordinates at index pt1 and the coordinates at index pt2.
    '''
    pos1 = coords[pt1]
    pos2 = coords[pt2]
    return np.sum((pos1 - pos2)**2)

def region_size(region):
    '''
        region - 2-dimensional numpy array with dimensions n*m, with all entries NaN or 1. 
            The region specified consists of those grid points for which the entries are 1.

        Outputs the number of entries in the region which are 1.
    '''
    return np.nansum(region)

def max_min_dist_sq(grid_dists_sq,region1,region2):
    '''
        grid_dists_sq - a 4-dimensional numpy array giving the distances between point (i,j) and (k,l) in the grid. Dimensions n*m*n*m
        region1 and region2 - 2-dimensional numpy arrays with dimensions n*m, with all entries NaN or 1. 
            The regions specified consist of those grid points for which the entries are 1.

        Returns the squared maximum (taken over all points in region1) of the minimum distance between that point and points in region 2.
    '''
    return np.nanmax(np.nanmin(grid_dists_sq*region2,axis=(2,3))*region1)

def hausdorff_dist(grid_dists_sq,region1,region2):
    ''' 
        grid_dists_sq - a 4-dimensional numpy array giving the distances between point (i,j) and (k,l) in the grid. Dimensions n*m*n*m
        region1 and region2 - 2-dimensional numpy arrays with dimensions n*m, with all entries NaN or 1. 
            The regions specified consist of those grid points for which the entries are 1.

        Returns the Hausdorff distance between the two regions specified.
    '''
    return np.sqrt(max(max_min_dist_sq(grid_dists_sq,region1,region2),max_min_dist_sq(grid_dists_sq,region2,region1)))

def l1_dist_helper(grid_dists_sq,region1,region2):
    '''
        grid_dists_sq - a 4-dimensional numpy array giving the distances between point (i,j) and (k,l) in the grid. Dimensions n*m*n*m
        region1 and region2 - 2-dimensional numpy arrays with dimensions n*m, with all entries NaN or 1. 
            The regions specified consist of those grid points for which the entries are 1.

        Returns the average (taken over all points in region1) of the minimum distance between that point and points in region2.
    '''
    return np.nansum(np.sqrt(np.nanmin(grid_dists_sq*region2,axis=(2,3))*region1)) / region_size(region1)

def l1_dist(grid_dists_sq,region1,region2):
    ''' 
        grid_dists_sq - a 4-dimensional numpy array giving the distances between point (i,j) and (k,l) in the grid. Dimensions n*m*n*m
        region1 and region2 - 2-dimensional numpy arrays with dimensions n*m, with all entries NaN or 1. 
            The regions specified consist of those grid points for which the entries are 1.

        Returns the symmetrized sum (taken over all points in region1) of the minimum distance between that point and points in region2
    '''
    return l1_dist_helper(grid_dists_sq,region1,region2) + l1_dist_helper(grid_dists_sq,region2,region1)

def region_diff(tornado_alley_list,predict_df,grid_dists_sq,positions,averaging_width,decision_threshold,dist_choice):
    ''' 
        tornado_alley_list - a dictionary of tuple pairs. Each key is a year with the value a 2-dimensional arrays of 0s and 1s of size n*m, 
            with 1 representing a grid point considered to be inside tornado alley in the ground truth data in that year.
        predict_df - a dataframe containing 'LATITUDE', 'LONGITUDE', 'DATE', and 'predictions' columns
        grid_dists_sq - a 4-dimensional numpy array giving the distances between point (i,j) and (k,l) in the grid. Dimensions n*m*n*m
        positions - a 2-dimensional numpy array representing an ordered list of longitude, latitude coordinates. Dimensions 2 by nm
        averaging_width - the latitude/longitude widths of the Gaussian convolved with the data
        decision_threshold - the predicted probability level above which a grid point will be considered 'accepted'.
        dist_choice - a string, should be either 'hausdorff' or 'l1' 

            The years in tornado_alley_list MUST encompass the set of distinct values in predict_df['DATE'].

        Returns the distance in the specified metric between the region denoted by tornado_alley and the 
            predicted truth values interpolated to a grid, locally averaged, with a prediction considered in the
            predicted tornado alley if it falls above the specified threshold
    '''
    years = predict_df['DATE'].unique()

    grid_pred_df = predictions_to_grid(predict_df,positions.transpose(),years,'predictions')

    score = 0
    for year in years:
        year_df = grid_pred_df[grid_pred_df['DATE'] == year]
        year_df = average_predicts(year_df, averaging_width)
        year_preds = np.array(year_df['preds']).reshape(tornado_alley_list[year].shape)
        pred_tornado_alley = (year_preds > decision_threshold)
        pred_tornado_alley = np.where(pred_tornado_alley == 0, np.nan, pred_tornado_alley)
        if region_size(pred_tornado_alley) == 0:
            return np.inf

        tornado_alley = tornado_alley_list[year]
        tornado_alley = np.where(tornado_alley == 0, np.nan, tornado_alley)

        if dist_choice == 'hausdorff':
            score += hausdorff_dist(grid_dists_sq,tornado_alley,pred_tornado_alley)
        elif dist_choice == 'l1':
            score += l1_dist(grid_dists_sq,tornado_alley,pred_tornado_alley)

    return score/len(years)