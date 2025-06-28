import pandas as pd
import numpy as np

from scipy.stats import gaussian_kde
from math import sin, cos, sqrt, atan2, radians

def dist_from_latlon(pt1, pt2):
    '''
        Computes the distance, in miles, between pt1 and pt2 on Earth's surface, where pt1 and pt2 are (latitude,longitude) pairs.
            Gives a valid formula for points in or near the US.
    '''
    R = 3963.1

    lat1 = radians(pt1[0])
    lon1 = radians(pt1[1])
    lat2 = radians(pt2[0])
    lon2 = radians(pt2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def yearly_tornado_distributions(tornado_df,bandwidth, years, quantile_to_remove):
    '''
        tornado_df - a dataframe containing tornado data. Year information stored in 'year_bin' column, latitude and longitude stored in 'begin_lat' and 'begin_lon' columns, respectively
        bandwidth - a float, the width the kde
        years - a list of all year_bins, all ints
        quantile_to_remove - a float between 0 and 1. Tornadoes whose distance from the mean tornado position in that bin 
            are further than this quantile of the distribution will be excluded from analysis. We want only to consider tornadoes in the 'center' of the region of susceptibility

        Outputs a function which takes a year (an int) and a position (a LONGITUDE, LATITUDE pair) as input and outputs the value of the kde at that point for that year
    '''

    yearly_tornado_dists = {}
    for year in years:
        year_tornadoes = tornado_df[tornado_df['year_bin'] == year].copy()

        mean_x = np.mean(year_tornadoes['begin_lon'].values)
        mean_y = np.mean(year_tornadoes['begin_lat'].values)

        year_tornadoes['dist_from_center'] = year_tornadoes.apply(lambda row: dist_from_latlon((mean_y,mean_x), (row['begin_lat'],row['begin_lon'])),axis=1)
        threshold = year_tornadoes['dist_from_center'].quantile(quantile_to_remove)
        year_tornadoes = year_tornadoes[year_tornadoes['dist_from_center'] < threshold]

        xvals = year_tornadoes['begin_lon'].values
        yvals = year_tornadoes['begin_lat'].values

        if len(xvals) == 0:
            print('Warning: No data is present in the bin ' + str(year))
        else:
            positions = np.vstack([xvals,yvals])

            dist = gaussian_kde(positions,bw_method=bandwidth)

            yearly_tornado_dists[year] = dist
    
    def output_func(year,position):
        return len(xvals) * yearly_tornado_dists[((year)//5)*5](position)

    return output_func

