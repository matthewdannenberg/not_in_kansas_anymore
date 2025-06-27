
To generate the figure maps: Run 1-data_and_cleaning.ipynb to generate the csv files. Then run 3-map_figure_making_all_tornadoes.ipynb for all tornadoes and 5-map_figure_making_strong_tornadoes.ipynb for strong tornadoes (for tornado alley).


1-data_and_cleaning: This notebook downloads the tornado data from NOAA, filters out the states outside mainland USA, filters out the tornado events below 31 degree latitude, and deleted the furthest 10 percent tornado events. Creates the csv files with all tornadoes for "tornado region" (regressdf_all.csv) and with high intensity tornadoes - "tornado alley"  (regressdf.csv).

2-Linear_regression_all_tornadoes: This notebook does a linear regression on all tornadoes, calculates the coefficients and predicts the parameters for tornado region for a given input year. Also does k-fold cross validation and prints our mean MSE and MAE scores.

3-map_figure_making_all_tornadoes : This notebook takes the csv files as input and creates the figures: map of mainland USA with our ellipse tornado region marked on it for the year which we input. It also does linear regression on the training data, and makes another map figure with predicted tornado region and "ground truth" tornado region (usually for testing the model on the test set of data).

4-regression_different_models_tornado_alley : This notebook does a linear, polynomial and kNN regression on strong tornadoes. Also does k-fold cross validations and prints our mean MSE and MAE scores.

5-map_figure_making_strong_tornadoes : This notebook takes the csv files as input and creates the figures: map of mainland USA with our ellipse tornado region marked on it for the year which we input. It also does linear regression on the training data, and makes another map figure with predicted tornado alley and "ground truth" tornado alley (usually for testing the model on the test set of data).
    
