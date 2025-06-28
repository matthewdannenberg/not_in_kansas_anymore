# not_in_kansas_anymore:
*An Exploratory Project Seeking to Understand the Shift of Tornado Alley*

Is Tornado Alley moving over time? Should we adjust our intuitive idea of where in the US is most susceptible to tornadoes? Would a modern Dorothy have started in another state before going to Oz? In this project, we've analyzed NOAA datasets containing tornado, hurricane, and overall climate data to try to answer these questions. We explore a number of different approaches, from the use of summary statistics to some time series style analysis to prediction based on climate data.

One should begin by asking: "What is Tornado Alley?" There is no formal definition, so we say that _in a given period of time, Tornado Alley is the region of the US most susceptible to high-strength tornadoes._ Our exact interpretation of that statement will vary in each of the three main portions of our analysis.

Method 1: 
    Regression -- Refer to the readme in the method_1_regression folder.

*Method 2: 
    Using Gaussian Kernel Density Estimation to obtain peak latitude and longitude data, and next, using this data on three prediction methods: Linear Regression, Gaussian White Noise model, and Gaussian Random Walk. The results of the first two methods seemed more accurate. A forecast for the year 2030 has been provided to show where the tornado alley will be (including high-intensity tornadoes only). A check using testing data has also been performed.

Climate Modeling of Tornado Alley:
    The relevant datasets can be downloaded using the two .py files in the data_download folder. Following that, all relevant analysis and discussion is found in the climate_tornado_model/tornado_alley_climate_model.ipynb notebook, with helper functions placed in scripts in the same folder.