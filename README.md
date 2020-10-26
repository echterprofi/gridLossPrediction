# gridLossPrediction

This is part of master thesis about using publicly available datasets to predict the power grid losses 24 hours into the future with 1-hour resolution.
The used data come mainly from two sources:

1- German Weather servive (dwd.de): weather data such as air temperature, relative humidity, dew point, zenith and others.

2- 50Hertz: This is a Transmission System OPerator in Germany and it makes available data such as Grid Losses, Grid Feed in, Vertical Grid Load, Forecasted and actual Solar Power, Forecasting and actual wind power.

The raw files from these sources needs to be processed. This is done with the help of the python script "dataUtilities.py".

After the data is processed, they can then be applied to a Deep Learning model inside the python script "gridLossPredictor.py"

