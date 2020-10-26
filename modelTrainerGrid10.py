# The following two lines will force the processing on the CPU not the GPU
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import *
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers.merge import concatenate
from bokeh.io import output_file, show
from bokeh.plotting import figure

# How many raw records to apply to the model (e.g. How many days worth of data to apply to the model)
dataPointsCount = 24 * 3

# How many predicted data points (e.g. 24 for the day ahead prediction)
outputCardinality = 24

# relative or absolute path (mind the escape characters) to the raw training data-set
fileNameTraining = "gridData/GridFeedIn/Netzverluste_Combined.csv"

# relative or absolute path (mind the escape characters) to the raw training data-set
fileNameValidation = "gridData/Netzverluste_2019.csv"


def prepareData(fileName, option="SLIDING_WINDOW", dateFormat="%d.%m.%Y %H"):
    """
    This function performs the pre-processing on raw data (weather, grid data and other).
    Inputs:
        - "fileName": String value representing the relative or absolute path (mind the escape characters) to the file with
        the raw data-set.

        - "option": String value with two possible values:
            -"SLIDING_WINDOW": A sliding window with a width equal to the value of the variable "dataPointsCount"
            traverses the entire data-set. The window starts at index 0 and moves one record per iteration.
            This option is to be used when the data-set is limited and there is a need to enlarge it. So instead of
            taking full days from 00:00 to 23:00, the starting hour will change by one each iteration and the length of
            the record will equal to value of "dataPointsCount".

            - "SEPARATE_INTERVALS": with each iteration the pointer will move not by 1 but by the value of the variable
            "outputCardinality" which usually has a value of 24, since the model produces 24 forecasted values
             for the day ahead.

        - "dateFormat": How should the date field be processed. Optional string with a default value of "%d.%m.%Y %H".
    Output:
    The functions outputs 3 numpy float64 arrays with equal lengths:
    - finalFileGridFeed: input data to the grid feed in model that predicts the value of the grid feed in
    - finalFileGridLoss: input data to the grid loss model that predicts the value of the grid loss
    - timeStamps: an array of the time stamps (date + hour) that correspond to each value of the two previous arrays
    """

    customDateParser = lambda x: datetime.strptime(x, "%d.%m.%Y %H")
    dataGrid = pd.read_csv(fileName, encoding='unicode_escape', parse_dates=['Date'], date_parser=customDateParser)
    data = pd.DataFrame.to_numpy(dataGrid.iloc[:, 0:])
    finalFileGridFeed = []
    finalFileGridLoss = []
    timeStamps = []

    loopRange = len(data) - dataPointsCount - outputCardinality

    if option == "SLIDING_WINDOW":
        loopRange = np.arange(0, loopRange, 1)
    elif option == "SEPARATE_INTERVALS":
        loopRange = np.arange(0, int(np.floor(loopRange / outputCardinality)) * outputCardinality, outputCardinality)

    for j in loopRange:
        i = j + dataPointsCount

        # index of the first record in the raw data representing the input to the DNN
        firstNow = j

        # index of the last record in the raw data representing the input to the DNN
        lastNow = i

        # index of the first record in the raw data representing the output (ground truth) to the DNN
        firstNext = i

        # index of the last record in the raw data representing the output (ground truth) to the DNN
        lastNext = firstNext + outputCardinality

        timeStampNow = data[firstNow:lastNow, 0]

        timeStampNext = data[firstNext:lastNext, 0]

        gridLossNow = data[firstNow:lastNow, 1] / 1000
        gridLossNow = np.reshape(gridLossNow, (1, -1)).squeeze()

        gridLossNext = data[firstNext:lastNext, 1] / 1000
        gridLossNext = np.reshape(gridLossNext, (1, -1)).squeeze()

        yearNext = (data[firstNext:lastNext, 2] - 2000) / 50
        yearNext = np.reshape(yearNext, (1, -1)).squeeze()

        dayOffFactorNext = data[firstNext:lastNext, 3]
        dayOffFactorNext = np.reshape(dayOffFactorNext, (1, -1)).squeeze()

        hourNext = data[firstNext:lastNext, 4] / 24
        hourNext = np.reshape(hourNext, (1, -1)).squeeze()

        weekdayNext = data[firstNext:lastNext, 5] / 7
        weekdayNext = np.reshape(weekdayNext, (1, -1)).squeeze()

        monthNext = data[firstNext:lastNext, 6] / 12
        monthNext = np.reshape(monthNext, (1, -1)).squeeze()

        gridFeedInNow = data[firstNow:lastNow, 7] / 100000
        gridFeedInNow = np.reshape(gridFeedInNow, (1, -1)).squeeze()

        gridFeedInNext = data[firstNext:lastNext, 7] / 100000
        gridFeedInNext = np.reshape(gridFeedInNext, (1, -1)).squeeze()

        airTempNext = (data[firstNext:lastNext, 8:15] + 50) / 100
        airTempNext = np.reshape(airTempNext, (1, -1)).squeeze()

        dewPointNext = (data[firstNext:lastNext, 15:22]) / 100
        dewPointNext = np.reshape(dewPointNext, (1, -1)).squeeze()

        soilTempNext = (data[firstNext:lastNext, 22:29] + 50) / 100
        soilTempNext = np.reshape(soilTempNext, (1, -1)).squeeze()

        windSpeedNext = (data[firstNext:lastNext, 29:36]) / 50
        windSpeedNext = np.reshape(windSpeedNext, (1, -1)).squeeze()

        solarZenithNext = (data[firstNext:lastNext, 36:42]) / 180
        solarZenithNext = np.reshape(solarZenithNext, (1, -1)).squeeze()

        solarPowerForecastNext = (data[firstNext:lastNext, 42]) / 50000
        solarPowerForecastNext = np.reshape(solarPowerForecastNext, (1, -1)).squeeze()

        windPowerForecastNext = (data[firstNext:lastNext, 43]) / 75000
        windPowerForecastNext = np.reshape(windPowerForecastNext, (1, -1)).squeeze()

        windPowerActualNow = (data[firstNow:lastNow, 44]) / 75000
        windPowerActualNow = np.reshape(windPowerActualNow, (1, -1)).squeeze()

        verticalGridLoadNow = (data[firstNow:lastNow, 45] + 40000) / 80000
        verticalGridLoadNow = np.reshape(verticalGridLoadNow, (1, -1)).squeeze()

        solarPowerActualNow = (data[firstNow:lastNow, 46]) / 50000
        solarPowerActualNow = np.reshape(solarPowerActualNow, (1, -1)).squeeze()

        airHumidityNext = (data[firstNext:lastNext, 47:54]) / 100
        airHumidityNext = np.reshape(airHumidityNext, (1, -1)).squeeze()

        outputGridFeed = np.concatenate(
            (gridFeedInNow, hourNext, weekdayNext, monthNext,  # yearNext,  # dayOffFactorNext,
             windSpeedNext,
             dewPointNext, airTempNext, soilTempNext, gridLossNow, verticalGridLoadNow,
             solarZenithNext, windPowerForecastNext, solarPowerForecastNext, windPowerActualNow,
             solarPowerActualNow, gridFeedInNext),
            axis=0)

        outputGridLoss = np.concatenate(
            (
                gridLossNow, windSpeedNext,  # hourNext, weekdayNext, monthNext, yearNext,
                soilTempNext, dewPointNext,
                airTempNext, solarZenithNext,  # dayOffFactorNext,
                windPowerActualNow,
                solarPowerForecastNext, windPowerForecastNext, verticalGridLoadNow, gridFeedInNow,
                gridLossNext),
            axis=0)

        finalFileGridFeed.append(outputGridFeed)
        finalFileGridLoss.append(outputGridLoss)
        timeStamps.append(timeStampNext)
    return np.asarray(finalFileGridFeed).astype('float64'), np.asarray(finalFileGridLoss).astype('float64'), np.asarray(
        timeStamps)


def trainModel(retrain=False, batchSize=100, dataYears=6, patience=20, kernelInitializer='normal'):
    """
    This function creates and trains a deep neural network with the following specs:
    Number of layers for both the grid feed in and the grid loss models: neuralNetworkDepth
    Number of neurons for grid feed in model: neuronsPerLayerGridFeedIn
    Number of neurons for grid loss model: neuronsPerLayerGridLoss

    The model has two separate networks that merge into a bigger model: One model for the grid feed in prediction
    and another for the gris loss prediction.

    Inputs:
    - "retrain": Boolean value when true, the pre-trained model will be loaded and retrained with the data-set. Otherwise a fresh model
    is created and trained. Optional argument with a default value of "False".

    - "batchSize": Integer value representing the batch size in records count to the for the fit function
    dataYears: Integer value representing the number of years take from the training dataset. a value of 1 means only the
    latest year data is applied to the model. Optional arguments with a default value of 100.

    - "patience": Integer value representing the number of epochs without improvement to the validation error before
    training auto stops. Optional arguments with a default value of 20.

    - "kernelInitializer": How should the weights of the model be initialized. example values: 'normal', 'random_normal',
    'zeros'. Optional arguments with a default value of 100. Optional arguments with a default value of 'normal'.

    The function calls the prepareData function to process the raw data then applies it to the model.
    For training data the option of "SLIDING_WINDOW" is used to increase the training data, however, for validation data
    the option "SEPARATE_INTERVALS" is used since it closely resembles the real life situation.

    With every epoch, the validation error (MSE) for the grid loss prediction is examined. Whenever a better
    validation error is attained, the model is saved to the hard disk under the name "model.h5".
    The training stops when either of the following conditions is met:
    1- Max number of epochs is reached (default value 50000)
    2- Patience condition is violated (i.e more epochs without improvement than defined by patience value -default value 20-)

    The following counter-measures are taken to avoid over-fitting:
    - A dropout layer with 20% is added to both models
    - The testing and validation data are completely separate, they even have separate files.
    """

    neuralNetworkDepth = 2
    neuronsPerLayerGridFeedIn = 100
    neuronsPerLayerGridLoss = 100
    trainingRecordsCount = dataYears * 365 * 24

    datasetGridFeedTraining, datasetGridLossTraining, timestampsTraining = prepareData(fileNameTraining,
                                                                                       "SLIDING_WINDOW")
    if trainingRecordsCount > len(datasetGridLossTraining):
        trainingRecordsCount = len(datasetGridLossTraining)

    datasetGridFeedTraining = datasetGridFeedTraining[-1 * trainingRecordsCount: -1, :]
    datasetGridLossTraining = datasetGridLossTraining[-1 * trainingRecordsCount: -1, :]

    datasetGridFeedValidation, datasetGridLossValidation, timestampsValidation = prepareData(fileNameValidation,
                                                                                             "SEPARATE_INTERVALS")
    inputCardinalityGridFeed = len(datasetGridFeedTraining[0]) - outputCardinality
    inputCardinalityGridLoss = len(datasetGridLossTraining[0]) - outputCardinality

    inputGridFeedIn = Input(shape=(inputCardinalityGridFeed,))
    hiddenGridFeedIn = Dense(neuronsPerLayerGridFeedIn, activation='relu', kernel_initializer=kernelInitializer)(
        inputGridFeedIn)
    for i in range(neuralNetworkDepth - 1):
        hiddenGridFeedIn = Dense(neuronsPerLayerGridFeedIn, activation='relu', kernel_initializer=kernelInitializer)(
            hiddenGridFeedIn)
        hiddenGridFeedIn = Dropout(0.2)(hiddenGridFeedIn)
    outputGridFeedIn = Dense(outputCardinality, activation="relu", kernel_initializer=kernelInitializer)(
        hiddenGridFeedIn)

    inputGridLoss = Input(shape=(inputCardinalityGridLoss,))
    merge = concatenate([inputGridLoss, outputGridFeedIn])
    hiddenGridLoss = Dense(neuronsPerLayerGridLoss, activation='relu', kernel_initializer=kernelInitializer)(merge)
    for i in range(neuralNetworkDepth - 1):
        hiddenGridLoss = Dense(neuronsPerLayerGridLoss, activation='relu', kernel_initializer=kernelInitializer)(
            hiddenGridLoss)
        hiddenGridLoss = Dropout(0.2)(hiddenGridLoss)
    outputGridLoss = Dense(outputCardinality, activation="linear", kernel_initializer=kernelInitializer)(hiddenGridLoss)

    model = Model(inputs=[inputGridFeedIn, inputGridLoss], outputs=[outputGridFeedIn, outputGridLoss])
    model.summary()
    model.compile(loss="mean_squared_error", optimizer="Adamax", metrics=["mean_squared_error"])
    if retrain:
        model = load_model('model.h5')

    x_train_gridFeed = datasetGridFeedTraining[:, 0:inputCardinalityGridFeed]
    y_train_gridFeed = datasetGridFeedTraining[:, inputCardinalityGridFeed:]

    x_train_gridLoss = datasetGridLossTraining[:, 0:inputCardinalityGridLoss]
    y_train_gridLoss = datasetGridLossTraining[:, inputCardinalityGridLoss:]

    x_test_gridFeed = datasetGridFeedValidation[:, 0:inputCardinalityGridFeed]
    y_test_gridFeed = datasetGridFeedValidation[:, inputCardinalityGridFeed:]

    x_test_gridLoss = datasetGridLossValidation[:, 0:inputCardinalityGridLoss]
    y_test_gridLoss = datasetGridLossValidation[:, inputCardinalityGridLoss:]

    es = EarlyStopping(monitor='val_dense_5_loss', mode='min', verbose=1, patience=patience)
    mc = ModelCheckpoint('model.h5', monitor='val_dense_5_loss', mode='min', verbose=1, save_best_only=True)

    history = model.fit([x_train_gridFeed, x_train_gridLoss], [y_train_gridFeed, y_train_gridLoss], epochs=50000,
                        batch_size=batchSize,
                        verbose=1,
                        validation_data=([x_test_gridFeed, x_test_gridLoss], [y_test_gridFeed, y_test_gridLoss]),
                        callbacks=[es, mc])

    keras.backend.clear_session()
    del model
    # testModelQuality(x_test, y_test)
    print(history.history.keys())


def testModelQuality(xGridFeedIn=[], xGridLosses=[], yGridFeedIn=[], yGridLosses=[], timestamps=[],
                     modelFileName='model.h5'):
    """
    This function test the quality of a pre-trained model, it can either receive the preprocessed input and output data
    through its arguments or it can read the raw data from the validation file defined in the
    global variable "fileNameValidation" if no data was passed as arguments since these arguments are optional.

    Inputs:
        -  xGridFeedIn: Numpy float64 array representing the input to the grid feed in model.
        Optional with a default value of an empty array.

        -  xGridLosses: Numpy float64 array representing the input to the grid loss model.
        Optional with a default value of an empty array.

        -  yGridFeedIn: Numpy float64 array representing the output of the grid feed in model (ground truth).
        Optional with a default value of an empty array.

        -  yGridLosses: Numpy float64 array representing the output of the grid loss model (ground truth).
        Optional with a default value of an empty array.

        - timestamps: Numpy timestamp array representing the time stamps values corresponding to each row in any of
        the 4 data arrays discussed above.

        - modelFileName: relative or absolute (mind the escape characters) path of the pre-trained model in h5 format.
        Optional with a default value of "model.h5".

    If any of the 5 numpy arrays are empty, then the function will load the raw data from the file defined by the
    global variable "fileNameValidation".

    The function loads the model and applies the inputs and examines the predicted values.
    Absolute relative error (ARE) is calculated and the following is provided to the user:
        - Average ARE
        - Median ARE
        - Max ARE
        - Percentage of AREs above 20 percent
        - A Bokeh line graph with date as X-axis and ARE as the Y-axis. The Graph allows for magnification, panning
        and exporting as image file.
    """

    model = load_model(modelFileName)
    model.summary()
    if len(xGridFeedIn) > 0 and len(xGridLosses) > 0 and len(yGridFeedIn) > 0 and len(yGridLosses) > 0:
        x_gridFeed = xGridFeedIn
        y_gridFeed = yGridFeedIn
        x_gridLoss = xGridLosses
        y_gridLoss = yGridLosses
        timestamps = timestamps
    else:
        datasetGridFeed, datasetGridLoss, timestamps = prepareData(fileNameValidation, "SEPARATE_INTERVALS")

        inputCardinalityGridFeed = len(datasetGridFeed[0]) - outputCardinality
        inputCardinalityGridLoss = len(datasetGridLoss[0]) - outputCardinality

        x_gridFeed = datasetGridFeed[:, 0:inputCardinalityGridFeed]
        y_gridFeed = datasetGridFeed[:, inputCardinalityGridFeed:]

        x_gridLoss = datasetGridLoss[:, 0:inputCardinalityGridLoss]
        y_gridLoss = datasetGridLoss[:, inputCardinalityGridLoss:]

    pred = model.predict([x_gridFeed, x_gridLoss])
    pred = np.reshape(pred[1], (-1, 1)).squeeze()
    y_gridLoss = np.reshape(y_gridLoss, (-1, 1)).squeeze()
    timestamps = np.reshape(timestamps, (-1, 1)).squeeze()
    error = np.abs(y_gridLoss - pred) / y_gridLoss
    error = np.reshape(error, (-1, 1)).squeeze()
    print("Average absolute error: " + str(np.average(error)))
    print("Median absolute error: " + str(np.median(error)))
    print("Max absolute error: " + str(np.max(error)))
    print("Percentage of errors above 20 percent: " + str(len(error[error >= 0.2]) / len(error)))

    p1 = figure(plot_width=1800, plot_height=400, x_axis_type='datetime',
                title="Prediction Relative Absolute Error", x_axis_label='Time')
    p1.line(timestamps, error, line_width=2, x='Time', y='Error')

    output_file("GridLossErrorGraph.html")

    show(p1)


######################################### Test ###############################################

# testModelQuality()

trainModel(False, 300, 7, 300)
