"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    ##########################################################################
    #                START OF TEAM 10 FEATURE ENGINEERING FUNCTIONS
    ##########################################################################

    ################################### drop_columns  ########################
    # To drop columns
    features_to_drop = ['User Id',
                    'Rider Id',
                    'Order No', # not for test
                    'Vehicle Type',
                    'Precipitation in millimeters',
                    'Arrival at Destination - Day of Month',
                    'Arrival at Destination - Weekday (Mo = 1)',
                    'Confirmation - Day of Month',
                    'Confirmation - Weekday (Mo = 1)',
                    'Arrival at Pickup - Day of Month',
                    'Arrival at Pickup - Weekday (Mo = 1)',
                    'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)',
                    'Pickup - Weekday (Mo = 1)'
                    'Arrival at Destination - Day of Month',
                    ]

    def drop_columns(df, features_to_drop = features_to_drop):
        
        return df.drop([column for column in df.columns if column in features_to_drop], axis=1)
    

    ################################## to_platform_type  ########################################
    def to_platform_type(df):
        """Returns a dataframe with encoded platform feature"""

        types = {1:'Type 1',2:'Type 2',3:'Type 3',4:'Type 4'}

        df['Platform Type'] = df['Platform Type'].map(types)
        
        return df

    ############################## to_weekday_name ########################################
    def to_weekday_name(df):
        """Returns a dataframe with names of days of the week"""


        ends_with = 'Weekday (Mo = 1)'
        weekday_names = {1:"Monday", 2:"Tuesday", 3: "Wednesday", 4:"Thursday", 5:"Friday", 6: "Saturday", 7:"Sunday"}
        for col in df.columns:
            if col.endswith(ends_with):
                df[col] = df[col].map(weekday_names) 
        
        return df

    ###############################Time Based Features###################################
    # Time Based Features
    #--------------------
    # 'Placement - Time'
    # 'Confirmation - Time' 
    # 'Arrival at Pickup - Time'
    # 'Pickup - Time'
    # 'Arrival at Destination - Time'
    # ------------------


    time_features = ['Placement - Time',
                    'Confirmation - Time' ,
                    'Arrival at Pickup - Time',
                    'Pickup - Time',
                    'Arrival at Destination - Time',
                    ]


    def to_seconds(df, time_features=time_features):
        """Returns a Dataframe with time features converted to seconds"""

        for column in time_features:

            if column in df:
            df[column] = pd.to_datetime(df[column])

            df[column.split(' -')[0]+'_hour'] = df[column].dt.hour
            df[column.split(' -')[0]+'_minute'] = df[column].dt.minute
            df[column.split(' -')[0]+'_second'] = df[column].dt.second
            df[column]= (df[column] - pd.to_datetime(pd.to_datetime('today').date())).astype('timedelta64[s]')
        
        return df


    ##########################################################################
    # Delta Features
    # ----------------------

    # -----------------------
    # Train and Test
    # ----------------
    # 'delta-Time-Confirmation_Placement'           ---> 'Confirmation - Time', 'Placement - Time'
    # 'delta-Time-Arrival-at-Pickup_Confirmation'   ---> 'Arrival at Pickup - Time', 'Confirmation - Time'
    # 'delta-Time-Pickup_Arrival-at-Pickup'         ---> 'Pickup - Time', 'Arrival at Pickup - Time'
    # ----------------
    # Train Not in TEST
    # ----------------
    # 'delta-Time-Arrival-at-Destination_Pickup'    ---> 'Arrival at Destination - Time', 'Pickup - Time'
    # 'delta-Time-Arrival-at-Destination_Placement' ---> 'Arrival at Destination - Time', 'Placement - Time'

    delta_cols = [
                ('delta-Time-Confirmation_Placement',  'Confirmation - Time', 'Placement - Time' ),
                ('delta-Time-Arrival-at-Pickup_Confirmation' ,'Arrival at Pickup - Time', 'Confirmation - Time'),
                ('delta-Time-Pickup_Arrival-at-Pickup', 'Pickup - Time', 'Arrival at Pickup - Time'),
                ('delta-Time-Arrival-at-Destination_Pickup', 'Arrival at Destination - Time', 'Pickup - Time'),
                ('delta-Time-Arrival-at-Destination_Placement','Arrival at Destination - Time', 'Placement - Time')
                ]

    # Function to compute to delta
    def to_delta(df, delta_cols=delta_cols):
        """Returns delta features from existing features"""

        for deltas in delta_cols:                               # Loop over the List of tuple [(output_name, col1, col2 ), (output_name, col1, col2 )]

            if deltas[1] in df and deltas[2] in df:             # check if columns exists
                df[deltas[0]] = df[deltas[1]] - df[deltas[2]]   # make delta feaure

        return df



    ##########################################################################
    #                END OF TEAM 10 FEATURE ENGINEERING FUNCTIONS
    ##########################################################################



    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    predict_vector = feature_vector_df[['Pickup Lat','Pickup Long',
                                        'Destination Lat','Destination Long']]
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
