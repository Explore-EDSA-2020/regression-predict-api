"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
import numpy as np

# pipeline based
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# preprocessing based
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.compose import TransformedTargetRegressor

# model based
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold 
################################################################



#################################################################
# 
#################################################################



#################################################################
# 
#################################################################

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')
riders = pd.read_csv('data/riders.csv')
train = train.merge(riders, how='left', on='Rider Id')


def preprocess(train):


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


    def customer_encoder(df):
        dict_cast = {'Business': 1, 'Personal':0}
        df['Personal or Business']= df['Personal or Business'].map(dict_cast)
        return df

    ##########################################################################
    #                END OF TEAM 10 FEATURE ENGINEERING FUNCTIONS
    ##########################################################################


    # ------------------------------------------------------------------------

    ##########################################################################
    #                START OF FUNCTIONS CALLS
    ##########################################################################
    
    # Drop redundant columns
    train = drop_columns(train)

    # Encode weekdays to weekday names category 
    # train= to_weekday_name(train)

    # Platform type to category
    # train= to_platform_type(train)
    train= customer_encoder(train)

    # cast time to datetime then create hours, minute, seconds
    train= to_seconds(train)

    # Create the delta features
    train= to_delta(train)

    ##########################################################################
    #                 END OF FUNCTIONS CALLS
    ##########################################################################

    numerical_features = ['Placement - Day of Month', 'Placement - Time', 'Confirmation - Time',
                        'Arrival at Pickup - Time', 'Pickup - Time', 'Distance (KM)', 'Temperature',
                        'Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long', 'No_Of_Orders',
                        'Age', 'Average_Rating', 'No_of_Ratings', 'Placement_hour', 'Placement_minute',
                        'Placement_second', 'Confirmation_hour', 'Confirmation_minute', 'Confirmation_second',
                        'Arrival at Pickup_hour', 'Arrival at Pickup_minute', 'Arrival at Pickup_second',
                         'Pickup_hour','Pickup_minute', 'Pickup_second', 'delta-Time-Confirmation_Placement',
                        'delta-Time-Arrival-at-Pickup_Confirmation', 'delta-Time-Pickup_Arrival-at-Pickup',
                        'Time from Pickup to Arrival', 'Platform Type',]


    categorical_features = [  'Personal or Business', 'Placement - Weekday (Mo = 1)']


    return train[numerical_features + categorical_features]
    # return predict_vector
    
#-------------------------------------------------------------
train_prep = preprocess(train)

# separate predictors from the target
X = train_prep.drop('Time from Pickup to Arrival', axis=1)
y = train_prep['Time from Pickup to Arrival']


#------------------------features-------------------------------------
numerical_features = ['Placement - Day of Month', 'Placement - Time', 'Confirmation - Time',
                    'Arrival at Pickup - Time', 'Pickup - Time', 'Distance (KM)', 'Temperature',
                    'Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long', 'No_Of_Orders',
                    'Age', 'Average_Rating', 'No_of_Ratings', 'Placement_hour', 'Placement_minute',
                    'Placement_second', 'Confirmation_hour', 'Confirmation_minute', 'Confirmation_second',
                    'Arrival at Pickup_hour', 'Arrival at Pickup_minute', 'Arrival at Pickup_second',
                        'Pickup_hour','Pickup_minute', 'Pickup_second', 'delta-Time-Confirmation_Placement',
                    'delta-Time-Arrival-at-Pickup_Confirmation', 'delta-Time-Pickup_Arrival-at-Pickup', 'Platform Type',
                     'Placement - Weekday (Mo = 1)'
                    ]

categorical_features = ['Personal or Business']
#-------------------------------------------------------------------------

# numeric Transformers
# 1. StandardScaler
# 2. PowerTransformer ---> Box-Cox or Yeo-Johnson
# 3. RobustScaler
numerical_transformer = Pipeline(steps=[
                            ('impute', SimpleImputer(strategy='mean')),
                            ('scale', PowerTransformer())
                        ])

# --------------------------------------------------------------------------------
# Categorical Transformers
categorical_transformer =  Pipeline(steps=[
                                    ('impute',SimpleImputer(strategy='most_frequent')),
                                    ('encode', OneHotEncoder(handle_unknown='ignore'))
                                ])

# --------------------------------------------------------------------------------
# Preprocesing  pipeline
preprocessor = ColumnTransformer(transformers=[
                                            ('num', numerical_transformer, numerical_features ),
                                            ('cat', categorical_transformer, categorical_features)
                                            ],
                                            remainder='passthrough') # might remove pass through, if we got all our features done



# --------------------------------------------------------------------------------
# define a function to complete the model by adding the model 
def full_pipeline(model, preprocessor=preprocessor):
    
    return Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('selector', SelectKBest(score_func=f_regression, k=10)),
                    ('model', model)
                ])

#-----------------------model--------------------------------------------
########################################################################################
#-----------------------------Cat Boost Regressor--------------------------------------
########################################################################################

catb_model = CatBoostRegressor(iterations=500,
                             learning_rate=0.05,
                             depth=10,
                             eval_metric='RMSE',
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)

# ------------------complete pipeline--------------------
catb_pipe = full_pipeline(catb_model)

#----------------------Train---------------------------------
print ("Training Model...")
catb_pipe.fit(X, y)
print ("Training Model Complete...")


# Pickle model for use within our API
save_path = '../assets/trained-models/catboost_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(catb_model, open(save_path,'wb'))



