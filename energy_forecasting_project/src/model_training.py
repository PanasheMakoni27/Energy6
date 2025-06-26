
#These are the librabries that I have made use. As you can see one of the most import imports here is the xgboost library as its the model of choice that I will be utalising 

import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib
import os

def train_models(X_train, y_train, X_test, y_test):
#Through my analysis of the problem, I understood that im actually training 6 models. The question said 6 hours ahead so I am breaking it down to according all 6 hours. In this instance im basically testing and training as passed in args. 

    models = []
    for i in range(6):
#In this instance, I have a loop. Now this loop will run 6 times creating a ne wxgboost model for each run which predicts each separate hour. So from hour 1 until we get to hour 6. 
        model = xgb.XGBRegressor()

#So basically in this instance Im teachning my model to use the different features that have been applied in the preprocessing file to actually predict the energy that will be created in each hour from 1 to hour 6. 
        model.fit(X_train, y_train.iloc[:, i])

        #Making predictions on the test set. 
        pred = model.predict(X_test)

        #Here im calculating the errors that might have happened between the predictions and the actual values
        mae = mean_absolute_error(y_test.iloc[:, i], pred)
        print(f"Step {i+1} MAE: {mae:.2f}")
        
        #I want the trained model to be savel in the blank list that I ctreated at the start of the program. 
        models.append(model)
    return models


#This is important. Forecasting is happening here, I am used my trained model to forecast for the next 6 hours, and I want to return the values from the past 6 hours. 
def forecast_next_6_hours(latest_row, models):

    predictions = []

    #Making the preictions
    for model in models:
        pred = model.predict(latest_row.values.reshape(1, -1))
        predictions.append(pred[0])
    return predictions

def save_models(models, path_prefix="models/xgb_model_t+"):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    for i, model in enumerate(models):

          #In this code, Im basically saving each model with a unique filename for its forest step
        joblib.dump(model, f"{path_prefix}{i+1}.pkl")
