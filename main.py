import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import warnings

warnings.simplefilter("ignore")


def take_input_feature(feature, min_value=0, max_value=1000):
    while True:
        if feature == "percip_type":
            percip_type = str(input("Please enter the Precipitation Type value: "))
            if percip_type == "rain" or percip_type == "snow":
                return percip_type
            print(f"The value of percip type should be either 'rain' or 'snow'\n")
            continue
        if (
            feature == "Year"
            or feature == "Month"
            or feature == "Day"
            or feature == "Hour"
        ):
            input_feature = int(input(f"\nPlease enter the value of {feature}: "))
            if input_feature >= min_value and input_feature <= max_value:
                return input_feature
            print(
                f"The value of {feature} should be in the range of {min_value} and {max_value}\n"
            )
            continue

        else:
            input_feature = float(input(f"\nPlease enter the value of {feature}: "))
            if input_feature >= min_value and input_feature <= max_value:
                return input_feature
            print(
                f"The value of {feature} should be in the range of {min_value} and {max_value}\n"
            )
            continue


def main():
    print(
        f"******************Welcome to Weather Forecasting Application Using Machine Learning******************\n"
    )
    print(
        f"Here are some guidelines about the range of the values of input: \n1. The value of Humidity should be from 0.0 to 1.0 with a step size of 0.1\n2. The value of Time(Hour) should be from 0 to 23 with a step size of 1\n3. The value of Year should be from 2006 to 2023\n4. The value of Month should be from 1 to 12\n5. The value of Day should be from 1 to 31\n6. The value of Wind Speed should be from 0 to 100 with a step size of 1\n7. The value of Wind Bearing should be from 0 to 360 with a step size of 1.\n8. The value of Visibility should be from 0.0 to 20.0 with a step size of 0.1\n9. The value of Pressure should be from 800.0 to 1100.0 with a step size of 1.0\n10. The value of Precipitation Type should be either 'rain' or 'snow'\n11. The value of Temprature should be from 25 to 40 with a step size of 1\n"
    )

    # taking input from the user end
    percip_type = take_input_feature("percip_type")
    temprature = take_input_feature("Temprature (C)", -10, 40)
    humidity = take_input_feature("Humidity", 0.0, 1.0)
    wind_speed = take_input_feature("Wind Speed", 0.0, 100.0)
    wind_bearing = take_input_feature("Wind Bearing", 0, 360)
    visibility = take_input_feature("Visibility", 0.0, 20.0)
    pressure = take_input_feature("Pressure", 800.0, 1100.0)
    year = take_input_feature("Year", 2006, 2023)
    month = take_input_feature("Month", 1, 12)
    day = take_input_feature("Day", 1, 31)
    hour = take_input_feature("Hour", 0, 23)

    # Coding the input data frame:
    input_data = pd.DataFrame(
        {
            "Precip Type": percip_type,
            "Temperature (C)": temprature,
            "Humidity": humidity,
            "Wind Speed (km/h)": wind_speed,
            "Wind Bearing (degrees)": wind_bearing,
            "Visibility (km)": visibility,
            "Pressure (millibars)": pressure,
            "Year": year,
            "Month": month,
            "Day": day,
            "Hour": hour,
        },
        index=[0],
    )

    # encoding the input data frame values:
    mapping = {"rain": 0, "snow": 1}
    input_data["Precip Type"] = input_data["Precip Type"].map(mapping)

    # Standardizing the input data:
    scaler = joblib.load("./models/scaler.pkl")
    scaled_input_data = pd.DataFrame(
        scaler.transform(input_data), columns=input_data.columns
    )

    # Output Mappings:
    classes = {0: "Clear", 1: "Foggy", 2: "Overcast"}

    # Models mapping:
    models = {
        1: joblib.load("./models/logistic_regression.pkl"),
        2: joblib.load("./models/gaussian_nb.pkl"),
        3: joblib.load("./models/sgd.pkl"),
        4: joblib.load("./models/decision_tree.pkl"),
        5: joblib.load("./models/KNN.pkl"),
        6: joblib.load("./models/random_forest.pkl"),
        7: tf.keras.models.load_model("./models/ANN1.h5"),
        8: tf.keras.models.load_model("./models/ANN2.h5"),
        9: tf.keras.models.load_model("./models/ANN3.h5"),
    }

    print(
        f"\nPlease select any one of the algorithms by their id to see the weather forecast:\n1. Logistic Regresssion\n2. Gaussian Naive Bayes\n3. Stochastic Gradient Descent\n4. Decision Tree\n5. K Nearest Neighbours Classifier\n6. Random Forest\n7. 16-256 ANN with 'tanh'\n8. 32-64-512 ANN with 'relu'\n9. 32-1024 ANN with 'sigmoid'"
    )

    model_no = int(input("Please enter the model serial id: "))

    predicted_class = ""
    predictions = models[model_no].predict(scaled_input_data)[0]

    if model_no == 7 or model_no == 8 or model_no == 9:
        predicted_class = np.argmax(predictions)
        print(f"The predicted class is: {classes[predicted_class]}")
        return

    print(f"\n***********The predicted class is: {classes[predictions]}***********")
    return


main()


# -----------------------------------------------Testing Data-------------------------------------------#

# Summary  | Precip Type |	Temperature (C) |	Humidity |	Wind Speed (km/h)
# Overcast   snow	         -1.738888889	    0.88	    20.3343
# Foggy      rain            5.211111111        0.92        4.7656
# Clear	     rain	         13.31111111	    0.82	    3.542
# Overcast	 rain	         17.01111111	    0.91	    5.4096
# Foggy	     snow	        -1.088888889	    0.93	    10.9319
# Clear	     rain	        15.01111111	        0.93	    3.2039


# Wind Bearing (degrees) | Visibility (km)   |	Pressure (millibars) |	Year |	Month |  Day |  Hour
# 281	                      4.3953	          1010.86	            2006  	 2	     26	    22
# 178                         1.2236              1013.4                2006     4       14     4
# 73	                      15.8263	          1018.44	            2006	 4	     25	    22
# 178	                      13.685	          1013.23	            2006	 8	     14	    0
# 180	                      0.322	              1032.88	            2006	12	     13	    7
# 341	                     15.8263	          1014.37	            2016	9	     9	    2
