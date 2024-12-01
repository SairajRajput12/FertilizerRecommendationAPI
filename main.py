from fastapi import FastAPI 
from pydantic import BaseModel 
import pickle
import uvicorn
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import joblib


class FertilizerInfo(BaseModel): 
    Temperature: int 
    Humidity: int 
    Moisture: int 
    Nitrogen: int 
    Potassium: int 
    Phosphorous: int 
    Soil_Code: str 
    Crop_Code: str 


app = FastAPI() 

origins = {"*"} 

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Load the trained model
loaded_model = joblib.load('fertilizer_prediction_model.pkl')

@app.get('/')
def basic_function():
    return {'message': 'Welcome to the Fertilizer Prediction API'} 

@app.post('/post') 
def predict_fertilizer(data: FertilizerInfo): 
    data = data.dict() 
    Temperature = data['Temperature'] 
    Humidity = data['Humidity'] 
    Moisture = data['Moisture'] 
    Nitrogen = data['Nitrogen'] 
    Potassium = data['Potassium'] 
    Phosphorous = data['Phosphorous'] 
    Soil_Code = data['Soil_Code'] 
    Crop_Code = data['Crop_Code'] 
    
    # Mapping dictionaries
    crop_map = {
        "barley": 0, "cotton": 1, "ground nuts": 2, "maize": 3, "millets": 4, 
        "oil seeds": 5, "paddy": 6, "pulses": 7, "sugarcane": 8, "tobacco": 9, 
        "wheat": 10, "coffee": 11, "kidneybeans": 12, "orange": 13, 
        "pomegranate": 14, "rice": 15, "watermelon": 16
    }
    
    soil_map = {
        "black": 0, "clayey": 1, "loamy": 2, "red": 3, "sandy": 4
    }
    
    soil_type = Soil_Code.lower()
    crop_type = Crop_Code.lower()
    
    fertilizer_map = {
        0: "10-10-10", 1: "10-26-26", 2: "14-14-14", 3: "14-35-14", 
        4: "15-15-15", 5: "17-17-17", 6: "20-20", 7: "28-28", 8: "DAP", 
        9: "Potassium chloride", 10: "Potassium sulfate", 11: "Superphosphate", 
        12: "TSP", 13: "Urea"
    }
    
    # Convert the string inputs to integer codes
    crop_code = crop_map.get(crop_type, -1) 
    soil_code = soil_map.get(soil_type, -1) 

    cc = '' 
    st = '' 
    if crop_code == 0: 
        cc = 'Barley' 
    elif crop_code == 1: 
        cc = 'Cotton' 
    elif crop_code == 2: 
        cc = 'Ground Nuts' 
    elif crop_code == 3: 
        cc = 'Maize' 
    elif crop_code == 4: 
        cc = 'Millets'
    elif crop_code == 5: 
        cc = 'Oil seeds'
    elif crop_code == 6: 
        cc = 'Paddy' 
    elif crop_code == 7: 
        cc = 'Pulses' 
    elif crop_code == 8: 
        cc = 'Sugarcane' 
    elif crop_code == 9: 
        cc = 'Tobacco' 
    elif crop_code == 10: 
        cc = 'Wheat' 
    elif crop_code == 11: 
        cc = 'Wheat' 
    elif crop_code == 12: 
        cc = 'Cotton' 
    elif crop_code == 13: 
        cc = 'Ground Nuts' 
    elif crop_code == 14: 
        cc = 'Maize' 
    elif crop_code == 15: 
        cc = 'Tobacco' 
    else: 
        cc = 'Paddy'


    if soil_code == 0: 
        st = 'Loamy'
    elif soil_code == 1:
        st = 'Sandy'
    elif soil_code == 2: 
        st = 'Clayey'
    elif soil_code == 3: 
        st = 'Black'
    else: 
        st = 'Red'

    if crop_code == -1 or soil_code == -1:
        return {'error': 'Invalid Crop_Code or Soil_Code'}

    input_data = {
        'Soil Type': [st],
        'Crop Type': [cc],
        'Nitrogen': [Nitrogen],
        'Potassium': [Potassium],
        'Phosphorous': [Phosphorous]
    }

    input_df = pd.DataFrame(input_data)

    # Perform the prediction
    prediction = classifier.predict(input_df)

    # Get the fertilizer name from the prediction
    response = fertilizer_map.get(int(prediction[0]), 'Unknown Fertilizer')
    
    return {'prediction': response} 


if __name__ == '__main__': 
    uvicorn.run(app, host='127.0.0.1', port=8000)


# from fastapi import FastAPI 
# from pydantic import BaseModel 
# import pickle
# import uvicorn
# import pandas
# from fastapi.middleware.cors import CORSMiddleware


# class FertilizerInfo(BaseModel): 
#     Temperature: int 
#     Humidity: int 
#     Moisture: int 
#     Nitrogen: int 
#     Potassium: int 
#     Phosphorous: int 
#     Soil_Code: str 
#     Crop_Code: str 


# app = FastAPI() 

# origins = {"*"} 

# app.add_middleware(
#     CORSMiddleware, 
#     allow_origins=origins, 
#     allow_credentials=True, 
#     allow_methods=["*"], 
#     allow_headers=["*"], 
# )

# # Load the trained model
# pickle_in = open("fertilizer_prediction_model.pkl", "rb") 
# classifier = pickle.load(pickle_in) 

# @app.get('/')
# def basic_function():
#     return {'message': 'Welcome to the Fertilizer Prediction API'} 

# @app.post('/post') 
# def predict_fertilizer(data: FertilizerInfo): 
#     data = data.dict() 
#     Temperature = data['Temperature'] 
#     Humidity = data['Humidity'] 
#     Moisture = data['Moisture'] 
#     Nitrogen = data['Nitrogen'] 
#     Potassium = data['Potassium'] 
#     Phosphorous = data['Phosphorous'] 
#     Soil_Code = data['Soil_Code'] 
#     Crop_Code = data['Crop_Code'] 
    
#     # Mapping dictionaries
#     crop_map = {
#         "barley": 0, "cotton": 1, "ground nuts": 2, "maize": 3, "millets": 4, 
#         "oil seeds": 5, "paddy": 6, "pulses": 7, "sugarcane": 8, "tobacco": 9, 
#         "wheat": 10, "coffee": 11, "kidneybeans": 12, "orange": 13, 
#         "pomegranate": 14, "rice": 15, "watermelon": 16
#     }
    
    
#     soil_map = {
#         "black": 0, "clayey": 1, "Loamy": 2, "red": 3, "sandy": 4
#     }
    
#     soil_type = Soil_Code.lower()
#     crop_type = Crop_Code.lower()
    
#     fertilizer_map = {
#         0: "10-10-10", 1: "10-26-26", 2: "14-14-14", 3: "14-35-14", 
#         4: "15-15-15", 5: "17-17-17", 6: "20-20", 7: "28-28", 8: "DAP", 
#         9: "Potassium chloride", 10: "Potassium sulfate", 11: "Superphosphate", 
#         12: "TSP", 13: "Urea"
#     }
    
#     # Convert the string inputs to integer codes
#     crop_code = crop_map.get(crop_type, -1) 
#     soil_code = soil_map.get(soil_type, -1) 

#     cc = '' 
#     st = '' 
#     if crop_code == 0: 
#         cc = 'Barley' 
#     else if crop_code == 1: 
#         cc = 'Cotton' 
#     else if crop_code == 2: 
#         cc = 'Ground Nuts' 
#     else if crop_code == 3: 
#         cc  = 'Maize' 
#     else if crop_code == 4: 
#         cc = 'Millets'
#     else if crop_code == 5: 
#         cc = 'Oil seeds'
#     else if crop_code == 6: 
#         cc = 'Paddy' 
#     else if crop_code == 7: 
#         cc = 'Pulses' 
#     else if crop_code == 8: 
#         cc = 'Sugarcane' 
#     else if crop_code == 9: 
#         cc = 'Tobacco' 
#     else if crop_code == 10: 
#         cc = 'Wheat' 
#     else if crop_code == 11: 
#         cc = 'Wheat' 
#     else if crop_code == 12: 
#         cc = 'Cotton' 
#     else if crop_code == 13: 
#         cc = 'Ground Nuts' 
#     else if crop_code == 14: 
#         cc = 'Maize' 
#     else if crop_code == 15: 
#         cc = 'Tobacco' 
#     else: 
#         cc = 'Paddy'


#     if soil_code == 0: 
#         st = 'Loamy'
#     else if soil_code == 1:
#         st = 'Sandy'
#     else if soil_code == 2: 
#         st = 'Clayey'
#     else if soil_code == 3: 
#         st = 'Black'
#     else: 
#         st = 'Red'

    

    
    
    
#     if crop_code == -1 or soil_code == -1:
#         return {'error': 'Invalid Crop_Code or Soil_Code'}


#     input_data = {
#         'Soil Type': [st],
#         'Crop Type': [cc],
#         'Nitrogen': [Nirogen],
#         'Potassium': [Potassium],
#         'Phosphorous': [Phosphorous]
#     }

#     input_df = pd.DataFrame(input_data)


#     # Perform the prediction
#     prediction = model.predict(input_df)

    
#     # Extracting the first element from the prediction array
#     # prediction_code = int(prediction[0])
    
#     # Get the fertilizer name from the prediction
#     response = fertilizer_map.get(prediction, 'Unknown Fertilizer')
    
#     return {'prediction': response} 


# if __name__ == '__main__': 
#     uvicorn.run(app, host='127.0.0.1', port=8000)


# from fastapi import FastAPI 
# from pydantic import BaseModel 
# import pickle
# import uvicorn
# from fastapi.middleware.cors import CORSMiddleware

# class FertilizerInfo(BaseModel): 
#     Temperature: int 
#     Humidity: int 
#     Moisture: int 
#     Nitrogen: int 
#     Potassium: int 
#     Phosphorous: int 
#     Soil_Code: str 
#     Crop_Code: str 


# app = FastAPI() 

# origins = {"*"} 

# app.add_middleware(
#     CORSMiddleware, 
#     allow_origins=origins, 
#     allow_credentials=True, 
#     allow_methods=["*"], 
#     allow_headers=["*"], 
# )

# # Load the trained model
# # pickle_in = open("fertilizer3.pkl", "rb") 
# # classifier = pickle.load(pickle_in) 

# @app.get('/')
# def basic_function():
#     return {'message': 'Welcome to the Fertilizer Prediction API'} 

# @app.post('/post') 
# def predict_fertilizer(data: FertilizerInfo): 
#     # data = data.dict() 
#     # Nitrogen = data['Nitrogen'] 
#     # Potassium = data['Potassium'] 
#     # Phosphorous = data['Phosphorous'] 
#     # Soil_Code = data['Soil_Code'] 
#     # Crop_Code = data['Crop_Code'] 
    
#     # # Mapping dictionaries
#     # crop_map = {
#     #     "Barley": 0, "Cotton": 1, "Ground Nuts": 2, "Maize": 3, "Millets": 4, 
#     #     "Oil seeds": 5, "Paddy": 6, "Pulses": 7, "Sugarcane": 8, "Tobacco": 9, 
#     #     "Wheat": 10, "coffee": 11, "kidneybeans": 12, "orange": 13, 
#     #     "pomegranate": 14, "rice": 15, "watermelon": 16
#     # }
    
#     # soil_map = {
#     #     "Black": 0, "Clayey": 1, "Loamy": 2, "Red": 3, "Sandy": 4
#     # }
    
#     # ferti = pickle.load(open('fertilizer.pkl','rb'))
#     # p = ferti.classes_
#     # fertilizer_map = p  
    
#     # # Convert the string inputs to integer codes
#     # crop_code = crop_map.get(Crop_Code, -1) 
#     # soil_code = soil_map.get(Soil_Code, -1) 
    
#     # if crop_code == -1 or soil_code == -1:
#     #     return {'error': 'Invalid Crop_Code or Soil_Code'}

#     # # Perform the prediction
#     # classifier = pickle.load(open('classifier5.pkl','rb'))
#     # prediction = classifier.predict([[soil_code, crop_code, Nitrogen, Potassium, Phosphorous]])
#     # print('prediction line 73 ',prediction)
#     # print(type(prediction))
#     # # Extracting the first element from the prediction array
#     # prediction_code = int(prediction[0])
#     # print(p)
#     # # Get the fertilizer name from the prediction
#     # response = p[prediction[0]]
#     # print(response)
#     # return {'prediction': response} 
#     model = joblib.load('xgboost_fertilizer_model.pkl')  # New model file path
#     scaler = joblib.load('scaler.pkl')  # New scaler file path
#     label_encoders = joblib.load('label_encoders.pkl')  # New label encoders file path
    
#     # Encode input data using the new label encoders
#     soil_type_encoded = label_encoders['Soil Type'].transform([data.Soil_Type])[0]
#     crop_type_encoded = label_encoders['Crop Type'].transform([data.Crop_Type])[0]
#     scaled_values = scaler.transform([[data.Nitrogen, data.Potassium, data.Phosphorous]])[0]
#     features = [soil_type_encoded, crop_type_encoded, scaled_values[0], scaled_values[1], scaled_values[2]]
    
#     # Make prediction using the new model
#     encoded_prediction = model.predict([features])
    
#     # Decode the prediction back to fertilizer name
#     predicted_fertilizer = label_encoders['Fertilizer Name'].inverse_transform(encoded_prediction)
    
#     return {"Predicted Fertilizer": predicted_fertilizer[0]}



# if __name__ == '__main__': 
#     uvicorn.run(app, host='127.0.0.1', port=8000)
