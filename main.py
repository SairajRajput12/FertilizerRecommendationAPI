from fastapi import FastAPI 
from pydantic import BaseModel 
import pickle
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

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
# pickle_in = open("fertilizer3.pkl", "rb") 
# classifier = pickle.load(pickle_in) 

@app.get('/')
def basic_function():
    return {'message': 'Welcome to the Fertilizer Prediction API'} 

@app.post('/post') 
def predict_fertilizer(data: FertilizerInfo): 
    # data = data.dict() 
    # Nitrogen = data['Nitrogen'] 
    # Potassium = data['Potassium'] 
    # Phosphorous = data['Phosphorous'] 
    # Soil_Code = data['Soil_Code'] 
    # Crop_Code = data['Crop_Code'] 
    
    # # Mapping dictionaries
    # crop_map = {
    #     "Barley": 0, "Cotton": 1, "Ground Nuts": 2, "Maize": 3, "Millets": 4, 
    #     "Oil seeds": 5, "Paddy": 6, "Pulses": 7, "Sugarcane": 8, "Tobacco": 9, 
    #     "Wheat": 10, "coffee": 11, "kidneybeans": 12, "orange": 13, 
    #     "pomegranate": 14, "rice": 15, "watermelon": 16
    # }
    
    # soil_map = {
    #     "Black": 0, "Clayey": 1, "Loamy": 2, "Red": 3, "Sandy": 4
    # }
    
    # ferti = pickle.load(open('fertilizer.pkl','rb'))
    # p = ferti.classes_
    # fertilizer_map = p  
    
    # # Convert the string inputs to integer codes
    # crop_code = crop_map.get(Crop_Code, -1) 
    # soil_code = soil_map.get(Soil_Code, -1) 
    
    # if crop_code == -1 or soil_code == -1:
    #     return {'error': 'Invalid Crop_Code or Soil_Code'}

    # # Perform the prediction
    # classifier = pickle.load(open('classifier5.pkl','rb'))
    # prediction = classifier.predict([[soil_code, crop_code, Nitrogen, Potassium, Phosphorous]])
    # print('prediction line 73 ',prediction)
    # print(type(prediction))
    # # Extracting the first element from the prediction array
    # prediction_code = int(prediction[0])
    # print(p)
    # # Get the fertilizer name from the prediction
    # response = p[prediction[0]]
    # print(response)
    # return {'prediction': response} 
    model = joblib.load('xgboost_fertilizer_model.pkl')  # New model file path
    scaler = joblib.load('scaler.pkl')  # New scaler file path
    label_encoders = joblib.load('label_encoders.pkl')  # New label encoders file path
    
    # Encode input data using the new label encoders
    soil_type_encoded = label_encoders['Soil Type'].transform([input_data.Soil_Type])[0]
    crop_type_encoded = label_encoders['Crop Type'].transform([input_data.Crop_Type])[0]
    scaled_values = scaler.transform([[input_data.Nitrogen, input_data.Potassium, input_data.Phosphorous]])[0]
    features = [soil_type_encoded, crop_type_encoded, scaled_values[0], scaled_values[1], scaled_values[2]]
    
    # Make prediction using the new model
    encoded_prediction = model.predict([features])
    
    # Decode the prediction back to fertilizer name
    predicted_fertilizer = label_encoders['Fertilizer Name'].inverse_transform(encoded_prediction)
    
    return {"Predicted Fertilizer": predicted_fertilizer[0]}



if __name__ == '__main__': 
    uvicorn.run(app, host='127.0.0.1', port=8000)
