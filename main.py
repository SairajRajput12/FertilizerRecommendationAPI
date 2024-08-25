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
pickle_in = open("fertilizer2.pkl", "rb") 
classifier = pickle.load(pickle_in) 

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
        "Barley": 0, "Cotton": 1, "Ground Nuts": 2, "Maize": 3, "Millets": 4, 
        "Oil seeds": 5, "Paddy": 6, "Pulses": 7, "Sugarcane": 8, "Tobacco": 9, 
        "Wheat": 10, "coffee": 11, "kidneybeans": 12, "orange": 13, 
        "pomegranate": 14, "rice": 15, "watermelon": 16
    }
    
    soil_map = {
        "Black": 0, "Clayey": 1, "Loamy": 2, "Red": 3, "Sandy": 4
    }
    
    fertilizer_map = {
        0: "10-10-10", 1: "10-26-26", 2: "14-14-14", 3: "14-35-14", 
        4: "15-15-15", 5: "17-17-17", 6: "20-20", 7: "28-28", 8: "DAP", 
        9: "Potassium chloride", 10: "Potassium sulfate", 11: "Superphosphate", 
        12: "TSP", 13: "Urea"
    }
    
    # Convert the string inputs to integer codes
    crop_code = crop_map.get(Crop_Code, -1) 
    soil_code = soil_map.get(Soil_Code, -1) 
    
    if crop_code == -1 or soil_code == -1:
        return {'error': 'Invalid Crop_Code or Soil_Code'}

    # Perform the prediction
    prediction = classifier.predict([[Temperature, Humidity, Moisture, Nitrogen, Potassium, Phosphorous, soil_code, crop_code]])
    
    # Extracting the first element from the prediction array
    prediction_code = int(prediction[0])
    
    # Get the fertilizer name from the prediction
    response = fertilizer_map.get(prediction_code, 'Unknown Fertilizer')
    
    return {'prediction': response} 


if __name__ == '__main__': 
    uvicorn.run(app, host='127.0.0.1', port=8000)
