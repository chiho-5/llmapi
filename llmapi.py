from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
# Initialize FastAPI app
app = FastAPI()
api_key = api_key = os.getenv("HF_API_KEY")

# Define the Hugging Face Inference Client
client = InferenceClient(api_key=api_key)

# Define the input schema
class PredictionInput(BaseModel):
    prediction: str  # "pneumonia" or "normal"

# LLM prompt template
def generate_prompt(prediction: str) -> str:
    if prediction == "pneumonia":
        return """
        The AI has detected a positive case of pneumonia. 
        Generate a supportive and empathetic message with recommendations for the next steps. 
        Also, provide at least one health tip for the patient.
        """
    elif prediction == "normal":
        return """
        The AI has determined no signs of pneumonia (normal case). 
        Generate a warm, encouraging message thanking the user for monitoring their health. 
        """
    else:
        return None

# API route to handle predictions
@app.post("/predict/")
async def predict_next_step(input_data: PredictionInput):
    prediction = input_data.prediction.lower()

    # Validate the input
    if prediction not in ["pneumonia", "normal"]:
        raise HTTPException(status_code=400, detail="Invalid prediction value. Use 'pneumonia' or 'normal'.")

    # Generate the prompt
    prompt = generate_prompt(prediction)
    if not prompt:
        raise HTTPException(status_code=500, detail="Error generating prompt.")

    messages = [
    {
        "role": "user",
        "content": prompt,
    }
]
    # Call the LLM for a response
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            max_tokens=500,
        )
        message = response.choices[0].message
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Pneumonia Prediction API. Use /predict/ to get next steps based on predictions."}
