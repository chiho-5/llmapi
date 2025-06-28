from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
from typing import List, Dict
from collections import defaultdict

# Initialize FastAPI app
app = FastAPI()

# Load API key from environment variable
api_key = os.getenv("HF_API_KEY")
# api_key = "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"
if not api_key:
    raise ValueError("HF_API_KEY environment variable is not set.")

# Define the Hugging Face Inference Client
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token=api_key)

# Thread-safe in-memory store for conversation history
conversation_history = defaultdict(lambda: {"history": []})

# Define the input schema for predictions
class PredictionInput(BaseModel):
    user_id: str
    prediction: str  # "pneumonia" or "normal"

# Define the input schema for chat
class ChatInput(BaseModel):
    user_id: str
    message: str

# Dependency to get conversation history
def get_conversation(user_id: str):
    if user_id not in conversation_history:
        conversation_history[user_id] = {"history": []}
    return conversation_history[user_id]


# LLM prompt template for predictions
def generate_prediction_prompt(prediction: str) -> str:
    prompts = {
        "pneumonia": """The AI has detected a positive case of pneumonia.
        Generate a supportive and empathetic message with recommendations for the next steps.
        Also, provide at least one health tip for the patient.""",
        "normal": """The AI has determined no signs of pneumonia (normal case).
        Generate a warm, encouraging message thanking the user for monitoring their health."""
    }
    return prompts.get(prediction, None)

# API route to handle predictions
@app.post("/predict/")
async def predict_next_step(input_data: PredictionInput):
    conversation = get_conversation(input_data.user_id)  # Get history based on body data
    prediction = input_data.prediction.lower()

    if prediction not in ["pneumonia", "normal"]:
        raise HTTPException(status_code=400, detail="Invalid prediction value. Use 'pneumonia' or 'normal'.")

    prompt = generate_prediction_prompt(prediction)
    if not prompt:
        raise HTTPException(status_code=500, detail="Error generating prompt.")

    conversation["history"].append({"role": "user", "content": prompt})

    try:
        response = client.text_generation(prompt, max_new_tokens=500)
        conversation["history"].append({"role": "assistant", "content": response})
        return {"message": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")


# API route to handle chat interactions
@app.post("/chat/")
async def chat(input_data: ChatInput):
    user_id = input_data.user_id  # Extract user_id from request body
    conversation = get_conversation(user_id)  # Get conversation history

    user_message = input_data.message
    conversation["history"].append({"role": "user", "content": user_message})

    try:
        response = client.text_generation(user_message, max_new_tokens=500)
        conversation["history"].append({"role": "assistant", "content": response})
        return {"message": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Pneumonia Prediction and Chat API.",
        "endpoints": {
            "/predict/": "Submit a prediction (pneumonia or normal) and get a response.",
            "/chat/": "Interact with the LLM in a conversational manner.",
        },
        "instructions": "Use POST requests with the required input schema for each endpoint.",
    }
