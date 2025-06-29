from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from collections import defaultdict
import os

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")

client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN,
)

# FastAPI app
app = FastAPI()

# Store conversation history per user
conversation_history = defaultdict(lambda: {"history": []})

# Request models
class PredictionInput(BaseModel):
    user_id: str
    prediction: str  # "pneumonia" or "normal"

class ChatInput(BaseModel):
    user_id: str
    message: str

# Prompt generator for prediction outcomes
def generate_prediction_prompt(prediction: str) -> str:
    prompts = {
        "pneumonia": """The AI has detected a positive case of pneumonia.
Generate a supportive and empathetic message with recommendations for the next steps.
Also, provide at least one health tip for the patient.""",
        "normal": """The AI has determined no signs of pneumonia (normal case).
Generate a warm, encouraging message thanking the user for monitoring their health."""
    }
    return prompts.get(prediction)

@app.post("/predict/")
async def predict_next_step(input_data: PredictionInput):
    user_id = input_data.user_id
    prediction = input_data.prediction.lower()

    if prediction not in ["pneumonia", "normal"]:
        raise HTTPException(status_code=400, detail="Invalid prediction value. Use 'pneumonia' or 'normal'.")

    prompt = generate_prediction_prompt(prediction)
    if not prompt:
        raise HTTPException(status_code=500, detail="Prompt generation failed.")

    history = conversation_history[user_id]["history"]
    history.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=history,
        )
        assistant_msg = response.choices[0].message
        history.append({"role": "assistant", "content": assistant_msg["content"]})
        return {"message": assistant_msg["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@app.post("/chat/")
async def chat(input_data: ChatInput):
    user_id = input_data.user_id
    user_msg = input_data.message

    history = conversation_history[user_id]["history"]
    history.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=history,
        )
        assistant_msg = response.choices[0].message
        history.append({"role": "assistant", "content": assistant_msg["content"]})
        return {"message": assistant_msg["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Pneumonia Prediction and Chat API.",
        "endpoints": {
            "/predict/": "Submit a prediction (pneumonia or normal) and get a response.",
            "/chat/": "Interact with the LLM in a conversational manner.",
        },
    }

