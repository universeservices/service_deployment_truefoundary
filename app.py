from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

@app.post("/predict/")
async def predict(hf_pipeline: str, model_deployed_url: str, inputs: Any, parameters: dict):
    # Convert the input to the V2 inference protocol.
    if hf_pipeline == "text-generation":
        inputs = {"inputs": inputs}
    elif hf_pipeline == "zero-shot-classification":
        inputs = {"inputs": inputs, "candidate_labels": parameters["candidate_labels"]}
    elif hf_pipeline == "object-detection":
        inputs = {"inputs": inputs}

    # Get the model.
    model = pipeline(hf_pipeline, model_url=model_deployed_url)

    # Predict.
    prediction = model(**inputs)

    # Return the prediction.
    return prediction

