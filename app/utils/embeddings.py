# In utils/embeddings.py
import vertexai
from vertexai.language_models import TextEmbeddingModel
from typing import List

# It's good practice to initialize the model once to avoid reloading it on every call.
MODEL_NAME = "gemini-embedding-001"
model = None

def generate_combined_embedding(text_to_embed: str) -> List[float]:
    """Generates a vector embedding for a given text string using Vertex AI."""
    global model
    # Lazy initialization of the model
    if model is None:
        print(f"Initializing Vertex AI TextEmbeddingModel: {MODEL_NAME}")
        model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
        
    print(f"Generating embedding for text: '{text_to_embed}'")
    try:
        # The API expects a list of texts and returns a list of embeddings.
        embeddings = model.get_embeddings([text_to_embed])
        # We need the vector values from the first (and only) embedding object.
        vector_embedding = embeddings[0].values
        print(f"Successfully generated embedding with dimension: {len(vector_embedding)}")
        return vector_embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return an empty list as a clear sign of failure.
        return []