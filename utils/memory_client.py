import os
from google import genai
from mem0 import Memory

# Set the appropriate API keys
os.environ["OPENAI_API_KEY"] = "sk-your-openai-key"           # For embedding (optional if using Gemini for embedder)
os.environ["GEMINI_API_KEY"] = "AIzaSy..."                    # Your valid Gemini API key


config = {
    "llm": {  # Generate content with Gemini
        "provider": "gemini",
        "config": {
            "model": "gemini-2.0-flash",
            "api_key": os.getenv("GEMINI_API_KEY")
        }
    },
    "embedder": {  # Use Gemini embedder (768-dim)
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": os.getenv("GEMINI_API_KEY")
        }
    },
    "vector_store": {  # Explicitly match embedding dims
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "trading_mem",
            "embedding_model_dims": 768
        }
    }
}

m = Memory.from_config(config)

def add_context(messages, user="moti", metadata=None):
    return m.add(messages, user_id=user, metadata=metadata)

def get_memories(message, user="moti"):
    return m.search(query=message, user_id=user, limit=5)
