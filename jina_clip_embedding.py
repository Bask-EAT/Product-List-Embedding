import os
import torch
import numpy as np
from typing import List, Optional
from PIL import Image
from transformers import AutoModel
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

# GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JinaCLIPEmbedding:
    """Jina CLIP v2 model for text and image embeddings"""

    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load Jina CLIP v2 model with GPU support"""
        try:
            model_name = os.getenv("MODEL_NAME", "jinaai/jina-clip-v2")
            torch_dtype = getattr(torch, os.getenv("TORCH_DTYPE", "bfloat16"))

            print(f"📦 Loading model: {model_name}")

            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
            ).to(device)

            self.model.eval()
            print(f"✅ Model loaded successfully on {device}")

        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            raise e

    def encode_text(
        self, text: str, task: str = "retrieval.query"
    ) -> Optional[List[float]]:
        """Encode text to embedding vector"""
        try:
            with torch.no_grad():
                embeddings = self.model.encode_text([text], task=task)

                if isinstance(embeddings, torch.Tensor):
                    return embeddings[0].cpu().numpy().tolist()
                elif isinstance(embeddings, np.ndarray):
                    return embeddings[0].tolist()
                else:
                    return embeddings[0]

        except Exception as e:
            st.error(f"❌ Text encoding error: {e}")
            return None

    def encode_image(self, image: Image.Image) -> Optional[List[float]]:
        """Encode image to embedding vector"""
        try:
            with torch.no_grad():
                embeddings = self.model.encode_image([image])

                if isinstance(embeddings, torch.Tensor):
                    return embeddings[0].cpu().numpy().tolist()
                elif isinstance(embeddings, np.ndarray):
                    return embeddings[0].tolist()
                else:
                    return embeddings[0]

        except Exception as e:
            st.error(f"❌ Image encoding error: {e}")
            return None
