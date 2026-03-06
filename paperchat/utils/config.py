import torch

from .api_keys import get_api_key

DEVICE = "auto"


def get_device():
    if DEVICE == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return DEVICE


# LLM
DEFAULT_PROVIDER = "gemini" if get_api_key("gemini") else "openai"
DEFAULT_MODEL = "gemini-2.0-flash" if get_api_key("gemini") else "gpt-4.1-nano"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

DEFAULT_SYSTEM_PROMPT = """
You are a helpful academic research assistant helping with scientific papers.
Act as an expert in the field of the paper you're discussing.
Answer questions based on your knowledge of the paper and the context provided.
When citing information, refer to the specific section of the paper where it appears.
Be precise, scholarly, and focus on the factual content of the paper.
"""
