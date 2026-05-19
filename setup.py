import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
print("Model loaded on:", next(model.parameters()).device)
tokens = model.to_tokens("Hello, world!")
print("Ready.")
