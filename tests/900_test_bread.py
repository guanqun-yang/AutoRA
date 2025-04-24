from typing import Dict

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim

def get_bread_query(query):
    return f'Represent this sentence for searching relevant passages: {query}'

# The model works really well with cls pooling (default) but also with mean pooling.
def pooling(outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
    else:
        raise NotImplementedError
    return outputs.detach().cpu().numpy()

model_id = 'mixedbread-ai/mxbai-embed-large-v1'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).cuda()

docs = [
    get_bread_query('A man is eating a piece of bread'),
    "A man is eating food.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
]

inputs = tokenizer(docs, padding=True, return_tensors='pt')
for k, v in inputs.items():
    inputs[k] = v.cuda()
outputs = model(**inputs).last_hidden_state
embeddings = pooling(outputs, inputs, 'cls')

similarities = cos_sim(embeddings[0], embeddings[1:])
print('similarities:', similarities)