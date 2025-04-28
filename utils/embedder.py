import torch

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
)


def get_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    return model, tokenizer


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()  # (1, seq_len, hidden_size)
    summed = (last_hidden_state * mask).sum(1)  # (1, hidden_size)
    counts = mask.sum(1).clamp(min=1e-9) # (1, hidden_size); avoid zero-division by setting the minimal possible value to 1e-9
    embedding = summed / counts  # (1, hidden_size)

    return embedding


def get_hf_embedding(
        model,
        tokenizer,
        texts,
        pooling_func=mean_pooling,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        batch_size=2,
        max_length=512
):
    is_cuda_available = device.type == "cuda"
    model.eval()
    if is_cuda_available:
        model = model.half()

    dataloader = DataLoader(
        texts,
        batch_size=batch_size,
        collate_fn=lambda batch: tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    )

    embeddings = list()
    with torch.inference_mode():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            output = model(**inputs).last_hidden_state
            batch_embeddings = pooling_func(output, inputs["attention_mask"])
            embeddings.extend(batch_embeddings.cpu())

    return embeddings