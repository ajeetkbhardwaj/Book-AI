import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        droprate: float = 0.0,
        causal: bool = True,
        training: bool = True):
    # 1. cosine similarity via dot product between query and key 
    logits_score = torch.matmul(query, key.transpose(-2, -1))
    # 2. normalize/scale the logit score
    d_k = key.size(-1)
    scaled_logits_score = logits_score / math.sqrt(d_k)
    # 3. Then Apply padding mask
    if mask is not None:
        scaled_logits_score= scaled_logits_score.masked_fill(mask, -1e9)
    # 4. Apply causal(look-ahead) mask
    if causal:
        l_q = query.size(-2)
        l_k = key.size(-2)
        # create upper triangular mask
        causal_mask = torch.triu(
            torch.ones((l_q, l_k),
                       dtype=torch.bool,
                       device=query.device,
                       diagonal=1)
        )

    # 5. softmax for weights score
    weights = F.softmax(scaled_logits_score, dim=-1)
                                
    # 6. addition of dropout requires training flag
    weights = F.dropout(weights, p=droprate, training=training)
    # 7. attention scores
    attention = torch.matmul(weights, value)
    return attention, weights

if __name__ == "__main__":
    query = torch.rand(1, 2, 3) 
    key = torch.rand(1, 2, 3)
    value = torch.rand(1, 2, 3)
    attention, weights = scaled_dot_product_attention(query, key, value)
    print(attention)
    print(weights)