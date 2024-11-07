"""Transform padded data seq to unpadded continuous seq
"""
import numpy as np

def append_line(tokens: np.ndarray, current_tokens: list, seq_length) -> tuple[list, list|None]:
    tokens: list = list(tokens)
    if len(tokens) + len(current_tokens) > seq_length:
        exceed_tokens = tokens[seq_length - len(current_tokens):]
        not_exceed_tokens = tokens[:seq_length - len(current_tokens)]
        current_tokens += not_exceed_tokens
        return current_tokens, exceed_tokens
    else:
        current_tokens += tokens
        return current_tokens, None

def depadding(data: np.ndarray, pad_token_id, seq_length=4096, context=200):
    assert data.ndim == 2, "Data should be a 2D array"
    
    new_data: list[np.ndarray] = []
    current_tokens: list[np.ndarray] = [] 
    for line in data:
        # find the index of first pad_token_id in data
        first_pad = np.where(line == pad_token_id)[0]
        if len(first_pad) == 0:
            # no pad_token_id in this line
            current_tokens, exceed_tokens = append_line(line, current_tokens, seq_length)
        else:
            first_pad = first_pad[0]
            current_tokens, exceed_tokens = append_line(line[:first_pad], current_tokens, seq_length)
            
        if exceed_tokens is not None:
            new_data.append(np.array(current_tokens, dtype=np.uint16))
            if len(exceed_tokens) > context:
                current_tokens = exceed_tokens
            else:
                current_tokens = []
                
    if current_tokens:
        new_data