import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def summarize_data(path, is_text=False):
    if not is_text:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = path
    
    vocab_size = len(set(text))

    print(f"Dataset Length: {len(text)} characters")
    print(f"Vocabulary Size: {vocab_size}")

    if not is_text:
        print(f"First 200 characters: {text[:200]}")
    
    print(f"Sorted Vocab: {sorted(list(set(text)))}")

    char_to_idx = {char: idx for idx, char in enumerate(sorted(list(set(text))))}
    idx_to_char = {idx: char for idx, char in enumerate(sorted(list(set(text))))}

    return vocab_size, char_to_idx, idx_to_char
    
    



class Tokenizer():
    """
    Character level tokenizer

    Given a string, we split it into characters
    each character is mapped to an embedding vector
    """
    def __init__(self, encoding_map, decoding_map):
        self.encoding_map = encoding_map
        self.decoding_map = decoding_map


    def autoregressive_tokenize(self, text):
        """
        Given string, return tensor of input tokens and target tokens
        """
        text_list = list(text)

        input_indices = [self.encoding_map[char] for char in text_list[:-1]]    # we don't make predictions for the last character since we don't have a next character
        target_indices = [self.encoding_map[char] for char in text_list[1:]]    # we don't use the first character as a target since we don't have a previous character

        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)
    
    def encode(self, text):
        """
        Given string, return tensor of input tokens
        """
        return torch.tensor([self.encoding_map[char] for char in text[:-1]])

    def make_target(self, text):
        """
        Given string, return tensor of target tokens
        """
        return torch.tensor([self.encoding_map[char] for char in text[1:]])

    def get_encoded_loader(self, text, batch_size=32, is_text=False, seq_len=200, step_size=20):
        """
        Given string or path to file, return DataLoader
        """
        if not is_text:
            with open(text, "r", encoding="utf-8") as f:
                text = f.read()
        
        input_idx, target_idx = self.autoregressive_tokenize(text)
        # input_idx.shape = (len(text) - 1,)
        # target_idx.shape = (len(text) - 1,)

        # split into overlapping sequences of length seq_len
        # not too much overlap - avoid redundancy
        input_seq = input_idx.unfold(0, seq_len, step_size)
        target_seq = target_idx.unfold(0, seq_len, step_size)

        print(f"Input Sequence Shape: {input_seq.shape}")
        print(f"Target Sequence Shape: {target_seq.shape}")

        dataloader = DataLoader(TensorDataset(input_seq, target_seq), batch_size=batch_size, shuffle=True)

        return dataloader


    def decode(self, indices: torch.Tensor):
        """
        Given tensor of indices, return string
        """

        return "".join([self.decoding_map[idx] for idx in indices.tolist()])