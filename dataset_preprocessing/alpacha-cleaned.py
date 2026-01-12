import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import tiktoken
import numpy as np
from datasets import load_dataset
import random

# --- ANALYSIS TOOL ---
def analyze_token_counts(dataset, tokenizer, thresholds=[1024, 2048, 4096], sample_size=None):
    print(f"\n--- Starting Length Analysis ---")
    
    data_to_scan = dataset
    if sample_size and sample_size < len(dataset):
        print(f"Sampling {sample_size} examples from {len(dataset)} total...")
        indices = random.sample(range(len(dataset)), sample_size)
        data_to_scan = dataset.select(indices)
    else:
        print(f"Scanning full dataset ({len(dataset)} examples)...")

    lengths = []
    user_prefix = "User: "
    assist_prefix = "\nAssistant: "
    eot_token = tokenizer._special_tokens['<|endoftext|>']

    for item in data_to_scan:
        instruction = item.get('instruction') or item.get('prompt')
        response = item.get('response') or item.get('output')
        
        full_text = f"{user_prefix}{instruction}{assist_prefix}{response}"
        
        # --- CRITICAL CHANGE: SANDWICH STRATEGY ---
        # 1. Start with EOS (Matches your pretraining code: tokens = [eot])
        tokens = [eot_token]
        # 2. Add text
        tokens.extend(tokenizer.encode_ordinary(full_text))
        # 3. End with EOS (Required for Chat model to learn when to stop)
        tokens.append(eot_token)
        
        lengths.append(len(tokens))

    total = len(lengths)
    lengths = np.array(lengths)
    
    # Calculate statistics
    mean_length = lengths.mean()
    std_length = lengths.std()
    q1 = np.percentile(lengths, 25)
    q3 = np.percentile(lengths, 75)
    iqr = q3 - q1
    
    print(f"\nResults (Total scanned: {total}):")
    print(f"Min length: {lengths.min()}")
    print(f"Max length: {lengths.max()}")
    print(f"Average length: {mean_length:.2f}")
    print(f"Standard deviation: {std_length:.2f}")
    print(f"Q1 (25th percentile): {q1:.2f}")
    print(f"Q3 (75th percentile): {q3:.2f}")
    print(f"IQR (Interquartile Range): {iqr:.2f}")
    
    for t in thresholds:
        count = np.sum(lengths > t)
        percentage = (count / total) * 100
        print(f"Examples > {t} tokens: {count} ({percentage:.2f}%)")
    
    print("--------------------------------\n")

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import tiktoken
from datasets import load_dataset

class ChatDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=1024):
        self.enc = tokenizer
        self.max_length = max_length
        self.eot_token = tokenizer._special_tokens['<|endoftext|>']
        self.user_prefix = "User: "
        self.assist_prefix = "\nAssistant: "
        
        # --- FILTERING STEP ---
        # We pre-process the dataset to remove the 20 examples > 1024.
        # This prevents "truncation" which might cut a good answer in half.
        print(f"Original size: {len(hf_dataset)}")
        self.data = []
        
        for item in hf_dataset:
            # We have to check length here to filter. 
            # Ideally, you'd cache this so you don't re-tokenize every startup, 
            # but for 50k examples, this takes < 5 seconds.
            full_text = f"{self.user_prefix}{item['instruction']}{self.assist_prefix}{item['output']}"
            tokens = self.enc.encode_ordinary(full_text)
            
            # Add 2 for the start/end sandwich
            total_len = len(tokens) + 2 
            
            if total_len <= self.max_length:
                # Store the tokens directly so we don't tokenize twice
                self.data.append(tokens)
        
        print(f"Filtered size (<= {max_length}): {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # We already have the tokens from the __init__ filter step
        raw_tokens = self.data[index]
        
        # Apply the Sandwich Strategy
        # [EOS] + tokens + [EOS]
        tokens = [self.eot_token] + raw_tokens + [self.eot_token]
            
        return torch.tensor(tokens, dtype=torch.long)

def custom_collate_fn(batch):
    # Dynamic Padding: Pads to the longest sequence in THIS batch, not 1024.
    padded_inputs = pad_sequence(batch, batch_first=True, padding_value=50256)
    labels = padded_inputs.clone()
    
    for i, item in enumerate(batch):
        length = len(item)
        labels[i, length:] = -100
        
    return padded_inputs, labels

# --- USAGE ---
enc = tiktoken.get_encoding('gpt2')
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# This will print "Original: 51760, Filtered: 51740"
train_ds = ChatDataset(dataset, enc, max_length=1024)

train_loader = DataLoader(
    train_ds, 
    batch_size=4, 
    shuffle=True, # Random shuffle is good for convergence
    collate_fn=custom_collate_fn
)

# Verify efficiency
for x, y in train_loader:
    print(f"Batch dimensions: {x.shape}")
    # You will likely see shapes like [4, 180] or [4, 250]
    # You will NOT see [4, 1024] unless you got very unlucky and hit a long file.
    break