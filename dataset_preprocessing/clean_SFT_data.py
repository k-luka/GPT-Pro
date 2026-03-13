import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import random
import os

# CONFIG
TRAIN_FILE = "data/sft_data/sft_train.pt"
VAL_FILE = "data/sft_data/sft_val.pt"
MAX_LENGTH = 2048 
DATASET_NAME = "yahma/alpaca-cleaned"
VAL_SPLIT = 0.05  # 5% for validation

def preprocess():
    # Ensure directory exists
    os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)
    
    enc = tiktoken.get_encoding('gpt2')
    eot_token = enc.eot_token
    dataset = load_dataset(DATASET_NAME, split="train")

    processed_data = []
    
    user_header = "User: "
    assist_header = "\nAssistant: "

    print("Tokenizing data...")
    for item in tqdm(dataset):
        instruction = item.get('instruction') # pyrefly: ignore
        response = item.get('output') # pyrefly: ignore

        if not instruction or not response:
            continue
        
        # We prepend EOS as a BOS/start token, which also acts as an attention sink.
        prompt_text = f"{user_header}{instruction}{assist_header}"
        prompt_tokens = [eot_token] + enc.encode_ordinary(prompt_text)
        
        # Format & Tokenize Response
        response_tokens = enc.encode_ordinary(response) + [eot_token]
        
        # Combine
        full_tokens = prompt_tokens + response_tokens
        
        if len(full_tokens) <= MAX_LENGTH:
            mask_len = len(prompt_tokens)
            processed_data.append({
                "tokens": torch.tensor(full_tokens, dtype=torch.long),
                "mask_len": mask_len
            })

    # SPLITTING
    print(f"\nTotal examples: {len(processed_data)}")
    print("Shuffling and splitting...")
    
    # Shuffle deterministically so the split is always the same if you re-run
    random.seed(42) 
    random.shuffle(processed_data)
    
    split_idx = int(len(processed_data) * (1 - VAL_SPLIT))
    train_data = processed_data[:split_idx]
    val_data = processed_data[split_idx:]
    
    print(f"Train size: {len(train_data)}")
    print(f"Val size:   {len(val_data)}")

    print(f"Saving to {TRAIN_FILE} and {VAL_FILE}...")
    torch.save(train_data, TRAIN_FILE)
    torch.save(val_data, VAL_FILE)
    print("Done!")

def inspect_file(file_path, num_samples=3):
    print(f"\n--- INSPECTING: {file_path} ---")
    data = torch.load(file_path)
    enc = tiktoken.get_encoding('gpt2')
    num_samples = min(num_samples, len(data))
    
    for i in range(num_samples):
        item = data[i]
        tokens = item['tokens']
        mask_len = item['mask_len']
        
        print(f"\n[Example {i}]")
        print(f"Tensor Shape: {tokens.shape}")
        
        # 1. Check EOS at the end
        last_token = tokens[-1].item()
        is_eos = (last_token == enc.eot_token)
        print(f"Ends with EOS (50256)? {is_eos} (Token: {last_token})")
        
        # 2. Decode the Prompt (Masked part)
        prompt_tokens = tokens[:mask_len]
        prompt_text = enc.decode(prompt_tokens.tolist())
        print(f"PROMPT (Masked): {repr(prompt_text)}")
        
        # 3. Decode the Response (Training part)
        response_tokens = tokens[mask_len:]
        response_text = enc.decode(response_tokens.tolist())
        print(f"RESPONSE (Train):  {repr(response_text)}")
        
    print("\n-------------------------------")

# Add this to your main block
if __name__ == "__main__":
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(VAL_FILE):
        preprocess()

    inspect_file(TRAIN_FILE)