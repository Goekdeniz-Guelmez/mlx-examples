# In your training script
from orpo_trainer import train_orpo
import mlx_lm
import mlx.optimizers as optim
from typing import List, Dict

data = [
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nThe capital of France is London.<|im_end|>"
    },
    {
        "prompt": "What's 2+2?",
        "chosen": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat's 2+2?<|im_end|>\n<|im_start|>assistant\n2+2 equals 4<|im_end|>",
        "rejected": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat's 2+2?<|im_end|>\n<|im_start|>assistant\n2+2 equals 5<|im_end|>"
    },
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nThe capital of France is London.<|im_end|>"
    },
    {
        "prompt": "What's 2+2?",
        "chosen": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat's 2+2?<|im_end|>\n<|im_start|>assistant\n2+2 equals 4<|im_end|>",
        "rejected": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat's 2+2?<|im_end|>\n<|im_start|>assistant\n2+2 equals 5<|im_end|>"
    }
]

class SimpleDataset:
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = SimpleDataset(data)
val_dataset = SimpleDataset(data)  # Just use first example for validation

model, tokenizer = mlx_lm.load("mlx-community/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1-4bit")
optimizer = optim.Adam(learning_rate=1e-5)

# Training
train_orpo(
    model=model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset
)