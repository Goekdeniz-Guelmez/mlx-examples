from orpo_trainer import train_orpo, ORPOTrainingArgs
import mlx_lm
import mlx.optimizers as optim
from typing import List, Dict

data = [
    {
        "prompt": "What is the capital of France?",
        "chosen": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>",
        "rejected": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nThe capital of France is London.<|im_end|>"
    },
    {
        "prompt": "What's 2+2?",
        "chosen": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat's 2+2?<|im_end|>\n<|im_start|>assistant\n2+2 equals 4<|im_end|>",
        "rejected": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat's 2+2?<|im_end|>\n<|im_start|>assistant\n2+2 equals 5<|im_end|>"
    },
    {
        "prompt": "What is the capital of France?",
        "chosen": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>",
        "rejected": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nThe capital of France is London.<|im_end|>"
    },
    {
        "prompt": "What's 2+2?",
        "chosen": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat's 2+2?<|im_end|>\n<|im_start|>assistant\n2+2 equals 4<|im_end|>",
        "rejected": "<|im_start|>system\nYou are a cool assistant.<|im_end|>\n<|im_start|>user\nWhat's 2+2?<|im_end|>\n<|im_start|>assistant\n2+2 equals 5<|im_end|>"
    }
]

class SimpleDataset:
    def __init__(self, data: List[Dict], tokenizer):
        # Pre-tokenize all the data
        self.tokenized_data = []
        for item in data:
            tokenized_item = {
                "prompt": item["prompt"],
                "chosen": tokenizer.encode(item["chosen"]),
                "rejected": tokenizer.encode(item["rejected"])
            }
            self.tokenized_data.append(tokenized_item)
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]

model, tokenizer = mlx_lm.load("mlx-community/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1-4bit")

train_dataset = SimpleDataset(data, tokenizer)
val_dataset = SimpleDataset(data, tokenizer)

optimizer = optim.Adam(learning_rate=1e-5)

training_args = ORPOTrainingArgs(
    batch_size=2,
    iters=1000,
    max_seq_length=2048,
    steps_per_eval=100,
    steps_per_report=10,
    alpha=0.1
)

# Training
train_orpo(
    model=model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    args=training_args
)