import time
import mlx.core as mx
from pathlib import Path
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from dataclasses import dataclass, field
from typing import Dict, Optional
from trainer import TrainingArgs, TrainingCallback

def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn

@dataclass
class ORPOTrainingArgs(TrainingArgs):
    alpha: float = field(default=1.0, metadata={"help": "Weight for the ORPO loss term"})
    pad_token_id: int = field(default=0, metadata={"help": "Padding token ID"})
    disable_prompt_loss: bool = field(
        default=False, 
        metadata={"help": "Whether to disable loss computation on prompt tokens"}
    )

def compute_batch_metrics(batch, loss):
    """
    Compute additional metrics for ORPO training.
    Returns dict with pos_prob, neg_prob, log_odds, etc.
    """
    # Extract values from the last forward pass
    pos_logps = batch.get('_cached_pos_logps')
    neg_logps = batch.get('_cached_neg_logps')
    log_odds = batch.get('_cached_log_odds')
    
    if pos_logps is None or neg_logps is None or log_odds is None:
        return {}
        
    return {
        'positive_geometric_mean': mx.mean(pos_logps).item(),
        'negative_geometric_mean': mx.mean(neg_logps).item(),
        'log_odds_mean': mx.mean(log_odds).item(),
        'log_odds_ratio': mx.mean(mx.log(mx.sigmoid(log_odds))).item()
    }


def orpo_loss(model, inputs: Dict):
    """
    Compute ORPO loss with enhanced numerical stability.
    """
    # Create new arrays for labels
    pos_labels = mx.array(inputs['positive_input_ids'])
    neg_labels = mx.array(inputs['negative_input_ids'])
    
    pad_id = inputs.get('pad_token_id', 0)
    pos_labels = mx.where(pos_labels == pad_id, -100, pos_labels)
    neg_labels = mx.where(neg_labels == pad_id, -100, neg_labels)

    try:
        # Forward passes
        pos_logits = model(inputs['positive_input_ids'])
        neg_logits = model(inputs['negative_input_ids'])
        
        # Handle NaN/Inf in logits immediately
        pos_logits = mx.where(mx.isnan(pos_logits), 0.0, pos_logits)
        pos_logits = mx.where(mx.isinf(pos_logits), 0.0, pos_logits)
        neg_logits = mx.where(mx.isnan(neg_logits), 0.0, neg_logits)
        neg_logits = mx.where(mx.isinf(neg_logits), 0.0, neg_logits)
        
        # Convert to float32 for better numerical stability
        pos_logits = pos_logits.astype(mx.float32)
        neg_logits = neg_logits.astype(mx.float32)
        
        # Apply layernorm-style normalization to logits
        pos_mean = mx.mean(pos_logits, axis=-1, keepdims=True)
        pos_std = mx.std(pos_logits, axis=-1, keepdims=True) + 1e-5
        pos_logits = (pos_logits - pos_mean) / pos_std
        
        neg_mean = mx.mean(neg_logits, axis=-1, keepdims=True)
        neg_std = mx.std(neg_logits, axis=-1, keepdims=True) + 1e-5
        neg_logits = (neg_logits - neg_mean) / neg_std

        # Compute cross entropy with careful masking
        valid_mask = (pos_labels != -100).astype(mx.float32)
        safe_labels = mx.where(pos_labels == -100, 0, pos_labels)
        
        # Use cross entropy with stable computation
        pos_loss = nn.losses.cross_entropy(
            pos_logits,
            safe_labels,
            reduction='none'
        )
        pos_loss = mx.where(mx.isnan(pos_loss), 0.0, pos_loss)
        
        # Apply mask and compute safe mean
        masked_loss = pos_loss * valid_mask
        token_count = mx.maximum(mx.sum(valid_mask), 1.0)
        pos_loss = mx.sum(masked_loss) / token_count
        
        # Compute logits for preference loss with normalization
        pos_logps = compute_stable_logps(
            pos_logits,
            inputs['attention_mask'],
            inputs['positive_input_ids'],
            inputs['positive_attention_mask']
        )
        neg_logps = compute_stable_logps(
            neg_logits,
            inputs['attention_mask'],
            inputs['negative_input_ids'],
            inputs['negative_attention_mask']
        )

        # Compute preference loss
        alpha = inputs.get('alpha', 1.0)
        log_diff = (pos_logps - neg_logps)
        
        # Use stable sigmoid computation
        x = mx.clip(log_diff, -10.0, 10.0)
        sigmoid = 1.0 / (1.0 + mx.exp(-x))
        sigmoid = mx.clip(sigmoid, 1e-7, 1.0 - 1e-7)
        
        pref_loss = -alpha * mx.mean(mx.log(sigmoid))
        pref_loss = mx.where(mx.isnan(pref_loss), 0.0, pref_loss)
        
        # Store metrics
        inputs['_cached_pos_logps'] = pos_logps
        inputs['_cached_neg_logps'] = neg_logps
        inputs['_cached_log_odds'] = log_diff

        # Combine losses with stability checks
        loss = pos_loss + pref_loss
        loss = mx.where(mx.isnan(loss), pos_loss, loss)

        return loss, pos_logits.shape[0]
        
    except Exception as e:
        print(f"Error in loss computation: {e}")
        return mx.array(0.0), 0

def compute_stable_logps(logits, prompt_mask, chosen_inputs, chosen_mask):
    """
    Compute log probabilities with enhanced numerical stability.
    """
    try:
        batch_size, seq_len, vocab_size = logits.shape
        seq_len = seq_len - 1
        
        # Find prompt lengths
        prompt_lens = mx.sum(prompt_mask, axis=1)
        prompt_lens = mx.clip(prompt_lens, 0, seq_len)
        
        # Create response mask
        full_seq_positions = mx.expand_dims(mx.arange(seq_len), 0)
        prompt_lens_expanded = mx.expand_dims(prompt_lens, 1)
        response_mask = (full_seq_positions >= prompt_lens_expanded) * chosen_mask[:, :-1]
        
        # Get shifted logits
        shifted_logits = logits[:, :-1, :]
        
        # Apply stable softmax using double normalization
        max_logits = mx.max(shifted_logits, axis=-1, keepdims=True)
        logits_norm = shifted_logits - max_logits
        exp_logits = mx.exp(logits_norm)
        sum_exp = mx.sum(exp_logits, axis=-1, keepdims=True)
        sum_exp = mx.maximum(sum_exp, 1e-7)
        log_sum_exp = mx.log(sum_exp)
        log_probs = logits_norm - log_sum_exp
        
        # Get target token probabilities
        indices = mx.expand_dims(chosen_inputs[:, 1:], axis=2)
        per_token_logps = mx.take_along_axis(log_probs, indices, axis=2).squeeze(2)
        
        # Apply response mask
        masked_logps = per_token_logps * response_mask
        token_counts = mx.maximum(mx.sum(response_mask, axis=1), 1.0)
        
        # Safe mean computation
        sum_logps = mx.sum(masked_logps, axis=1)
        result = sum_logps / token_counts
        
        # Final clipping for stability
        return mx.clip(result, -10.0, 0.0)
        
    except Exception as e:
        print(f"Error in log_probs computation: {e}")
        return mx.zeros((batch_size,))

def iterate_orpo_batches(
    dataset,
    tokenizer,
    batch_size,
    max_seq_length,
    train=False
):
    """Create batches from preference pairs dataset.
    
    Args:
        dataset: Dataset containing preference pairs
        tokenizer: Tokenizer for processing inputs
        batch_size: Desired batch size (will be reduced if dataset is smaller)
        max_seq_length: Maximum sequence length for truncation
        train: Whether this is for training (affects shuffling)
        
    Yields:
        dict: Batch of processed examples
    """
    # Get actual batch size based on dataset size
    actual_batch_size = min(batch_size, len(dataset))
    if actual_batch_size == 0:
        raise ValueError("Dataset cannot be empty")
        
    # Sort by length
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]["prompt"]))
    
    # Make the batches
    batch_idx = [
        idx[i : i + actual_batch_size] 
        for i in range(0, len(idx), actual_batch_size)
    ]

    while True:
        indices = np.random.permutation(len(batch_idx)) if train else range(len(batch_idx))
        for i in indices:
            # Get examples for this batch
            examples = [dataset[j] for j in batch_idx[i]]
            curr_batch_size = len(examples)
            
            # Encode each sequence in the batch
            batch_prompts = []
            batch_chosen = []
            batch_rejected = []
            
            for ex in examples:
                # Encode texts directly without applying templates
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": ex["prompt"]}],
                    tokenize=False
                )
                batch_prompts.append(tokenizer.encode(prompt_text))
                batch_chosen.append(tokenizer.encode(ex["chosen"]))
                batch_rejected.append(tokenizer.encode(ex["rejected"]))

            # Get lengths for padding
            prompt_lengths = [len(x) for x in batch_prompts]
            chosen_lengths = [len(x) for x in batch_chosen]
            rejected_lengths = [len(x) for x in batch_rejected]

            # Handle sequence length limits
            max_prompt_len = min(max(prompt_lengths), max_seq_length)
            max_chosen_len = min(max(chosen_lengths), max_seq_length)
            max_rejected_len = min(max(rejected_lengths), max_seq_length)

            # Create numpy arrays for the batch
            pad_id = tokenizer.pad_token_id
            prompt_arr = np.full((curr_batch_size, max_prompt_len), pad_id, np.int32)
            chosen_arr = np.full((curr_batch_size, max_chosen_len), pad_id, np.int32)
            rejected_arr = np.full((curr_batch_size, max_rejected_len), pad_id, np.int32)

            # Create attention masks
            prompt_mask = np.zeros((curr_batch_size, max_prompt_len))
            chosen_mask = np.zeros((curr_batch_size, max_chosen_len))
            rejected_mask = np.zeros((curr_batch_size, max_rejected_len))

            # Fill arrays with actual sequences
            for j in range(curr_batch_size):
                # Prompt
                p_len = min(len(batch_prompts[j]), max_prompt_len)
                prompt_arr[j, :p_len] = batch_prompts[j][:p_len]
                prompt_mask[j, :p_len] = 1

                # Chosen
                c_len = min(len(batch_chosen[j]), max_chosen_len)
                chosen_arr[j, :c_len] = batch_chosen[j][:c_len]
                chosen_mask[j, :c_len] = 1

                # Rejected
                r_len = min(len(batch_rejected[j]), max_rejected_len)
                rejected_arr[j, :r_len] = batch_rejected[j][:r_len]
                rejected_mask[j, :r_len] = 1

            # Convert to MLX arrays
            batch = {
                "attention_mask": mx.array(prompt_mask),
                "positive_input_ids": mx.array(chosen_arr),
                "positive_attention_mask": mx.array(chosen_mask),
                "negative_input_ids": mx.array(rejected_arr),
                "negative_attention_mask": mx.array(rejected_mask),
                "pad_token_id": pad_id,
                "lengths": mx.array([min(l, max_seq_length) for l in chosen_lengths])
            }

            yield batch

        if not train:
            break


def evaluate_orpo(
    model,
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss: callable = orpo_loss,
):
    """
    Evaluate model on preference pairs dataset.
    
    Args:
        model: The MLX model to evaluate
        dataset: Dataset containing preference pairs
        tokenizer: Tokenizer for processing inputs
        batch_size: Batch size for evaluation
        num_batches: Number of batches to evaluate (-1 for full dataset)
        max_seq_length: Maximum sequence length for truncation
        loss: Loss function to use for evaluation
        
    Returns:
        tuple: (average loss, dictionary of additional metrics)
    """
    total_loss = 0
    total_tokens = 0
    metrics_sum = {}
    
    # Create batch iterator
    batch_iter = iterate_orpo_batches(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        train=False
    )
    
    # Determine number of iterations
    if num_batches == -1:
        num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for _ in range(num_batches):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
            
        # Compute loss and number of tokens
        loss_val, num_tokens = loss(model, batch)
        
        # Force evaluation to get concrete values
        loss_val = loss_val.item()
        # Check if num_tokens is already an int or needs .item()
        if hasattr(num_tokens, 'item'):
            num_tokens = num_tokens.item()
        
        # Update running totals
        total_loss += loss_val * num_tokens
        total_tokens += num_tokens
        
        # Compute additional metrics
        batch_metrics = compute_batch_metrics(batch, loss_val)
        
        # Accumulate weighted metrics
        for key, value in batch_metrics.items():
            if key not in metrics_sum:
                metrics_sum[key] = 0
            metrics_sum[key] += value * num_tokens
    
    # Compute averages
    if total_tokens == 0:
        return 0.0, {}
        
    avg_loss = total_loss / total_tokens
    avg_metrics = {
        key: value / total_tokens 
        for key, value in metrics_sum.items()
    }
    
    return avg_loss, avg_metrics

class ORPODataset:
    """Dataset for ORPO training with prompt, chosen, rejected format"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get the components
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Create full sequences by combining prompt with responses
        chosen_seq = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen}
            ],
            tokenize=False
        )
        rejected_seq = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected}
            ],
            tokenize=False
        )
        prompt_only = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False
        )
        
        # Tokenize sequences
        chosen_tokens = self.tokenizer.encode(chosen_seq)
        rejected_tokens = self.tokenizer.encode(rejected_seq)
        prompt_tokens = self.tokenizer.encode(prompt_only)
        
        # Create attention masks (1 for tokens, 0 for padding)
        chosen_mask = [1] * len(chosen_tokens)
        rejected_mask = [1] * len(rejected_tokens)
        prompt_mask = [1] * len(prompt_tokens)
        
        # Truncate if needed
        max_len = self.max_length
        if len(chosen_tokens) > max_len:
            chosen_tokens = chosen_tokens[:max_len]
            chosen_mask = chosen_mask[:max_len]
        if len(rejected_tokens) > max_len:
            rejected_tokens = rejected_tokens[:max_len]
            rejected_mask = rejected_mask[:max_len]
        if len(prompt_tokens) > max_len:
            prompt_tokens = prompt_tokens[:max_len]
            prompt_mask = prompt_mask[:max_len]
            
        return {
            'attention_mask': mx.array(prompt_mask),
            'positive_input_ids': mx.array(chosen_tokens),
            'positive_attention_mask': mx.array(chosen_mask),
            'negative_input_ids': mx.array(rejected_tokens),
            'negative_attention_mask': mx.array(rejected_mask),
        }

    def __len__(self):
        return len(self.data)

def train_orpo(
    model: nn.Module,
    tokenizer,
    optimizer,
    train_dataset: ORPODataset,
    val_dataset: Optional[ORPODataset] = None,
    args: ORPOTrainingArgs = ORPOTrainingArgs(),
    training_callback: Optional[TrainingCallback] = None,
):
    """
    Train a model using ORPO with proper gradient clipping.
    """
    print(f"Starting ORPO training..., iters: {args.iters}")
    
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, batch)
        
        # Clip gradients to prevent explosions using MLX arrays
        for g in tree_flatten(grad)[0]:
            if isinstance(g, mx.array):
                g[:] = mx.clip(g, mx.array(-1.0), mx.array(1.0))
            
        # Model update
        optimizer.update(model, grad)
        return lvalue, toks

    loss_value_and_grad = nn.value_and_grad(model, orpo_loss)

    losses = []
    n_tokens = 0
    trained_tokens = 0
    start = time.perf_counter()
    
    # Main training loop
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_orpo_batches(
            dataset=train_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        # Report validation loss if needed
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss, val_metrics = evaluate_orpo(
                model=model,
                dataset=val_dataset,
                loss=orpo_loss,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
            )
            val_time = time.perf_counter() - stop
            print(
                f"Iter {it}: Val loss {val_loss:.3f}, Val took {val_time:.3f}s"
            )

            if training_callback is not None:
                val_info = {
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_time": val_time,
                    **val_metrics
                }
                training_callback.on_val_loss_report(val_info)

            start = time.perf_counter()

        try:
            # Perform training step
            lvalue, toks = step(batch)
            mx.eval(state)  # Ensure gradients are applied
            
            # Convert values safely
            try:
                loss_val = float(lvalue.item()) if hasattr(lvalue, 'item') else float(lvalue)
                num_tokens = int(toks.item()) if hasattr(toks, 'item') else int(toks)
                
                # Only record if values are valid
                if not (mx.isnan(loss_val) or mx.isinf(loss_val)):
                    losses.append(loss_val)
                    n_tokens += num_tokens
                    trained_tokens += num_tokens
                else:
                    print(f"Skipping invalid loss value at iteration {it}")
                    
            except (ValueError, TypeError) as e:
                print(f"Error processing loss values at iteration {it}: {e}")
                continue

        except Exception as e:
            print(f"Error in training step at iteration {it}: {e}")
            continue

        # Report training metrics if needed
        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()
            
            if losses:  # Only compute stats if we have valid losses
                train_loss = np.mean(losses)
                learning_rate = optimizer.learning_rate.item()
                it_sec = args.steps_per_report / (stop - start)
                tokens_sec = float(n_tokens) / (stop - start)
                peak_mem = mx.metal.get_peak_memory() / 2**30
                
                print(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem:.3f} GB"
                )

                if training_callback is not None:
                    train_info = {
                        "iteration": it,
                        "train_loss": train_loss,
                        "learning_rate": learning_rate,
                        "iterations_per_second": it_sec,
                        "tokens_per_second": tokens_sec,
                        "trained_tokens": trained_tokens,
                        "peak_memory": peak_mem
                    }
                    training_callback.on_train_loss_report(train_info)
            else:
                print(f"Iter {it}: No valid losses to report")

            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Save adapter weights
        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    # Save final weights
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.")