import mlx.core as mx
import mlx.nn as nn
import time
from pathlib import Path
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten
from mlx.nn.losses import cross_entropy
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import numpy as np
from trainer import TrainingCallback, grad_checkpoint, TrainingArgs

@dataclass
class ORPOTrainingArgs(TrainingArgs):
    alpha: float = field(
        default=0.1,
        metadata={"help": "Weight for the preference loss term"}
    )
    disable_prompt_loss: bool = field(
        default=False,
        metadata={"help": "Whether to disable loss computation on prompt tokens"}
    )
    pad_token_id: int = field(
        default=0,
        metadata={"help": "Token ID for padding"}
    )

def orpo_loss(model, inputs: mx.array, targets: mx.array, 
              lengths: mx.array, alpha: float, 
              pad_token_id: int) -> Tuple[mx.array, mx.array]:
    """
    Compute ORPO loss combining NLL and preference learning
    """
    def compute_logps(logits: mx.array, inputs: mx.array, 
                     attn_mask: mx.array, prompt_mask: mx.array) -> mx.array:
        # Compute per-token log probabilities
        log_probs = nn.log_softmax(logits[:, :-1, :], axis=-1)
        
        # Create mask for response tokens (excluding prompt)
        resp_mask = attn_mask[:, :-1] - prompt_mask[:, 1:]
        
        # Gather log probs for actual tokens
        token_indices = mx.expand_dims(inputs[:, 1:], axis=-1)
        per_token_logps = mx.take_along_axis(log_probs, token_indices, axis=-1)
        per_token_logps = mx.squeeze(per_token_logps, axis=-1)
        
        # Mask and average
        masked_logps = per_token_logps * resp_mask
        token_counts = mx.sum(resp_mask, axis=1)
        return mx.sum(masked_logps, axis=1) / token_counts

    # Create attention masks
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]
    
    # Forward passes for positive and negative samples
    pos_logits = model(inputs)
    
    # Standard NLL loss on positive samples
    ce_loss = cross_entropy(pos_logits.astype(mx.float32), targets)
    ce_loss = ce_loss * length_mask
    n_tokens = mx.sum(length_mask)
    ce_loss = mx.sum(ce_loss) / n_tokens

    # Compute log probabilities for preference learning
    pos_logps = compute_logps(pos_logits, inputs, length_mask, length_mask)
    
    # Compute log odds and final loss
    preference_loss = -mx.mean(pos_logps)
    
    total_loss = ce_loss + alpha * preference_loss
    return total_loss, n_tokens

def orpo_iterate_batches(dataset: List[Dict[str, Any]], 
                        tokenizer: Any,
                        batch_size: int,
                        max_seq_length: int,
                        train: bool = False):
    """
    Iterator for ORPO training data format
    """
    # Sort by length of chosen responses
    idx = sorted(range(len(dataset)), 
                key=lambda i: len(dataset[i]["chosen"]))
    
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    # Handle distributed training
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError(
            "The batch size must be divisible by the number of workers"
        )

    # Create batches
    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        indices = np.random.permutation(len(batch_idx)) if train else range(len(batch_idx))
        
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            
            # Get sequence lengths
            lengths = [len(x["chosen"]) for x in batch]
            max_len = min(max(lengths), max_seq_length)
            
            # Pad to multiple of 8
            pad_to = 8
            max_len = pad_to * ((max_len + pad_to - 1) // pad_to)
            
            # Create arrays
            batch_size = len(batch)
            inputs = np.zeros((batch_size, max_len), np.int32)
            targets = np.zeros((batch_size, max_len), np.int32)
            
            # Fill arrays
            for j, item in enumerate(batch):
                seq_len = min(len(item["chosen"]), max_len)
                inputs[j, :seq_len] = item["chosen"][:seq_len]
                targets[j, :seq_len] = item["rejected"][:seq_len]
                lengths[j] = seq_len
                
            yield (mx.array(inputs), 
                   mx.array(targets),
                   mx.array(lengths))

        if not train:
            break

def train_orpo(
    model: nn.Module,
    tokenizer: Any,
    optimizer: Any,
    train_dataset: List[Dict[str, Any]],
    val_dataset: List[Dict[str, Any]],
    args: ORPOTrainingArgs = ORPOTrainingArgs(),
    training_callback: TrainingCallback = None,
):
    """
    Main training loop for ORPO
    """
    print(f"Starting ORPO training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    # Define step function with detailed loss computation
    def step(batch):
        def compute_batch_loss(model, inputs, targets, lengths):
            # Get the logits and compute basic NLL loss
            logits = model(inputs)
            length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]
            
            # Compute NLL loss
            ce_loss = cross_entropy(logits.astype(mx.float32), targets) * length_mask
            n_tokens = mx.sum(length_mask)
            ce_loss = mx.sum(ce_loss) / n_tokens
            
            # Compute preference logits
            log_probs = nn.log_softmax(logits[:, :-1, :], axis=-1)
            resp_mask = length_mask[:, :-1]
            
            # Gather token probabilities
            token_indices = mx.expand_dims(inputs[:, 1:], axis=-1)
            per_token_logps = mx.take_along_axis(log_probs, token_indices, axis=-1)
            per_token_logps = mx.squeeze(per_token_logps, axis=-1)
            
            # Compute log probabilities
            masked_logps = per_token_logps * resp_mask
            token_counts = mx.sum(resp_mask, axis=1)
            pos_logps = mx.sum(masked_logps, axis=1) / token_counts
            
            # Compute preference loss
            pref_loss = -mx.mean(pos_logps)
            
            # Combine losses
            total_loss = ce_loss + args.alpha * pref_loss
            
            metrics = {
                'nll_loss': ce_loss.item(),
                'pref_loss': pref_loss.item(),
                'pos_logp_mean': mx.mean(pos_logps).item()
            }
            
            return total_loss, (n_tokens, metrics)
        
        def loss_fn(model, *batch):
            return compute_batch_loss(model, *batch)[0]  # Return only the loss
        
        # Pass the function directly to value_and_grad
        loss_and_grad = nn.value_and_grad(model=model, fn=loss_fn)
        loss, grad = loss_and_grad(model, *batch)
        
        # Compute the actual loss and metrics separately
        _, (toks, metrics) = compute_batch_loss(model, *batch)
        
        grad = average_gradients(grad)
        optimizer.update(model, grad)
        return loss, toks, metrics

    # Training loop
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    metrics_sum = {'nll_loss': 0.0, 'pref_loss': 0.0, 'pos_logp_mean': 0.0}
    
    start = time.perf_counter()
    
    for it, batch in zip(
        range(1, args.iters + 1),
        orpo_iterate_batches(
            dataset=train_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        # Validation
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            
            val_loss = evaluate_orpo(
                model=model,
                dataset=val_dataset,
                loss=orpo_loss,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=orpo_iterate_batches,
            )
            val_time = time.perf_counter() - stop
            
            if rank == 0:
                print(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                val_info = {
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            start = time.perf_counter()

        # Training step
        lvalue, toks, batch_metrics = step(batch)
        losses += lvalue
        n_tokens += toks
        steps += 1
        
        # Accumulate metrics
        for k, v in batch_metrics.items():
            metrics_sum[k] += v
        
        mx.eval(state, losses, n_tokens)

        # Report training metrics
        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item()
            train_loss /= steps * world_size
            n_tokens = mx.distributed.all_sum(n_tokens).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.metal.get_peak_memory() / 1e9
            
            # Average metrics
            avg_metrics = {k: v / steps for k, v in metrics_sum.items()}
            
            if rank == 0:
                print(
                    f"Iter {it}: "
                    f"Train loss {train_loss:.3f}, "
                    f"NLL loss {avg_metrics['nll_loss']:.3f}, "
                    f"Pref loss {avg_metrics['pref_loss']:.3f}, "
                    f"Pos logp {avg_metrics['pos_logp_mean']:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "nll_loss": avg_metrics['nll_loss'],
                    "pref_loss": avg_metrics['pref_loss'],
                    "pos_logp_mean": avg_metrics['pos_logp_mean'],
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            metrics_sum = {k: 0.0 for k in metrics_sum}
            start = time.perf_counter()

        # Save adapter weights
        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    # Save final weights
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.")
    
    return model

def evaluate_orpo(
    model,
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss: callable = orpo_loss,
    iterate_batches: callable = orpo_iterate_batches,
    alpha: float = 0.1,  # Add default alpha
    pad_token_id: int = 0,  # Add default pad_token_id
):
    all_losses = 0
    ntokens = 0

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        # Pass the additional arguments to the loss function
        losses, toks = loss(model, *batch, alpha=alpha, pad_token_id=pad_token_id)
        all_losses += losses * toks
        ntokens += toks
        mx.eval(all_losses, ntokens)

    all_losses = mx.distributed.all_sum(all_losses)
    ntokens = mx.distributed.all_sum(ntokens)

    return (all_losses / ntokens).item()