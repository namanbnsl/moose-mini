import torch
from model import *
import time, json, inspect
from dataclasses import asdict
import os

version = "1_train_0"

params = ModelArgs(vocab_size=50304)
model = Moose(params)
model.to(params.device)

train_loader = DataLoaderLite(B=params.max_batch_size, T=params.max_seq_len, split="train")
val_loader = DataLoaderLite(B=params.max_batch_size, T=params.max_seq_len, split="val")

max_lr = 5e-4
min_lr = max_lr * 0.1
warmup_steps = 48
max_steps = 1600
    

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

weight_decay = 0.1
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and 'cuda' in params.device
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=weight_decay, fused=use_fused)

total_batch_size = 2**16
assert total_batch_size % (params.max_batch_size * params.max_seq_len) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (params.max_batch_size * params.max_seq_len)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

for iter in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        xb, yb = train_loader.next_batch() # by default this is train
        xb, yb = xb.to(params.device), yb.to(params.device)
        logits, loss = model(xb, targets=yb)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {iter:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

    if iter % 25 == 0 or max_steps - 1:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(params.device), y.to(params.device)
                logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

    if iter > 0 and (iter % 100 == 0 or iter == max_steps - 1):
        checkpoint_path = os.path.join(log_dir, f"model_{iter:05d}.pt")
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': iter,
            'val_loss': val_loss_accum.item()
        }

        torch.save(checkpoint, checkpoint_path)

name = f'models/moose_{version}'
torch.save(model.state_dict(), f'{name}.pth')

params_dict = asdict(params)

with open(f'{name}.json', 'w') as f:
    json.dump(params_dict, f)
