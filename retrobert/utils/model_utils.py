"""Seeding, checkpoint I/O, optimizer and scheduler."""

import os
import random
import math
import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def set_seed(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_model(model, optimizer, scheduler, step, best_metric, args, name="best"):
    os.makedirs(args.save_model_path, exist_ok=True)
    save_file_path = os.path.join(args.save_model_path, f"checkpoint_{name}.pth.tar")
    state_dict = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": args,
        "best_metric": best_metric,
    }
    torch.save(state_dict, save_file_path)
    print(f"Model checkpoint '{name}' saved successfully to {save_file_path}.")

def set_optim(model, args, loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_batches = max(1, len(loader))
    steps_per_epoch = math.ceil(total_batches / args.gradient_accumulation_steps)
    total_training_steps = max(1, steps_per_epoch * args.train_epochs)
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_training_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    return optimizer, scheduler, total_training_steps, warmup_steps

def load_model(model, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
