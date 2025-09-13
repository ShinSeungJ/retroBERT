import os
import random
import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup


def set_seed(args):
    """ Ensure reproducibility by setting seed for random number generation """
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)   # if using multiple GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(model, optimizer, scheduler, step, best_metric, args, name="best"):
    """ Save model checkpoints """
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
    """ Initialize optimizer and learning rate scheduler for the model """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # adjust the learning rate decay to decrease faster during training
    t_total = (len(loader.dataset) // (args.train_batch_size)) * args.train_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    return optimizer, scheduler


def load_model(model, model_path):
    """ Load saved model checkpoint for inference """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode
    return model
