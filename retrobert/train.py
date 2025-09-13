import gc
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.model_utils import save_model

def train(model, train_loader, optimizer, criterion, step=0, valid_loader=None, best_f1=-1, scheduler=None, args=None):
    """
    model training function
    """
    train_loss_list = []
    batch_idx = 0
    if best_f1 == -1:
        best_f1 = -np.inf

    early_stop = False
    
    # training loop
    for epoch in range(1, args.train_epochs +1):
        model.train()
        for batch in train_loader:
            # extract & send to device
            source_seq = batch['input'].to(args.device)
            attention_mask = batch['mask'].to(args.device)
            labels = batch['target'].to(args.device)

            outputs = model(source_seq, attention_mask=attention_mask)
            logits = outputs.logits
            # logits = outputs
            loss = criterion(logits, labels)

            loss = torch.mean(loss) / args.gradient_accumulation_steps
            loss.backward()

            # gradient accumulation logic
            if batch_idx % args.gradient_accumulation_steps == 0:
                # clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                model.zero_grad()
                step += 1
            
            train_loss_list.append(loss.item() * args.gradient_accumulation_steps)
            
            # current learning rate from scheduler or optimizer
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]

            # logging
            if batch_idx % (args.report_every_step * args.gradient_accumulation_steps) == 0:
                log = f"epoch: {epoch}, (step: {step}) | "
                log += f"train loss: {sum(train_loss_list)/len(train_loss_list):.10f} | "
                log += f"lr: {lr:.10f}"
                # wandb.log({"train_loss": sum(train_loss_list) / len(train_loss_list), "lr": lr})
                print(log)
                train_loss_list = []
                
            # validation loop
            if valid_loader and batch_idx % (args.eval_every_step * args.gradient_accumulation_steps) == 0:
                best_f1 = validate(model, valid_loader, criterion, epoch, step, best_f1, optimizer, scheduler, args)
                model.train()
            
            if early_stop:
                break

            batch_idx += 1

            # Clear CUDA cache if it's a good time
            if batch_idx % (args.eval_every_step * args.gradient_accumulation_steps) == 0:
                torch.cuda.empty_cache()
                gc.collect()  # Trigger Python garbage collection

        if early_stop:
            print("Training completed early due to low validation loss.")
            break
        
        # Save a checkpoint at the end of each epoch if specified in args
        if args.save_every_epoch:
            save_model(model, optimizer, scheduler, epoch, best_f1, args, name=f"{epoch}")

    save_model(model, optimizer, scheduler, step, best_f1, args, name="last")

def validate(model, valid_loader, criterion, epoch, step, best_f1, optimizer, scheduler, args):
    """
    Validation function extracted from training loop
    Returns updated best_f1
    """
    model.eval()
    valid_loss_list = []
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for i, v_batch in enumerate(valid_loader):
            v_seq = v_batch['input'].to(args.device)
            v_mask = v_batch['mask'].to(args.device)
            v_labels = v_batch['target'].to(args.device)

            v_outputs = model(v_seq, attention_mask=v_mask)
            v_logits = v_outputs.logits
            v_loss = criterion(v_logits, v_labels)

            valid_loss_list.append(v_loss.item())
            probabilities = F.softmax(v_logits, dim=1)
            preds = torch.argmax(probabilities, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(v_labels.cpu().numpy())
        
        valid_loss = sum(valid_loss_list) / len(valid_loss_list)

        # compute classification metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=None)
        class_avg_f1 = np.mean(f1)

        log = f"epoch: {epoch}, (step: {step})"
        log += f" | valid loss: {valid_loss:.10f}"
        print(log)

        for i, (prec, rec, f1_) in enumerate(zip(precision, recall, f1)):
            print(f"Class {i} - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1_:.4f}")
        print(f"Overall Validation - Accuracy: {accuracy:.4f}, F1: {class_avg_f1:.4f}")

        if best_f1 < class_avg_f1:
            best_f1 = class_avg_f1
            save_model(model, optimizer, scheduler, step, best_f1, args, name="best_f1")
    
    return best_f1