"""Training and validation loops."""

import gc
import numpy as np
import torch
import torch.nn.functional as F

from .log import print_early_stop, print_train_step, print_validation_metrics
from .metric import compute_sequence_metrics
from .utils.model_utils import save_model

def train(model, train_loader, optimizer, criterion, step=0, valid_loader=None, best_metric=-1, best_f1=-1, scheduler=None, args=None):
    train_loss_list = []
    batch_idx = 0
    if best_metric == -1:
        best_metric = np.inf
    if best_f1 == -1:
        best_f1 = -np.inf

    best_epoch = 0

    for epoch in range(1, args.train_epochs +1):
        model.train()
        optimizer.zero_grad()
        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            source_seq = batch['input'].to(args.device)
            attention_mask = batch['mask'].to(args.device)
            labels = batch['target'].to(args.device)

            outputs = model(source_seq, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            loss = torch.mean(loss) / args.gradient_accumulation_steps
            loss.backward()

            should_step = ((batch_idx + 1) % args.gradient_accumulation_steps == 0) or ((batch_idx + 1) == num_batches)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                step += 1

            train_loss_list.append(loss.item() * args.gradient_accumulation_steps)

            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]

            if batch_idx % (args.report_every_step * args.gradient_accumulation_steps) == 0:
                print_train_step(epoch, step, sum(train_loss_list)/len(train_loss_list), lr)
                train_loss_list = []

            if valid_loader and batch_idx % (args.eval_every_step * args.gradient_accumulation_steps) == 0:
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

                    accuracy, precision, recall, f1 = compute_sequence_metrics(all_labels, all_preds)
                    class_avg_f1 = np.mean(f1)

                    print_validation_metrics(epoch, step, valid_loss,
                                             accuracy, precision, recall, f1)

                    if best_metric > valid_loss:
                        best_metric = valid_loss
                        best_epoch = epoch

                    if best_f1 < class_avg_f1:
                        best_f1 = class_avg_f1
                        best_epoch = epoch
                        save_model(model, optimizer, scheduler, step, best_f1, args, name="best_f1")

                    model.train()

            if (batch_idx + 1) % (args.eval_every_step * args.gradient_accumulation_steps) == 0:
                torch.cuda.empty_cache()
                gc.collect()

        if valid_loader and args.early_stop_patience > 0:
            epochs_without_improvement = epoch - best_epoch
            if epochs_without_improvement >= args.early_stop_patience:
                print_early_stop(epoch, args.train_epochs, epochs_without_improvement,
                                 best_metric, best_f1, best_epoch)
                break

        if args.save_every_epoch:
            save_model(model, optimizer, scheduler, epoch, best_metric, args, name=f"{epoch}")
