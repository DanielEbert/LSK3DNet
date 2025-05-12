import os
import random
import time
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
import spconv.pytorch as spconv

from tqdm import tqdm
from utils.metric_util import per_class_iu, fast_hist
from builder import data_builder, loss_builder, optim_builder
from network.largekernel_model import get_model_class
from easydict import EasyDict
import shutil

from utils.load_util import load_yaml
from utils.load_save_util import load_checkpoint_old, load_checkpoint_model_mask
from utils.erk_sparse_core import Masking, CosineDecay

import warnings
warnings.filterwarnings("ignore")

def setup_env_and_configs():
    parser = argparse.ArgumentParser(description='Single GPU Training Script for SemanticKITTI')
    parser.add_argument('--config_path', default='./config/lk-semantickitti_erk_finetune.yaml',
                        help='Path to the configuration YAML file.')
    parser.add_argument('--load_weights', action='store_true', help='Initialize model weights and biases from previous model')
    args = parser.parse_args()

    configs = EasyDict(load_yaml(args.config_path))
    configs.update(vars(args))

    model_save_path = configs['model_params']['model_save_path']
    # Use dirname of the model save path for exp_dir if it's a path, else use current dir's subfolder
    if os.path.dirname(model_save_path):
        # TODO: maybe use original code
        exp_dir = os.path.dirname(model_save_path)
        path_parts = model_save_path.split('/')
        exp_code_backup_root = path_parts[0] if len(path_parts) > 1 else 'experiments'
        exp_code_backup_dir = os.path.join('.', exp_code_backup_root)
    else: # model_save_path is just a filename
        exp_dir = '.'
        exp_code_backup_dir = os.path.join('.', 'experiments', os.path.splitext(model_save_path)[0])

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(exp_code_backup_dir, exist_ok=True)

    files_to_copy = [
        __file__,
        'dataloader/dataset2.py',
        'dataloader/pc_dataset.py',
        'dataloader/utils.py',
        'builder/data_builder.py',
        'network/largekernel_model.py',
        'utils/erk_sparse_core.py',
        args.config_path
    ]
    print(f"Backing up code to: {exp_code_backup_dir}")
    for f_path in files_to_copy:
        if os.path.exists(f_path):
            try:
                shutil.copy(f_path, exp_code_backup_dir)
            except shutil.SameFileError:
                pass # Source and destination are the same
            except Exception as e:
                print(f"Warning: Could not copy {f_path} to {exp_code_backup_dir}: {e}")
        else:
            print(f"Warning: File {f_path} not found for backup.")

    return configs, exp_dir


def reinitialize_model_weights(model, leaky_relu_slope=0.1, check_coverage=True):
    """
    Re-initializes model weights and biases to common good defaults.
    - Kaiming Normal for Conv/Linear/SubMConv3d weights.
    - Zeros for Conv/Linear/SubMConv3d biases (if they exist).
    - Ones for BatchNorm/LayerNorm weights, Zeros for their biases.

    Args:
        model (nn.Module): The model to re-initialize.
        leaky_relu_slope (float): Slope for LeakyReLU for Kaiming init.
        check_coverage (bool): If True, prints a report of modules with parameters
                               that were not explicitly handled by an init rule.
    """
    print("Applying custom re-initialization to model parameters...")

    parameter_holding_module_names = set()
    initialized_module_names = set()

    if check_coverage:
        for name, m_check in model.named_modules():
            # Check if the module m_check itself has parameters (not its children)
            # and if any of them require gradients.
            if any(p.requires_grad for p in m_check.parameters(recurse=False)):
                parameter_holding_module_names.add(name)

    for name, m in model.named_modules():
        was_initialized = False
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, spconv.SubMConv3d)):
            if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, a=leaky_relu_slope, mode='fan_in', nonlinearity='leaky_relu')
                was_initialized = True
            if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                nn.init.constant_(m.bias, 0)
                was_initialized = True

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                nn.init.constant_(m.weight, 1)
                was_initialized = True
            if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                nn.init.constant_(m.bias, 0)
                was_initialized = True

        elif isinstance(m, nn.Embedding):
            if m.weight.requires_grad:
                nn.init.normal_(m.weight, mean=0, std=0.02)
                was_initialized = True

        if was_initialized and name in parameter_holding_module_names:
            initialized_module_names.add(name)

    if check_coverage:
        unhandled_modules = parameter_holding_module_names - initialized_module_names

        if unhandled_modules:
            print("WARNING: The following modules with trainable parameters were NOT explicitly handled by an initialization rule:")
            for name in sorted(list(unhandled_modules)):
                module_ptr = model
                for part in name.split('.'):
                    if part:
                        module_ptr = getattr(module_ptr, part)
                print(f"  - Name: '{name}', Type: {type(module_ptr).__name__}")


def train(configs, exp_dir):
    torch.autograd.set_detect_anomaly(True)

    dataset_config = configs['dataset_params']
    model_config = configs['model_params']
    train_hypers = configs['train_params']
    sparse_config = configs['sparse_params']

    if torch.cuda.is_available():
        pytorch_device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        pytorch_device = torch.device('cpu')
        print("CUDA not available, running on CPU.")

    # Seed everything
    seed = train_hypers.get('seed', random.randint(0, 2**32 - 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if pytorch_device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Using seed: {seed}")

    binary_label_name = {0: 'ground', 1: 'obstacle'}
    unique_label_str = [binary_label_name[x] for x in sorted(binary_label_name.keys())]
    num_classes = dataset_config.get('num_classes', len(unique_label_str))

    my_model = get_model_class(model_config['model_architecture'])(configs)

    pre_weight = None # For sparse mask initialization
    if model_config.get('model_load_path') and os.path.exists(model_config['model_load_path']):
        print(f"Loading pre-trained weights from: {model_config['model_load_path']}")
        try:
            my_model, pre_weight = load_checkpoint_model_mask(model_config['model_load_path'], my_model, pytorch_device)
        except Exception as e_mask:
            print(f"Failed to load checkpoint with mask: {e_mask}. Trying old method.")
            try:
                my_model = load_checkpoint_old(model_config['model_load_path'], my_model)
            except Exception as e_old:
                print(f"Failed to load checkpoint with old method: {e_old}. Starting from scratch.")

    if not configs.load_weights:
        reinitialize_model_weights(my_model, leaky_relu_slope=0.1)

    my_model.to(pytorch_device)

    train_hypers.distributed = False
    train_dataset_loader, val_dataset_loader, _ = data_builder.build(dataset_config, train_hypers)

    configs.train_params.total_steps = train_hypers['max_num_epochs'] * len(train_dataset_loader)
    print(f"Total training steps: {configs.train_params.total_steps}, Batches per epoch: {len(train_dataset_loader)}")
    sparse_config['stop_sparse_epoch'] = sparse_config['stop_sparse_epoch'] * len(train_dataset_loader)

    optimizer, scheduler = optim_builder.build(configs, my_model)
    criterion = loss_builder.criterion(configs, pytorch_device)
    scaler = amp.GradScaler(enabled=train_hypers['amp_enabled'])

    mask_object = None
    if sparse_config['use_sparse']:
        decay = CosineDecay(sparse_config['prune_rate'], int(configs.train_params.total_steps))
        mask_object = Masking(optimizer, scaler, # Pass the main optimizer and scaler
                       spatial_partition=model_config['spatial_group_partition'],
                       prune_mode=sparse_config['prune'], prune_rate_decay=decay,
                       growth_mode=sparse_config['growth'], redistribution_mode=sparse_config['redistribution'],
                       fp16=train_hypers['amp_enabled'], update_frequency=sparse_config['update_frequency'],
                       sparsity=sparse_config['sparsity'], sparse_init=sparse_config['sparse_init'],
                       device=pytorch_device, distributed=False,
                       stop_iter=sparse_config['stop_sparse_epoch'])
        try:
            mask_object.add_module(my_model, pre_weight)
        except Exception as e_add_module: # Catch specific errors if possible
            print(f"Masking.add_module failed (possibly pre_weight issue): {e_add_module}. Adding module without pre_weight.")
            mask_object.add_module(my_model)

    best_val_miou = 0.0
    global_iter = 0
    eval_every_n_steps = train_hypers['eval_every_n_steps']
    # sche_epoch_update is True if scheduler is an epoch-wise scheduler (e.g. StepLR, MultiStepLR)
    # False if it's a step-wise scheduler (e.g. CosineAnnealingLR, OneCycleLR)
    # This info should ideally come from optim_builder or config. Assuming it's step-wise if not specified.
    sche_epoch_update = configs.train_params.get('scheduler_update_per_epoch', True)

    for epoch in range(train_hypers['max_num_epochs']):
        my_model.train()
        epoch_loss_list = []
        
        progress_bar = tqdm(total=len(train_dataset_loader), ncols=100,
                            desc=f'Epoch {epoch}/{train_hypers["max_num_epochs"]-1}')

        for i_iter, train_data_dict in enumerate(train_dataset_loader):
            # --- Evaluation Block ---
            if global_iter > 0 and global_iter % eval_every_n_steps == 0:
                my_model.eval()
                hist_list = []
                total_inference_time = 0.0
                num_val_batches_processed = 0
                print(f"\nRunning validation at global_iter {global_iter}...")
                with torch.no_grad():
                    for i_iter_val, val_data_dict in enumerate(val_dataset_loader):
                        if i_iter_val > 300: # Limit validation batches as in original
                            print(f"Validation limited to first {i_iter_val} batches.")
                            break
                        num_val_batches_processed += 1
                        for key in val_data_dict:
                            if isinstance(val_data_dict[key], torch.Tensor):
                                val_data_dict[key] = val_data_dict[key].to(pytorch_device)
                        
                        if pytorch_device.type == 'cuda': torch.cuda.synchronize()
                        start_time = time.time()
                        val_data_dict = my_model(val_data_dict)
                        if pytorch_device.type == 'cuda': torch.cuda.synchronize()
                        end_time = time.time()
                        total_inference_time += (end_time - start_time)

                        predict_labels = torch.argmax(val_data_dict['logits'], dim=1).cpu().numpy()
                        val_pt_labs = val_data_dict['labels'].cpu().numpy()
                        hist_list.append(fast_hist(predict_labels, val_pt_labs, num_classes))
                
                if num_val_batches_processed > 0:
                    avg_inf_time = total_inference_time / num_val_batches_processed
                    print(f'Avg. val inference time per batch: {avg_inf_time:.4f} s ({num_val_batches_processed} batches)')
                
                if hist_list:
                    iou = per_class_iu(sum(hist_list))
                    print('Validation per class iou: ')
                    for class_name, class_iou_val in zip(unique_label_str, iou):
                        print(f'  {class_name} : {class_iou_val * 100:.2f}%')
                    val_miou = np.nanmean(iou) * 100

                    if best_val_miou < val_miou:
                        best_val_miou = val_miou
                        save_dict = {'checkpoint': my_model.state_dict()}
                        if sparse_config['use_sparse'] and mask_object:
                            save_dict['mask'] = mask_object.masks
                        
                        current_save_path = model_config['model_save_path']
                        torch.save(save_dict, current_save_path, _use_new_zipfile_serialization=False)
                        print(f'Best model saved to: {current_save_path} (mIoU: {best_val_miou:.3f}%)')
                    
                    print(f'Current val mIoU: {val_miou:.3f} | Best val mIoU: {best_val_miou:.3f}')
                else:
                    print("No validation batches processed or hist_list empty.")

                my_model.train()
                if pytorch_device.type == 'cuda': torch.cuda.empty_cache()
                epoch_loss_list = [] # Reset loss list after eval

            # --- Training Step ---
            for key in train_data_dict:
                if isinstance(train_data_dict[key], torch.Tensor):
                    train_data_dict[key] = train_data_dict[key].to(pytorch_device)
            
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(enabled=train_hypers['amp_enabled']):
                train_data_dict = my_model(train_data_dict)
                loss = criterion(train_data_dict)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"NaN or Inf loss detected: {loss.item()}. Skipping step.")
                if global_iter % eval_every_n_steps == 0 and epoch_loss_list:
                    print(f'Epoch {epoch}, Iter {i_iter}, Avg Loss before skip: {np.mean(epoch_loss_list):.3f}')
                progress_bar.update(1)
                global_iter += 1
                continue


            epoch_loss_list.append(loss.item())
            optimizer_step_skipped = False

            if sparse_config['use_sparse'] and mask_object:
                clip_max_norm = 0.1 if train_hypers['amp_enabled'] else 0.25
                if train_hypers['amp_enabled']:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=clip_max_norm)
                    
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    mask_object.step()
                    scaler.update()
                    new_scale = scaler.get_scale()
                    optimizer_step_skipped = (new_scale < prev_scale)
                else: # Sparse, No AMP
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=clip_max_norm)
                    mask_object.step() # This includes optimizer.step() and mask updates
            else: # Not Sparse
                clip_max_norm = 0.25 # Same for AMP and No-AMP in non-sparse original
                if train_hypers['amp_enabled']:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=clip_max_norm)

                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer) # Optimizer step
                    scaler.update()
                    new_scale = scaler.get_scale()
                    optimizer_step_skipped = (new_scale < prev_scale)
                else: # Not Sparse, No AMP
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=clip_max_norm)
                    optimizer.step()
            
            if not sche_epoch_update: # If step-wise scheduler
                if not optimizer_step_skipped:
                    scheduler.step()
                # else:
                #     print(f"LR scheduler step skipped at iter {global_iter} due to optimizer step skip.")

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}',
                                      'lr': f'{optimizer.param_groups[0]["lr"]:.1e}',
                                      'best_val_miou': f'{best_val_miou:.4f}'})
            progress_bar.update(1)
            global_iter += 1

            # Log accumulated loss at interval if not eval iteration
            if global_iter % eval_every_n_steps == 0 and i_iter +1 < len(train_dataset_loader) : # Avoid double log if eval also logs
                if epoch_loss_list:
                    print(f'\nEpoch {epoch}, Iter {i_iter}, Avg Train Loss: {np.mean(epoch_loss_list):.3f}')
                # epoch_loss_list = [] # Reset here if logging mid-epoch, or after eval

        if sche_epoch_update: # If epoch-wise scheduler
            scheduler.step()

        if pytorch_device.type == 'cuda': torch.cuda.empty_cache()
        progress_bar.close()
        if epoch_loss_list: # Log final epoch avg loss
            print(f"End of Epoch {epoch}, Avg Train Loss: {np.mean(epoch_loss_list):.3f}, LR: {optimizer.param_groups[0]['lr']:.1e}")

    print(f"Training finished. Best validation mIoU: {best_val_miou:.3f}")
    print(f"Final model saved at: {model_config['model_save_path']}")


if __name__ == '__main__':
    loaded_configs, experiment_directory = setup_env_and_configs()
    print("Effective configurations:")
    for key, value in loaded_configs.items():
        print(f"  {key}: {value}")
    print(f"Experiment directory for model: {experiment_directory}")
    
    train(loaded_configs, experiment_directory)

