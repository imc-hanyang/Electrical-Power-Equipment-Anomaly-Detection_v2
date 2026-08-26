import os
import time
import argparse
import datetime
import json
import pandas as pd
from tqdm import tqdm as tqdm
from logger import create_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from utils import *
from dataset import (
    CLAdapterDataset,
    PoissonPerlinSyntheticMaskCLAdapterDataset,
    SyntheticMaskCLAdapterDataset,
)
from build_model import CLAdapter_CLIP_ViT
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score


if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-mode', type=str, required=True)
    parser.add_argument('--finetune-mode', type=str, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--csv-dir', type=str, required=True)
    parser.add_argument('--config-name', type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--init-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nbatch_log', type=int, default=500)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--val_fold', type=int, default=0)
    parser.add_argument('--test_fold', type=int, default=1)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--normal-loss-multiplier', type=float, default=1.0)
    parser.add_argument('--anomaly-loss-multiplier', type=float, default=1.0)
    parser.add_argument('--no-validation', action='store_true')
    parser.add_argument('--selection-metric', type=str, default='acc', choices=['acc', 'f1', 'roc', 'ap', 'loss'])
    parser.add_argument('--backbone-name', type=str, default=None)
    parser.add_argument('--backbone-out-dim', type=int, default=None)
    parser.add_argument('--backbone-num-patch', type=int, default=None)
    parser.add_argument('--finetune-ckpt', type=str, default=None)
    parser.add_argument('--norm', type=str, default='clip', choices=['clip', 'imagenet'])
    parser.add_argument('--pooling-mode', type=str, default='mean', choices=['mean', 'topk', 'attention', 'gated_mil', 'rank_mil'])
    parser.add_argument('--topk-ratio', type=float, default=0.10)
    parser.add_argument('--attention-hidden-dim', type=int, default=192)
    parser.add_argument('--mil-roi-top', type=float, default=0.28)
    parser.add_argument('--mil-roi-bottom', type=float, default=0.72)
    parser.add_argument('--mil-gate-init', type=float, default=0.10)
    parser.add_argument('--ranking-margin', type=float, default=1.0)
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--synthetic-mask-supervision', action='store_true')
    parser.add_argument('--synthetic-mask-probability', type=float, default=0.5)
    parser.add_argument('--synthetic-mask-seed', type=int, default=3107)
    parser.add_argument(
        '--synthetic-mask-mode',
        type=str,
        default='legacy',
        choices=['legacy', 'poisson_perlin'],
    )
    args, _ = parser.parse_known_args()
    config = config_from_name(args.config_name)
    return args, config

def train_epoch(cur_epoch, model, train_loader, optimizer, criterion, scaler, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    bar = tqdm(train_loader)
    steps = 0
    for batch in bar:
        if args.synthetic_mask_supervision:
            images, labels, synthetic_images, patch_targets, synthetic_flags = batch
            synthetic_images = synthetic_images.cuda(non_blocking=True)
            patch_targets = patch_targets.cuda(non_blocking=True)
            synthetic_flags = synthetic_flags.cuda(non_blocking=True)
        else:
            images, labels = batch
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True).long()
        if cur_epoch<=args.warmup_epochs:
            lr = get_warm_up_lr(cur_epoch, steps, args.warmup_epochs, args.init_lr, len(bar))
            set_lr(optimizer, lr)
        else:
            lr = get_train_epoch_lr(cur_epoch, steps, args.epochs, args.init_lr, len(bar))
            set_lr(optimizer, lr)
        with torch.cuda.amp.autocast():
            if args.pooling_mode == 'rank_mil':
                real_batch_size = images.shape[0]
                if args.synthetic_mask_supervision and synthetic_flags.any():
                    combined_images = torch.cat(
                        [images, synthetic_images[synthetic_flags]], dim=0
                    )
                    combined_preds, combined_auxiliary = model(
                        combined_images, return_attention=True
                    )
                    preds = combined_preds[:real_batch_size]
                    auxiliary = {
                        key: (
                            value[:real_batch_size]
                            if torch.is_tensor(value)
                            and value.ndim > 0
                            and value.shape[0] == combined_images.shape[0]
                            else value
                        )
                        for key, value in combined_auxiliary.items()
                    }
                    synthetic_evidence = combined_auxiliary['roi_patch_evidence'][
                        real_batch_size:
                    ]
                    synthetic_roi_mask = combined_auxiliary['roi_mask']
                else:
                    preds, auxiliary = model(images, return_attention=True)
                    synthetic_evidence = None
                    synthetic_roi_mask = None
                fused_loss = criterion(preds, labels)
                global_loss = criterion(auxiliary['global_logits'], labels)
                bag_loss = criterion(auxiliary['bag_logits'], labels)
                components = [fused_loss, global_loss, bag_loss]
                positive = labels == 1
                negative = labels == 0
                if positive.any() and negative.any():
                    positive_evidence = auxiliary['bag_evidence'][positive].mean()
                    negative_evidence = auxiliary['bag_evidence'][negative].mean()
                    ranking_loss = torch.relu(
                        torch.as_tensor(args.ranking_margin, device=images.device)
                        - positive_evidence
                        + negative_evidence
                    )
                    components.append(ranking_loss)
                image_loss = torch.stack(components).mean()
                if args.synthetic_mask_supervision:
                    patch_components = []
                    if negative.any():
                        normal_patch_loss = F.softplus(
                            auxiliary['roi_patch_evidence'][negative]
                        ).mean()
                        patch_components.append(normal_patch_loss)
                    if synthetic_flags.any():
                        synthetic_targets = patch_targets[synthetic_flags][
                            :, synthetic_roi_mask
                        ]
                        positive_patches = synthetic_targets >= 0.5
                        negative_patches = ~positive_patches
                        supervised_parts = []
                        if positive_patches.any():
                            supervised_parts.append(
                                F.softplus(-synthetic_evidence[positive_patches]).mean()
                            )
                        if negative_patches.any():
                            supervised_parts.append(
                                F.softplus(synthetic_evidence[negative_patches]).mean()
                            )
                        if supervised_parts:
                            patch_components.append(torch.stack(supervised_parts).mean())
                    if not patch_components:
                        raise RuntimeError('synthetic patch supervision produced no valid loss')
                    patch_loss = torch.stack(patch_components).mean()
                    loss = 0.5 * image_loss + 0.5 * patch_loss
                else:
                    loss = image_loss
            else:
                preds = model(images)
                loss = criterion(preds, labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        reduced_loss = reduce_tensor(loss.data)
        losses.update(reduced_loss, images.size(0))
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        if args.local_rank==0:
            bar.set_description('lr: %.6f, loss_cur: %.5f, loss_avg: %.5f' % (lr, losses.val, losses.avg))
        if batch_time.count%args.nbatch_log==0 and args.local_rank==0:
            mu = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info('epoch: %d, iter: [%d/%d] || lr: %.6f, memory_used: %.0fMB, loss_cur: %.5f, loss_avg: %.5f, \
                        time_avg: %.3f, time_total: %.3f' % (cur_epoch, batch_time.count, len(train_loader), lr, mu, losses.val, losses.avg, batch_time.avg, batch_time.sum))
        steps += 1
    return losses.avg

def auc_from_logits(preds, labels):
    probs = torch.softmax(preds.float(), dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    if len(np.unique(labels_np)) < 2:
        return None, None
    if probs.shape[1] == 2:
        scores = probs[:, 1]
        return float(roc_auc_score(labels_np, scores)), float(average_precision_score(labels_np, scores))
    return float(roc_auc_score(labels_np, probs, multi_class='ovr')), None


def val_epoch(model, valid_loader, criterion, num_classes=None):
    model.eval()
    bar = tqdm(valid_loader)
    with torch.no_grad():
        preds = []
        labels = []
        for (image, label) in bar:
            image, label = image.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            pred = model(image)
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        loss = criterion(preds, labels)
        acc = accuracy(preds, labels, topk=(1, ))[0]
        prec = precision(preds, labels)
        reca = recall(preds, labels)
        f1 = f1_score(prec, reca)
        roc, ap = auc_from_logits(preds, labels)
        reduced_loss = reduce_tensor(loss)
        reduced_acc = reduce_tensor(acc)
        reduced_prec = reduce_tensor(prec)
        reduced_reca = reduce_tensor(reca)
        reduced_f1 = reduce_tensor(f1)
    return reduced_loss, reduced_acc, reduced_prec, reduced_reca, reduced_f1, roc, ap


def test_epoch(model, test_loader, criterion, num_classes=None):
    model.eval()
    bar = tqdm(test_loader)
    with torch.no_grad():
        preds = []
        labels = []
        for (image, label) in bar:
            image, label = image.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            pred = model(image)
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        loss = criterion(preds, labels)
        acc = accuracy(preds, labels, topk=(1, ))[0]
        prec = precision(preds, labels)
        reca = recall(preds, labels)
        f1 = f1_score(prec, reca)
        roc, ap = auc_from_logits(preds, labels)
    return {
        'loss': float(loss.item()),
        'acc': float(acc.item()),
        'prec': float(prec.item()),
        'reca': float(reca.item()),
        'f1': float(f1.item()),
        'roc': roc,
        'ap': ap,
    }
    
def main(config):
    df = pd.read_csv(args.csv_dir)
    is_malignant = 'malignant' in args.csv_dir
    if args.synthetic_mask_supervision:
        if args.pooling_mode != 'rank_mil':
            raise ValueError('synthetic mask supervision requires --pooling-mode rank_mil')
        dataset_class = (
            PoissonPerlinSyntheticMaskCLAdapterDataset
            if args.synthetic_mask_mode == 'poisson_perlin'
            else SyntheticMaskCLAdapterDataset
        )
        dataset_train = dataset_class(
            is_malignant,
            df,
            args.val_fold,
            args.test_fold,
            'train',
            config.MODEL.img_size,
            config.data_root,
            args.norm,
            synthetic_probability=args.synthetic_mask_probability,
            synthetic_seed=args.synthetic_mask_seed,
            patch_size=16,
            roi_top=args.mil_roi_top,
            roi_bottom=args.mil_roi_bottom,
        )
    else:
        dataset_train = CLAdapterDataset(is_malignant, df, args.val_fold, args.test_fold, 'train', config.MODEL.img_size, config.data_root, args.norm)
    dataset_valid = CLAdapterDataset(is_malignant, df, args.val_fold, args.test_fold, 'valid', config.MODEL.img_size, config.data_root, args.norm)
    has_valid = (not args.no_validation) and len(dataset_valid) > 0
    train_sampler = DistributedSampler(dataset_train)
    valid_sampler = DistributedSampler(dataset_valid) if has_valid else None
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler,
                                               drop_last=False)
    valid_loader = None
    if has_valid:
        valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers,
                                                   shuffle=False, pin_memory=True, sampler=valid_sampler, drop_last=False)
    model = CLAdapter_CLIP_ViT(config)
    if config.MODEL.finetune != None:
        load_ckpt_finetune(config.MODEL.finetune, model, logger=logger, args=args)
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        if args.local_rank == 0:
            logger.info('Backbone frozen for supervised patch-ranking training.')
    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None, find_unused_parameters=True) #find_unused_parameters=True
    optimizer = get_optim_from_config(config, model, 'embed')
    class_weights = torch.tensor(
        [args.normal_loss_multiplier, args.anomaly_loss_multiplier],
        dtype=torch.float32,
        device='cuda',
    )
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights if config.MODEL.num_classes == 2 else None).cuda()
    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    best_score = float('inf') if args.selection_metric == 'loss' else -1
    best_record = {}
    last_train_loss = None
    final_epoch = args.epochs
    args.epochs += 1
    for epoch in range(1, args.epochs):
        if args.local_rank==0:
            logger.info(f"----------[Epoch {epoch}]----------")
        train_sampler.set_epoch(epoch)
        if args.synthetic_mask_supervision:
            dataset_train.set_epoch(epoch)
        train_loss = train_epoch(epoch, model, train_loader, optimizer, criterion, scaler, args)
        last_train_loss = float(train_loss)
        if not has_valid:
            if args.local_rank==0:
                logger.info(f"epoch: {epoch} || loss_train: {train_loss:.5f}")
                logger.info(f'Epoch {epoch} time cost: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')
            continue
        val_loss, acc, val_prec, val_reca, val_f1, val_roc, val_ap = val_epoch(model, valid_loader, criterion, config.MODEL.num_classes)
        if args.local_rank==0:
            logger.info(f"epoch: {epoch} || loss_train: {train_loss:.5f}, loss_val: {val_loss:.5f}, val_acc: {acc:.5f}, val_prec: {val_prec:.5f}, val_reca: {val_reca:.5f}, val_f1: {val_f1:.5f}, val_roc: {val_roc if val_roc is not None else -1:.5f}")
            metric_values = {
                'acc': acc,
                'f1': val_f1,
                'roc': val_roc if val_roc is not None else -1,
                'ap': val_ap if val_ap is not None else -1,
                'loss': val_loss,
            }
            score = metric_values[args.selection_metric]
            improved = score <= best_score if args.selection_metric == 'loss' else score >= best_score
            if improved:
                best_score = score
                save_path = os.path.join(config.MODEL.output_dir, f'{config.MODEL.backbone.model_name}_best.pth')
                logger.info(f"Save best model to {save_path}, with best {args.selection_metric}: {best_score}")
                save_checkpoint(model, save_path)
                best_record = {
                    'best_epoch': epoch,
                    'selection_metric': args.selection_metric,
                    'best_val_score': float(best_score),
                    'val': {
                        'loss': float(val_loss),
                        'acc': float(acc),
                        'prec': float(val_prec),
                        'reca': float(val_reca),
                        'f1': float(val_f1),
                        'roc': val_roc,
                        'ap': val_ap,
                    },
                    'test': None,
                    'test_evaluation': 'forbidden_during_development',
                    'args': vars(args),
                    'output_dir': config.MODEL.output_dir,
                    'model_name': config.MODEL.backbone.model_name,
                    'num_classes': config.MODEL.num_classes,
                }
                with open(os.path.join(config.MODEL.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
                    json.dump(best_record, f, indent=2, ensure_ascii=False)
            logger.info(f'Epoch {epoch} time cost: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')
    if args.local_rank==0:
        if not has_valid:
            save_path = os.path.join(config.MODEL.output_dir, f'{config.MODEL.backbone.model_name}_final.pth')
            logger.info(f"Save final model to {save_path}, no validation split was used.")
            save_checkpoint(model, save_path)
            best_record = {
                'best_epoch': final_epoch,
                'selection_metric': 'final_epoch_no_validation',
                'best_val_score': None,
                'train': {
                    'loss': last_train_loss,
                },
                'val': None,
                'test': None,
                'test_evaluation': 'forbidden_during_development',
                'args': vars(args),
                'output_dir': config.MODEL.output_dir,
                'model_name': config.MODEL.backbone.model_name,
                'num_classes': config.MODEL.num_classes,
            }
            with open(os.path.join(config.MODEL.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(best_record, f, indent=2, ensure_ascii=False)
        logger.info(f"Best val {args.selection_metric}: {best_score if has_valid else 'N/A'}")
        if best_record:
            logger.info(json.dumps(best_record, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    args, config = parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    config.defrost()
    if 'malignant' in args.csv_dir:
        config.MODEL.num_classes = 4
        config.MODEL.output_dir += '/malignant'
    elif 'cotton' in args.csv_dir:
        config.MODEL.num_classes = 80
        config.MODEL.output_dir += '/agricultural_cotton'
    elif 'soyloc' in args.csv_dir:
        config.MODEL.num_classes = 200
        config.MODEL.output_dir += '/agricultural_soyloc'
    elif 'plant' in args.csv_dir:
        config.MODEL.num_classes = 4
        config.MODEL.output_dir += '/agricultural_plant_pathology'
    elif 'WHU-RS19' in args.csv_dir:
        config.MODEL.num_classes = 19
        config.MODEL.output_dir += '/RemoteSensing_plant_WHU-RS19'
    elif 'glass-insulator' in args.csv_dir:
        config.MODEL.num_classes = 2
        config.MODEL.output_dir += '/Industry_DefectSupervised_glass-insulator'
    elif 'lightning-rod-suspension' in args.csv_dir:
        config.MODEL.num_classes = 2
        config.MODEL.output_dir += '/Industry_DefectSupervised_lightning-rod-suspension'
    elif 'polymer-insulator-upper-shackle' in args.csv_dir:
        config.MODEL.num_classes = 2
        config.MODEL.output_dir += '/Industry_DefectSupervised_polymer-insulator-upper-shackle'
    elif 'vari-grip' in args.csv_dir:
        config.MODEL.num_classes = 3
        config.MODEL.output_dir += '/Industry_DefectSupervised_vari-grip'
    elif 'yoke-suspension' in args.csv_dir:
        config.MODEL.num_classes = 2
        config.MODEL.output_dir += '/Industry_DefectSupervised_yoke-suspension'
    elif 'KTH-TIPS2-b' in args.csv_dir:
        config.MODEL.num_classes = 11
        config.MODEL.output_dir += '/Material_KTH-TIPS2-b'
    elif 'tiny-imagenet' in args.csv_dir:
        config.MODEL.num_classes = 200
        config.MODEL.output_dir += '/tiny-imagenet'
    elif 'PACS' in args.csv_dir:
        config.MODEL.num_classes = 7
        config.MODEL.output_dir += '/OOD_PACS'
    else:
        config.MODEL.num_classes = 2
        config.MODEL.output_dir += '/gastric'
    if 'kepco' in args.csv_dir.lower():
        config.MODEL.num_classes = 2
        config.MODEL.output_dir += '/kepco'
    if args.model_mode == 'conv':
        config.MODEL.backbone.out_dim = 1024
        config.MODEL.backbone.num_patch = 49
    elif args.model_mode == 'res_xcep':
        config.MODEL.backbone.out_dim = 2048
        config.MODEL.backbone.num_patch = 49
    else:
        config.MODEL.backbone.out_dim = 768
        config.MODEL.backbone.num_patch = 196
        # config.MODEL.backbone.out_dim = 1024
        # config.MODEL.backbone.num_patch = 576
    if args.backbone_name is not None:
        config.MODEL.backbone.model_name = args.backbone_name
    if args.backbone_out_dim is not None:
        config.MODEL.backbone.out_dim = args.backbone_out_dim
    if args.backbone_num_patch is not None:
        config.MODEL.backbone.num_patch = args.backbone_num_patch
    if args.finetune_ckpt is not None:
        config.MODEL.finetune = args.finetune_ckpt
    config.MODEL.m_mode = args.model_mode
    config.MODEL.f_mode = args.finetune_mode
    config.MODEL.pooling_mode = args.pooling_mode
    config.MODEL.topk_ratio = args.topk_ratio
    config.MODEL.attention_hidden_dim = args.attention_hidden_dim
    config.MODEL.mil_roi_top = args.mil_roi_top
    config.MODEL.mil_roi_bottom = args.mil_roi_bottom
    config.MODEL.mil_gate_init = args.mil_gate_init
    config.MODEL.output_dir += '/'+args.model_mode
    config.MODEL.output_dir += '/'+args.finetune_mode
    if config.MODEL.finetune is not None:
        args.init_lr /= 10
        config.MODEL.output_dir += '/'+config.MODEL.backbone.model_name+'-unfreeze'
    else:
        config.MODEL.output_dir += '/'+config.MODEL.backbone.model_name
    config.MODEL.img_size = args.image_size
    if args.output_dir is not None:
        config.MODEL.output_dir = args.output_dir
    config.init_lr = args.init_lr
    config.batch_size = args.batch_size
    config.Optimizer.name = args.optimizer
    config.local_rank = args.local_rank
    config.world_size = args.world_size
    config.data_root = args.data_root
    config.freeze()
    torch.cuda.set_device(args.local_rank + args.gpu_id)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    set_seed(config.SEED)
    os.makedirs(config.MODEL.output_dir, exist_ok=True)
    logger = create_logger(output_dir=config.MODEL.output_dir, dist_rank=args.local_rank, name=f"{config.MODEL.backbone.model_name}")
    if args.local_rank==0:
        logger.info(config.dump())
    main(config)
