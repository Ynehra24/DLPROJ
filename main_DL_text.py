import os
import sys
import math
import time
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup

ORIGINAL_PAPER_PATH = "/Users/yatharthnehva/Downloads/emnlp.pdf"

DATA_CONFIG = {
    'task1': {
        'train_tsv': '/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_train.tsv',
        'dev_tsv':   '/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_dev.tsv'
    },
    'task2': {
        'train_tsv': '/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv',
        'dev_tsv':   '/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv'
    },
    'task3': {
        'train_tsv': '/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_train.tsv',
        'dev_tsv':   '/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_dev.tsv'
    }
}

DEFAULTS = {
    'clip_model': 'openai/clip-vit-base-patch32',
    'batch_size': 32,
    'epochs': 12,
    'lr': 5e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'checkpoints_CLIP_ARCHITECTURE_23112025',
}

def read_tsv(path: str, task_name: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        print(f"Warning: TSV not found: {path}")
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        col_idx = {c: i for i, c in enumerate(header)}
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < len(header):
                parts += [''] * (len(header) - len(parts))
            r = {c: parts[i] for c, i in col_idx.items()}
            if task_name == "task1":
                try:
                    r["t1"] = int(r.get("label", -1)) if r.get("label", "") != "" else -1
                except:
                    r["t1"] = -1
                r["t2"] = r["t3a"] = r["t3b"] = r["t4"] = -1
            elif task_name == "task2":
                try:
                    r["t2"] = int(r.get("label_text_image", -1)) if r.get("label_text_image", "") != "" else -1
                except:
                    r["t2"] = -1
                r["t1"] = r["t3a"] = r["t3b"] = r["t4"] = -1
            elif task_name == "task3":
                try:
                    r["t4"] = int(r.get("label_text_image", -1)) if r.get("label_text_image", "") != "" else -1
                except:
                    r["t4"] = -1
                r["t1"] = r["t2"] = r["t3a"] = r["t3b"] = -1
            else:
                r["t1"] = r["t2"] = r["t3a"] = r["t3b"] = r["t4"] = -1
            rows.append(r)
    return rows

class CrisisMultimodalDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], processor: CLIPProcessor, image_transform=None):
        self.rows = rows
        self.processor = processor
        self.image_transform = image_transform

    def __len__(self):
        return len(self.rows)

    def _load_image(self, path: str) -> Optional[Image.Image]:
        if path is None or path == '':
            return None
        if not os.path.exists(path):
            return None
        try:
            im = Image.open(path).convert('RGB')
            return im
        except Exception:
            return None

    def __getitem__(self, idx):
        r = self.rows[idx]
        text = r.get('tweet_text', r.get('text', r.get('tweet', '')))
        image = self._load_image(r.get('image', r.get('image_path', '')))
        if image is None:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.image_transform:
            image = self.image_transform(image)
        inputs = self.processor(text=[text], images=image, return_tensors='pt', padding=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = {
            't1': torch.tensor(r.get('t1', -1), dtype=torch.long),
            't2': torch.tensor(r.get('t2', -1), dtype=torch.long),
            't3a': torch.tensor(r.get('t3a', -1), dtype=torch.long),
            't3b': torch.tensor(r.get('t3b', -1), dtype=torch.long),
            't4': torch.tensor(r.get('t4', -1), dtype=torch.long),
        }
        return inputs, labels

def collate_batch(batch):
    inputs_list = [b[0] for b in batch]
    labels_list = [b[1] for b in batch]
    input_ids = [i['input_ids'] for i in inputs_list]
    attention_mask = [i['attention_mask'] for i in inputs_list]
    pixel_values = [i['pixel_values'] for i in inputs_list]
    max_len = max([x.shape[0] for x in input_ids])
    ids = torch.stack([F.pad(x, (0, max_len - x.shape[0]), value=0) if x.shape[0] < max_len else x for x in input_ids])
    masks = torch.stack([F.pad(x, (0, max_len - x.shape[0]), value=0) if x.shape[0] < max_len else x for x in attention_mask])
    collated = {
        'input_ids': ids,
        'attention_mask': masks,
        'pixel_values': torch.stack(pixel_values)
    }
    labels = {}
    for k in ['t1', 't2', 't3a', 't3b', 't4']:
        labels[k] = torch.stack([l[k] for l in labels_list])
    return collated, labels

class QueryingTransformer(nn.Module):
    def __init__(self, embed_dim=512, n_query=16, n_layer=3, n_head=8, dropout=0.1):
        super().__init__()
        self.n_query = n_query
        self.query_tokens = nn.Parameter(torch.randn(n_query, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head,
                                                   dim_feedforward=embed_dim*4, dropout=dropout,
                                                   activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, img_embs, txt_embs):
        B = img_embs.size(0)
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        cat = torch.cat([q, img_embs, txt_embs], dim=1)
        out = self.transformer(cat.permute(1, 0, 2)).permute(1, 0, 2)
        pooled = out[:, :self.n_query, :].mean(dim=1)
        return self.proj(pooled)

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class MultimodalMultiTask(nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32', embed_dim=512, freeze_backbone=True):
        super().__init__()
        try:
            self.clip = CLIPModel.from_pretrained(
                clip_model_name,
                use_safetensors=True,
                ignore_mismatched_sizes=True
            )
        except Exception:
            self.clip = CLIPModel.from_pretrained(clip_model_name, ignore_mismatched_sizes=True)
        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False
        self.img_proj = nn.Linear(self.clip.vision_model.config.hidden_size, embed_dim)
        self.txt_proj = nn.Linear(self.clip.text_model.config.hidden_size, embed_dim)
        self.querying = QueryingTransformer(embed_dim=embed_dim, n_query=16, n_layer=3, n_head=8)
        self.head_t1 = nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_t2 = ClassificationHead(embed_dim, 3)
        self.head_t3a = ClassificationHead(embed_dim, 3)
        self.head_t3b = ClassificationHead(embed_dim, 3)
        self.head_t4 = ClassificationHead(embed_dim, 3)

    def forward(self, batch_inputs):
        clip_out = self.clip(
            input_ids=batch_inputs['input_ids'],
            attention_mask=batch_inputs['attention_mask'],
            pixel_values=batch_inputs['pixel_values'],
            return_loss=False,
            output_hidden_states=True,
            return_dict=True
        )
        try:
            img_pool = clip_out.vision_model_output.pooler_output
        except Exception:
            img_pool = clip_out.vision_model_output.last_hidden_state.mean(dim=1)
        try:
            txt_pool = clip_out.text_model_output.pooler_output
        except Exception:
            txt_pool = clip_out.text_model_output.last_hidden_state.mean(dim=1)
        img_tok = self.img_proj(img_pool).unsqueeze(1)
        txt_tok = self.txt_proj(txt_pool).unsqueeze(1)
        mm = self.querying(img_tok, txt_tok)
        out = {
            't1_logits': self.head_t1(mm).squeeze(-1),
            't2_logits': self.head_t2(mm),
            't3a_logits': self.head_t3a(mm),
            't3b_logits': self.head_t3b(mm),
            't4_logits': self.head_t4(mm),
            'mm': mm
        }
        return out

def compute_masked_losses(outputs, labels, device):
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    losses = {}
    t1_mask = labels['t1'] >= 0
    if t1_mask.any():
        t1_logits = outputs['t1_logits'][t1_mask]
        t1_target = labels['t1'][t1_mask].float().to(device)
        l1 = F.binary_cross_entropy_with_logits(t1_logits, t1_target)
        losses['t1'] = float(l1.detach().cpu())
        loss = loss + l1
    t2_mask = labels['t2'] >= 0
    if t2_mask.any():
        t2_logits = outputs['t2_logits'][t2_mask]
        t2_target = labels['t2'][t2_mask].to(device)
        l2 = F.cross_entropy(t2_logits, t2_target)
        losses['t2'] = float(l2.detach().cpu())
        loss = loss + l2
    t3_mask = labels['t3a'] >= 0
    if t3_mask.any():
        l3a = F.cross_entropy(outputs['t3a_logits'][t3_mask], labels['t3a'][t3_mask].to(device))
        l3b = F.cross_entropy(outputs['t3b_logits'][t3_mask], labels['t3b'][t3_mask].to(device))
        losses['t3a'] = float(l3a.detach().cpu()); losses['t3b'] = float(l3b.detach().cpu())
        loss = loss + 0.8 * (l3a + l3b)
    t4_mask = labels['t4'] >= 0
    if t4_mask.any():
        l4 = F.cross_entropy(outputs['t4_logits'][t4_mask], labels['t4'][t4_mask].to(device))
        losses['t4'] = float(l4.detach().cpu())
        loss = loss + 0.8 * l4
    return loss, losses

from collections import defaultdict

def compute_metrics(preds: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]):
    metrics = {}
    mask = labels['t1'] >= 0
    if mask.any():
        logits = preds['t1_logits'][mask].detach().cpu()
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).long()
        true = labels['t1'][mask].cpu()
        metrics['t1_acc'] = (pred == true).float().mean().item()
    mask = labels['t2'] >= 0
    if mask.any():
        logits = preds['t2_logits'][mask].detach().cpu()
        pred = logits.argmax(dim=1)
        true = labels['t2'][mask].cpu()
        metrics['t2_acc'] = (pred == true).float().mean().item()
    mask = labels['t3a'] >= 0
    if mask.any():
        logits = preds['t3a_logits'][mask].detach().cpu()
        pred = logits.argmax(dim=1)
        true = labels['t3a'][mask].cpu()
        metrics['t3a_acc'] = (pred == true).float().mean().item()
    mask = labels['t3b'] >= 0
    if mask.any():
        logits = preds['t3b_logits'][mask].detach().cpu()
        pred = logits.argmax(dim=1)
        true = labels['t3b'][mask].cpu()
        metrics['t3b_acc'] = (pred == true).float().mean().item()
    mask = labels['t4'] >= 0
    if mask.any():
        logits = preds['t4_logits'][mask].detach().cpu()
        pred = logits.argmax(dim=1)
        true = labels['t4'][mask].cpu()
        metrics['t4_acc'] = (pred == true).float().mean().item()
    return metrics

def train(model: nn.Module, train_loader: DataLoader, dev_loader: DataLoader, cfg: Dict[str, Any]):
    device = cfg['device']
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=1e-2)
    total_steps = len(train_loader) * cfg['epochs']
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=200, num_training_steps=total_steps)
    best_dev = -1.0
    os.makedirs(cfg['save_dir'], exist_ok=True)
    for epoch in range(cfg['epochs']):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = {k: v.to(device) for k, v in labels.items()}
            outputs = model(inputs)
            loss, loss_dict = compute_masked_losses(outputs, labels, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch} step {i}/{len(train_loader)} loss={loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        t1 = time.time()
        print(f"Epoch {epoch} finished. avg_loss={avg_loss:.4f}. time={t1-t0:.1f}s")
        dev_metrics = evaluate(model, dev_loader, device)
        print(f"Dev metrics: {dev_metrics}")
        score = dev_metrics.get('t2_acc', dev_metrics.get('t1_acc', 0.0))
        if score > best_dev:
            best_dev = score
            ckpt = os.path.join(cfg['save_dir'], f"best_epoch_{epoch}.pt")
            torch.save({'model_state': model.state_dict(), 'cfg': cfg}, ckpt)
            print(f"Saved best model to {ckpt}")

def evaluate(model: nn.Module, loader: DataLoader, device):
    model.eval()
    all_preds = defaultdict(list)
    all_labels = defaultdict(list)
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Eval"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(inputs)
            for k in ['t1_logits', 't2_logits', 't3a_logits', 't3b_logits', 't4_logits']:
                all_preds[k].append(outputs[k].cpu())
            for k in ['t1', 't2', 't3a', 't3b', 't4']:
                all_labels[k].append(labels[k])
    preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
    labels = {k: torch.cat(v, dim=0) for k, v in all_labels.items()}
    metrics = compute_metrics({
        't1_logits': preds['t1_logits'],
        't2_logits': preds['t2_logits'],
        't3a_logits': preds['t3a_logits'],
        't3b_logits': preds['t3b_logits'],
        't4_logits': preds['t4_logits'],
    }, labels)
    return metrics

def build_dataloaders(data_cfg, clip_model_name, batch_size):
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    image_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
    ])
    train_rows = []
    dev_rows = []
    for tg, paths in data_cfg.items():
        train_rows += read_tsv(paths['train_tsv'], tg)
        dev_rows += read_tsv(paths['dev_tsv'], tg)
    def unique(rows):
        seen = set(); out = []
        for r in rows:
            idv = r.get('tweet_id', r.get('tweetid', r.get('id', None)))
            key = idv if idv is not None else str(len(out))
            if key in seen:
                continue
            seen.add(key); out.append(r)
        return out
    train_rows = unique(train_rows)
    dev_rows = unique(dev_rows)
    train_ds = CrisisMultimodalDataset(train_rows, processor, image_transform=image_transforms)
    dev_ds = CrisisMultimodalDataset(dev_rows, processor, image_transform=image_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_loader, dev_loader

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model', default=DEFAULTS['clip_model'])
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--device', default=DEFAULTS['device'])
    parser.add_argument('--save_dir', default=DEFAULTS['save_dir'])
    args, _ = parser.parse_known_args(argv)
    cfg = {
        'clip_model': args.clip_model,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'device': args.device,
        'save_dir': args.save_dir,
    }
    train_loader, dev_loader = build_dataloaders(DATA_CONFIG, cfg['clip_model'], cfg['batch_size'])
    print(f"Train batches: {len(train_loader)}, Dev batches: {len(dev_loader)}")
    model = MultimodalMultiTask(clip_model_name=cfg['clip_model'], embed_dim=512, freeze_backbone=True)
    train(model, train_loader, dev_loader, cfg)

if __name__ == '__main__':
    main()
