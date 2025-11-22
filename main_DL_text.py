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

# ---------- Config (edit these filepaths to point to your TSVs) ----------
ORIGINAL_PAPER_PATH = "/Users/yatharthnehva/Downloads/emnlp.pdf"  # path to the uploaded EMNLP paper

# For clarity we expect three "task groups" each with train/dev tsvs (you said "all three tsv files")
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

# training options
DEFAULTS = {
    'clip_model': 'openai/clip-vit-base-patch32',
    'batch_size': 32,
    'epochs': 12,
    'lr': 5e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'checkpoints_CLIP_ARCHITECTURE_23112025',
}

# ---------- Utilities: TSV loader ----------

def read_tsv(path: str) -> List[Dict[str, Any]]:
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
                # pad
                parts += [''] * (len(header) - len(parts))
            r = {c: parts[i] for c, i in col_idx.items()}
            # normalize label fields to int or -1
            for lbl in ['t1', 't2', 't3a', 't3b', 't4']:
                if lbl in r and r[lbl] != '':
                    try:
                        r[lbl] = int(r[lbl])
                    except:
                        r[lbl] = -1
                else:
                    r[lbl] = -1
            rows.append(r)
    return rows

# ---------- Dataset ----------
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
            # corrupted image
            return None

    def __getitem__(self, idx):
        r = self.rows[idx]
        text = r.get('text', '')

        # load image if available
        image = self._load_image(r.get('image_path', ''))

        # If image missing or corrupted, create a placeholder image (black)
        if image is None:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # apply transforms if provided (do this before processor so pixel_values are consistent)
        if self.image_transform:
            image = self.image_transform(image)

        # processor will handle both text and images and return pixel_values always
        inputs = self.processor(text=[text], images=image, return_tensors='pt', padding=True)
        # squeeze batch dim
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = {
            't1': torch.tensor(r.get('t1', -1), dtype=torch.long),
            't2': torch.tensor(r.get('t2', -1), dtype=torch.long),
            't3a': torch.tensor(r.get('t3a', -1), dtype=torch.long),
            't3b': torch.tensor(r.get('t3b', -1), dtype=torch.long),
            't4': torch.tensor(r.get('t4', -1), dtype=torch.long),
        }
        return inputs, labels

# Collate function for dataloader
def collate_batch(batch):
    # batch is list of (inputs, labels)
    inputs_list = [b[0] for b in batch]
    labels_list = [b[1] for b in batch]
    # stack manually: each inputs has keys input_ids, attention_mask, pixel_values
    collated = {}
    # text tokens -> pad to same length
    input_ids = [i['input_ids'] for i in inputs_list]
    attention_mask = [i['attention_mask'] for i in inputs_list]
    pixel_values = [i['pixel_values'] for i in inputs_list]
    # pad input_ids
    max_len = max([x.shape[0] for x in input_ids])
    ids = torch.stack([F.pad(x, (0, max_len - x.shape[0]), value=0) if x.shape[0] < max_len else x for x in input_ids])
    masks = torch.stack([F.pad(x, (0, max_len - x.shape[0]), value=0) if x.shape[0] < max_len else x for x in attention_mask])
    collated['input_ids'] = ids
    collated['attention_mask'] = masks
    collated['pixel_values'] = torch.stack(pixel_values)
    # collect labels as tensors
    labels = {}
    for k in ['t1', 't2', 't3a', 't3b', 't4']:
        labels[k] = torch.stack([l[k] for l in labels_list])
    return collated, labels

# ---------- Model components ----------
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
        # img_embs: B x N x D  (N may be 1 depending on backbone)
        # txt_embs: B x L x D
        B = img_embs.size(0)
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        cat = torch.cat([q, img_embs, txt_embs], dim=1)  # B x S x D
        # transformer expects seq_len x batch x dim
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
        self.clip = CLIPModel.from_pretrained(
            clip_model_name,
            use_safetensors=True,
            ignore_mismatched_sizes=True
        )

        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False
        # projection to common dim
        self.img_proj = nn.Linear(self.clip.vision_model.config.hidden_size, embed_dim)
        self.txt_proj = nn.Linear(self.clip.text_model.config.hidden_size, embed_dim)
        # Querying transformer
        self.querying = QueryingTransformer(embed_dim=embed_dim, n_query=16, n_layer=3, n_head=8)
        # heads
        self.head_t1 = nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_t2 = ClassificationHead(embed_dim, 3)
        self.head_t3a = ClassificationHead(embed_dim, 3)
        self.head_t3b = ClassificationHead(embed_dim, 3)
        self.head_t4 = ClassificationHead(embed_dim, 3)

    def forward(self, batch_inputs):
        # batch_inputs: dict with input_ids, attention_mask, pixel_values
        clip_out = self.clip(
            input_ids=batch_inputs['input_ids'],
            attention_mask=batch_inputs['attention_mask'],
            pixel_values=batch_inputs['pixel_values'],
            return_loss=False,
            output_hidden_states=True,
            return_dict=True
        )
        # Get pooled outputs (may adapt depending on CLIP version)
        try:
            img_pool = clip_out.vision_model_output.pooler_output
        except:
            # fallback: use last hidden state mean
            img_pool = clip_out.vision_model_output.last_hidden_state.mean(dim=1)
        try:
            txt_pool = clip_out.text_model_output.pooler_output
        except:
            txt_pool = clip_out.text_model_output.last_hidden_state.mean(dim=1)

        # For querying transformer we want token sequences; create fake token sequences with length=1
        img_tok = self.img_proj(img_pool).unsqueeze(1)  # B x 1 x D
        txt_tok = self.txt_proj(txt_pool).unsqueeze(1)  # B x 1 x D

        mm = self.querying(img_tok, txt_tok)  # B x D

        out = {
            't1_logits': self.head_t1(mm).squeeze(-1),
            't2_logits': self.head_t2(mm),
            't3a_logits': self.head_t3a(mm),
            't3b_logits': self.head_t3b(mm),
            't4_logits': self.head_t4(mm),
            'mm': mm
        }
        return out

# ---------- Loss and training helpers ----------

def compute_masked_losses(outputs, labels, device):
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    losses = {}

    # Task1 BCE
    t1_mask = labels['t1'] >= 0
    if t1_mask.any():
        t1_logits = outputs['t1_logits'][t1_mask]
        t1_target = labels['t1'][t1_mask].float().to(device)
        l1 = F.binary_cross_entropy_with_logits(t1_logits, t1_target)
        losses['t1'] = float(l1.detach().cpu())
        loss = loss + l1

    # Task2 CE
    t2_mask = labels['t2'] >= 0
    if t2_mask.any():
        t2_logits = outputs['t2_logits'][t2_mask]
        t2_target = labels['t2'][t2_mask].to(device)
        l2 = F.cross_entropy(t2_logits, t2_target)
        losses['t2'] = float(l2.detach().cpu())
        loss = loss + l2

    # Task3 (structure & severity)
    t3_mask = labels['t3a'] >= 0
    if t3_mask.any():
        l3a = F.cross_entropy(outputs['t3a_logits'][t3_mask], labels['t3a'][t3_mask].to(device))
        l3b = F.cross_entropy(outputs['t3b_logits'][t3_mask], labels['t3b'][t3_mask].to(device))
        losses['t3a'] = float(l3a.detach().cpu())
        losses['t3b'] = float(l3b.detach().cpu())
        loss = loss + 0.8 * (l3a + l3b)

    # Task4
    t4_mask = labels['t4'] >= 0
    if t4_mask.any():
        l4 = F.cross_entropy(outputs['t4_logits'][t4_mask], labels['t4'][t4_mask].to(device))
        losses['t4'] = float(l4.detach().cpu())
        loss = loss + 0.8 * l4

    return loss, losses


# ---------- Metrics ----------
from collections import defaultdict

def compute_metrics(preds: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]):
    # preds: logits; labels: tensors with -1
    # return simple accuracy/recall/f1 per task (micro/macro omitted for brevity)
    metrics = {}
    # t1
    mask = labels['t1'] >= 0
    if mask.any():
        logits = preds['t1_logits'][mask].detach().cpu()
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).long()
        true = labels['t1'][mask].cpu()
        acc = (pred == true).float().mean().item()
        metrics['t1_acc'] = acc
    # t2
    mask = labels['t2'] >= 0
    if mask.any():
        logits = preds['t2_logits'][mask].detach().cpu()
        pred = logits.argmax(dim=1)
        true = labels['t2'][mask].cpu()
        acc = (pred == true).float().mean().item()
        metrics['t2_acc'] = acc
    # t3a
    mask = labels['t3a'] >= 0
    if mask.any():
        logits = preds['t3a_logits'][mask].detach().cpu()
        pred = logits.argmax(dim=1)
        true = labels['t3a'][mask].cpu()
        metrics['t3a_acc'] = (pred == true).float().mean().item()
    # t3b
    mask = labels['t3b'] >= 0
    if mask.any():
        logits = preds['t3b_logits'][mask].detach().cpu()
        pred = logits.argmax(dim=1)
        true = labels['t3b'][mask].cpu()
        metrics['t3b_acc'] = (pred == true).float().mean().item()
    # t4
    mask = labels['t4'] >= 0
    if mask.any():
        logits = preds['t4_logits'][mask].detach().cpu()
        pred = logits.argmax(dim=1)
        true = labels['t4'][mask].cpu()
        metrics['t4_acc'] = (pred == true).float().mean().item()
    return metrics

# ---------- Training loop ----------

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
            # move to device
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

        # validation
        dev_metrics = evaluate(model, dev_loader, device)
        print(f"Dev metrics: {dev_metrics}")
        # choose a scalar to save best, e.g. t2_acc if present otherwise t1
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
            # collect logits and labels (cpu)
            for k in ['t1_logits', 't2_logits', 't3a_logits', 't3b_logits', 't4_logits']:
                all_preds[k].append(outputs[k].cpu())
            for k in ['t1', 't2', 't3a', 't3b', 't4']:
                all_labels[k].append(labels[k])
    # concat
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

# ---------- Main runner ----------

def build_dataloaders(data_cfg, clip_model_name, batch_size):
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    image_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
    ])
    # merge rows from the three task-specific TSVs into a single dataset object per split
    train_rows = []
    dev_rows = []
    for tg, paths in data_cfg.items():
        train_rows += read_tsv(paths['train_tsv'])
        dev_rows += read_tsv(paths['dev_tsv'])
    # deduplicate by id if necessary
    def unique(rows):
        seen = set(); out = []
        for r in rows:
            id = r.get('id', None)
            key = id if id is not None else str(len(out))
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
