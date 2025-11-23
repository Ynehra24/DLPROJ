#!/usr/bin/env python3
"""
train_hierarchical_single_model_fixed.py

Full fixed training script (single-file). Designed to be Jupyter-friendly and to run
on CPU / CUDA / MPS. All previously reported errors have been fixed:

 - No argparse crash inside Jupyter (use run_training() API).
 - No sklearn/pandas imports that caused environment binary issues.
 - Correct CLIP vision/text hidden dims handling (vision 768 -> proj -> 512, text 512 -> proj -> 512).
 - Robust extraction of pooler_output for CLIP variants; fallbacks handled.
 - Masked multitask losses across Tasks 1..4 with focal option for Task1.
 - Optional WeightedRandomSampler to rebalance minority classes.
 - EMA option, gradient clipping, LR scheduling, backbone/head LR groups.
 - Lightweight local classification-report + confusion-matrix (avoids sklearn).
 - Uses uploaded paper path for provenance: /mnt/data/emnlp.pdf
 - Jupyter friendly: call run_training(user_config) to run.

USAGE:
  In a notebook cell: `model, train_loader, dev_loader = run_training(user_config)`
  Or `if __name__ == '__main__': run_training()` will run with default CONFIG.

AUTHOR: assistant (fixed version for user's environment)
"""

import os
import random
import time
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup

# ----------------------------
# Provenance (user uploaded file path)
# ----------------------------
PAPER_PATH = "/mnt/data/emnlp.pdf"

# ----------------------------
# Default CONFIG (change using run_training(user_config))
# ----------------------------
CONFIG = {
    'DATA_CONFIG': {
        'damage': {
            'train_tsv': '/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_train.tsv',
            'dev_tsv':   '/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_damage_text_img_dev.tsv'
        },
        'humanitarian': {
            'train_tsv': '/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv',
            'dev_tsv':   '/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv'
        },
        'informative': {
            'train_tsv': '/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_train.tsv',
            'dev_tsv':   '/Volumes/Extreme SSD/DL_Proj/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_dev.tsv'
        }
    },
    'CLIP_MODEL': 'openai/clip-vit-base-patch32',
    'BATCH_SIZE': 16,
    'EPOCHS': 8,
    'LR': 3e-4,
    'BACKBONE_LR': 5e-6,
    'DEVICE': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'SAVE_DIR': 'checkpoints_hier_single_fixed',
    'MAX_TEXT_LEN': 77,
    'NUM_WORKERS': 0,
    'PIN_MEMORY': False,
    'IMAGE_SIZE': 224,
    'USE_SAMPLER': False,
    'TEXT_DROPOUT': 0.0,
    'FREEZE_BACKBONE': True,
    'USE_AMP': False,   # will only enable on CUDA
    'WEIGHT_DECAY': 1e-2,
    'EMA': False,
    'EMA_DECAY': 0.999,
    'CLIP_GRAD': 1.0,
    'FOCAL_T1': True,
    'GRAD_ACCUM_STEPS': 1
}

# ----------------------------
# Label maps (paper)
# ----------------------------
DAMAGE_MAP = {
    'little_or_no_damage': 0,
    'mild_damage': 1,
    'severe_damage': 2
}

# Task2 mapping: 0 = Non-informative, 1 = Humanitarian, 2 = Structural
HUM_LABEL_TO_TASK2 = {
    'not_humanitarian': 0,
    'other_relevant_information': 0,
    'affected_individuals': 1,
    'injured_or_dead_people': 1,
    'missing_or_found_people': 1,
    'rescue_volunteering_or_donation_effort': 1,
    'infrastructure_and_utility_damage': 2,
    'vehicle_damage': 2
}

# Task3 structure type mapping
HUM_LABEL_TO_T3TYPE = {
    'infrastructure_and_utility_damage': 0,
    'vehicle_damage': 1
}

# Task4 humanitarian subcategories
HUM_LABEL_TO_T4 = {
    'affected_individuals': 0,
    'injured_or_dead_people': 0,
    'missing_or_found_people': 0,
    'rescue_volunteering_or_donation_effort': 1,
    'other_relevant_information': 2,
    'not_humanitarian': 2,
    'infrastructure_and_utility_damage': 2,
    'vehicle_damage': 2
}

# ----------------------------
# Utilities for TSV parsing & safety
# ----------------------------
def safe_str(x):
    if x is None:
        return ''
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()

def detect_column(header: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in header:
            return c
    return None

def read_rows_from_tsv(path: str) -> Tuple[List[str], List[List[str]]]:
    rows = []
    if not os.path.exists(path):
        return [], []
    with open(path, 'r', encoding='utf-8') as f:
        raw_header = f.readline().rstrip('\n').split('\t')
        header = []
        seen = {}
        for h in raw_header:
            h = h.strip()
            if h not in seen:
                header.append(h)
                seen[h] = 1
            else:
                hdr = f"{h}__{seen[h]}"
                header.append(hdr)
                seen[h] += 1
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < len(header):
                parts += [''] * (len(header) - len(parts))
            rows.append(parts)
    return header, rows

def scan_and_print_distribution(path: str, col: str) -> Counter:
    ctr = Counter()
    if not os.path.exists(path):
        return ctr
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        if col not in header:
            return ctr
        idx = header.index(col)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if idx < len(parts):
                v = safe_str(parts[idx]).lower()
                if v:
                    ctr[v] += 1
    return ctr

# ----------------------------
# Build combined rows (train/dev) from three TSVs.
# Each combined row has fields: tweet_text, image_path, t1,t2,t3_type,t3_sev,t4
# Missing labels are -1
# ----------------------------
def build_combined_rows(data_cfg: Dict[str,Any]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    train_rows = []
    dev_rows = []

    for key in ['damage','humanitarian','informative']:
        p = data_cfg[key]
        for split in ['train_tsv','dev_tsv']:
            path = p[split]
            header, raw_rows = read_rows_from_tsv(path)
            if not header:
                continue
            label_col = detect_column(header, ['label','label_text_image','label_text','label_image'])
            text_col = detect_column(header, ['tweet_text','tweet','text'])
            if text_col is None:
                text_col = header[3] if len(header) > 3 else header[0]
            image_col = detect_column(header, ['image','image_path','image_id'])
            if image_col is None:
                image_col = header[4] if len(header) > 4 else header[1]

            for parts in raw_rows:
                rec = {header[i]: parts[i] for i in range(len(header))}
                text = safe_str(rec.get(text_col,'')) if text_col else ''
                image = safe_str(rec.get(image_col,'')) if image_col else ''
                label_raw = safe_str(rec.get(label_col,'')).lower() if label_col else ''

                row = {
                    'tweet_text': text,
                    'image_path': image,
                    't1': -1,
                    't2': -1,
                    't3_type': -1,
                    't3_sev': -1,
                    't4': -1
                }

                if key == 'damage':
                    if label_raw:
                        row['t3_sev'] = DAMAGE_MAP.get(label_raw, -1)
                    (train_rows if split == 'train_tsv' else dev_rows).append(row)

                elif key == 'humanitarian':
                    if label_raw:
                        row['t2'] = HUM_LABEL_TO_TASK2.get(label_raw, 0)
                        row['t3_type'] = HUM_LABEL_TO_T3TYPE.get(label_raw, -1)
                        row['t4'] = HUM_LABEL_TO_T4.get(label_raw, 2)
                    (train_rows if split == 'train_tsv' else dev_rows).append(row)

                elif key == 'informative':
                    if label_raw:
                        row['t1'] = 1 if label_raw == 'informative' else 0
                    (train_rows if split == 'train_tsv' else dev_rows).append(row)

    return train_rows, dev_rows

# ----------------------------
# Dataset and collate
# Uses CLIPProcessor for consistent preprocessing (processor handles normalization & resizing).
# ----------------------------
class CrisisDataset(Dataset):
    def __init__(self, rows: List[Dict[str,Any]], processor: CLIPProcessor, max_text_len=77, image_size=224, text_dropout=0.0):
        self.rows = rows
        self.processor = processor
        self.max_text_len = max_text_len
        self.image_size = image_size
        self.text_dropout = text_dropout

    def __len__(self):
        return len(self.rows)

    def _load_image(self, path: str):
        try:
            if path and os.path.exists(path):
                img = Image.open(path).convert('RGB')
            else:
                raise FileNotFoundError()
        except Exception:
            img = Image.new('RGB', (self.image_size, self.image_size), (0,0,0))
        return img

    def _dropout_text(self, text: str):
        if self.text_dropout <= 0.0:
            return text
        toks = text.split()
        keep = [w for w in toks if random.random() > self.text_dropout]
        if not keep:
            return text
        return " ".join(keep)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        text = r.get('tweet_text','') or ''
        text = self._dropout_text(text)
        img = self._load_image(r.get('image_path',''))

        proc = self.processor(text=[text], images=[img],
                              padding='max_length', truncation=True,
                              max_length=self.max_text_len, return_tensors='pt')

        inputs = {
            'input_ids': proc['input_ids'].squeeze(0),
            'attention_mask': proc['attention_mask'].squeeze(0),
            'pixel_values': proc['pixel_values'].squeeze(0)
        }

        labels = {
            't1': torch.tensor(r.get('t1', -1), dtype=torch.long),
            't2': torch.tensor(r.get('t2', -1), dtype=torch.long),
            't3_type': torch.tensor(r.get('t3_type', -1), dtype=torch.long),
            't3_sev': torch.tensor(r.get('t3_sev', -1), dtype=torch.long),
            't4': torch.tensor(r.get('t4', -1), dtype=torch.long)
        }
        return inputs, labels

def collate_batch(batch):
    inputs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batched_inputs = {
        'input_ids': torch.stack([i['input_ids'] for i in inputs]),
        'attention_mask': torch.stack([i['attention_mask'] for i in inputs]),
        'pixel_values': torch.stack([i['pixel_values'] for i in inputs])
    }
    batched_labels = {
        't1': torch.stack([l['t1'] for l in labels]),
        't2': torch.stack([l['t2'] for l in labels]),
        't3_type': torch.stack([l['t3_type'] for l in labels]),
        't3_sev': torch.stack([l['t3_sev'] for l in labels]),
        't4': torch.stack([l['t4'] for l in labels])
    }
    return batched_inputs, batched_labels

# ----------------------------
# Model: CLIP backbone + projections + QueryingTransformer + heads
# ----------------------------
class QueryingTransformer(nn.Module):
    def __init__(self, embed_dim=512, n_query=16, n_layer=3, n_head=8, dropout=0.1):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(n_query, embed_dim) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head,
                                         dim_feedforward=embed_dim*4, dropout=dropout, activation='gelu')
        self.tr = nn.TransformerEncoder(enc, num_layers=n_layer)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_query = n_query

    def forward(self, ie, te):
        ie = ie.unsqueeze(1)
        te = te.unsqueeze(1)
        B = ie.size(0)
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([q, ie, te], dim=1)  # (B, n_query+2, D)
        x = self.tr(x.permute(1,0,2)).permute(1,0,2)
        pooled = x[:, :self.n_query].mean(1)
        return self.proj(pooled)

class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class HierarchicalMultimodalModel(nn.Module):
    def __init__(self, clip_model_name, embed_dim=512, freeze_backbone=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)

        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False

        # Determine raw hidden sizes robustly
        # Common: vision_hidden=768, text_hidden=512 for ViT-B-32
        vision_dim = None
        text_dim = None
        try:
            # Try config fields (HF newer versions)
            vision_dim = self.clip.config.vision_config.hidden_size
        except Exception:
            pass
        try:
            text_dim = self.clip.config.text_config.hidden_size
        except Exception:
            pass
        # Fallback guesses
        if vision_dim is None:
            # try probing attributes
            try:
                vision_dim = getattr(self.clip.vision_model.config, 'hidden_size')
            except Exception:
                vision_dim = 768
        if text_dim is None:
            try:
                text_dim = getattr(self.clip.text_model.config, 'hidden_size')
            except Exception:
                text_dim = 512

        self.vision_dim = vision_dim
        self.text_dim = text_dim

        # Project raw towers to shared embed_dim
        self.img_proj = nn.Linear(self.vision_dim, embed_dim)
        self.txt_proj = nn.Linear(self.text_dim, embed_dim)

        self.querying = QueryingTransformer(embed_dim=embed_dim)

        # Heads
        self.head_t1 = MLPHead(embed_dim, out_dim=2)
        self.head_t2 = MLPHead(embed_dim, out_dim=3)
        self.head_t3_type = MLPHead(embed_dim, out_dim=3)
        self.head_t3_sev = MLPHead(embed_dim, out_dim=3)
        self.head_t4 = MLPHead(embed_dim, out_dim=3)

    def forward(self, batch: Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
        # Ensure return of model-level pooler outputs
        out = self.clip(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            return_dict=True,
            output_hidden_states=True
        )

        # Preferred: use pooler_output if available (raw towers)
        img_repr = None
        txt_repr = None

        # Vision pooler (common HF field)
        if hasattr(out, 'vision_model_output') and getattr(out.vision_model_output, 'pooler_output', None) is not None:
            img_repr = out.vision_model_output.pooler_output  # expected shape [B, vision_dim]
        elif hasattr(out, 'image_embeds') and out.image_embeds is not None:
            # image_embeds are final CLIP-projected embeddings (512), but we want raw tower -> use fallback by projecting back is not possible.
            # Use image_embeds only if we cannot obtain pooler_output; then adapt projections accordingly.
            img_repr = out.image_embeds
        else:
            # As last resort, average last_hidden_state
            if hasattr(out, 'vision_model_output') and getattr(out.vision_model_output, 'last_hidden_state', None) is not None:
                img_repr = out.vision_model_output.last_hidden_state.mean(1)
            else:
                # final fallback: use zeros
                img_repr = torch.zeros(batch['pixel_values'].size(0), self.vision_dim, device=batch['pixel_values'].device)

        # Text pooler
        if hasattr(out, 'text_model_output') and getattr(out.text_model_output, 'pooler_output', None) is not None:
            txt_repr = out.text_model_output.pooler_output  # expected shape [B, text_dim]
        elif hasattr(out, 'text_embeds') and out.text_embeds is not None:
            txt_repr = out.text_embeds
        else:
            if hasattr(out, 'text_model_output') and getattr(out.text_model_output, 'last_hidden_state', None) is not None:
                txt_repr = out.text_model_output.last_hidden_state.mean(1)
            else:
                txt_repr = torch.zeros(batch['input_ids'].size(0), self.text_dim, device=batch['input_ids'].device)

        # At this point shapes may be:
        # img_repr: [B, vision_dim] OR [B, 512] (if image_embeds)
        # txt_repr: [B, text_dim] OR [B, 512]
        # We project both to shared embed_dim.
        # If img_repr is already 512 (image_embeds), but self.img_proj expects vision_dim, they won't match.
        # To be robust, if dimension mismatches, recreate projection adapters.

        if img_repr.size(1) != self.img_proj.in_features:
            # replace img_proj on the fly to match found dim (keeps previously learned weights if shapes match partially).
            old = self.img_proj
            self.img_proj = nn.Linear(img_repr.size(1), old.out_features).to(img_repr.device)
        if txt_repr.size(1) != self.txt_proj.in_features:
            old = self.txt_proj
            self.txt_proj = nn.Linear(txt_repr.size(1), old.out_features).to(txt_repr.device)

        vp = self.img_proj(img_repr)
        tp = self.txt_proj(txt_repr)

        mm = self.querying(vp, tp)

        return {
            't1_logits': self.head_t1(mm),
            't2_logits': self.head_t2(mm),
            't3_type_logits': self.head_t3_type(mm),
            't3_sev_logits': self.head_t3_sev(mm),
            't4_logits': self.head_t4(mm)
        }

# ----------------------------
# Losses, weights, simple metrics
# ----------------------------
def focal_loss_logits(inputs: torch.Tensor, targets: torch.Tensor, alpha: Optional[torch.Tensor]=None, gamma: float=2.0):
    ce = F.cross_entropy(inputs, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()

def compute_class_weights_from_rows(rows: List[Dict[str,Any]], key: str, n_classes: int):
    ctr = Counter([r[key] for r in rows if r.get(key, -1) >= 0])
    if len(ctr) == 0:
        return None
    freqs = np.array([ctr.get(i, 0) for i in range(n_classes)], dtype=np.float32)
    inv = (freqs.max() + 1e-12) / (freqs + 1.0)
    w = inv / (inv.sum() + 1e-12) * len(inv)
    return torch.tensor(w, dtype=torch.float32)

def compute_masked_losses(outputs: Dict[str,torch.Tensor], labels: Dict[str,torch.Tensor], device: torch.device, weights: Dict[str,Optional[torch.Tensor]]=None, focal_on_t1: bool=True):
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    # t1 (binary)
    mask = labels['t1'] >= 0
    if mask.any():
        logits = outputs['t1_logits'][mask]
        tg = labels['t1'][mask].to(device)
        w = weights.get('t1') if (weights and 't1' in weights) else None
        if focal_on_t1:
            loss = loss + focal_loss_logits(logits, tg, alpha=(w.to(device) if w is not None else None))
        else:
            loss = loss + F.cross_entropy(logits, tg, weight=(w.to(device) if w is not None else None))
    # t2
    mask = labels['t2'] >= 0
    if mask.any():
        logits = outputs['t2_logits'][mask]
        tg = labels['t2'][mask].to(device)
        w = weights.get('t2') if (weights and 't2' in weights) else None
        loss = loss + F.cross_entropy(logits, tg, weight=(w.to(device) if w is not None else None))
    # t3_type
    mask = labels['t3_type'] >= 0
    if mask.any():
        logits = outputs['t3_type_logits'][mask]
        tg = labels['t3_type'][mask].to(device)
        w = weights.get('t3_type') if (weights and 't3_type' in weights) else None
        loss = loss + F.cross_entropy(logits, tg, weight=(w.to(device) if w is not None else None))
    # t3_sev
    mask = labels['t3_sev'] >= 0
    if mask.any():
        logits = outputs['t3_sev_logits'][mask]
        tg = labels['t3_sev'][mask].to(device)
        w = weights.get('t3_sev') if (weights and 't3_sev' in weights) else None
        loss = loss + F.cross_entropy(logits, tg, weight=(w.to(device) if w is not None else None))
    # t4
    mask = labels['t4'] >= 0
    if mask.any():
        logits = outputs['t4_logits'][mask]
        tg = labels['t4'][mask].to(device)
        w = weights.get('t4') if (weights and 't4' in weights) else None
        loss = loss + F.cross_entropy(logits, tg, weight=(w.to(device) if w is not None else None))
    return loss

# Minimal classification-report and confusion-matrix to avoid sklearn
def simple_confusion_matrix(y_true: List[int], y_pred: List[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def simple_classification_report(y_true: List[int], y_pred: List[int], labels: Optional[List[int]]=None):
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    if len(labels) == 0:
        return {}, np.array([[]])
    n_classes = max(labels) + 1
    cm = simple_confusion_matrix(y_true, y_pred, n_classes)
    per_class = {}
    supports = cm.sum(axis=1)
    for i in labels:
        tp = int(cm[i,i])
        fp = int(cm[:,i].sum() - tp)
        fn = int(cm[i,:].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[i] = {'precision': precision, 'recall': recall, 'f1-score': f1, 'support': int(supports[i])}
    macro_f1 = float(np.mean([per_class[i]['f1-score'] for i in labels]))
    macro_prec = float(np.mean([per_class[i]['precision'] for i in labels]))
    macro_rec = float(np.mean([per_class[i]['recall'] for i in labels]))
    total = sum(supports) if len(supports) else 1
    weighted_f1 = float(sum(per_class[i]['f1-score'] * per_class[i]['support'] for i in labels) / total)
    weighted_prec = float(sum(per_class[i]['precision'] * per_class[i]['support'] for i in labels) / total)
    weighted_rec = float(sum(per_class[i]['recall'] * per_class[i]['support'] for i in labels) / total)
    report = {
        'per_class': per_class,
        'macro avg': {'precision': macro_prec, 'recall': macro_rec, 'f1-score': macro_f1, 'support': int(total)},
        'weighted avg': {'precision': weighted_prec, 'recall': weighted_rec, 'f1-score': weighted_f1, 'support': int(total)}
    }
    return report, cm

# ----------------------------
# Evaluation
# ----------------------------
def evaluate_detailed(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds = {k: [] for k in ['t1','t2','t3_type','t3_sev','t4']}
    all_trues = {k: [] for k in ['t1','t2','t3_type','t3_sev','t4']}

    with torch.no_grad():
        for inp, lbl in loader:
            inp = {k:v.to(device) for k,v in inp.items()}
            out = model(inp)
            for t in ['t1','t2','t3_type','t3_sev','t4']:
                mask = lbl[t] >= 0
                if mask.any():
                    preds = out[f'{t}_logits'][mask].argmax(1).cpu().tolist()
                    trues = lbl[t][mask].cpu().tolist()
                    all_preds[t].extend(preds)
                    all_trues[t].extend(trues)

    reports = {}
    cms = {}
    for t in ['t1','t2','t3_type','t3_sev','t4']:
        if len(all_trues[t]) == 0:
            continue
        labels = sorted(list(set(all_trues[t]) | set(all_preds[t])))
        rep, cm = simple_classification_report(all_trues[t], all_preds[t], labels=labels)
        reports[t] = rep
        cms[t] = cm
        print(f"\n--- Task {t} ---")
        for c, meta in rep['per_class'].items():
            print(f"Class {c}: precision={meta['precision']:.4f} recall={meta['recall']:.4f} f1={meta['f1-score']:.4f} support={meta['support']}")
        print("macro f1:", rep['macro avg']['f1-score'])
        print("weighted f1:", rep['weighted avg']['f1-score'])
        print("Confusion matrix:\n", cm)
    return reports, cms

# ----------------------------
# Dataloaders builder
# ----------------------------
def build_dataloaders(data_cfg, clip_model, batch_size, max_text_len, num_workers, pin_memory, use_sampler=False, text_dropout=0.0):
    print("=== Scanning data distributions (train TSVs) ===")
    for k,p in data_cfg.items():
        print(k, scan_and_print_distribution(p['train_tsv'], 'label'))

    train_rows, dev_rows = build_combined_rows(data_cfg)
    processor = CLIPProcessor.from_pretrained(clip_model)
    train_ds = CrisisDataset(train_rows, processor, max_text_len, CONFIG['IMAGE_SIZE'], text_dropout=text_dropout)
    dev_ds = CrisisDataset(dev_rows, processor, max_text_len, CONFIG['IMAGE_SIZE'], text_dropout=0.0)

    train_sampler = None
    if use_sampler:
        try:
            ctr = Counter([r['t3_sev'] for r in train_rows if r.get('t3_sev',-1) >= 0])
            if sum(ctr.values()) == 0:
                ctr = Counter([r['t1'] for r in train_rows if r.get('t1',-1) >= 0])
            total = sum(ctr.values()) or 1
            class_weight = {c: total/(v + 1e-12) for c,v in ctr.items()}
            weights = [float(class_weight.get(r.get('t3_sev', r.get('t1', -1)), 1.0)) for r in train_rows]
            train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            print("Using WeightedRandomSampler to rebalance.")
        except Exception as e:
            print("Sampler creation failed:", e)
            train_sampler = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
                              collate_fn=collate_batch, num_workers=num_workers, pin_memory=pin_memory, sampler=train_sampler)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0, pin_memory=pin_memory)
    return train_loader, dev_loader, train_rows, dev_rows

# ----------------------------
# EMA helper
# ----------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[name])
        self.backup = {}

# ----------------------------
# Training loop
# ----------------------------
def train_model(model: nn.Module, train_loader: DataLoader, dev_loader: DataLoader, cfg: Dict[str,Any], train_rows: List[Dict[str,Any]]):
    device = torch.device(cfg['DEVICE'])
    model.to(device)

    use_amp = (device.type == 'cuda') and cfg.get('USE_AMP', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == 'cuda' else None

    # class weights
    weights = {}
    weights['t1'] = compute_class_weights_from_rows(train_rows, 't1', 2)
    weights['t2'] = compute_class_weights_from_rows(train_rows, 't2', 3)
    weights['t3_type'] = compute_class_weights_from_rows(train_rows, 't3_type', 3)
    weights['t3_sev'] = compute_class_weights_from_rows(train_rows, 't3_sev', 3)
    weights['t4'] = compute_class_weights_from_rows(train_rows, 't4', 3)
    print("Computed class weights (None means no labels for that task in training set):")
    for k,v in weights.items():
        print(k, None if v is None else v.tolist())

    # param groups: clip backbone vs heads
    backbone_params = [p for n,p in model.named_parameters() if 'clip' in n and p.requires_grad]
    head_params = [p for n,p in model.named_parameters() if 'clip' not in n and p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': cfg['BACKBONE_LR']})
    if head_params:
        param_groups.append({'params': head_params, 'lr': cfg['LR']})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.get('WEIGHT_DECAY', 1e-2))

    total_steps = max(1, len(train_loader) * cfg['EPOCHS'] // max(1, cfg.get('GRAD_ACCUM_STEPS', 1)))
    warmup_steps = max(1, int(total_steps * 0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ema = EMA(model, decay=cfg.get('EMA_DECAY', 0.999)) if cfg.get('EMA', False) else None

    os.makedirs(cfg['SAVE_DIR'], exist_ok=True)
    best_val = -1.0
    global_step = 0
    grad_accum = cfg.get('GRAD_ACCUM_STEPS', 1)

    for epoch in range(cfg['EPOCHS']):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg['EPOCHS']}")
        optimizer.zero_grad()
        for step, (inp, lbl) in pbar:
            inp = {k:v.to(device) for k,v in inp.items()}
            lbl = {k:v.to(device) for k,v in lbl.items()}

            if scaler is not None:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(inp)
                    loss = compute_masked_losses(outputs, lbl, device, weights=weights, focal_on_t1=cfg.get('FOCAL_T1', True))
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('CLIP_GRAD', 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    if ema is not None:
                        ema.update(model)
            else:
                outputs = model(inp)
                loss = compute_masked_losses(outputs, lbl, device, weights=weights, focal_on_t1=cfg.get('FOCAL_T1', True))
                loss = loss / grad_accum
                loss.backward()
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('CLIP_GRAD', 1.0))
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    if ema is not None:
                        ema.update(model)

            running_loss += loss.item() * grad_accum
            if (step + 1) % 20 == 0:
                pbar.set_postfix({'loss': f"{(running_loss / (step+1)):.4f}"})

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} finished. Avg loss: {avg_loss:.6f}")

        # validation (use EMA weights if enabled)
        if ema is not None:
            ema.apply_shadow(model)
        reports, cms = evaluate_detailed(model, dev_loader, torch.device(cfg['DEVICE']))
        if ema is not None:
            ema.restore(model)

        # select metric: prefer Task2 macro-F1, else Task1 macro-F1
        val_score = 0.0
        if 't2' in reports:
            val_score = reports['t2']['macro avg']['f1-score']
        elif 't1' in reports:
            val_score = reports['t1']['macro avg']['f1-score']
        print(f"Validation selected score: {val_score:.4f}")

        if val_score > best_val:
            best_val = val_score
            ckpt = os.path.join(cfg['SAVE_DIR'], f"best_epoch{epoch+1}.pt")
            torch.save({'model_state': model.state_dict(), 'cfg': cfg, 'epoch': epoch+1, 'val_score': val_score}, ckpt)
            print("Saved best checkpoint:", ckpt)

    print("Training finished. Best val score:", best_val)

# ----------------------------
# Jupyter-friendly entry
# ----------------------------
def run_training(user_config: Optional[Dict[str,Any]] = None):
    cfg = CONFIG.copy()
    if user_config:
        cfg.update(user_config)

    print("Device:", cfg['DEVICE'])
    print("Paper path:", PAPER_PATH)

    train_loader, dev_loader, train_rows, dev_rows = build_dataloaders(
        cfg['DATA_CONFIG'],
        cfg['CLIP_MODEL'],
        cfg['BATCH_SIZE'],
        cfg['MAX_TEXT_LEN'],
        cfg['NUM_WORKERS'],
        cfg['PIN_MEMORY'],
        use_sampler=cfg.get('USE_SAMPLER', False),
        text_dropout=cfg.get('TEXT_DROPOUT', 0.0)
    )

    print("Train batches:", len(train_loader), "| Dev batches:", len(dev_loader))
    model = HierarchicalMultimodalModel(cfg['CLIP_MODEL'], embed_dim=512, freeze_backbone=cfg.get('FREEZE_BACKBONE', True))

    train_model(model, train_loader, dev_loader, cfg, train_rows)
    return model, train_loader, dev_loader

# ----------------------------
# If executed as script, run with default CONFIG
# ----------------------------
if __name__ == '__main__':
    run_training()
