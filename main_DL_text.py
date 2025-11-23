# new training loop 

# train.py
import os
import time
import argparse
from typing import List, Dict, Any, Optional
from collections import Counter
from pathlib import Path
from PIL import Image
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# try import torchvision transforms if available
try:
    from torchvision import transforms as T
    TV_AVAILABLE = True
except Exception:
    TV_AVAILABLE = False

# --- user-uploaded paper path (developer instruction) ---
PAPER_PATH = "/Users/yatharthnehva/Downloads/emnlp.pdf"

# ============================================================
# DEVICE
# ============================================================
if torch.backends.mps.is_available():
    DEVICE = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# KEEP YOUR PATHS / MAPS UNCHANGED
# ============================================================
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
    'device': DEVICE,
    'save_dir': 'checkpoints_final',
    'max_text_len': 77,
    'num_workers': 0,
    'pin_memory': False
}

TARGET_IMAGE_SIZE = (224, 224)

DAMAGE_MAP = {
    'little_or_no_damage': 0,
    'mild_damage': 1,
    'severe_damage': 2
}
BINARY_MAP = {
    'negative': 0,
    'positive': 1
}

# ============================================================
# HELPERS: TSV reading & safe string
# ============================================================
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

def read_and_map(path: str, task_name: str) -> List[Dict[str,Any]]:
    rows = []
    if not os.path.exists(path):
        return rows

    with open(path, 'r', encoding='utf-8') as f:
        raw_header = f.readline().rstrip('\n').split('\t')

        # Fix duplicate column names
        header = []
        seen = {}
        for h in raw_header:
            if h not in seen:
                header.append(h)
                seen[h] = 1
            else:
                new_h = f"{h}__{seen[h]}"
                header.append(new_h)
                seen[h] += 1

        col_idx = {c: i for i, c in enumerate(header)}

        # Text column
        text_col = detect_column(raw_header, ['tweet_text','tweet','text'])
        if text_col is None:
            text_col = raw_header[2] if len(raw_header) > 2 else raw_header[0]

        # Image column
        image_col = detect_column(raw_header, ['image','image_path','image_id'])

        # Label column
        if task_name == 'task1':
            label_col = detect_column(raw_header, ['label','damage','label_text_image'])
        else:
            label_col = detect_column(raw_header, ['label_text_image','label','label_text','label_image'])

        # remap label_col to deduped header if needed
        if label_col and label_col not in header:
            for h in header:
                if h.startswith(label_col):
                    label_col = h
                    break

        # read rows
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < len(header):
                parts += [''] * (len(header) - len(parts))

            rec = {c: parts[i] for c, i in col_idx.items()}

            def fetch(colname):
                if colname in rec:
                    return rec[colname]
                for k in rec:
                    if k.startswith(colname):
                        return rec[k]
                return ""

            text = safe_str(fetch(text_col))
            img  = safe_str(fetch(image_col))
            raw = safe_str(fetch(label_col)).lower()

            mapped = -1
            if raw:
                if task_name == 'task1':
                    mapped = DAMAGE_MAP.get(raw, -1)
                else:
                    mapped = BINARY_MAP.get(raw, -1)

            rows.append({
                'tweet_text': text,
                'image_path': img,
                't1': mapped if task_name == 'task1' else -1,
                't2': mapped if task_name == 'task2' else -1,
                't4': mapped if task_name == 'task3' else -1
            })

    return rows

# ============================================================
# DATASET with augmentation
# ============================================================
class CrisisDataset(Dataset):
    def __init__(self, rows, processor, max_text_len=77, target_size=(224,224), augment=False, text_dropout_prob=0.0):
        self.rows = rows
        self.processor = processor
        self.max_text_len = max_text_len
        self.target_size = target_size
        self.augment = augment
        self.text_dropout_prob = text_dropout_prob

        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
        self.std  = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

        if self.augment and TV_AVAILABLE:
            self.aug = T.Compose([
                T.RandomResizedCrop(self.target_size[0], scale=(0.8,1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.1,0.1,0.1,0.0),
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.rows)

    def _load_image(self, path):
        try:
            if path and os.path.exists(path):
                img = Image.open(path).convert("RGB")
            else:
                raise Exception()
        except:
            img = Image.fromarray(np.zeros((self.target_size[1], self.target_size[0], 3), np.uint8))
        return img

    def _process_image(self, img):
        if self.aug:
            try:
                img = self.aug(img)
            except Exception:
                img = img.resize(self.target_size, Image.BICUBIC)
        else:
            img = img.resize(self.target_size, Image.BICUBIC)

        arr = np.array(img).astype("float32") / 255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2,0,1)
        return torch.tensor(arr, dtype=torch.float32)

    def _text_dropout(self, text:str):
        if self.text_dropout_prob <= 0.0:
            return text
        toks = text.split()
        keep = [w for w in toks if random.random() > self.text_dropout_prob]
        if len(keep) == 0:
            return text  # avoid empty
        return " ".join(keep)

    def __getitem__(self, idx):
        row = self.rows[idx]

        text = row.get('tweet_text','')
        if self.text_dropout_prob > 0.0:
            text = self._text_dropout(text)

        img = self._load_image(row.get('image_path',''))
        pixel_values = self._process_image(img)

        tok = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt"
        )

        inputs = {
            'input_ids': tok['input_ids'].squeeze(0),
            'attention_mask': tok['attention_mask'].squeeze(0),
            'pixel_values': pixel_values
        }

        labels = {
            't1': torch.tensor(row.get('t1', -1), dtype=torch.long),
            't2': torch.tensor(row.get('t2', -1), dtype=torch.long),
            't4': torch.tensor(row.get('t4', -1), dtype=torch.long)
        }

        return inputs, labels

# ============================================================
# COLLATE (same as yours)
# ============================================================
def collate_batch(batch):
    inputs = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    return {
        'input_ids': torch.stack([i['input_ids'] for i in inputs]),
        'attention_mask': torch.stack([i['attention_mask'] for i in inputs]),
        'pixel_values': torch.stack([i['pixel_values'] for i in inputs])
    }, {
        't1': torch.stack([l['t1'] for l in labels]),
        't2': torch.stack([l['t2'] for l in labels]),
        't4': torch.stack([l['t4'] for l in labels])
    }

# ============================================================
# MODEL (exactly your original architecture)
# ============================================================
class QueryingTransformer(nn.Module):
    def __init__(self, embed_dim=512, n_query=16, n_layer=3, n_head=8, dropout=0.1):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(n_query, embed_dim) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            activation='gelu'
        )
        self.tr = nn.TransformerEncoder(enc, num_layers=n_layer)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_query = n_query

    def forward(self, ie, te):
        B = ie.size(0)
        q = self.query_tokens.unsqueeze(0).expand(B,-1,-1)
        x = torch.cat([q, ie, te], dim=1)
        x = self.tr(x.permute(1,0,2)).permute(1,0,2)
        pooled = x[:, :self.n_query].mean(1)
        return self.proj(pooled)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256,out_dim)
        )
    def forward(self, x):
        return self.net(x)


class MultimodalMultiTask(nn.Module):
    def __init__(self, clip_model_name, embed_dim=512, freeze_backbone=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)

        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.img_proj = nn.Linear(self.clip.vision_model.config.hidden_size, embed_dim)
        self.txt_proj = nn.Linear(self.clip.text_model.config.hidden_size, embed_dim)

        self.querying = QueryingTransformer(embed_dim)
        self.head_t1 = ClassificationHead(embed_dim,3)
        self.head_t2 = ClassificationHead(embed_dim,2)
        self.head_t4 = ClassificationHead(embed_dim,2)

    def forward(self, batch):
        out = self.clip(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            output_hidden_states=True,
            return_dict=True
        )
        vp = out.vision_model_output.pooler_output
        tp = out.text_model_output.pooler_output

        vp = self.img_proj(vp).unsqueeze(1)
        tp = self.txt_proj(tp).unsqueeze(1)

        mm = self.querying(vp, tp)

        return {
            't1_logits': self.head_t1(mm),
            't2_logits': self.head_t2(mm),
            't4_logits': self.head_t4(mm)
        }

# ============================================================
# Losses, metrics, and utility functions (focal loss, weights)
# ============================================================
def focal_loss_logits(inputs, targets, alpha=None, gamma=2.0, reduction='mean'):
    # inputs: logits (N, C), targets: (N,)
    ce = F.cross_entropy(inputs, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def compute_class_weights_from_rows(rows, label_key, n_classes):
    ctr = Counter([r[label_key] for r in rows if r.get(label_key, -1) >= 0])
    freqs = [ctr.get(i, 0) for i in range(n_classes)]
    freqs = np.array(freqs, dtype=np.float32)
    # inverse frequency with smoothing
    inv = (freqs.max() / (freqs + 1.0))
    w = inv / (inv.sum() + 1e-12) * n_classes
    return torch.tensor(w, dtype=torch.float32)

def make_weighted_sampler(rows, label_key, n_classes):
    # compute per-sample weights (inverse class freq)
    ctr = Counter([r[label_key] for r in rows if r.get(label_key, -1) >= 0])
    total = sum(ctr.values()) or 1
    class_weight = {c: total / (v + 1e-12) for c, v in ctr.items()}
    weights = []
    for r in rows:
        v = r.get(label_key, -1)
        weights.append(float(class_weight.get(v, 1.0)))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# Helper: unfreeze last N layers of CLIP (best-effort)
def unfreeze_clip_last_n_layers(model, n=1):
    # vision encoder: possible attr paths depend on HF CLIP version; try the common ones
    try:
        v_blocks = model.clip.vision_model.encoder.layers
    except Exception:
        v_blocks = getattr(model.clip.vision_model, "encoder", None)
        if v_blocks is not None:
            v_blocks = getattr(v_blocks, "layers", None)
    try:
        t_blocks = model.clip.text_model.encoder.layers
    except Exception:
        t_blocks = getattr(model.clip.text_model, "encoder", None)
        if t_blocks is not None:
            t_blocks = getattr(t_blocks, "layers", None)

    def _unfreeze(blocks):
        if blocks is None:
            return
        for blk in list(blocks)[-n:]:
            for p in blk.parameters():
                p.requires_grad = True

    _unfreeze(v_blocks)
    _unfreeze(t_blocks)

# EMA helper
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

# masked losses with optional focal on t1
def compute_masked_losses(outputs, labels, device, weights=None, use_focal_for_t1=True):
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    for t, ncls in [('t1',3), ('t2',2), ('t4',2)]:
        mask = labels[t] >= 0
        if mask.any():
            logits = outputs[f'{t}_logits'][mask]
            tg = labels[t][mask]
            w = None
            if weights and weights.get(t) is not None:
                w = weights[t].to(device)
            if t == 't1' and use_focal_for_t1:
                loss_t = focal_loss_logits(logits, tg, alpha=w, gamma=2.0, reduction='mean')
            else:
                loss_t = F.cross_entropy(logits, tg, weight=w)
            loss = loss + loss_t
    return loss

# compute per-task accuracy on loader
def compute_metrics(preds, labels):
    out = {}
    for t in ['t1','t2','t4']:
        mask = labels[t] >= 0
        if mask.any():
            pr = preds[f'{t}_logits'][mask].argmax(1).cpu()
            tr = labels[t][mask].cpu()
            out[f'{t}_acc'] = float((pr==tr).float().mean())
    return out

# detailed evaluation (classification_report + confusion matrix)
def evaluate_detailed(model, loader, device):
    model.eval()
    all_preds = {t: [] for t in ['t1','t2','t4']}
    all_trues = {t: [] for t in ['t1','t2','t4']}

    with torch.no_grad():
        for inp, lbl in loader:
            inp = {k:v.to(device) for k,v in inp.items()}
            out = model(inp)
            for t in ['t1','t2','t4']:
                mask = lbl[t] >= 0
                if mask.any():
                    y_pred = out[f'{t}_logits'][mask].argmax(1).cpu().tolist()
                    y_true = lbl[t][mask].cpu().tolist()
                    all_preds[t].extend(y_pred)
                    all_trues[t].extend(y_true)

    reports = {}
    cms = {}
    for t in ['t1','t2','t4']:
        if len(all_trues[t]) == 0:
            continue
        print(f"\n--- Task {t} ---")
        print(classification_report(all_trues[t], all_preds[t], digits=4, zero_division=0))
        cm = confusion_matrix(all_trues[t], all_preds[t])
        print("Confusion matrix:\n", cm)
        reports[t] = classification_report(all_trues[t], all_preds[t], output_dict=True, zero_division=0)
        cms[t] = cm
    return reports, cms

# ============================================================
# DATALOADER BUILD WITH SAMPLER + RETURNS rows
# ============================================================
def build_dataloaders(data_cfg, clip_model, batch_size, max_text_len, num_workers, pin_memory,
                      augment=False, text_dropout_prob=0.0, use_sampler=False):
    print("=== Scanning label distributions ===")
    for tg, paths in data_cfg.items():
        col = 'label' if tg=='task1' else 'label_text_image'
        print(tg, scan_and_print_distribution(paths['train_tsv'], col))

    train_rows = []
    dev_rows   = []

    for tg, p in data_cfg.items():
        train_rows += read_and_map(p['train_tsv'], tg)
        dev_rows   += read_and_map(p['dev_tsv'],   tg)

    processor = CLIPProcessor.from_pretrained(clip_model)

    train_ds = CrisisDataset(train_rows, processor, max_text_len, TARGET_IMAGE_SIZE, augment=augment, text_dropout_prob=text_dropout_prob)
    dev_ds   = CrisisDataset(dev_rows, processor, max_text_len, TARGET_IMAGE_SIZE, augment=False, text_dropout_prob=0.0)

    train_sampler = None
    if use_sampler:
        try:
            train_sampler = make_weighted_sampler(train_rows, 't1', 3)
            print("Using WeightedRandomSampler over t1 to rebalance severity")
        except Exception as e:
            print("Sampler creation failed:", e)
            train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
        collate_fn=collate_batch, num_workers=num_workers,
        pin_memory=pin_memory, sampler=train_sampler
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=0,
        pin_memory=pin_memory
    )
    return train_loader, dev_loader, train_rows, dev_rows

# ============================================================
# TRAIN LOOP with AMP, accumulation, EMA, LR groups, early stop
# ============================================================
from torch.cuda.amp import autocast, GradScaler

def train(model, train_loader, dev_loader, cfg, train_rows=None):
    device = cfg['device']
    model.to(device)

    # compute class weights
    weights = {}
    if train_rows is not None:
        weights['t1'] = compute_class_weights_from_rows(train_rows, 't1', 3)
        weights['t2'] = compute_class_weights_from_rows(train_rows, 't2', 2)
        weights['t4'] = compute_class_weights_from_rows(train_rows, 't4', 2)
        print("Class weights (t1,t2,t4):", weights['t1'].tolist(), weights['t2'].tolist(), weights['t4'].tolist())

    # optional unfreeze
    if cfg.get('unfreeze_last_n', 0) > 0:
        print("Unfreezing last", cfg['unfreeze_last_n'], "layers of CLIP")
        unfreeze_clip_last_n_layers(model, cfg['unfreeze_last_n'])

    # param groups: backbone vs heads
    backbone_params = [p for n,p in model.named_parameters() if 'clip' in n and p.requires_grad]
    head_params = [p for n,p in model.named_parameters() if 'clip' not in n and p.requires_grad]
    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({'params': backbone_params, 'lr': cfg.get('backbone_lr', 5e-6)})
    if len(head_params) > 0:
        param_groups.append({'params': head_params, 'lr': cfg.get('head_lr', cfg['lr'])})

    opt = torch.optim.AdamW(param_groups, weight_decay=cfg.get('weight_decay', 1e-2))

    total_steps = max(1, len(train_loader) * cfg['epochs'] // cfg.get('grad_accum_steps', 1))
    warmup_steps = max(1, int(0.03 * total_steps))
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    scaler = GradScaler()
    ema = EMA(model, decay=cfg.get('ema_decay', 0.999)) if cfg.get('use_ema', False) else None

    os.makedirs(cfg['save_dir'], exist_ok=True)
    best = -1.0
    no_improve = 0
    patience = cfg.get('early_stop_patience', 3)
    global_step = 0

    for ep in range(cfg['epochs']):
        model.train()
        total_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {ep}")

        opt.zero_grad()
        for i, (inp, lbl) in pbar:
            inp = {k:v.to(device) for k,v in inp.items()}
            lbl = {k:v.to(device) for k,v in lbl.items()}

            with autocast():
                out = model(inp)
                loss = compute_masked_losses(out, lbl, device, weights=weights, use_focal_for_t1=cfg.get('use_focal_t1', True))
                loss = loss / cfg.get('grad_accum_steps', 1)

            scaler.scale(loss).backward()

            if (i + 1) % cfg.get('grad_accum_steps', 1) == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip_grad', 1.0))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
                global_step += 1
                if ema is not None:
                    ema.update(model)

            total_loss += loss.item() * cfg.get('grad_accum_steps', 1)
            pbar.set_postfix({"loss": f"{(total_loss / (i+1)):.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {ep} avg loss: {avg_loss:.6f}")

        # evaluation on dev
        model.eval()
        if ema is not None:
            # swap to ema weights for evaluation
            ema.apply_shadow(model)

        reports, cms = evaluate_detailed(model, dev_loader, device)

        if ema is not None:
            ema.restore(model)

        # pick scoring metric - prefer t2_acc (humanitarian) fallback to t1_acc
        # compute dev score
        dev_score = 0.0
        if 't2' in reports:
            # use macro F1 if available
            dev_score = reports['t2'].get('macro avg', {}).get('f1-score', 0.0)
        elif 't1' in reports:
            dev_score = reports['t1'].get('macro avg', {}).get('f1-score', 0.0)
        else:
            dev_score = 0.0

        print("Dev score (selection):", dev_score)

        if dev_score > best:
            best = dev_score
            no_improve = 0
            ckpt = os.path.join(cfg['save_dir'], f"best_{ep}.pt")
            torch.save({'model_state': model.state_dict(), 'cfg': cfg}, ckpt)
            print("Saved", ckpt)
        else:
            no_improve += 1
            print(f"No improvement count: {no_improve}/{patience}")
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    print("Training finished. Best score:", best)

# ============================================================
# MAIN
# ============================================================
def scan_and_print_distribution(path: str, col: str):
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
                if v != '':
                    ctr[v] += 1
    return ctr

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model', default=DEFAULTS['clip_model'])
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--device', default=DEFAULTS['device'])
    parser.add_argument('--save_dir', default=DEFAULTS['save_dir'])
    parser.add_argument('--num_workers', type=int, default=DEFAULTS['num_workers'])
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--use_sampler', action='store_true', help="Use WeightedRandomSampler on t1")
    parser.add_argument('--augment', action='store_true', help="Use simple image augmentations")
    parser.add_argument('--text_dropout', type=float, default=0.0, help="Text word dropout prob")
    parser.add_argument('--unfreeze_last_n', type=int, default=0, help="Unfreeze last N CLIP layers")
    parser.add_argument('--use_ema', action='store_true', help="Use EMA of weights during training")
    parser.add_argument('--grad_accum_steps', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--backbone_lr', type=float, default=5e-6)
    parser.add_argument('--head_lr', type=float, default=5e-4)
    parser.add_argument('--use_focal_t1', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=3)
    args, _ = parser.parse_known_args(argv)

    cfg = {
        'clip_model': args.clip_model,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'device': args.device,
        'save_dir': args.save_dir,
        'max_text_len': DEFAULTS['max_text_len'],
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'unfreeze_last_n': args.unfreeze_last_n,
        'use_ema': args.use_ema,
        'grad_accum_steps': args.grad_accum_steps,
        'backbone_lr': args.backbone_lr,
        'head_lr': args.head_lr,
        'use_focal_t1': args.use_focal_t1,
        'early_stop_patience': args.early_stop_patience,
        'ema_decay': 0.999
    }

    print("Device:", cfg['device'])
    train_loader, dev_loader, train_rows, dev_rows = build_dataloaders(
        DATA_CONFIG, cfg['clip_model'], cfg['batch_size'], cfg['max_text_len'],
        args.num_workers, args.pin_memory, augment=args.augment, text_dropout_prob=args.text_dropout,
        use_sampler=args.use_sampler
    )

    print("Train batches:", len(train_loader), "| Dev batches:", len(dev_loader))

    model = MultimodalMultiTask(cfg['clip_model'], freeze_backbone=True)
    train(model, train_loader, dev_loader, cfg, train_rows=train_rows)

if __name__ == "__main__":
    main()
