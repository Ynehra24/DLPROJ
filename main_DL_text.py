import os
import time
import argparse
from typing import List, Dict, Any, Optional
from collections import Counter
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

# ============================================================
# DEVICE SELECTION
# ============================================================
if torch.backends.mps.is_available():
    DEVICE = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CONFIG
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

# **IMPORTANT:** Correct CLIP input size
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
# HELPERS
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


# ============================================================
# SAFE TSV PARSER WITH DUPLICATE HEADER HANDLING
# ============================================================
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
# DATASET
# ============================================================
class CrisisDataset(Dataset):
    def __init__(self, rows, processor, max_text_len=77, target_size=(224,224)):
        self.rows = rows
        self.processor = processor
        self.max_text_len = max_text_len
        self.target_size = target_size

        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
        self.std  = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

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
        img = img.resize(self.target_size, Image.BICUBIC)
        arr = np.array(img).astype("float32") / 255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2,0,1)
        return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.rows[idx]

        text = row.get('tweet_text','')
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
# COLLATE
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
# MODEL (ARCHITECTURE EXACTLY SAME AS YOUR ORIGINAL)
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
# LOSS / METRIC
# ============================================================
def compute_masked_losses(outputs, labels, device):
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    for t in ['t1','t2','t4']:
        mask = labels[t] >= 0
        if mask.any():
            l = F.cross_entropy(outputs[f'{t}_logits'][mask], labels[t][mask])
            loss = loss + l
    return loss

def compute_metrics(preds, labels):
    out = {}
    for t in ['t1','t2','t4']:
        mask = labels[t] >= 0
        if mask.any():
            pr = preds[f'{t}_logits'][mask].argmax(1).cpu()
            tr = labels[t][mask].cpu()
            out[f'{t}_acc'] = float((pr==tr).float().mean())
    return out


# ============================================================
# EVAL / TRAIN
# ============================================================
def evaluate(model, loader, device):
    model.eval()
    p = {f'{t}_logits': [] for t in ['t1','t2','t4']}
    l = {t: [] for t in ['t1','t2','t4']}
    with torch.no_grad():
        for inp, lbl in loader:
            inp = {k:v.to(device) for k,v in inp.items()}
            out = model(inp)
            for t in ['t1','t2','t4']:
                p[f'{t}_logits'].append(out[f'{t}_logits'].cpu())
                l[t].append(lbl[t])
    preds = {k:torch.cat(v,0) for k,v in p.items()}
    labs  = {k:torch.cat(v,0) for k,v in l.items()}
    return compute_metrics(preds,labs)


def train(model, train_loader, dev_loader, cfg):
    device = cfg['device']
    model.to(device)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg['lr'], weight_decay=1e-2
    )

    sched = get_cosine_schedule_with_warmup(
        opt, 200, len(train_loader)*cfg['epochs']
    )

    os.makedirs(cfg['save_dir'], exist_ok=True)
    best = -1.0

    for ep in range(cfg['epochs']):
        model.train()
        total = 0.0

        for i,(inp,lbl) in enumerate(tqdm(train_loader, desc=f"Epoch {ep}")):
            inp = {k:v.to(device) for k,v in inp.items()}
            lbl = {k:v.to(device) for k,v in lbl.items()}

            out = model(inp)
            loss = compute_masked_losses(out, lbl, device)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            total += loss.item()

        print(f"Epoch {ep} Loss: {total/len(train_loader):.6f}")

        m = evaluate(model, dev_loader, device)
        print("Dev:", m)

        score = m.get('t2_acc', m.get('t1_acc', 0))
        if score > best:
            best = score
            ckpt = os.path.join(cfg['save_dir'], f"best_{ep}.pt")
            torch.save({'model_state': model.state_dict(), 'cfg': cfg}, ckpt)
            print("Saved", ckpt)


# ============================================================
# BUILD DATALOADERS
# ============================================================
def build_dataloaders(data_cfg, clip_model, batch_size, max_text_len, num_workers, pin_memory):
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

    train_ds = CrisisDataset(train_rows, processor, max_text_len, TARGET_IMAGE_SIZE)
    dev_ds   = CrisisDataset(dev_rows, processor, max_text_len, TARGET_IMAGE_SIZE)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_batch, num_workers=num_workers,
        pin_memory=pin_memory
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=0,
        pin_memory=pin_memory
    )
    return train_loader, dev_loader


# ============================================================
# MAIN
# ============================================================
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
    args, _ = parser.parse_known_args(argv)

    cfg = {
        'clip_model': args.clip_model,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'device': args.device,
        'save_dir': args.save_dir,
        'max_text_len': DEFAULTS['max_text_len']
    }

    print("Device:", cfg['device'])

    train_loader, dev_loader = build_dataloaders(
        DATA_CONFIG, cfg['clip_model'], cfg['batch_size'], cfg['max_text_len'],
        args.num_workers, args.pin_memory
    )

    print("Train batches:", len(train_loader), "| Dev batches:", len(dev_loader))

    model = MultimodalMultiTask(cfg['clip_model'])
    train(model, train_loader, dev_loader, cfg)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
