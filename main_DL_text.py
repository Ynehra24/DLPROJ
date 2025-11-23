# full_fixed_final_pipeline.py
import os
import time
import argparse
from typing import List, Dict, Any, Optional
from collections import Counter
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

# ----------------- CONFIG -----------------
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
    'save_dir': 'checkpoints_final',
    'max_text_len': 77
}

# ----------------- KNOWN LABEL MAPPINGS (from your distributions) -----------------
DAMAGE_MAP = {
    'little_or_no_damage': 0,
    'mild_damage': 1,
    'severe_damage': 2
}
BINARY_MAP = {
    'negative': 0,
    'positive': 1
}

# ----------------- UTILITIES -----------------
def safe_str(x):
    if x is None: return ''
    if isinstance(x, str): return x.strip()
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

# ----------------- TSV reading and mapping -----------------
def read_and_map(path: str, task_name: str) -> List[Dict[str,Any]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        col_idx = {c:i for i,c in enumerate(header)}
        text_col = detect_column(header, ['tweet_text','tweet','text'])
        if text_col is None:
            text_col = header[2] if len(header) > 2 else header[0]
        image_col = detect_column(header, ['image','image_path','image_id'])
        label_col = None
        if task_name == 'task1':
            label_col = detect_column(header, ['label','damage','label_text_image'])
        else:
            label_col = detect_column(header, ['label_text_image','label','label_text','label_image'])
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < len(header):
                parts += [''] * (len(header) - len(parts))
            rec = {c: parts[i] for c,i in col_idx.items()}
            text = safe_str(rec.get(text_col, ''))
            img = safe_str(rec.get(image_col, ''))
            mapped = -1
            raw = ''
            if label_col and label_col in rec:
                raw = safe_str(rec[label_col]).lower()
                if raw != '':
                    if task_name == 'task1':
                        mapped = DAMAGE_MAP.get(raw, -1)
                    else:
                        mapped = BINARY_MAP.get(raw, -1)
            rec['tweet_text'] = text
            rec['image_path'] = img
            rec['t1'] = mapped if task_name == 'task1' else -1
            rec['t2'] = mapped if task_name == 'task2' else -1
            rec['t4'] = mapped if task_name == 'task3' else -1
            rows.append(rec)
    return rows

# ----------------- Dataset & Collate -----------------
class CrisisDataset(Dataset):
    def __init__(self, rows: List[Dict[str,Any]], processor: CLIPProcessor, max_text_len:int=77):
        self.rows = rows
        self.processor = processor
        # enforce CLIP text limit (usually 77)
        self.max_text_len = min(max_text_len, DEFAULTS['max_text_len'])
    def __len__(self): return len(self.rows)
    def _load_image(self, path: str):
        if not path:
            return None
        if not os.path.exists(path):
            return None
        try:
            im = Image.open(path).convert('RGB')
            return im
        except:
            return None
    def __getitem__(self, idx):
        r = self.rows[idx]
        text = r.get('tweet_text','')
        img_path = r.get('image_path','')
        img = self._load_image(img_path)
        if img is None:
            img = Image.new('RGB', (224,224), (0,0,0))
        # try with safe parameters; if processor complains about padding, retry with padding=True
        try:
            inputs = self.processor(
                text=[text],
                images=img,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len
            )
        except ValueError:
            inputs = self.processor(
                text=[text],
                images=img,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_text_len
            )
        inputs = {k: v.squeeze(0) for k,v in inputs.items()}
        labels = {
            't1': torch.tensor(r.get('t1', -1), dtype=torch.long),
            't2': torch.tensor(r.get('t2', -1), dtype=torch.long),
            't4': torch.tensor(r.get('t4', -1), dtype=torch.long)
        }
        return inputs, labels

def collate_batch(batch):
    inputs_list = [b[0] for b in batch]
    labels_list = [b[1] for b in batch]
    # gather keys safely
    input_ids = torch.stack([i['input_ids'] for i in inputs_list])
    attention_mask = torch.stack([i['attention_mask'] for i in inputs_list])
    # pixel_values might be present and should be B x C x H x W
    if 'pixel_values' in inputs_list[0]:
        pixel_values = torch.stack([i['pixel_values'] for i in inputs_list])
    else:
        # fallback zero tensor
        bs = len(inputs_list)
        pixel_values = torch.zeros((bs, 3, 224, 224), dtype=torch.float)
    collated = {'input_ids': input_ids, 'attention_mask': attention_mask, 'pixel_values': pixel_values}
    labels = {k: torch.stack([l[k] for l in labels_list]) for k in labels_list[0].keys()}
    return collated, labels

# ----------------- Model -----------------
class QueryingTransformer(nn.Module):
    def __init__(self, embed_dim=512, n_query=16, n_layer=3, n_head=8, dropout=0.1):
        super().__init__()
        self.n_query = n_query
        self.query_tokens = nn.Parameter(torch.randn(n_query, embed_dim)*0.02)
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, dim_feedforward=embed_dim*4, dropout=dropout, activation='gelu')
        self.tr = nn.TransformerEncoder(enc, num_layers=n_layer)
        self.proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, ie, te):
        B = ie.size(0)
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        cat = torch.cat([q, ie, te], dim=1)
        out = self.tr(cat.permute(1,0,2)).permute(1,0,2)
        pooled = out[:, :self.n_query, :].mean(dim=1)
        return self.proj(pooled)

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256,out_dim))
    def forward(self,x): return self.net(x)

class MultimodalMultiTask(nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32', embed_dim=512, freeze_backbone=True, t1_classes=3, t2_classes=2, t4_classes=2):
        super().__init__()
        try:
            self.clip = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True, ignore_mismatched_sizes=True)
        except Exception:
            self.clip = CLIPModel.from_pretrained(clip_model_name, ignore_mismatched_sizes=True)
        if freeze_backbone:
            for p in self.clip.parameters(): p.requires_grad = False
        self.img_proj = nn.Linear(self.clip.vision_model.config.hidden_size, embed_dim)
        self.txt_proj = nn.Linear(self.clip.text_model.config.hidden_size, embed_dim)
        self.querying = QueryingTransformer(embed_dim=embed_dim, n_query=16, n_layer=3, n_head=8)
        self.head_t1 = ClassificationHead(embed_dim, t1_classes)
        self.head_t2 = ClassificationHead(embed_dim, t2_classes)
        self.head_t4 = ClassificationHead(embed_dim, t4_classes)
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
        return {
            't1_logits': self.head_t1(mm),
            't2_logits': self.head_t2(mm),
            't4_logits': self.head_t4(mm),
            'mm': mm
        }

# ----------------- Loss & Metrics -----------------
def compute_masked_losses(outputs, labels, device):
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    losses = {}
    m1 = labels['t1'] >= 0
    if m1.any():
        l1 = F.cross_entropy(outputs['t1_logits'][m1].to(device), labels['t1'][m1].to(device))
        losses['t1'] = float(l1.detach().cpu()); loss = loss + l1
    m2 = labels['t2'] >= 0
    if m2.any():
        l2 = F.cross_entropy(outputs['t2_logits'][m2].to(device), labels['t2'][m2].to(device))
        losses['t2'] = float(l2.detach().cpu()); loss = loss + l2
    m4 = labels['t4'] >= 0
    if m4.any():
        l4 = F.cross_entropy(outputs['t4_logits'][m4].to(device), labels['t4'][m4].to(device))
        losses['t4'] = float(l4.detach().cpu()); loss = loss + l4
    return loss, losses

def compute_metrics(preds, labels):
    metrics = {}
    m1 = labels['t1'] >= 0
    if m1.any():
        pr = preds['t1_logits'][m1].argmax(dim=1).cpu(); tr = labels['t1'][m1].cpu(); metrics['t1_acc'] = (pr==tr).float().mean().item()
    m2 = labels['t2'] >= 0
    if m2.any():
        pr = preds['t2_logits'][m2].argmax(dim=1).cpu(); tr = labels['t2'][m2].cpu(); metrics['t2_acc'] = (pr==tr).float().mean().item()
    m4 = labels['t4'] >= 0
    if m4.any():
        pr = preds['t4_logits'][m4].argmax(dim=1).cpu(); tr = labels['t4'][m4].cpu(); metrics['t4_acc'] = (pr==tr).float().mean().item()
    return metrics

# ----------------- Train / Eval -----------------
def evaluate(model: nn.Module, loader: DataLoader, device):
    model.eval()
    all_preds = {'t1_logits': [], 't2_logits': [], 't4_logits': []}
    all_labels = {'t1': [], 't2': [], 't4': []}
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Eval"):
            inputs = {k: v.to(device) for k,v in inputs.items()}
            outputs = model(inputs)
            all_preds['t1_logits'].append(outputs['t1_logits'].cpu())
            all_preds['t2_logits'].append(outputs['t2_logits'].cpu())
            all_preds['t4_logits'].append(outputs['t4_logits'].cpu())
            all_labels['t1'].append(labels['t1'])
            all_labels['t2'].append(labels['t2'])
            all_labels['t4'].append(labels['t4'])
    preds = {k: torch.cat(v, 0) for k,v in all_preds.items()}
    labs = {k: torch.cat(v, 0) for k,v in all_labels.items()}
    return compute_metrics(preds, labs)

def train(model: nn.Module, train_loader: DataLoader, dev_loader: DataLoader, cfg: Dict[str,Any]):
    device = cfg['device']
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=1e-2)
    total_steps = len(train_loader) * cfg['epochs']
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=200, num_training_steps=total_steps)
    best = -1.0
    os.makedirs(cfg['save_dir'], exist_ok=True)
    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
            inputs = {k: v.to(device) for k,v in inputs.items()}
            labels = {k: v.to(device) for k,v in labels.items()}
            outputs = model(inputs)
            loss, loss_dict = compute_masked_losses(outputs, labels, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch} step {i}/{len(train_loader)} loss={loss.item():.6f} losses={loss_dict}")
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} avg_loss={avg_loss:.6f}")
        dev_metrics = evaluate(model, dev_loader, device)
        print("Dev metrics:", dev_metrics)
        score = dev_metrics.get('t2_acc', dev_metrics.get('t1_acc', 0.0))
        if score > best:
            best = score
            ckpt = os.path.join(cfg['save_dir'], f"best_epoch_{epoch}.pt")
            torch.save({'model_state': model.state_dict(), 'cfg': cfg}, ckpt)
            print("Saved", ckpt)

# ----------------- Build dataloaders (uses fixed mappings above) -----------------
def build_dataloaders(data_cfg, clip_model_name, batch_size, max_text_len=77):
    print("=== Scanning label distributions (first few lines) ===")
    for tg, paths in data_cfg.items():
        train_p = paths['train_tsv']
        dev_p = paths['dev_tsv']
        col = 'label' if tg == 'task1' else 'label_text_image'
        ctr_train = scan_and_print_distribution(train_p, col)
        ctr_dev = scan_and_print_distribution(dev_p, col)
        if ctr_train:
            print(f"{tg} train {col} distribution (top 10): {ctr_train.most_common(10)}")
        if ctr_dev:
            print(f"{tg} dev {col} distribution (top 10): {ctr_dev.most_common(10)}")
    train_rows = []
    dev_rows = []
    for tg, paths in data_cfg.items():
        train_rows += read_and_map(paths['train_tsv'], tg)
        dev_rows += read_and_map(paths['dev_tsv'], tg)
    def count_lab(rows):
        c = {'t1':0,'t2':0,'t4':0}
        for r in rows:
            for k in c:
                if r.get(k, -1) >= 0:
                    c[k] += 1
        return c
    print("Train labeled counts:", count_lab(train_rows))
    print("Dev labeled counts:", count_lab(dev_rows))
    # create processor with fallback
    try:
        processor = CLIPProcessor.from_pretrained(clip_model_name)
    except Exception:
        processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=False)
    train_ds = CrisisDataset(train_rows, processor, max_text_len)
    dev_ds = CrisisDataset(dev_rows, processor, max_text_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_loader, dev_loader

# ----------------- main -----------------
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
        'max_text_len': DEFAULTS['max_text_len']
    }
    train_loader, dev_loader = build_dataloaders(DATA_CONFIG, cfg['clip_model'], cfg['batch_size'], cfg['max_text_len'])
    print(f"Train batches: {len(train_loader)}, Dev batches: {len(dev_loader)}")
    model = MultimodalMultiTask(clip_model_name=cfg['clip_model'], embed_dim=512, freeze_backbone=True, t1_classes=3, t2_classes=2, t4_classes=2)
    train(model, train_loader, dev_loader, cfg)

if __name__ == '__main__':
    main()
