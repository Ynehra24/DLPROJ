import os
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup

PAPER_PATH = "/mnt/data/emnlp.pdf"  

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
    'lr': 5e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'checkpoints_improved',
    'max_text_len': 77,
    'num_workers': 4,
    'pin_memory': False,
    'image_size': 224,
    'warmup_frac': 0.03,
    'unfreeze_last_n_clip_layers': 2,
    'use_focal': False,
    'focal_gamma': 2.0,
    'weight_decay': 1e-2,
    'grad_clip': 1.0,
    'amp': True, 
    'gate_by_t1': True  
}


DAMAGE_MAP = {
    'little_or_no_damage': 0,
    'mild_damage': 1,
    'severe_damage': 2
}
BINARY_MAP = {'negative': 0, 'positive': 1}


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

def scan_and_print_distribution(path: str, col: str) -> Counter:
    ctr = Counter()
    if not os.path.exists(path):
        return ctr
    with open(path, 'r', encoding='utf-8') as f:
        raw_header = f.readline().rstrip('\n').split('\t')
        if col not in raw_header:
            return ctr
        idx = raw_header.index(col)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if idx < len(parts):
                v = safe_str(parts[idx]).lower()
                if v != '':
                    ctr[v] += 1
    return ctr

def read_and_map(path: str, task_name: str) -> List[Dict[str,Any]]:
    rows = []
    if not os.path.exists(path):
        return rows

    with open(path, 'r', encoding='utf-8') as f:
        raw_header = f.readline().rstrip('\n').split('\t')

        header = []
        seen = {}
        for h in raw_header:
            h = h.strip()
            if h == '':
                h = 'col'
            if h not in seen:
                header.append(h)
                seen[h] = 1
            else:
                new_h = f"{h}__{seen[h]}"
                header.append(new_h)
                seen[h] += 1

        col_idx = {c: i for i, c in enumerate(header)}

        text_col = detect_column(raw_header, ['tweet_text','tweet','text'])
        if text_col is None:
            text_col = header[2] if len(header) > 2 else header[0]

        image_col = detect_column(raw_header, ['image','image_path','image_id'])
        if image_col is None:
            image_col = header[1] if len(header) > 1 else header[0]

        if task_name == 'task1':
            label_col = detect_column(raw_header, ['label','damage','label_text_image'])
        else:
            label_col = detect_column(raw_header, ['label_text_image','label','label_text','label_image'])

        if label_col and label_col not in header:
            for h in header:
                if h.startswith(label_col):
                    label_col = h
                    break

        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < len(header):
                parts += [''] * (len(header) - len(parts))
            rec = {c: parts[i] for c,i in col_idx.items()}

            def fetch(colname):
                if colname in rec:
                    return rec[colname]
                for k in rec:
                    if k.startswith(colname):
                        return rec[k]
                return ""

            text = safe_str(fetch(text_col))
            img  = safe_str(fetch(image_col))
            raw = safe_str(fetch(label_col)).lower() if label_col else ""

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
class CrisisDataset(Dataset):
    def __init__(self, rows, processor: CLIPProcessor, max_text_len=77, image_size=224):
        self.rows = rows
        self.processor = processor
        self.max_text_len = max_text_len
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def _load_image(self, path: str) -> Image.Image:
        try:
            if path and os.path.exists(path):
                img = Image.open(path).convert("RGB")
            else:
                raise FileNotFoundError()
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size))
        return img

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        text = row.get('tweet_text','')
        img_path = row.get('image_path','')
        image = self._load_image(img_path)

        proc_out = self.processor(
            text=[text],
            images=[image],
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        inputs = {
            'input_ids': proc_out['input_ids'].squeeze(0),
            'attention_mask': proc_out['attention_mask'].squeeze(0),
            'pixel_values': proc_out['pixel_values'].squeeze(0)
        }

        labels = {
            't1': torch.tensor(row.get('t1', -1), dtype=torch.long),
            't2': torch.tensor(row.get('t2', -1), dtype=torch.long),
            't4': torch.tensor(row.get('t4', -1), dtype=torch.long)
        }
        return inputs, labels

def collate_batch(batch):
    inputs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    return {
        'input_ids': torch.stack([i['input_ids'] for i in inputs]),
        'attention_mask': torch.stack([i['attention_mask'] for i in inputs]),
        'pixel_values': torch.stack([i['pixel_values'] for i in inputs]),
    }, {
        't1': torch.stack([l['t1'] for l in labels]),
        't2': torch.stack([l['t2'] for l in labels]),
        't4': torch.stack([l['t4'] for l in labels])
    }

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

class AdapterHead(nn.Module):
    def __init__(self, in_dim, out_dim, bottleneck=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class MultimodalMultiTask(nn.Module):
    def __init__(self, clip_model_name, embed_dim=512, freeze_backbone=True, unfreeze_last_n=0):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)

        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False

        if unfreeze_last_n > 0:
            self._unfreeze_last_n_layers(unfreeze_last_n)

        self.img_proj = nn.Linear(self.clip.vision_model.config.hidden_size, embed_dim)
        self.txt_proj = nn.Linear(self.clip.text_model.config.hidden_size, embed_dim)

        self.querying = QueryingTransformer(embed_dim=embed_dim)
        self.head_t1 = AdapterHead(embed_dim, 3, bottleneck=256)
        self.head_t2 = AdapterHead(embed_dim, 2, bottleneck=128)
        self.head_t4 = AdapterHead(embed_dim, 2, bottleneck=128)

    def _unfreeze_last_n_layers(self, n:int):
        def _unfreeze(module, name_prefix, n):
            blocks = None
            for attr in ['encoder.layers','encoder.layer','transformer.layer','layers','encoder.encoder.layer']:
                try:
                    obj = module
                    for part in attr.split('.'):
                        obj = getattr(obj, part)
                    if isinstance(obj, (list, tuple)) or hasattr(obj, '__len__'):
                        blocks = obj
                        break
                except Exception:
                    continue
            if blocks is None:
                return
            L = len(blocks)
            for i in range(max(0, L-n), L):
                for p in blocks[i].parameters():
                    p.requires_grad = True

        try:
            _unfreeze(self.clip.vision_model, 'vision_model', n)
        except Exception:
            pass
        try:
            _unfreeze(self.clip.text_model, 'text_model', n)
        except Exception:
            pass

    def forward(self, batch):
        out = self.clip(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            output_hidden_states=True,
            return_dict=True
        )
        try:
            vp = out.vision_model_output.pooler_output
            tp = out.text_model_output.pooler_output
        except Exception:
            vp = out.vision_model_output.last_hidden_state.mean(1)
            tp = out.last_hidden_state[:,0,:]

        vp = self.img_proj(vp).unsqueeze(1)
        tp = self.txt_proj(tp).unsqueeze(1)

        mm = self.querying(vp, tp)

        return {
            't1_logits': self.head_t1(mm),
            't2_logits': self.head_t2(mm),
            't4_logits': self.head_t4(mm)
        }
def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha=None, gamma=2.0):
    """
    inputs: logits (N, C)
    targets: (N,) long
    """
    logpt = -F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(logpt)
    loss = -((1-pt)**gamma) * logpt
    if alpha is not None:
        at = alpha.gather(0, targets)
        loss = loss * at
    return loss.mean()

def compute_masked_losses(outputs: Dict[str,torch.Tensor],
                           labels: Dict[str,torch.Tensor],
                           device: torch.device,
                           class_weights: Dict[str,Optional[torch.Tensor]],
                           use_focal: bool=False,
                           focal_gamma: float=2.0) -> torch.Tensor:
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    for t in ['t1','t2','t4']:
        mask = labels[t] >= 0
        if mask.any():
            logits = outputs[f'{t}_logits'][mask]
            targets = labels[t][mask].to(device)
            weight = class_weights.get(t, None)
            if use_focal:
                if weight is not None:
                    alpha = weight.to(device)
                else:
                    alpha = None
                l = focal_loss(logits, targets, alpha=alpha, gamma=focal_gamma)
            else:
                if weight is not None:
                    l = F.cross_entropy(logits, targets, weight=weight.to(device))
                else:
                    l = F.cross_entropy(logits, targets)
            loss = loss + l
    return loss

def compute_metrics_from_preds(preds: Dict[str,torch.Tensor],
                               labels: Dict[str,torch.Tensor],
                               gate_by_t1: bool = True) -> Dict[str,float]:
    out = {}
    for t in ['t1','t2','t4']:
        pred_logits = preds.get(f'{t}_logits')
        if pred_logits is None:
            continue
        pred = pred_logits.argmax(1)
        lab = labels[t]
        mask = lab >= 0
        if mask.sum().item() == 0:
            continue

        if gate_by_t1 and t != 't1' and ('t1' in preds):
            t1_pred = preds['t1_logits'].argmax(1)
            gate_mask = t1_pred >= 1
            mask = mask & gate_mask

        if mask.sum().item() == 0:
            out[f'{t}_acc'] = float('nan')
            out[f'{t}_prec'] = float('nan')
            out[f'{t}_rec'] = float('nan')
            out[f'{t}_f1'] = float('nan')
            continue

        p = pred[mask].cpu().long()
        r = lab[mask].cpu().long()

        acc = float((p==r).float().mean().item())

        classes = torch.unique(torch.cat([p, r])).tolist()
        precisions = []
        recalls = []
        f1s = []
        for c in classes:
            tp = ((p==c) & (r==c)).sum().item()
            fp = ((p==c) & (r!=c)).sum().item()
            fn = ((p!=c) & (r==c)).sum().item()
            prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
            rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
            f1 = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        out[f'{t}_acc'] = acc
        out[f'{t}_prec'] = float(np.mean(precisions))
        out[f'{t}_rec'] = float(np.mean(recalls))
        out[f'{t}_f1'] = float(np.mean(f1s))
    return out

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             gate_by_t1: bool=True) -> Dict[str,float]:
    model.eval()
    preds = {f'{t}_logits': [] for t in ['t1','t2','t4']}
    labs = {t: [] for t in ['t1','t2','t4']}
    with torch.no_grad():
        for inp, lbl in loader:
            inp = {k:v.to(device) for k,v in inp.items()}
            out = model(inp)
            for t in ['t1','t2','t4']:
                preds[f'{t}_logits'].append(out[f'{t}_logits'].cpu())
                labs[t].append(lbl[t])
    preds = {k: torch.cat(v, 0) for k,v in preds.items()}
    labs  = {k: torch.cat(v, 0) for k,v in labs.items()}
    return compute_metrics_from_preds(preds, labs, gate_by_t1)

def train(model: nn.Module, train_loader: DataLoader, dev_loader: DataLoader, cfg: Dict[str,Any]):
    device = torch.device(cfg['device'])
    model.to(device)

    print("Computing class weights from training dataset...")
    label_counts = defaultdict(Counter)
    for _, lbl in train_loader.dataset: 
        for t in ['t1','t2','t4']:
            v = int(lbl[t].item())
            if v >= 0:
                label_counts[t][v] += 1

    class_weights = {}
    for t in ['t1','t2','t4']:
        cnt = label_counts[t]
        if len(cnt) == 0:
            class_weights[t] = None
            continue
        total = sum(cnt.values())
        freqs = [cnt.get(i,0) for i in range(max(cnt.keys())+1)]
        inv = [total/(f+1e-12) for f in freqs]
        inv = np.array(inv, dtype=np.float32)
        inv = inv / inv.sum() * len(inv)
        class_weights[t] = torch.tensor(inv, dtype=torch.float32)

        print(f"Task {t} class counts: {dict(cnt)} -> weights: {inv.tolist()}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    total_steps = len(train_loader) * cfg['epochs']
    warmup_steps = max(1, int(total_steps * cfg['warmup_frac']))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg['amp'] and device.type=='cuda'))

    os.makedirs(cfg['save_dir'], exist_ok=True)
    best_scores = {'t2_f1': -1.0, 't1_f1': -1.0}
    global_step = 0

    for ep in range(cfg['epochs']):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{cfg['epochs']}", leave=False)
        for batch_idx, (inp, lbl) in enumerate(pbar):
            inp = {k:v.to(device) for k,v in inp.items()}
            lbl = {k:v.to(device) for k,v in lbl.items()}

            with torch.cuda.amp.autocast(enabled=(cfg['amp'] and device.type=='cuda')):
                out = model(inp)
                loss = compute_masked_losses(out, lbl, device, class_weights,
                                             use_focal=cfg['use_focal'], focal_gamma=cfg['focal_gamma'])

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['grad_clip'])
            scaler.step(opt)
            scaler.update()
            sched.step()

            running_loss += loss.item()
            global_step += 1
            if batch_idx % 20 == 0:
                pbar.set_postfix({'loss': running_loss/(batch_idx+1)})
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {ep+1} train_loss: {avg_loss:.6f}")
        metrics = evaluate(model, dev_loader, device, gate_by_t1=cfg['gate_by_t1'])
        print(f"Dev metrics epoch {ep+1}:", metrics)
        t2_f1 = metrics.get('t2_f1', float('nan'))
        t1_f1 = metrics.get('t1_f1', float('nan'))
        criterion_score = t2_f1 if not np.isnan(t2_f1) else t1_f1

        if not np.isnan(criterion_score) and criterion_score > best_scores.get('t2_f1', -1.0):
            best_scores['t2_f1'] = criterion_score
            ckpt = os.path.join(cfg['save_dir'], f"best_t2_epoch{ep+1}.pt")
            torch.save({'model_state': model.state_dict(),
                        'cfg': cfg,
                        'epoch': ep+1,
                        'metrics': metrics}, ckpt)
            print("Saved best_t2 checkpoint:", ckpt)
        last_ckpt = os.path.join(cfg['save_dir'], f"last_epoch{ep+1}.pt")
        torch.save({'model_state': model.state_dict(),
                    'cfg': cfg,
                    'epoch': ep+1,
                    'metrics': metrics}, last_ckpt)

    print("Training finished. Best t2_f1:", best_scores['t2_f1'])

def build_dataloaders(data_cfg, clip_model, batch_size, max_text_len, num_workers, pin_memory, image_size):
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

    train_ds = CrisisDataset(train_rows, processor, max_text_len, image_size)
    dev_ds   = CrisisDataset(dev_rows, processor, max_text_len, image_size)

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
    parser.add_argument('--unfreeze_last_n', type=int, default=DEFAULTS['unfreeze_last_n_clip_layers'])
    parser.add_argument('--use_focal', action='store_true')
    parser.add_argument('--focal_gamma', type=float, default=DEFAULTS['focal_gamma'])
    parser.add_argument('--amp', action='store_true')
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
        'image_size': DEFAULTS['image_size'],
        'warmup_frac': DEFAULTS['warmup_frac'],
        'unfreeze_last_n': args.unfreeze_last_n,
        'use_focal': args.use_focal,
        'focal_gamma': args.focal_gamma,
        'weight_decay': DEFAULTS['weight_decay'],
        'grad_clip': DEFAULTS['grad_clip'],
        'amp': args.amp,
        'gate_by_t1': DEFAULTS['gate_by_t1']
    }

    print("Device:", cfg['device'])
    train_loader, dev_loader = build_dataloaders(
        DATA_CONFIG, cfg['clip_model'], cfg['batch_size'], cfg['max_text_len'],
        cfg['num_workers'], cfg['pin_memory'], cfg['image_size']
    )
    print("Train batches:", len(train_loader), "| Dev batches:", len(dev_loader))

    model = MultimodalMultiTask(cfg['clip_model'],
                                embed_dim=512,
                                freeze_backbone=True,
                                unfreeze_last_n=cfg['unfreeze_last_n'])
    train(model, train_loader, dev_loader, cfg)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
