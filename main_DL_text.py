#!/usr/bin/env python3
import os
import sys
import argparse
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# CONFIG - edit these paths if needed
# ----------------------------
DATA_CONFIG = {
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
}

# local uploaded paper path (developer note / provenance)
PAPER_PATH = "/mnt/data/emnlp.pdf"

DEFAULTS = {
    'clip_model': 'openai/clip-vit-base-patch32',
    'batch_size': 16,
    'epochs': 8,
    'lr': 3e-4,
    'backbone_lr': 5e-6,
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'save_dir': 'checkpoints_hier_final',
    'max_text_len': 77,
    'num_workers': 0,
    'pin_memory': False,
    'image_size': 224,
    'embed_dim': 512,
}

# mappings
DAMAGE_MAP = {'little_or_no_damage':0,'mild_damage':1,'severe_damage':2}
HUM_LABEL_TO_TASK2 = {
    'not_humanitarian':0,'other_relevant_information':0,
    'affected_individuals':1,'injured_or_dead_people':1,'missing_or_found_people':1,'rescue_volunteering_or_donation_effort':1,
    'infrastructure_and_utility_damage':2,'vehicle_damage':2
}
HUM_LABEL_TO_T3TYPE = {'infrastructure_and_utility_damage':0,'vehicle_damage':1}
HUM_LABEL_TO_T4 = {
    'affected_individuals':0,'injured_or_dead_people':0,'missing_or_found_people':0,
    'rescue_volunteering_or_donation_effort':1,'other_relevant_information':2,'not_humanitarian':2,
    'infrastructure_and_utility_damage':2,'vehicle_damage':2
}

# ----------------------------
# TSV reading + row builder
# ----------------------------
def safe_str(x):
    if x is None: return ''
    if isinstance(x, str): return x.strip()
    return str(x).strip()

def detect_column(header: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in header:
            return c
    return None

def read_rows_from_tsv(path: str):
    if not os.path.exists(path):
        return [], []
    with open(path, 'r', encoding='utf-8') as f:
        raw_header = f.readline().rstrip('\n').split('\t')
        header=[]
        seen={}
        for h in raw_header:
            h=h.strip()
            if h not in seen:
                header.append(h); seen[h]=1
            else:
                new_h=f"{h}__{seen[h]}"; header.append(new_h); seen[h]+=1
        rows=[]
        for line in f:
            parts=line.rstrip('\n').split('\t')
            if len(parts)<len(header): parts += ['']*(len(header)-len(parts))
            rows.append(parts)
    return header, rows

def build_combined_rows(data_cfg: Dict[str,Any]):
    train_rows=[]
    dev_rows=[]
    for key in ['damage','humanitarian','informative']:
        p = data_cfg.get(key, {})
        for split in ['train_tsv','dev_tsv']:
            path = p.get(split)
            if not path: continue
            header, raw_rows = read_rows_from_tsv(path)
            if not header: continue
            label_col = detect_column(header, ['label','label_text_image','label_text','label_image'])
            text_col = detect_column(header, ['tweet_text','tweet','text'])
            if text_col is None:
                text_col = header[3] if len(header)>3 else header[0]
            image_col = detect_column(header, ['image','image_path','image_id'])
            if image_col is None:
                image_col = header[4] if len(header)>4 else (header[1] if len(header)>1 else header[0])
            for parts in raw_rows:
                rec = {header[i]: parts[i] for i in range(len(header))}
                text = safe_str(rec.get(text_col,'')) if text_col else ''
                image = safe_str(rec.get(image_col,'')) if image_col else ''
                label_raw = safe_str(rec.get(label_col,'')).lower() if label_col else ''
                row = {'tweet_text':text,'image_path':image,'t1':-1,'t2':-1,'t3_type':-1,'t3_sev':-1,'t4':-1}
                if key=='damage':
                    if label_raw: row['t3_sev']=DAMAGE_MAP.get(label_raw,-1)
                elif key=='humanitarian':
                    if label_raw:
                        row['t2']=HUM_LABEL_TO_TASK2.get(label_raw,0)
                        row['t3_type']=HUM_LABEL_TO_T3TYPE.get(label_raw,-1)
                        row['t4']=HUM_LABEL_TO_T4.get(label_raw,2)
                elif key=='informative':
                    if label_raw:
                        row['t1'] = 1 if label_raw=='informative' else 0
                if split=='train_tsv': train_rows.append(row)
                else: dev_rows.append(row)
    return train_rows, dev_rows

# ----------------------------
# Dataset class (CLIPProcessor handles both text+image)
# ----------------------------
class CrisisDataset(Dataset):
    def __init__(self, rows: List[Dict[str,Any]], processor: CLIPProcessor, max_text_len=77, image_size=224):
        self.rows = rows
        self.proc = processor
        self.max_text_len = max_text_len
        self.image_size = image_size

    def __len__(self): return len(self.rows)

    def _load_image(self, path):
        try:
            if path and os.path.exists(path):
                img=Image.open(path).convert('RGB')
            else:
                raise FileNotFoundError()
        except Exception:
            img=Image.new('RGB', (self.image_size,self.image_size), (0,0,0))
        return img

    def __getitem__(self, idx):
        r = self.rows[idx]
        text = r.get('tweet_text','') or ''
        img = self._load_image(r.get('image_path',''))
        proc = self.proc(text=[text], images=[img], padding='max_length', truncation=True, max_length=self.max_text_len, return_tensors='pt')
        inputs = {'input_ids': proc['input_ids'].squeeze(0),
                  'attention_mask': proc['attention_mask'].squeeze(0),
                  'pixel_values': proc['pixel_values'].squeeze(0)}
        labels = {'t1': torch.tensor(r.get('t1',-1),dtype=torch.long),
                  't2': torch.tensor(r.get('t2',-1),dtype=torch.long),
                  't3_type': torch.tensor(r.get('t3_type',-1),dtype=torch.long),
                  't3_sev': torch.tensor(r.get('t3_sev',-1),dtype=torch.long),
                  't4': torch.tensor(r.get('t4',-1),dtype=torch.long)}
        return inputs, labels

def collate_batch(batch):
    inputs = [x[0] for x in batch]; labels=[x[1] for x in batch]
    batched_inputs={'input_ids': torch.stack([i['input_ids'] for i in inputs]),
                    'attention_mask': torch.stack([i['attention_mask'] for i in inputs]),
                    'pixel_values': torch.stack([i['pixel_values'] for i in inputs])}
    batched_labels={'t1': torch.stack([l['t1'] for l in labels]),
                    't2': torch.stack([l['t2'] for l in labels]),
                    't3_type': torch.stack([l['t3_type'] for l in labels]),
                    't3_sev': torch.stack([l['t3_sev'] for l in labels]),
                    't4': torch.stack([l['t4'] for l in labels])}
    return batched_inputs, batched_labels

# ----------------------------
# Model (CLIP backbone with lazy projections + heads)
# ----------------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))
    def forward(self,x): return self.net(x)

class HierarchicalMultimodalModel(nn.Module):
    def __init__(self, clip_model_name, embed_dim=512, freeze_backbone=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        # freeze entire CLIP backbone if requested (vision+text)
        if freeze_backbone:
            for p in self.clip.parameters(): p.requires_grad=False

        self.embed_dim = embed_dim
        # lazy modules (will be created on first forward)
        self._img_proj = None
        self._txt_proj = None
        self.head_t1 = None; self.head_t2=None; self.head_t3_type=None; self.head_t3_sev=None; self.head_t4=None

    def _lazy_init(self, img_repr, txt_repr):
        d_img = img_repr.size(-1); d_txt = txt_repr.size(-1)
        # create linear projections + heads on device, with trainable params
        if self._img_proj is None:
            self._img_proj = nn.Linear(d_img, self.embed_dim).to(img_repr.device)
        if self._txt_proj is None:
            self._txt_proj = nn.Linear(d_txt, self.embed_dim).to(txt_repr.device)
        if self.head_t1 is None:
            self.head_t1 = MLPHead(self.embed_dim, out_dim=2).to(img_repr.device)
            self.head_t2 = MLPHead(self.embed_dim, out_dim=3).to(img_repr.device)
            self.head_t3_type = MLPHead(self.embed_dim, out_dim=3).to(img_repr.device)
            self.head_t3_sev = MLPHead(self.embed_dim, out_dim=3).to(img_repr.device)
            self.head_t4 = MLPHead(self.embed_dim, out_dim=3).to(img_repr.device)

    def forward(self, batch):
        # CLIP forward: we ask for image_embeds and text_embeds (pooled)
        out = self.clip(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], pixel_values=batch['pixel_values'], return_dict=True)
        # prefer official fields
        if hasattr(out, 'image_embeds') and hasattr(out, 'text_embeds'):
            img_repr = out.image_embeds
            txt_repr = out.text_embeds
        else:
            # fallbacks
            if hasattr(out, 'vision_model_output') and hasattr(out.vision_model_output, 'last_hidden_state'):
                img_repr = out.vision_model_output.last_hidden_state.mean(1)
            else:
                img_repr = out.last_hidden_state.mean(1)
            if hasattr(out, 'text_model_output') and hasattr(out.text_model_output, 'last_hidden_state'):
                txt_repr = out.text_model_output.last_hidden_state.mean(1)
            else:
                txt_repr = out.last_hidden_state[:,0,:]

        # lazy init trainable heads/projections (ensures correct dims)
        if self._img_proj is None or self._txt_proj is None:
            self._lazy_init(img_repr, txt_repr)

        vp = self._img_proj(img_repr)
        tp = self._txt_proj(txt_repr)
        mm = (vp + tp) * 0.5  # simple fusion

        return {'t1_logits': self.head_t1(mm), 't2_logits': self.head_t2(mm),
                't3_type_logits': self.head_t3_type(mm), 't3_sev_logits': self.head_t3_sev(mm),
                't4_logits': self.head_t4(mm)}

# ----------------------------
# Loss helpers, evaluate
# ----------------------------
def focal_loss_logits(inputs, targets, alpha=None, gamma=2.0, reduction='mean'):
    ce = F.cross_entropy(inputs, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean() if reduction=='mean' else loss.sum()

def compute_class_weights_from_rows(rows: List[Dict[str,Any]], key: str, n_classes: int):
    ctr = Counter([r[key] for r in rows if r.get(key,-1) >= 0])
    if len(ctr)==0: return None
    freqs = np.array([ctr.get(i,0) for i in range(n_classes)], dtype=np.float32)
    inv = (freqs.max()+1e-12) / (freqs + 1.0)
    w = inv / (inv.sum()+1e-12) * len(inv)
    return torch.tensor(w, dtype=torch.float32)

def compute_masked_losses(outputs, labels, device, weights=None, focal_on_t1=True):
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    per_task={}
    # t1
    mask = labels['t1'] >= 0
    if mask.any():
        logits = outputs['t1_logits'][mask]; tg = labels['t1'][mask].to(device)
        w = weights.get('t1') if (weights and 't1' in weights) else None
        l = focal_loss_logits(logits, tg, alpha=(w.to(device) if w is not None else None)) if focal_on_t1 else F.cross_entropy(logits, tg, weight=(w.to(device) if w is not None else None))
        loss = loss + l; per_task['t1']=l
    else: per_task['t1']=torch.tensor(0.0, device=device)
    # t2
    mask = labels['t2'] >= 0
    if mask.any():
        logits = outputs['t2_logits'][mask]; tg = labels['t2'][mask].to(device)
        w = weights.get('t2') if (weights and 't2' in weights) else None
        l = F.cross_entropy(logits, tg, weight=(w.to(device) if w is not None else None))
        loss = loss + l; per_task['t2']=l
    else: per_task['t2']=torch.tensor(0.0, device=device)
    # t3_type
    mask = labels['t3_type'] >= 0
    if mask.any():
        logits = outputs['t3_type_logits'][mask]; tg = labels['t3_type'][mask].to(device)
        w = weights.get('t3_type') if (weights and 't3_type' in weights) else None
        l = F.cross_entropy(logits, tg, weight=(w.to(device) if w is not None else None))
        loss = loss + l; per_task['t3_type']=l
    else: per_task['t3_type']=torch.tensor(0.0, device=device)
    # t3_sev
    mask = labels['t3_sev'] >= 0
    if mask.any():
        logits = outputs['t3_sev_logits'][mask]; tg = labels['t3_sev'][mask].to(device)
        w = weights.get('t3_sev') if (weights and 't3_sev' in weights) else None
        l = F.cross_entropy(logits, tg, weight=(w.to(device) if w is not None else None))
        loss = loss + l; per_task['t3_sev']=l
    else: per_task['t3_sev']=torch.tensor(0.0, device=device)
    # t4
    mask = labels['t4'] >= 0
    if mask.any():
        logits = outputs['t4_logits'][mask]; tg = labels['t4'][mask].to(device)
        w = weights.get('t4') if (weights and 't4' in weights) else None
        l = F.cross_entropy(logits, tg, weight=(w.to(device) if w is not None else None))
        loss = loss + l; per_task['t4']=l
    else: per_task['t4']=torch.tensor(0.0, device=device)
    return loss, per_task

def evaluate_detailed(model, loader, device):
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
                    all_preds[t].extend(preds); all_trues[t].extend(trues)
    reports={}; cms={}
    for t in ['t1','t2','t3_type','t3_sev','t4']:
        if len(all_trues[t])==0: continue
        print(f"\n--- Task {t} ---")
        reports[t] = classification_report(all_trues[t], all_preds[t], output_dict=True, zero_division=0)
        print(classification_report(all_trues[t], all_preds[t], digits=4, zero_division=0))
        cms[t] = confusion_matrix(all_trues[t], all_preds[t])
        print("Confusion matrix:\n", cms[t])
    return reports, cms

# ----------------------------
# DATALOADER builder
# ----------------------------
def build_dataloaders(data_cfg, clip_model_name, batch_size, max_text_len, num_workers, pin_memory, use_sampler=False):
    print("=== Scanning data distributions (train TSVs) ===")
    for k,p in data_cfg.items():
        ctr = scan_and_print_distribution(p.get('train_tsv',''), 'label')
        print(k, ctr)
    train_rows, dev_rows = build_combined_rows(data_cfg)
    print(f"Train rows: {len(train_rows)}, Dev rows: {len(dev_rows)}")
    if len(train_rows)==0:
        print("ERROR: no training rows found. Check DATA_CONFIG paths. Exiting."); sys.exit(1)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    train_ds = CrisisDataset(train_rows, processor, max_text_len, DEFAULTS['image_size'])
    dev_ds = CrisisDataset(dev_rows, processor, max_text_len, DEFAULTS['image_size'])
    train_sampler=None
    if use_sampler:
        ctr = Counter([r['t3_sev'] for r in train_rows if r.get('t3_sev',-1) >=0])
        if sum(ctr.values())==0:
            ctr = Counter([r['t1'] for r in train_rows if r.get('t1',-1) >=0])
        if sum(ctr.values())>0:
            total=sum(ctr.values()); class_weight={c: total/(v+1e-12) for c,v in ctr.items()}
            weights = [float(class_weight.get(r.get('t3_sev', r.get('t1', -1)), 1.0)) for r in train_rows]
            if len(weights)>0:
                train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
                print("Using WeightedRandomSampler.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None), collate_fn=collate_batch, num_workers=num_workers, pin_memory=pin_memory, sampler=train_sampler)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0, pin_memory=pin_memory)
    return train_loader, dev_loader, train_rows

def scan_and_print_distribution(path: str, col: str):
    ctr = Counter()
    if not path or not os.path.exists(path): return ctr
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        if col not in header: return ctr
        idx = header.index(col)
        for line in f:
            parts=line.rstrip('\n').split('\t')
            if idx < len(parts):
                v=safe_str(parts[idx]).lower()
                if v: ctr[v]+=1
    return ctr

# ----------------------------
# TRAIN loop
# ----------------------------
def train(model, train_loader, dev_loader, cfg, train_rows):
    device = torch.device(cfg['device']); model.to(device)
    use_amp = (device.type=='cuda') and cfg.get('use_amp', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    weights={}
    weights['t1']=compute_class_weights_from_rows(train_rows,'t1',2)
    weights['t2']=compute_class_weights_from_rows(train_rows,'t2',3)
    weights['t3_type']=compute_class_weights_from_rows(train_rows,'t3_type',3)
    weights['t3_sev']=compute_class_weights_from_rows(train_rows,'t3_sev',3)
    weights['t4']=compute_class_weights_from_rows(train_rows,'t4',3)
    print("Computed class weights:")
    for k,v in weights.items(): print(k, None if v is None else v.tolist())

    # param groups: use id() to compare parameters robustly
    backbone = [p for n,p in model.named_parameters() if 'clip' in n and p.requires_grad]
    backbone_ids = set(id(p) for p in backbone)
    others = [p for n,p in model.named_parameters() if p.requires_grad and id(p) not in backbone_ids]
    param_groups=[]
    if len(backbone)>0: param_groups.append({'params': backbone, 'lr': cfg['backbone_lr']})
    if len(others)>0: param_groups.append({'params': others, 'lr': cfg['lr']})
    if len(param_groups)==0:
        raise RuntimeError("No trainable params found. Check freeze settings.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
    total_steps = max(1, len(train_loader)*cfg['epochs'])
    warmup_steps = max(1, int(total_steps*0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(cfg['save_dir'], exist_ok=True)
    best_val=-1.0

    for epoch in range(cfg['epochs']):
        model.train(); running_loss=0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for step, (inp, lbl) in pbar:
            inp = {k:v.to(device) for k,v in inp.items()}; lbl = {k:v.to(device) for k,v in lbl.items()}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inp)
                loss, _ = compute_masked_losses(outputs, lbl, device, weights=weights, focal_on_t1=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            running_loss += float(loss.item())
            if (step+1)%20==0: pbar.set_postfix({'loss':f"{(running_loss/(step+1)):.4f}"})
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"\nEpoch {epoch+1} finished. Avg loss: {avg_loss:.6f}")

        # eval
        reports, cms = evaluate_detailed(model, dev_loader, device)
        val_score = 0.0
        if 't2' in reports: val_score = reports['t2'].get('macro avg',{}).get('f1-score',0.0)
        elif 't1' in reports: val_score = reports['t1'].get('macro avg',{}).get('f1-score',0.0)
        print(f"Validation selected score: {val_score:.4f}")
        if val_score > best_val:
            best_val = val_score
            ckpt = os.path.join(cfg['save_dir'], f"best_epoch{epoch+1}.pt")
            torch.save({'model_state': model.state_dict(), 'cfg': cfg, 'epoch': epoch+1, 'val_score': val_score}, ckpt)
            print("Saved best checkpoint:", ckpt)
    print("Training finished. Best val score:", best_val)

# ----------------------------
# CLI / main
# ----------------------------
def main(argv=None):
    # avoid ipykernel argv noise when running in notebook
    if argv is None and 'ipykernel' in sys.modules: argv=[]
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model', default=DEFAULTS['clip_model'])
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--backbone_lr', type=float, default=DEFAULTS['backbone_lr'])
    parser.add_argument('--device', default=DEFAULTS['device'])
    parser.add_argument('--save_dir', default=DEFAULTS['save_dir'])
    parser.add_argument('--num_workers', type=int, default=DEFAULTS['num_workers'])
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    args = parser.parse_args(argv)

    cfg = {'clip_model': args.clip_model, 'batch_size': args.batch_size, 'epochs': args.epochs, 'lr': args.lr, 'backbone_lr': args.backbone_lr, 'device': args.device, 'save_dir': args.save_dir, 'max_text_len': DEFAULTS['max_text_len'], 'num_workers': args.num_workers, 'pin_memory': args.pin_memory, 'use_amp': args.use_amp}
    print("Device:", cfg['device']); print("Paper path:", PAPER_PATH)

    # build dataloaders
    train_loader, dev_loader, train_rows = build_dataloaders(DATA_CONFIG, cfg['clip_model'], cfg['batch_size'], cfg['max_text_len'], cfg['num_workers'], cfg['pin_memory'], use_sampler=args.use_sampler)
    print("Train batches:", len(train_loader), "| Dev batches:", len(dev_loader))

    # create model
    model = HierarchicalMultimodalModel(cfg['clip_model'], embed_dim=DEFAULTS['embed_dim'], freeze_backbone=True)

    # ----------------------------
    # IMPORTANT FIX:
    # If model uses lazy init for heads (created on first forward),
    # perform a single forward with one batch to ensure trainable params exist
    # before optimizer creation. This prevents "No trainable params found".
    # ----------------------------
    try:
        # grab a single batch from train_loader and run a no_grad forward to initialize lazy modules
        sample_iter = iter(train_loader)
        sample_batch = next(sample_iter)  # (inputs, labels)
        sample_inputs, _ = sample_batch
        device = torch.device(cfg['device'])
        model.to(device)
        with torch.no_grad():
            # move sample to device
            sample_inputs = {k: v.to(device) for k, v in sample_inputs.items()}
            # if model.clip is on CPU and device is MPS/CUDA, ensure processor tensors are moved
            _ = model(sample_inputs)
        # done: lazy modules initialized
    except StopIteration:
        # no samples (should have been caught earlier), but ignore
        pass
    except Exception as e:
        print("Warning: initialization forward pass failed (this is non-fatal). Error:", repr(e))

    # enter training loop
    train(model, train_loader, dev_loader, cfg, train_rows)

if __name__ == '__main__':
    main()
