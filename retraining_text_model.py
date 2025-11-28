##############################################################
#   RESUME TRAINING FROM PUBLIC KAGGLE DATASET CHECKPOINT    #
##############################################################

import torch, os
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ==========================================================================
#  🔥 LOAD CHECKPOINT FROM PUBLIC KAGGLE DATASET
# ==========================================================================
DATASET_FOLDER = "/kaggle/input/itsacrisis-checkpoints"   # PUBLIC CHECKPOINT SOURCE
RESUME_FILE    = "E23.pt"                                 # << CHANGE THIS IF NEEDED
RESUME_PATH    = f"{DATASET_FOLDER}/{RESUME_FILE}"
assert os.path.exists(RESUME_PATH), f"❌ Missing checkpoint: {RESUME_PATH}"

print(f"\n🔄 Resuming training from:\n{RESUME_PATH}\n")

device="cuda"
model = MODEL().to(device)
model.load_state_dict(torch.load(RESUME_PATH, map_location=device), strict=False)
print("✔ Checkpoint loaded — continuing training\n")

# ==========================================================================
#  RELOAD TRAIN/DEV LIKE ORIGINAL SCRIPT
# ==========================================================================
train_rows, dev_rows = load_all()
stats = compute_class_stats(train_rows)
class_weights = {
    "t1": make_weights(stats["t1"],2),
    "t2": make_weights(stats["t2"],3),
    "t3t":make_weights(stats["t3t"],2),
    "t3s":make_weights(stats["t3s"],3),
    "t4": make_weights(stats["t4"],3)
}

tok = RobertaTokenizerFast.from_pretrained("roberta-base")
train_ds = CRISIS(train_rows, tok)
dev_ds   = CRISIS(dev_rows, tok)

sample_weights=[4 if r["t3s"]>=0 else 1 for r in train_rows]
sampler=WeightedRandomSampler(sample_weights,len(sample_weights),replacement=True)

TL = DataLoader(train_ds,batch_size=8,sampler=sampler,collate_fn=collate)
DL = DataLoader(dev_ds,batch_size=8,shuffle=False,collate_fn=collate)

# ==========================================================================
#  TRAIN SETTINGS (UNCHANGED)
# ==========================================================================
epochs_more=15
opt   = torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-2)
steps = len(TL)*epochs_more
sched = get_cosine_schedule_with_warmup(opt,int(0.03*steps),steps)
scaler=torch.cuda.amp.GradScaler()

TASKS=['t1','t2','t3t','t3s','t4']
print(f"\n🚀 Continuing training for {epochs_more} epochs...\n")

# ==========================================================================
#  TRAIN — METRICS — CONFUSION MATRIX — UPLOAD
# ==========================================================================
for ep in range(1,epochs_more+1):

    # --------------------------- TRAIN ----------------------------------
    model.train(); total=0
    for B,Y in tqdm(TL,desc=f"Resume Epoch {ep}/{epochs_more}"):
        B={k:v.to(device) for k,v in B.items()}
        Y={k:v.to(device) for k,v in Y.items()}
        opt.zero_grad()

        with torch.cuda.amp.autocast():
            out=model(B)
            loss=multi_task_loss(out,Y,class_weights,device)

        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step(); total+=loss.item()

    print(f"\n🟣 Resume Loss = {total/len(TL):.4f}\n")

    # --------------------------- EVAL -----------------------------------
    preds={k:[] for k in TASKS}
    truth={k:[] for k in TASKS}
    model.eval()

    with torch.no_grad():
        for B,Y in DL:
            B={k:v.to(device)for k,v in B.items()}
            out=model(B)
            for k in TASKS:
                mask=(Y[k]>=0)&(Y[k]<NUM_CLASSES[k])
                preds[k]+=out[k][mask].argmax(1).cpu().tolist()
                truth[k]+=Y[k][mask].tolist()

    # =====================================================================
    #  🔥 PRINT COMPLETE METRICS PER TASK *PER CLASS*
    # =====================================================================
    print("\n===================== 🧠 EPOCH EVALUATION SUMMARY =====================\n")
    for k in TASKS:
        y_true=np.array(truth[k]); y_pred=np.array(preds[k])
        cm = confusion_matrix(y_true,y_pred,labels=list(range(NUM_CLASSES[k])))
        report=classification_report(y_true,y_pred,output_dict=True,zero_division=0)

        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  TASK {k.upper()}  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Overall Accuracy: {report['accuracy']:.4f}")
        print("\nPer-class metrics:")
        for c in range(NUM_CLASSES[k]):
            print(f"  Class {c}:  "
                  f"Prec={report[str(c)]['precision']:.4f}   "
                  f"Rec={report[str(c)]['recall']:.4f}     "
                  f"F1={report[str(c)]['f1-score']:.4f}")

        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(cm)

    # ---------------------- SAVE ----------------------
    save_local=f"/kaggle/working/checkpoints_sota/RESUME_E{ep}.pt"
    torch.save(model.state_dict(),save_local)
    print(f"\n💾 Checkpoint saved → {save_local}")

    # ---------------------- UPLOAD ---------------------
    if kagglehub:
        try:
            kagglehub.dataset_upload("yathnehr/itsacrisis-checkpoints","/kaggle/working/checkpoints_sota")
            print("📤 Kaggle dataset updated\n")
        except Exception as e:
            print("⚠ Upload failed:",e)
    else:
        print("⚠ kagglehub unavailable, upload skipped\n")
