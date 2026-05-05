import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "/kaggle/input/datasets/mohitiiitb/gee-water/gee_water_dataset"
SAVE_DIR = "/kaggle/working/checkpoints"

EPOCHS = 30
BATCH_SIZE = 4
LR = 3e-4
NUM_WORKERS = 0
ACCUM_STEPS = 4

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- AUG ----------------
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
    A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
    A.Normalize(),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])

# ---------------- DATASET ----------------
class WaterDataset(Dataset):
    def __init__(self, img_dir, mask_dir, tf=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.tf = tf
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith(".tif")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]

        img = np.array(Image.open(os.path.join(self.img_dir, fname)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, fname)).convert("L"))

        mask = (mask > 127).astype(np.float32)

        if self.tf:
            out = self.tf(image=img, mask=mask)
            img = out["image"]
            mask = out["mask"].unsqueeze(0)

        return img, mask

# ---------------- LOADERS ----------------
train_loader = DataLoader(
    WaterDataset(f"{ROOT}/img/train", f"{ROOT}/label/train", train_tf),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)

val_loader = DataLoader(
    WaterDataset(f"{ROOT}/img/val", f"{ROOT}/label/val", val_tf),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# ---------------- MODEL ----------------
class SCSE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//16, 1),
            nn.ReLU(),
            nn.Conv2d(c//16, c, 1),
            nn.Sigmoid()
        )
        self.sse = nn.Sequential(
            nn.Conv2d(c, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)

class SpatialBranch(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class WaterFormerLite(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            "efficientnet-b2", in_channels=3, depth=5, weights="imagenet"
        )

        c = self.encoder.out_channels[-1]

        self.spatial = SpatialBranch(c)
        self.attn = SCSE(c)

        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(128, 64, 32, 16, 8),
            n_blocks=5
        )

        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        feats = self.encoder(x)
        enc = feats[-1]
    
        spa = nn.functional.interpolate(
            self.spatial(x),
            size=enc.shape[2:],
            mode="bilinear",
            align_corners=False
        )
    
        x = enc + spa
        x = self.attn(x)
    
        feats[-1] = x
    
        dec = self.decoder(feats)
        return self.head(dec)

model = WaterFormerLite().to(DEVICE)

# ---------------- LOSS ----------------
focal = smp.losses.FocalLoss(mode="binary", gamma=2.0)
tversky = smp.losses.TverskyLoss(mode="binary", alpha=0.3, beta=0.7)

def loss_fn(p, t):
    return 0.5 * focal(p, t) + 0.5 * tversky(p, t)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda")

# ---------------- METRICS ----------------
def compute_metrics(pred, tgt):
    pred = (torch.sigmoid(pred) > 0.5).float()

    tp = (pred * tgt).sum()
    tn = ((1-pred)*(1-tgt)).sum()
    fp = (pred*(1-tgt)).sum()
    fn = ((1-pred)*tgt).sum()

    eps = 1e-6

    iou = (tp + eps) / (tp + fp + fn + eps)
    f1  = (2*tp + eps) / (2*tp + fp + fn + eps)

    total = tp + tn + fp + fn
    po = (tp + tn) / (total + eps)
    pe = (((tp+fp)*(tp+fn) + (fn+tn)*(fp+tn)) / ((total+eps)**2))
    kappa = (po - pe) / (1 - pe + eps)

    return iou.item(), f1.item(), kappa.item()

# ---------------- TRAIN ----------------
best = 0.0

for epoch in range(EPOCHS):
    model.train()
    tl = 0.0

    optimizer.zero_grad(set_to_none=True)

    for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Ep {epoch+1:02d} train")):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            p = model(x)
            loss = loss_fn(p, y) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        tl += loss.item() * ACCUM_STEPS

    tl /= len(train_loader)

    model.eval()
    vl = vi = vf = vk = 0.0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Ep {epoch+1:02d} val"):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                p = model(x)
                loss = loss_fn(p, y)

            iou, f1, kappa = compute_metrics(p, y)

            vl += loss.item()
            vi += iou
            vf += f1
            vk += kappa

    vl /= len(val_loader)
    vi /= len(val_loader)
    vf /= len(val_loader)
    vk /= len(val_loader)

    print(f"{epoch+1:02d} | TL {tl:.4f} | VL {vl:.4f} | IoU {vi:.4f} | F1 {vf:.4f} | Kappa {vk:.4f}")

    if vi > best:
        best = vi
        torch.save(model.state_dict(), f"{SAVE_DIR}/best.pth")
        print(f"  saved best (IoU {best:.4f})")

print("Best IoU:", best)