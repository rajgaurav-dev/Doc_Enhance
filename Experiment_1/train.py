import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    get_matched_pairs,
    split_pairs,
    DocumentDataset
)

from loss import (
    l1_loss,
    ssim_loss,
    illumination_loss,
    PerceptualLoss
)

from utils import freeze_nafnet_layers, load_nafnet
# --------------------------------------------------
# CONFIG
# --------------------------------------------------
INPUT_DIR = "/content/raj_drive/MyDrive/Apps10x_data/Input"
GT_DIR    = "/content/raj_drive/MyDrive/Apps10x_data/GroundTruth"

PATCH_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 40
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# DATA
# --------------------------------------------------
pairs = get_matched_pairs(INPUT_DIR, GT_DIR)
train_pairs, val_pairs, test_pairs = split_pairs(pairs)

train_ds = DocumentDataset(INPUT_DIR, GT_DIR, train_pairs, training=True)
val_ds   = DocumentDataset(INPUT_DIR, GT_DIR, val_pairs, training=False)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = load_nafnet(
    weights_path="experiments/pretrained_models/NAFNet-SIDD-width64.pth",
    device=DEVICE,
    evluation = True
)

# Freeze early layers if needed
freeze_nafnet_layers(model, freeze_encoders=2)

model.to(DEVICE)

percep = PerceptualLoss().to(DEVICE)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)


# --------------------------------------------------
# TRAIN LOOP
# --------------------------------------------------
best_val_loss = float("inf")

for epoch in range(EPOCHS):

    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] TRAIN")

    for inp, gt in pbar:
        inp = inp.to(DEVICE)
        gt  = gt.to(DEVICE)

        out = model(inp)

        loss = (
            l1_loss(out, gt)
            + 0.2 * ssim_loss(out, gt)
            + 0.1 * percep(out, gt)
            + 0.1 * illumination_loss(out, gt)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss /= len(train_loader)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] VAL")

        for inp, gt in pbar:
            inp = inp.to(DEVICE)
            gt  = gt.to(DEVICE)

            out = model(inp)

            loss = (
                l1_loss(out, gt)
                + 0.2 * ssim_loss(out, gt)
                + 0.1 * percep(out, gt)
                + 0.1 * illumination_loss(out, gt)
            )

            val_loss += loss.item()
            pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
    )

    # ---------------- UNFREEZE ----------------
    if epoch == 10:
        print("Unfreezing all layers & lowering LR")
        for p in model.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # ---------------- SAVE BEST ----------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "nafnet_document_best.pth")
        print("Saved best model")
