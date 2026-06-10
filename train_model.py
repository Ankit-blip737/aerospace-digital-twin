"""
============================================================
 UAV Predictive Degradation Model — Training Script v3
 PRIMARY   : PyTorch  (Conv1D + BiGRU + Attention)
 FALLBACK  : scikit-learn  (RandomForest + GradientBoosting)
             — works on any machine, no DLL issues
============================================================
 Run from project root:
   python train_model.py
============================================================
"""

import os, sys, pickle, warnings, time, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline

SEED       = 42
np.random.seed(SEED)
BASE       = os.path.dirname(os.path.abspath(__file__))
SEQ_LEN    = 40
N_FEATURES = 59
N_CLASSES  = 5

# ── Try PyTorch; fall back to sklearn if DLLs fail ────────────────────────
USE_PYTORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _ = torch.zeros(2, 3)   # force actual DLL load
    USE_PYTORCH = True
    DEVICE = torch.device('cpu')
    print("  Backend: PyTorch (Deep Learning)")
except Exception as e:
    print(f"  PyTorch unavailable ({e.__class__.__name__})")
    print("  Backend: scikit-learn (Machine Learning Ensemble)")

print("=" * 62)
print(f"  UAV DEGRADATION MODEL — {'PyTorch' if USE_PYTORCH else 'scikit-learn'} backend")
print(f"  Features: {N_FEATURES}  |  Seq len: {SEQ_LEN}  |  Classes: {N_CLASSES}")
print("=" * 62)

# ─────────────────────────────────────────────────────────────────────────────
#  1.  LOAD ALL DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading datasets …")

def load_csv(rel_path, dedup_cols=False):
    full = os.path.join(BASE, rel_path)
    if not os.path.exists(full):
        print(f"  [WARN] Not found: {full}")
        return pd.DataFrame()
    df = pd.read_csv(full, low_memory=False)
    if dedup_cols:
        df = df.loc[:, ~df.columns.duplicated()]
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    print(f"  {os.path.basename(rel_path):42s}  {len(df):6d} rows")
    return df

att  = load_csv('dataset/ATT/ALL_FAIL_LOG_ATT.csv')
baro = load_csv('dataset/BARO/ALL_FAIL_LOG_BARO.csv')
bat  = load_csv('dataset/BAT/ALL_FAIL_LOG_BAT_0.csv')
ctun = load_csv('dataset/CTUN/ALL_FAIL_LOG_CTUN.csv')
gps  = load_csv('dataset/GPS/ALL_FAIL_LOG_GPS_0.csv')
imu  = load_csv('dataset/IMU/ALL_FAIL_LOG_IMU_0_Random.csv')
mag  = load_csv('dataset/MAG/ALL_FAIL_LOG_MAG_0.csv')
motb = load_csv('dataset/MOTB/ALL_FAIL_LOG_MOTB.csv')
pscd = load_csv('dataset/PSCD/ALL_FAIL_LOG_PSCD.csv')
rate = load_csv('dataset/RATE/ALL_FAIL_LOG_RATE.csv')
vibe = load_csv('dataset/VIBE/ALL_FAIL_LOG_VIBE_0_Random.csv', dedup_cols=True)
xkf1 = load_csv('dataset/XKF1/ALL_FAIL_LOG_XKF1_0_Random.csv')

# ─────────────────────────────────────────────────────────────────────────────
#  2.  BUILD 59-FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Building aligned feature matrix …")

N = len(att)
LABEL_COL = 'lables'

def get_val(df, idx, col, default=0.0):
    if df.empty or col not in df.columns:
        return default
    mapped = int((idx / N) * len(df)) % len(df)
    try:
        v = float(df.iloc[mapped].get(col, default))
        return v if np.isfinite(v) else default
    except Exception:
        return default

X_list, y_list = [], []

for i in range(N):
    g = lambda df, col, d=0.0: get_val(df, i, col, d)
    row = [
        g(att,'DesRoll'), g(att,'Roll'), g(att,'DesPitch'), g(att,'Pitch'),
        g(att,'DesYaw'),  g(att,'Yaw'),  g(att,'ErrRP'),    g(att,'ErrYaw'),
        g(bat,'Volt'),    g(bat,'VoltR'), g(bat,'Curr'),    g(bat,'CurrTot'),
        g(bat,'Temp'),    g(bat,'Res'),   g(bat,'RemPct'),
        g(baro,'Alt'),    g(baro,'Press'), g(baro,'CRt'),
        g(gps,'NSats'),   g(gps,'HDop'),  g(gps,'Spd'),
        g(imu,'abGyrX'),  g(imu,'abGyrY'), g(imu,'abGyrZ'),
        g(imu,'abAccX'),  g(imu,'abAccY'), g(imu,'abAccZ'), g(imu,'abT'),
        g(mag,'MagX'),    g(mag,'MagY'),  g(mag,'MagZ'),   g(mag,'Health'),
        g(motb,'LiftMax'), g(motb,'BatVolt'), g(motb,'BatRes'), g(motb,'ThLimit'),
        g(pscd,'TPD'),    g(pscd,'PD'),   g(pscd,'DVD'),   g(pscd,'VD'),
        g(rate,'RDes'),   g(rate,'R'),    g(rate,'PDes'),  g(rate,'P'),
        g(rate,'YDes'),   g(rate,'Y'),    g(rate,'ADes'),  g(rate,'A'),
        g(vibe,'VibeX'),  g(vibe,'VibeY'), g(vibe,'VibeZ'), g(vibe,'Clip'),
        g(xkf1,'VN'),     g(xkf1,'VE'),  g(xkf1,'VD'),
        g(xkf1,'PN'),     g(xkf1,'PE'),  g(xkf1,'OH'),
        g(ctun,'ThO'),
    ]
    assert len(row) == N_FEATURES
    X_list.append(row)
    y_list.append(int(att.iloc[i].get(LABEL_COL, 0)))

    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{N} rows processed …")

X_raw = np.array(X_list, dtype=np.float32)
y_raw = np.array(y_list, dtype=np.int64)
cls_names = ['Nominal', 'Vibration', 'Sensor Drift', 'GPS Fault', 'Battery Fault']

print(f"\n  Feature matrix : {X_raw.shape}")
for cls, cnt in zip(*np.unique(y_raw, return_counts=True)):
    print(f"  Label {cls} ({cls_names[cls]:15s}): {cnt:5d}  ({cnt/len(y_raw)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
#  3.  SCALE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Scaling …")

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_raw)
X_scaled = np.clip(X_scaled, -5.0, 5.0).astype(np.float32)

with open(os.path.join(BASE, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print(f"  Scaler saved → scaler.pkl")

# ─────────────────────────────────────────────────────────────────────────────
#  4.  SEQUENCE (for deep models) or FLATTEN WITH ROLLING STATS (for sklearn)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Creating features …")

if USE_PYTORCH:
    # Sliding window sequences for GRU/Conv1D
    def make_sequences(X, y, seq_len):
        Xs = np.lib.stride_tricks.sliding_window_view(X, (seq_len, X.shape[1]))[::1, 0]
        ys = y[seq_len:]
        return Xs.astype(np.float32), ys.astype(np.int64)
    X_feat, y_feat = make_sequences(X_scaled, y_raw, SEQ_LEN)
    print(f"  Sequences : {X_feat.shape}")
else:
    # For sklearn: flatten seq + add rolling mean/std for temporal context
    def make_seq_features(X, y, seq_len):
        feats, labels = [], []
        for i in range(seq_len, len(X)):
            window = X[i - seq_len:i]   # (seq_len, n_feat)
            flat   = window[-1]          # last frame = current state (59,)
            mu     = window.mean(axis=0) # rolling mean (59,)
            sd     = window.std(axis=0)  # rolling std  (59,)
            mx     = window.max(axis=0)  # rolling max  (59,)
            mn     = window.min(axis=0)  # rolling min  (59,)
            feat   = np.concatenate([flat, mu, sd, mx, mn])  # 295 features
            feats.append(feat)
            labels.append(int(y[i]))
        return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)

    X_feat, y_feat = make_seq_features(X_scaled, y_raw, SEQ_LEN)
    print(f"  Sklearn features (flat+stats): {X_feat.shape}")

# ─────────────────────────────────────────────────────────────────────────────
#  5.  SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Splitting …")

X_tmp, X_test, y_tmp, y_test = train_test_split(
    X_feat, y_feat, test_size=0.15, random_state=SEED, stratify=y_feat)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.15, random_state=SEED, stratify=y_tmp)

print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
cw_arr = compute_class_weight('balanced', classes=np.arange(N_CLASSES), y=y_train)
cw_dict = {i: float(cw_arr[i]) for i in range(N_CLASSES)}
print(f"  Class weights: { {k: round(v,2) for k,v in cw_dict.items()} }")

# ─────────────────────────────────────────────────────────────────────────────
#  6A.  PYTORCH TRAINING
# ─────────────────────────────────────────────────────────────────────────────
if USE_PYTORCH:
    print("\n[6/7] Training PyTorch model …")

    class UAVFaultNet(nn.Module):
        def __init__(self, n_feat, seq_len, n_cls):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_feat, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.Conv1d(128, 64,  kernel_size=3, padding=1),
                nn.BatchNorm1d(64),  nn.GELU(),
                nn.Dropout(0.2),
            )
            self.gru1  = nn.GRU(64,  128, bidirectional=True, batch_first=True)
            self.drop1 = nn.Dropout(0.3)
            self.gru2  = nn.GRU(256,  64, bidirectional=True, batch_first=True)
            self.drop2 = nn.Dropout(0.25)
            self.attn  = nn.MultiheadAttention(128, num_heads=4, dropout=0.1, batch_first=True)
            self.norm  = nn.LayerNorm(128)
            self.drop3 = nn.Dropout(0.15)
            self.head  = nn.Sequential(
                nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(128, 64),  nn.GELU(),           nn.Dropout(0.2),
                nn.Linear(64, n_cls),
            )

        def forward(self, x):
            xc = self.conv(x.transpose(1, 2)).transpose(1, 2)
            xg, _ = self.gru1(xc);  xg = self.drop1(xg)
            xg, _ = self.gru2(xg);  xg = self.drop2(xg)
            xa, _ = self.attn(xg, xg, xg)
            xa = self.norm(xg + self.drop3(xa))
            return self.head(xa.mean(dim=1))

    def to_loader(X, y, batch_size=32, shuffle=True):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_ld = to_loader(X_train, y_train, shuffle=True)
    val_ld   = to_loader(X_val,   y_val,   shuffle=False)
    test_ld  = to_loader(X_test,  y_test,  shuffle=False)

    model      = UAVFaultNet(N_FEATURES, SEQ_LEN, N_CLASSES).to(DEVICE)
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cw_tensor  = torch.tensor(cw_arr, dtype=torch.float32)
    criterion  = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer  = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, mode='max', factor=0.4, patience=5, min_lr=1e-6)

    print(f"  UAVFaultNet: {n_params:,} parameters")

    MAX_EPOCHS, PATIENCE = 80, 12
    best_val_acc, patience_ctr, best_state, history = 0.0, 0, None, []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * len(yb); tc += (model(Xb).argmax(1)==yb).sum().item(); tt += len(yb)
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_ld:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                lg = model(Xb)
                vl += criterion(lg, yb).item()*len(yb); vc += (lg.argmax(1)==yb).sum().item(); vt += len(yb)

        ta = tc/tt*100; va = vc/vt*100
        history.append({'epoch': epoch, 'train_acc': ta, 'val_acc': va})
        print(f"  Ep {epoch:3d}  train={ta:.1f}%  val={va:.1f}%  [{time.time()-t0:.1f}s]")
        scheduler.step(va)

        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            print(f"  ✓ Best val_acc={best_val_acc:.2f}%")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for Xb, yb in test_ld:
            all_p.extend(model(Xb.to(DEVICE)).argmax(1).cpu().tolist())
            all_l.extend(yb.tolist())

    test_acc = accuracy_score(all_l, all_p) * 100
    print(f"\n  Test Accuracy: {test_acc:.2f}%")
    print(classification_report(all_l, all_p, target_names=cls_names))

    torch.save({
        'model_state_dict': best_state,
        'n_features': N_FEATURES, 'seq_len': SEQ_LEN,
        'n_classes':  N_CLASSES,
        'best_val_acc': best_val_acc, 'test_acc': test_acc,
        'history': history,
    }, os.path.join(BASE, 'uav_model.pt'))

    meta = {'n_features': N_FEATURES, 'seq_len': SEQ_LEN, 'n_classes': N_CLASSES,
            'scaler_type': 'RobustScaler', 'backend': 'pytorch'}

# ─────────────────────────────────────────────────────────────────────────────
#  6B.  SKLEARN ENSEMBLE TRAINING (fallback when PyTorch DLL fails)
# ─────────────────────────────────────────────────────────────────────────────
else:
    print("\n[6/7] Training scikit-learn Ensemble …")
    print("  (RandomForest + GradientBoosting + ExtraTrees voting ensemble)")

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=18, min_samples_leaf=2,
        class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=0)

    et = ExtraTreesClassifier(
        n_estimators=300, max_depth=18, min_samples_leaf=2,
        class_weight='balanced', random_state=SEED+1, n_jobs=-1, verbose=0)

    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, random_state=SEED+2, verbose=0)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('et', et), ('gb', gb)],
        voting='soft', n_jobs=1)

    print("  Fitting Random Forest …")
    rf.fit(X_train, y_train)
    rf_val_acc = accuracy_score(y_val, rf.predict(X_val)) * 100
    print(f"  RF val_acc = {rf_val_acc:.2f}%")

    print("  Fitting Extra Trees …")
    et.fit(X_train, y_train)
    et_val_acc = accuracy_score(y_val, et.predict(X_val)) * 100
    print(f"  ET val_acc = {et_val_acc:.2f}%")

    print("  Fitting Gradient Boosting …")
    gb.fit(X_train, y_train)
    gb_val_acc = accuracy_score(y_val, gb.predict(X_val)) * 100
    print(f"  GB val_acc = {gb_val_acc:.2f}%")

    print("  Fitting Voting Ensemble …")
    ensemble.fit(X_train, y_train)
    val_acc  = accuracy_score(y_val,  ensemble.predict(X_val))  * 100
    test_acc = accuracy_score(y_test, ensemble.predict(X_test)) * 100

    print(f"\n  Ensemble val_acc  = {val_acc:.2f}%")
    print(f"  Ensemble test_acc = {test_acc:.2f}%")

    y_pred = ensemble.predict(X_test)
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=cls_names))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    for i, row in enumerate(cm):
        print(f"  {cls_names[i]:18s}", row)

    # Save sklearn ensemble
    model_path = os.path.join(BASE, 'uav_model_sklearn.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': ensemble, 'n_features_flat': X_train.shape[1],
                     'n_features': N_FEATURES, 'seq_len': SEQ_LEN,
                     'test_acc': test_acc, 'val_acc': val_acc}, f)
    print(f"\n  ✓ Sklearn model saved → uav_model_sklearn.pkl")

    meta = {'n_features': N_FEATURES, 'seq_len': SEQ_LEN, 'n_classes': N_CLASSES,
            'scaler_type': 'RobustScaler', 'backend': 'sklearn',
            'n_features_flat': int(X_train.shape[1])}

# ─────────────────────────────────────────────────────────────────────────────
#  7.  SAVE METADATA
# ─────────────────────────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'model_meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\n  ✓ Scaler     → scaler.pkl  (RobustScaler, {N_FEATURES} features)")
print(f"  ✓ Meta       → model_meta.pkl  (backend={meta['backend']})")
print(f"\n  Restart ml_api.py to serve the new model.")
print("\n" + "=" * 62)
print("  TRAINING COMPLETE!")
print("=" * 62)
