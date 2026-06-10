"""
UAV Predictive Diagnostics API v3
===================================
Dual-backend: loads PyTorch model (uav_model.pt) if available,
otherwise falls back to sklearn ensemble (uav_model_sklearn.pkl).
 - 59 features / 40 timestep rolling window
 - RobustScaler from scaler.pkl
 - 5-class fault classification
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pickle, os, collections
from scipy import stats

app = FastAPI(title="UAV Predictive Diagnostics API v3")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

BASE   = os.path.dirname(os.path.abspath(__file__))
print("=" * 55)
print("  UAV AI MICROSERVICE v3")
print("=" * 55)

# ── Load model metadata ─────────────────────────────────────────────────────
meta_path = os.path.join(BASE, 'model_meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    N_FEATURES     = meta.get('n_features', 59)
    SEQ_LEN        = meta.get('seq_len', 40)
    N_CLASSES      = meta.get('n_classes', 5)
    BACKEND        = meta.get('backend', 'sklearn')
    N_FEAT_FLAT    = meta.get('n_features_flat', N_FEATURES * 5)  # flat+stats for sklearn
    print(f"[META] backend={BACKEND}  n_features={N_FEATURES}  seq_len={SEQ_LEN}")
else:
    N_FEATURES, SEQ_LEN, N_CLASSES = 59, 40, 5
    BACKEND, N_FEAT_FLAT = 'sklearn', 59 * 5
    print("[META] model_meta.pkl not found — using defaults")

# ── Try PyTorch backend ─────────────────────────────────────────────────────
model_pytorch = None
if BACKEND == 'pytorch':
    try:
        import torch
        import torch.nn as nn

        class UAVFaultNet(nn.Module):
            def __init__(self, n_feat, seq_len, n_cls):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(n_feat, 128, kernel_size=5, padding=2),
                    nn.BatchNorm1d(128), nn.GELU(),
                    nn.Conv1d(128, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
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
                    nn.Linear(128, 64),  nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(64, n_cls),
                )

            def forward(self, x):
                xc = self.conv(x.transpose(1, 2)).transpose(1, 2)
                xg, _ = self.gru1(xc);  xg = self.drop1(xg)
                xg, _ = self.gru2(xg);  xg = self.drop2(xg)
                xa, _ = self.attn(xg, xg, xg)
                xa = self.norm(xg + self.drop3(xa))
                return self.head(xa.mean(dim=1))

        for fname in ['uav_model.pt']:
            pt_path = os.path.join(BASE, fname)
            if os.path.exists(pt_path):
                ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
                model_pytorch = UAVFaultNet(
                    ckpt.get('n_features', N_FEATURES),
                    ckpt.get('seq_len', SEQ_LEN),
                    ckpt.get('n_classes', N_CLASSES),
                ).to('cpu')
                model_pytorch.load_state_dict(ckpt['model_state_dict'])
                model_pytorch.eval()
                print(f"[MODEL] PyTorch loaded: {fname}")
                print(f"[MODEL] val_acc={ckpt.get('best_val_acc',0):.2f}%  "
                      f"test_acc={ckpt.get('test_acc',0):.2f}%")
                break
    except Exception as e:
        print(f"[MODEL] PyTorch unavailable: {e.__class__.__name__} — falling back to sklearn")
        BACKEND = 'sklearn'

# ── Sklearn backend ─────────────────────────────────────────────────────────
model_sklearn = None
if BACKEND == 'sklearn' or model_pytorch is None:
    sk_path = os.path.join(BASE, 'uav_model_sklearn.pkl')
    if os.path.exists(sk_path):
        try:
            with open(sk_path, 'rb') as f:
                sk_data = pickle.load(f)
            model_sklearn = sk_data['model']
            N_FEAT_FLAT   = sk_data.get('n_features_flat', N_FEATURES * 5)
            BACKEND       = 'sklearn'
            print(f"[MODEL] Sklearn ensemble loaded  test_acc={sk_data.get('test_acc',0):.2f}%")
        except Exception as e:
            print(f"[MODEL] Sklearn load failed: {e}")

if model_pytorch is None and model_sklearn is None:
    print("[MODEL] ⚠ No model found — run: python train_model.py")

# ── Scaler ──────────────────────────────────────────────────────────────────
scaler = None
sp = os.path.join(BASE, 'scaler.pkl')
if os.path.exists(sp):
    try:
        with open(sp, 'rb') as f:
            scaler = pickle.load(f)
        print(f"[SCALER] Loaded ({scaler.__class__.__name__}, {N_FEATURES} features)")
    except Exception as e:
        print(f"[SCALER] Failed: {e}")

if scaler is None:
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaler.fit(np.random.rand(200, N_FEATURES))
    print("[SCALER] ⚠ Dummy scaler — run train_model.py!")

# ── Prediction smoothing ────────────────────────────────────────────────────
pred_buffer = collections.deque(maxlen=5)

FAULT_CLASSES = {
    0: {"name": "nominal",         "label": "NOMINAL",           "color": "#10b981"},
    1: {"name": "vibration_fault", "label": "VIBRATION FAULT",   "color": "#f97316"},
    2: {"name": "sensor_drift",    "label": "SENSOR DRIFT",      "color": "#a855f7"},
    3: {"name": "gps_fault",       "label": "GPS / COMMS FAULT", "color": "#eab308"},
    4: {"name": "battery_fault",   "label": "BATTERY FAULT",     "color": "#ef4444"},
}

# ── 59 Feature Names (exact order used during training) ─────────────────────
FEATURE_NAMES = [
    'DesRoll', 'Roll', 'DesPitch', 'Pitch', 'DesYaw', 'Yaw', 'ErrRP', 'ErrYaw',
    'Volt', 'VoltR', 'Curr', 'CurrTot', 'Temp', 'Res', 'RemPct',
    'Alt', 'Press', 'CRt',
    'NSats', 'HDop', 'Spd',
    'GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ', 'IMU_Temp',
    'MagX', 'MagY', 'MagZ', 'Mag_Health',
    'LiftMax', 'MotBatVolt', 'MotBatRes', 'ThLimit',
    'TPD', 'PD', 'DVD', 'VD',
    'RDes', 'R_Rate', 'PDes', 'P_Rate', 'YDes', 'Y_Rate', 'ADes', 'A_Rate',
    'VibeX', 'VibeY', 'VibeZ', 'Clip',
    'VN', 'VE', 'VD_ekf', 'PN', 'PE', 'OH',
    'ThO',
]

# ── Sensor group mapping for human-readable explainability ──────────────────
FEATURE_GROUPS = {}
for _fn in FEATURE_NAMES:
    if _fn in ('DesRoll','Roll','DesPitch','Pitch','DesYaw','Yaw','ErrRP','ErrYaw'):
        FEATURE_GROUPS[_fn] = 'Attitude'
    elif _fn in ('Volt','VoltR','Curr','CurrTot','Temp','Res','RemPct'):
        FEATURE_GROUPS[_fn] = 'Battery'
    elif _fn in ('Alt','Press','CRt'):
        FEATURE_GROUPS[_fn] = 'Barometer'
    elif _fn in ('NSats','HDop','Spd'):
        FEATURE_GROUPS[_fn] = 'GPS'
    elif _fn in ('GyrX','GyrY','GyrZ','AccX','AccY','AccZ','IMU_Temp'):
        FEATURE_GROUPS[_fn] = 'IMU'
    elif _fn in ('MagX','MagY','MagZ','Mag_Health'):
        FEATURE_GROUPS[_fn] = 'Magnetometer'
    elif _fn in ('LiftMax','MotBatVolt','MotBatRes','ThLimit'):
        FEATURE_GROUPS[_fn] = 'Motor'
    elif _fn in ('TPD','PD','DVD','VD'):
        FEATURE_GROUPS[_fn] = 'Position'
    elif _fn in ('RDes','R_Rate','PDes','P_Rate','YDes','Y_Rate','ADes','A_Rate'):
        FEATURE_GROUPS[_fn] = 'Rate Control'
    elif _fn in ('VibeX','VibeY','VibeZ','Clip'):
        FEATURE_GROUPS[_fn] = 'Vibration'
    elif _fn in ('VN','VE','VD_ekf','PN','PE','OH'):
        FEATURE_GROUPS[_fn] = 'EKF'
    else:
        FEATURE_GROUPS[_fn] = 'Propulsion'

def health_from_class(cls: int, conf: float) -> float:
    base    = {0: 98.5, 1: 72.0, 2: 62.0, 3: 54.0, 4: 20.0}
    penalty = {0: 0.00, 1: 0.12, 2: 0.15, 3: 0.12, 4: 0.15}
    return round(float(max(0.0, min(100.0, base[cls] - penalty[cls] * conf))), 1)

def compute_explainability(model, arr_scaled, predicted_class):
    """
    Feature Ablation XAI — REAL model-derived explainability using ONLY forward passes.
    
    For each sensor group, we zero out all features in that group and re-run the model.
    The drop in the predicted class probability tells us how important that group is.
    This is mathematically equivalent to Occlusion Sensitivity / Feature Ablation
    (used by Microsoft InterpretML, Captum, SHAP KernelExplainer).
    
    Memory: same as a single forward pass (no backward, no gradient storage).
    """
    try:
        import torch

        with torch.no_grad():
            # Baseline: original prediction probability
            inp_base = torch.from_numpy(arr_scaled).unsqueeze(0)
            probs_base = torch.softmax(model(inp_base), dim=1)[0].cpu().numpy()
            base_prob = float(probs_base[predicted_class])

            # Group features by sensor subsystem for efficient ablation
            GROUP_INDICES = {}
            for i, fn in enumerate(FEATURE_NAMES):
                grp = FEATURE_GROUPS.get(fn, 'Other')
                if grp not in GROUP_INDICES:
                    GROUP_INDICES[grp] = []
                GROUP_INDICES[grp].append(i)

            # Ablate each group: zero out its features, measure probability drop
            group_impacts = []
            for grp_name, indices in GROUP_INDICES.items():
                ablated = arr_scaled.copy()
                ablated[:, indices] = 0.0  # zero out entire group across all timesteps
                inp_abl = torch.from_numpy(ablated).unsqueeze(0)
                probs_abl = torch.softmax(model(inp_abl), dim=1)[0].cpu().numpy()
                drop = base_prob - float(probs_abl[predicted_class])
                # Only keep positive drops (feature removal reduced confidence = feature was important)
                if drop > 0.001:
                    group_impacts.append((grp_name, drop, indices))

            if not group_impacts:
                return []

            # Sort groups by impact
            group_impacts.sort(key=lambda x: x[1], reverse=True)

            # For top 3 groups, drill down to individual features
            results = []
            for grp_name, grp_drop, indices in group_impacts[:3]:
                for idx in indices:
                    ablated = arr_scaled.copy()
                    ablated[:, idx] = 0.0
                    inp_abl = torch.from_numpy(ablated).unsqueeze(0)
                    probs_abl = torch.softmax(model(inp_abl), dim=1)[0].cpu().numpy()
                    feat_drop = base_prob - float(probs_abl[predicted_class])
                    if feat_drop > 0.001:
                        results.append((FEATURE_NAMES[idx], grp_name, feat_drop))

            if not results:
                return []

            # Normalize to percentages
            total_drop = sum(r[2] for r in results)
            results_pct = [(r[0], r[1], round((r[2] / total_drop) * 100, 1)) for r in results]
            results_pct.sort(key=lambda x: x[2], reverse=True)

            return [{"feature": r[0], "group": r[1], "impact_pct": r[2]} for r in results_pct[:5]]

    except Exception as e:
        print(f"[XAI] Ablation error: {e}")
        return []

def sk_seq_to_flat(seq: np.ndarray) -> np.ndarray:
    """Convert (SEQ_LEN, N_FEATURES) → flat+stats vector for sklearn."""
    flat = seq[-1]            # (59,)  current frame
    mu   = seq.mean(axis=0)  # (59,)  rolling mean
    sd   = seq.std(axis=0)   # (59,)  rolling std
    mx   = seq.max(axis=0)   # (59,)  rolling max
    mn   = seq.min(axis=0)   # (59,)  rolling min
    return np.concatenate([flat, mu, sd, mx, mn]).reshape(1, -1)  # (1, 295)

# ── Schemas ─────────────────────────────────────────────────────────────────
class TelemetryPayload(BaseModel):
    sequence:    list[list[float]]
    force_class: Optional[int] = None   # if set, compute saliency for this class on real data

class RULPayload(BaseModel):
    health_history: list[float]

# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status":     "ok",
        "backend":    BACKEND,
        "model":      model_pytorch is not None or model_sklearn is not None,
        "n_features": N_FEATURES,
        "seq_len":    SEQ_LEN,
        "scaler":     scaler.__class__.__name__,
    }


@app.post("/predict")
async def predict(payload: TelemetryPayload):
    if model_pytorch is None and model_sklearn is None:
        return {"error": "No model loaded. Run: python train_model.py"}
    try:
        arr = np.array(payload.sequence, dtype=np.float32)
        if arr.shape != (SEQ_LEN, N_FEATURES):
            return {"error": f"Expected ({SEQ_LEN},{N_FEATURES}), got {arr.shape}"}

        arr_scaled = np.clip(scaler.transform(arr), -5.0, 5.0).astype(np.float32)

        if model_pytorch is not None:
            import torch
            with torch.no_grad():
                inp   = torch.from_numpy(arr_scaled).unsqueeze(0)
                probs = torch.softmax(model_pytorch(inp), dim=1)[0].cpu().numpy()

            raw_class  = int(np.argmax(probs))
            confidence = float(np.max(probs) * 100)

            explainability = []
            saliency_class = payload.force_class if (payload.force_class is not None and payload.force_class != 0) else raw_class
            if saliency_class != 0:
                explainability = compute_explainability(model_pytorch, arr_scaled, saliency_class)

        else:
            flat  = sk_seq_to_flat(arr_scaled)
            probs = model_sklearn.predict_proba(flat)[0]
            raw_class  = int(np.argmax(probs))
            confidence = float(np.max(probs) * 100)

        all_probs  = {str(i): round(float(probs[i]) * 100, 2) for i in range(N_CLASSES)}

        pred_buffer.append(raw_class)
        smooth_class = raw_class
        if len(pred_buffer) == 5:
            counts = [pred_buffer.count(i) for i in range(N_CLASSES)]
            smooth_class = int(np.argmax(counts))

        fi     = FAULT_CLASSES.get(smooth_class, FAULT_CLASSES[0])
        health = health_from_class(smooth_class, confidence)

        return {
            "status":          fi["name"],
            "status_label":    fi["label"],
            "status_color":    fi["color"],
            "fault_class":     smooth_class,
            "raw_frame_class": raw_class,
            "confidence":      f"{confidence:.2f}%",
            "health_index":    health,
            "class_probs":     all_probs,
            "backend":         BACKEND,
            "explainability":  explainability if model_pytorch is not None else [],
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}


@app.post("/rul")
async def rul(payload: RULPayload):
    try:
        h = payload.health_history
        if len(h) < 8:
            return {"rul_minutes": None, "trend": "insufficient_data"}
        x = np.arange(len(h), dtype=float)
        y = np.array(h, dtype=float)
        slope, intercept, r, p, se = stats.linregress(x, y)
        if slope >= 0:
            return {"rul_minutes": None, "trend": "stable_or_improving",
                    "slope": round(slope, 4)}
        pts_to_zero = -y[-1] / slope if slope != 0 else float('inf')
        rul_min     = round(pts_to_zero / 240, 1)
        trend = "critical" if rul_min < 5 else "warning" if rul_min < 15 else "stable"
        return {
            "rul_minutes":    min(999.0, max(0.0, rul_min)),
            "trend":          trend,
            "slope":          round(slope, 4),
            "r_squared":      round(r**2, 3),
            "current_health": round(float(y[-1]), 1),
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)