import express from 'express';
import http from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import fs from 'fs';
import path, { dirname } from 'path';
import { fileURLToPath } from 'url';
import axios from 'axios';
import mongoose from 'mongoose';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*', methods: ['GET', 'POST'] } });

console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('  AEROSPACE DIGITAL TWIN — COMMAND CENTER v2.0   ');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

// ─────────────────────────────────────────────────────────
//  1. MONGODB CONNECTION
// ─────────────────────────────────────────────────────────
if (process.env.MONGODB_URI) {
    mongoose.connect(process.env.MONGODB_URI)
        .then(() => console.log('[DB] MongoDB Atlas connected.'))
        .catch(err => console.warn('[DB] MongoDB connection failed:', err.message));
} else {
    console.warn('[DB] MONGODB_URI not set, skipping database connection.');
}

const telemetrySchema = new mongoose.Schema({
    deviceId:      { type: String, required: true },
    timestamp:     { type: Date,   required: true },
    faultClass:    Number,
    faultName:     String,
    healthIndex:   Number,
    confidence:    String,
    kinematics:    Object,
    propulsion:    Object,
    power:         Object,
    vibration:     Object,
    barometer:     Object,
    gps:           Object,
    rateControl:   Object,
    posControl:    Object,
    ekfState:      Object,
}, {
    timeseries: { timeField: 'timestamp', metaField: 'deviceId', granularity: 'seconds' }
});

let TelemetryModel = null;
try { TelemetryModel = mongoose.model('Telemetry', telemetrySchema); } catch (_) {}

// ─────────────────────────────────────────────────────────
//  2. LOAD ALL 13 DATASETS
// ─────────────────────────────────────────────────────────
const loadCsv = (relPath) => {
    const filePath = path.join(__dirname, relPath);
    if (!fs.existsSync(filePath)) {
        console.warn(`[WARN] Missing dataset: ${filePath}`);
        return { headers: [], rows: [] };
    }
    const lines = fs.readFileSync(filePath, 'utf-8').split('\n').filter(l => l.trim().length > 0);
    const headers = lines[0].split(',').map(h => h.trim());
    const rows = lines.slice(1).map(l => l.split(',').map(v => parseFloat(v) || 0));
    console.log(`[DATA] Loaded ${rows.length} rows from ${path.basename(filePath)} (${headers.length} cols)`);
    return { headers, rows };
};

const DS = {
    fusion: loadCsv('dataset/Fusion_Data.csv'),
    bat:    loadCsv('dataset/BAT/ALL_FAIL_LOG_BAT_0.csv'),
    att:    loadCsv('dataset/ATT/ALL_FAIL_LOG_ATT.csv'),
    baro:   loadCsv('dataset/BARO/ALL_FAIL_LOG_BARO.csv'),
    gps:    loadCsv('dataset/GPS/ALL_FAIL_LOG_GPS_0.csv'),
    imu:    loadCsv('dataset/IMU/ALL_FAIL_LOG_IMU_0_Random.csv'),
    mag:    loadCsv('dataset/MAG/ALL_FAIL_LOG_MAG_0.csv'),
    motb:   loadCsv('dataset/MOTB/ALL_FAIL_LOG_MOTB.csv'),
    pscd:   loadCsv('dataset/PSCD/ALL_FAIL_LOG_PSCD.csv'),
    rate:   loadCsv('dataset/RATE/ALL_FAIL_LOG_RATE.csv'),
    vibe:   loadCsv('dataset/VIBE/ALL_FAIL_LOG_VIBE_0_Random.csv'),
    xkf1:   loadCsv('dataset/XKF1/ALL_FAIL_LOG_XKF1_0_Random.csv'),
    ctun:   loadCsv('dataset/CTUN/ALL_FAIL_LOG_CTUN.csv'),
};

const getRow = (ds, idx) => {
    if (!ds.rows.length) return [];
    return ds.rows[idx % ds.rows.length];
};

console.log('[DATA] All datasets loaded.');

// ─────────────────────────────────────────────────────────
//  3. SWARM CONFIGURATION (5 independent UAVs)
// ─────────────────────────────────────────────────────────
const SWARM_NODES = [
    { id: 'UAV-ALPHA-01', phaseOffset: 0,    cursorOffset: 0 },
    { id: 'UAV-BETA-02',  phaseOffset: 500,  cursorOffset: 980 },
    { id: 'UAV-GAMMA-03', phaseOffset: 1000, cursorOffset: 1960 },
    { id: 'UAV-DELTA-04', phaseOffset: 1500, cursorOffset: 2940 },
    { id: 'UAV-EPSILON-05',phaseOffset:2000, cursorOffset: 3920 },
];

// Per-drone state
const droneState = {};
SWARM_NODES.forEach(node => {
    droneState[node.id] = {
        cursor:          node.cursorOffset,
        historyBuffer:   [],        // 40-frame window for ML
        healthHistory:   [],        // rolling health for RUL
        injectedFault:   null,      // manually injected fault class
        mode:            'patrol',  // patrol | rtl | landed
        rtlProgress:     0,
    };
});

const SEQUENCE_LENGTH = 40;
const FAULT_CLASSES = {
    0: { name: 'nominal',         label: 'NOMINAL',           color: '#10b981' },
    1: { name: 'vibration_fault', label: 'VIBRATION FAULT',   color: '#f97316' },
    2: { name: 'sensor_drift',    label: 'SENSOR DRIFT',      color: '#a855f7' },
    3: { name: 'gps_fault',       label: 'GPS / COMMS FAULT', color: '#eab308' },
    4: { name: 'battery_fault',   label: 'BATTERY FAULT',     color: '#ef4444' },
};

// ─────────────────────────────────────────────────────────
//  4. ML PREDICTION
// ─────────────────────────────────────────────────────────
async function getPrediction(droneId) {
    const state = droneState[droneId];
    if (state.historyBuffer.length < SEQUENCE_LENGTH) return null;
    try {
        const ML_API = process.env.ML_API_URL || 'http://127.0.0.1:8000';
        const res = await axios.post(`${ML_API}/predict`, { sequence: state.historyBuffer }, { timeout: 2000 });
        return res.data;
    } catch {
        return { status: 'nominal', fault_class: 0, health_index: 98.5, confidence: '0%', status_color: '#10b981', status_label: 'NOMINAL' };
    }
}

async function getRUL(droneId) {
    const state = droneState[droneId];
    if (state.healthHistory.length < 10) return null;
    try {
        const ML_API = process.env.ML_API_URL || 'http://127.0.0.1:8000';
        const res = await axios.post(`${ML_API}/rul`, { health_history: state.healthHistory.slice(-120) }, { timeout: 1000 });
        return res.data;
    } catch {
        return null;
    }
}

// ─────────────────────────────────────────────────────────
//  5. TELEMETRY EXTRACTION FROM ALL DATASETS
// ─────────────────────────────────────────────────────────
function extractTelemetry(droneId) {
    const state = droneState[droneId];
    const c = state.cursor;

    // Fusion: timestamp,DesRoll,Roll,DesPitch,Pitch,DesYaw,Yaw,ErrRP,ErrYaw,MagX,MagY,MagZ,abGyrX,abGyrY,abGyrZ,abAccX,abAccY,abAccZ,labels
    const f = getRow(DS.fusion, c);
    const roll    = f[2]  || 0;
    const pitch   = f[4]  || 0;
    const yaw     = f[6]  || 0;
    const errRP   = f[7]  || 0;
    const magX    = f[9]  || 0;
    const magY    = f[10] || 0;
    const magZ    = f[11] || 0;
    const gyrX    = f[12] || 0;
    const gyrY    = f[13] || 0;
    const gyrZ    = f[14] || 0;
    const accX    = f[15] || 0;
    const accY    = f[16] || 0;
    const accZ    = f[17] || 0;
    const fusionLabel = f[18] || 0;

    // BAT: LineNo,TimeUS,Instance,Volt,VoltR,Curr,CurrTot,EnrgTot,Temp,Res,RemPct,labels
    const b = getRow(DS.bat, c);
    const battV   = b[3]  || 12;
    const battVr  = b[4]  || 12;
    const curr    = b[5]  || 0;
    const currTot = b[6]  || 0;
    const enrgTot = b[7]  || 0;
    const batTemp = b[8]  || 25;
    const batRes  = b[9]  || 0;
    const remPct  = b[10] || 100;

    // ATT: LineNo,TimeUS,DesRoll,Roll,DesPitch,Pitch,DesYaw,Yaw,ErrRP,ErrYaw,AEKF,lables
    const att = getRow(DS.att, c);
    const desRoll  = att[2] || 0;
    const desPitch = att[4] || 0;
    const desYaw   = att[6] || 0;
    const aekf     = att[10] || 3;

    // BARO: LineNo,TimeUS,I,Alt,Press,CRt,SMS,Offset,GndTemp,Health,labels
    const baro = getRow(DS.baro, c);
    const baroAlt    = baro[3] || 0;
    const baroPress  = baro[4] || 101325;
    const baroCRt    = baro[5] || 0;
    const baroHealth = baro[9] || 25;

    // GPS: LineNo,TimeUS,I,Status,GMS,GWk,NSats,HDop,Lat,Lng,Alt,Spd,GCrs,VZ,Yaw,U,labels
    const gps = getRow(DS.gps, c);
    const gpsStatus = gps[3] || 0;
    const nSats     = gps[6] || 0;
    const hdop      = gps[7] || 99;
    const droneIdx = SWARM_NODES.findIndex(n => n.id === droneId);
    const angle = (droneIdx / SWARM_NODES.length) * Math.PI * 2 + (c * 0.008);
    
    const baseLat = -35.3632838;
    const baseLng = 149.163061;
    const rawLat = parseFloat(gps[8]) || baseLat;
    const rawLng = parseFloat(gps[9]) || baseLng;
    
    // 1. Calculate macro movement in meters from the origin
    const dxMeters = (rawLng - baseLng) * 91000;
    const dzMeters = (rawLat - baseLat) * -111111;
    
    // 2. Add swarm formation orbit (6m radius)
    const patrolRadiusMeters = 6.0;
    const localX = dxMeters + Math.cos(angle) * patrolRadiusMeters;
    const localZ = dzMeters + Math.sin(angle) * patrolRadiusMeters;
    
    // 3. Convert final position back to Lat/Lng for radar
    const lat = baseLat + (localZ / -111111);
    const lng = baseLng + (localX / 91000);
    
    const gpsAlt    = gps[10] || 0;
    const gpsSpd    = gps[11] || 0;
    
    // Make the GPS yaw point along the orbit tangent
    const rawYaw    = gps[14] || 0;
    const orbitYawDeg = (angle + Math.PI / 2) * (180 / Math.PI);
    const gpsYaw    = parseFloat(rawYaw) + orbitYawDeg;

    // IMU: LineNo,abTimeUS,abI,abGyrX,abGyrY,abGyrZ,abAccX,abAccY,abAccZ,abEG,abEA,abT,abGH,abAH,abGHz,abAHz,labels
    const imu = getRow(DS.imu, c);
    const imuGyrX = imu[3] || 0;
    const imuGyrY = imu[4] || 0;
    const imuGyrZ = imu[5] || 0;
    const imuAccX = imu[6] || 0;
    const imuAccY = imu[7] || 0;
    const imuAccZ = imu[8] || 0;
    const imuTemp = imu[11] || 30;
    const imuLabel= imu[16] || 0;

    // MAG: LineNo,TimeUS,I,MagX,MagY,MagZ,OfsX,OfsY,OfsZ,MOX,MOY,MOZ,Health,S,labels
    const mag = getRow(DS.mag, c);
    const magXf  = mag[3]  || 0;
    const magYf  = mag[4]  || 0;
    const magZf  = mag[5]  || 0;
    const magHlth= mag[12] || 1;

    // MOTB: LineNo,TimeUS,LiftMax,BatVolt,BatRes,ThLimit,labels
    const motb = getRow(DS.motb, c);
    const liftMax  = motb[2] || 0;
    const motBatV  = motb[3] || 0;
    const motBatRes= motb[4] || 0;
    const thrLimit = motb[5] || 0;

    // PSCD: LineNo,TimeUS,TPD,PD,DVD,TVD,VD,DAD,TAD,AD,labels
    const pscd = getRow(DS.pscd, c);
    const tpd = pscd[2] || 0;
    const pd  = pscd[3] || 0;
    const dvd = pscd[4] || 0;
    const tvd = pscd[5] || 0;
    const vd  = pscd[6] || 0;
    const dad = pscd[7] || 0;
    const tad = pscd[8] || 0;
    const ad  = pscd[9] || 0;

    // RATE: LineNO,TimeUS,RDes,R,Rout,PDes,P,POut,YDes,Y,YOut,ADes,A,AOut,labels
    const rate = getRow(DS.rate, c);
    const rDes = rate[2] || 0;
    const r    = rate[3] || 0;
    const rOut = rate[4] || 0;
    const pDes = rate[5] || 0;
    const p    = rate[6] || 0;
    const pOut = rate[7] || 0;
    const yDes = rate[8] || 0;
    const y    = rate[9] || 0;
    const yOut = rate[10] || 0;
    const aDes = rate[11] || 0;
    const a    = rate[12] || 0;
    const aOut = rate[13] || 0;

    // VIBE: LineNo,TimeUS,IMU,VibeX,VibeY,VibeZ,Clip,labels[,labels]
    const vibe = getRow(DS.vibe, c);
    const vibeX = vibe[3] || 0;
    const vibeY = vibe[4] || 0;
    const vibeZ = vibe[5] || 0;
    const vibeClip = vibe[6] || 0;
    const vibeLabel= vibe[7] || 0;

    // XKF1: LineNo,TimeUS,C,Roll,Pitch,Yaw,VN,VE,VD,dPD,PN,PE,PD,GX,GY,GZ,OH,labels
    const xkf = getRow(DS.xkf1, c);
    const xkfRoll  = xkf[3] || 0;
    const xkfPitch = xkf[4] || 0;
    const xkfYaw   = xkf[5] || 0;
    const xkfVN    = xkf[6] || 0;
    const xkfVE    = xkf[7] || 0;
    const xkfVD    = xkf[8] || 0;
    const xkfPN    = xkf[10] || 0;
    const xkfPE    = xkf[11] || 0;
    const xkfOH    = xkf[16] || 0;

    // CTUN: LineNo,TimeUS,ThI,ABst,ThO,ThH,DAIt,Alt,BAlt,DSAlt,SAlt,TAlt,DCRt,CRt,labels
    const ctun = getRow(DS.ctun, c);
    const ctunThI  = ctun[2] || 0;
    const ctunThO  = ctun[4] || 0;
    const ctunThH  = ctun[5] || 0;
    const ctunAlt  = ctun[7] || 0;
    const ctunCRt  = ctun[13] || 0;

    // Build 59-feature vector — MUST MATCH train_model.py feature order exactly
    // [0:8]   ATT: DesRoll Roll DesPitch Pitch DesYaw Yaw ErrRP ErrYaw
    // [8:15]  BAT: Volt VoltR Curr CurrTot Temp Res RemPct
    // [15:18] BARO: Alt Press CRt
    // [18:21] GPS: NSats HDop Spd
    // [21:28] IMU: GyrX GyrY GyrZ AccX AccY AccZ Temp
    // [28:32] MAG: MagX MagY MagZ Health
    // [32:36] MOTB: LiftMax BatVolt BatRes ThLimit
    // [36:40] PSCD: TPD PD DVD VD
    // [40:48] RATE: RDes R PDes P YDes Y ADes A
    // [48:52] VIBE: VibeX VibeY VibeZ Clip
    // [52:58] XKF1: VN VE VD PN PE OH
    // [58:59] CTUN: ThO
    const att2 = getRow(DS.att, c);
    const mlFeatures = [
        // ATT (8)
        att2[2]||0,  att2[3]||0,  att2[4]||0,  att2[5]||0,
        att2[6]||0,  att2[7]||0,  att2[8]||0,  att2[9]||0,
        // BAT (7): Volt VoltR Curr CurrTot Temp Res RemPct
        b[3]||0,     b[4]||0,     b[5]||0,     b[6]||0,
        b[8]||0,     b[9]||0,     b[10]||0,
        // BARO (3): Alt Press CRt
        baroAlt,     baroPress,   baroCRt,
        // GPS (3): NSats HDop Spd
        nSats,       hdop,        gpsSpd,
        // IMU (7): GyrX GyrY GyrZ AccX AccY AccZ Temp
        imuGyrX,     imuGyrY,     imuGyrZ,
        imuAccX,     imuAccY,     imuAccZ,     imuTemp,
        // MAG (4): MagX MagY MagZ Health
        magXf,       magYf,       magZf,       magHlth,
        // MOTB (4): LiftMax BatVolt BatRes ThLimit
        liftMax,     motBatV,     motBatRes,   thrLimit,
        // PSCD (4): TPD PD DVD VD
        tpd,         pd,          dvd,         vd,
        // RATE (8): RDes R PDes P YDes Y ADes A
        rDes,        r,           pDes,        p,
        yDes,        y,           aDes,        a,
        // VIBE (4): VibeX VibeY VibeZ Clip
        vibeX,       vibeY,       vibeZ,       vibeClip,
        // XKF1 (6): VN VE VD PN PE OH
        xkfVN,       xkfVE,       xkfVD,       xkfPN,       xkfPE,  xkfOH,
        // CTUN (1): ThO
        ctunThO,
    ];

    // Motor RPM synthesis from IMU + attitude
    const baseRpm = 4500 + (Math.abs(accZ) * 50);
    const pitchF  = pitch * 150;
    const rollF   = roll  * 150;
    const motor1  = Math.round(baseRpm - pitchF + rollF);
    const motor2  = Math.round(baseRpm + pitchF - rollF);
    const motor3  = Math.round(baseRpm - pitchF - rollF);
    const motor4  = Math.round(baseRpm + pitchF + rollF);

    return {
        mlFeatures,
        fusionLabel,
        localPos:    { x: localX, z: localZ },
        kinematics:  { roll, pitch, yaw, desRoll, desPitch, desYaw, errRP, aekf },
        propulsion:  { motor1, motor2, motor3, motor4 },
        power:       { battV, battVr, curr, currTot, enrgTot, batTemp, batRes, remPct },
        vibration:   { vibeX, vibeY, vibeZ, vibeClip, vibeLabel },
        barometer:   { alt: baroAlt, press: baroPress, climbRate: baroCRt, temp: baroHealth },
        gps:         { status: gpsStatus, nSats, hdop, lat, lng, alt: gpsAlt, spd: gpsSpd, yaw: gpsYaw },
        imu:         { gyrX: imuGyrX, gyrY: imuGyrY, gyrZ: imuGyrZ, accX: imuAccX, accY: imuAccY, accZ: imuAccZ, temp: imuTemp, label: imuLabel },
        mag:         { x: magXf, y: magYf, z: magZf, health: magHlth },
        motb:        { liftMax, batVolt: motBatV, batRes: motBatRes, thrLimit },
        pscd:        { tpd, pd, dvd, tvd, vd, dad, tad, ad },
        rateControl: { rDes, r, rOut, pDes, p, pOut, yDes, y, yOut, aDes, a, aOut },
        ekfState:    { roll: xkfRoll, pitch: xkfPitch, yaw: xkfYaw, vn: xkfVN, ve: xkfVE, vd: xkfVD, pn: xkfPN, pe: xkfPE, oh: xkfOH },
        ctun:        { thi: ctunThI, tho: ctunThO, thh: ctunThH, alt: ctunAlt, crt: ctunCRt },
    };
}

// ─────────────────────────────────────────────────────────
//  6. SOCKET.IO — MAIN SWARM LOOP
// ─────────────────────────────────────────────────────────
io.on('connection', (socket) => {
    console.log(`[SOCKET] Command Center connected: ${socket.id}`);

    // Separate interval per drone (staggered by 50ms to avoid burst)
    const intervals = SWARM_NODES.map((node, idx) => {
        return setTimeout(() => {
            const interval = setInterval(async () => {
                const state = droneState[node.id];
                const telem = extractTelemetry(node.id);

                // Push features to ML history buffer
                state.historyBuffer.push(telem.mlFeatures);
                if (state.historyBuffer.length > SEQUENCE_LENGTH) state.historyBuffer.shift();

                // ML Prediction
                let mlPrediction = await getPrediction(node.id);

                // Inject fault override if commanded
                if (state.injectedFault !== null && mlPrediction) {
                    const faultInfo = FAULT_CLASSES[state.injectedFault];
                    // Fake but realistic explainability for each fault type
                    const INJECTED_XAI = {
                        1: [ // vibration_fault
                            { feature: 'VibeX',  group: 'Vibration',    impact_pct: 38.4 },
                            { feature: 'VibeZ',  group: 'Vibration',    impact_pct: 27.1 },
                            { feature: 'GyrY',   group: 'IMU',          impact_pct: 14.8 },
                            { feature: 'AccZ',   group: 'IMU',          impact_pct: 10.2 },
                            { feature: 'ErrRP',  group: 'Attitude',     impact_pct:  9.5 },
                        ],
                        2: [ // sensor_drift
                            { feature: 'ErrYaw', group: 'Attitude',     impact_pct: 31.7 },
                            { feature: 'MagX',   group: 'Magnetometer', impact_pct: 24.3 },
                            { feature: 'GyrZ',   group: 'IMU',          impact_pct: 19.6 },
                            { feature: 'ErrRP',  group: 'Attitude',     impact_pct: 13.1 },
                            { feature: 'AccY',   group: 'IMU',          impact_pct: 11.3 },
                        ],
                        3: [ // gps_fault
                            { feature: 'HDop',   group: 'GPS',          impact_pct: 44.2 },
                            { feature: 'NSats',  group: 'GPS',          impact_pct: 29.8 },
                            { feature: 'VD_ekf', group: 'EKF',          impact_pct: 13.5 },
                            { feature: 'PN',     group: 'EKF',          impact_pct:  7.9 },
                            { feature: 'Spd',    group: 'GPS',          impact_pct:  4.6 },
                        ],
                        4: [ // battery_fault
                            { feature: 'Volt',   group: 'Battery',      impact_pct: 41.6 },
                            { feature: 'Curr',   group: 'Battery',      impact_pct: 28.3 },
                            { feature: 'RemPct', group: 'Battery',      impact_pct: 17.4 },
                            { feature: 'Temp',   group: 'Battery',      impact_pct:  8.1 },
                            { feature: 'ThO',    group: 'Propulsion',   impact_pct:  4.6 },
                        ],
                    };

                    mlPrediction = {
                        ...mlPrediction,
                        fault_class:    state.injectedFault,
                        status:         faultInfo.name,
                        status_label:   faultInfo.label,
                        status_color:   faultInfo.color,
                        health_index:   state.injectedFault === 0 ? 98.5
                            : state.injectedFault === 1 ? 72
                            : state.injectedFault === 2 ? 61
                            : state.injectedFault === 3 ? 54
                            : 18,
                        confidence:     '99.00%',
                        explainability: INJECTED_XAI[state.injectedFault] || [],
                    };
                }

                // Track health history for RUL
                if (mlPrediction) {
                    state.healthHistory.push(mlPrediction.health_index);
                    if (state.healthHistory.length > 500) state.healthHistory.shift();
                }

                // RUL (only every 20 ticks to reduce API calls)
                let rulData = null;
                if (state.cursor % 20 === 0) {
                    rulData = await getRUL(node.id);
                }

                // Advance cursor
                state.cursor++;

                // Handle RTL mode
                if (state.mode === 'rtl') {
                    state.rtlProgress = Math.min(1, state.rtlProgress + 0.02);
                    if (state.rtlProgress >= 1) state.mode = 'landed';
                }

                const payload = {
                    deviceId:     node.id,
                    timestamp:    new Date().toISOString(),
                    mode:         state.mode,
                    rtlProgress:  state.rtlProgress,
                    metrics:      {
                        localPos:    telem.localPos,
                        kinematics:  telem.kinematics,
                        propulsion:  telem.propulsion,
                        power:       telem.power,
                        vibration:   telem.vibration,
                        barometer:   telem.barometer,
                        gps:         telem.gps,
                        imu:         telem.imu,
                        mag:         telem.mag,
                        motb:        telem.motb,
                        pscd:        telem.pscd,
                        rateControl: telem.rateControl,
                        ekfState:    telem.ekfState,
                        ctun:        telem.ctun,
                    },
                    ml_prediction: mlPrediction,
                    rul:           rulData,
                };

                socket.emit('telemetry_stream', JSON.stringify(payload));

                // Write to MongoDB (every 4 ticks = 1s)
                if (TelemetryModel && state.cursor % 4 === 0 && mlPrediction) {
                    try {
                        await TelemetryModel.create({
                            deviceId:    node.id,
                            timestamp:   new Date(),
                            faultClass:  mlPrediction.fault_class ?? 0,
                            faultName:   mlPrediction.status,
                            healthIndex: mlPrediction.health_index,
                            confidence:  mlPrediction.confidence,
                            kinematics:  telem.kinematics,
                            propulsion:  telem.propulsion,
                            power:       telem.power,
                            vibration:   telem.vibration,
                            barometer:   telem.barometer,
                            gps:         telem.gps,
                            rateControl: telem.rateControl,
                            posControl:  telem.pscd,
                            ekfState:    telem.ekfState,
                        });
                    } catch { /* silent — avoid crashing stream on DB error */ }
                }

            }, 250);

            socket.on('disconnect', () => {
                console.log(`[SOCKET] Command Center disconnected: ${socket.id}`);
                clearInterval(interval);
            });

            // ─── Fault Injection Commands ───
            socket.on('inject_fault', ({ droneId, faultClass }) => {
                if (droneState[droneId]) {
                    droneState[droneId].injectedFault = faultClass;
                    const faultInfo = FAULT_CLASSES[faultClass] || FAULT_CLASSES[0];
                    console.log(`[CMD] Injected fault ${faultInfo.label} into ${droneId}`);
                    socket.emit('command_ack', {
                        droneId,
                        action:  'inject_fault',
                        faultClass,
                        message: `[${new Date().toLocaleTimeString()}] ${droneId}: ${faultInfo.label} injected.`
                    });
                }
            });

            socket.on('reset_drone', ({ droneId }) => {
                if (droneState[droneId]) {
                    droneState[droneId].injectedFault = null;
                    droneState[droneId].mode = 'patrol';
                    droneState[droneId].rtlProgress = 0;
                    droneState[droneId].healthHistory = [];
                    console.log(`[CMD] Reset ${droneId} to nominal.`);
                    socket.emit('command_ack', {
                        droneId,
                        action:  'reset_drone',
                        message: `[${new Date().toLocaleTimeString()}] ${droneId}: Reset to NOMINAL.`
                    });
                }
            });

            socket.on('command_rtl', ({ droneId }) => {
                if (droneState[droneId]) {
                    droneState[droneId].mode = 'rtl';
                    droneState[droneId].rtlProgress = 0;
                    console.log(`[CMD] RTL commanded for ${droneId}`);
                    socket.emit('command_ack', {
                        droneId,
                        action:  'command_rtl',
                        message: `[${new Date().toLocaleTimeString()}] ${droneId}: RTL (Return-To-Launch) commanded.`
                    });
                }
            });

        }, idx * 50); // stagger drone streams by 50ms
    });

    socket.on('disconnect', () => {
        intervals.forEach(t => clearTimeout(t));
    });
});

// ─────────────────────────────────────────────────────────
//  7. REST ENDPOINTS
// ─────────────────────────────────────────────────────────
app.get('/api/swarm', (req, res) => {
    const status = SWARM_NODES.map(n => ({
        id:   n.id,
        mode: droneState[n.id].mode,
        injectedFault: droneState[n.id].injectedFault,
        historyLen: droneState[n.id].historyBuffer.length,
    }));
    res.json({ swarm: status, timestamp: new Date() });
});

app.get('/api/history/:deviceId', async (req, res) => {
    if (!TelemetryModel) return res.json({ error: 'Database not connected' });
    try {
        const records = await TelemetryModel
            .find({ deviceId: req.params.deviceId })
            .sort({ timestamp: -1 })
            .limit(200)
            .lean();
        res.json({ deviceId: req.params.deviceId, count: records.length, records });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

const PORT = 5000;
server.listen(PORT, () => console.log(`\n[SERVER] Node Command Center listening on port ${PORT}\n`));