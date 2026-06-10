import mongoose from 'mongoose';

const telemetrySchema = new mongoose.Schema({
    // 1. The Meta Field (Who is sending the data?)
    deviceId: { type: String, required: true },
    
    // 2. The Time Field (When did it happen?)
    timestamp: { type: Date, required: true },
    
    // 3. The Telemetry Payload (What are the physics?)
    voltageMv: Number,
    currentMa: Number,
    cpuTempC: Number,
    batteryTempC: Number,
    cpuLoadPct: Number,
    gpuLoadPct: Number,
    isCameraActive: Boolean,
    batteryLevelPct: Number,

    // 4. The AI Inference (What did the ML engine think?)
    aiConfidencePct: Number,
    anomalyDetected: Boolean

}, {
    // THIS IS THE MAGIC: Converting a standard collection into an IoT Time-Series collection
    timeseries: {
        timeField: 'timestamp',
        metaField: 'deviceId',
        granularity: 'seconds' // Optimizes storage for high-frequency 1-second interval writes
    }
});

export default mongoose.model('Telemetry', telemetrySchema);