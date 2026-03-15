/**
 * Audio engine — decodes WAV from base64, creates ConvolverNode for real-time
 * "hear the room" playback.
 */

let audioCtx = null;
let currentSource = null;

function getCtx() {
    if (!audioCtx) audioCtx = new AudioContext();
    return audioCtx;
}

/** Decode a base64 WAV string into an AudioBuffer. */
export async function decodeWav(base64) {
    const ctx = getCtx();
    const binary = atob(base64);
    const buf = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) buf[i] = binary.charCodeAt(i);
    return ctx.decodeAudioData(buf.buffer);
}

/** Stop any currently playing sound. */
export function stop() {
    if (currentSource) {
        try { currentSource.stop(); } catch {}
        currentSource = null;
    }
}

/** Play the raw impulse response (the IR itself as audio). */
export function playIR(irBuffer) {
    stop();
    const ctx = getCtx();
    const src = ctx.createBufferSource();
    src.buffer = irBuffer;
    src.connect(ctx.destination);
    src.start();
    currentSource = src;
    src.onended = () => { currentSource = null; };
}

/**
 * Generate a short clap (noise burst) convolved with the IR.
 * This lets users "hear" the room — the clap excites the impulse response.
 */
export function playConvolved(irBuffer) {
    stop();
    const ctx = getCtx();
    const sr = irBuffer.sampleRate;

    // Generate a 5ms exponential noise burst (clap)
    const clapLen = Math.floor(0.005 * sr);
    const clapBuf = ctx.createBuffer(1, clapLen, sr);
    const clapData = clapBuf.getChannelData(0);
    for (let i = 0; i < clapLen; i++) {
        clapData[i] = (Math.random() * 2 - 1) * Math.exp(-i / (0.001 * sr));
    }

    const src = ctx.createBufferSource();
    src.buffer = clapBuf;

    const convolver = ctx.createConvolver();
    convolver.buffer = irBuffer;

    // Slight gain to make it audible
    const gain = ctx.createGain();
    gain.gain.value = 3.0;

    src.connect(convolver);
    convolver.connect(gain);
    gain.connect(ctx.destination);
    src.start();
    currentSource = src;
    src.onended = () => { currentSource = null; };
}
