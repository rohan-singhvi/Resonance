import { RoomScene } from './scene.js';
import { decodeWav, decodeAudioFile, playIR, playConvolved, playUserConvolved, stop } from './audio.js';

const API_BASE = window.location.origin + '/resonance/api';

// ── State ──────────────────────────────────────────────────────
let scene;
let currentIRBuffer = null;
let userAudioBuffer = null;
let simulating = false;

const MATERIAL_LABELS = {
    concrete: 'Concrete', brick: 'Brick', drywall: 'Drywall',
    plaster: 'Plaster', glass: 'Glass / Ceramic', wood_floor: 'Wood Floor',
    carpet_thin: 'Thin Carpet', carpet_thick: 'Thick Carpet',
    acoustic_foam: 'Acoustic Foam', acoustic_panel: 'Acoustic Panel',
    curtain: 'Curtain', audience: 'Audience',
};

const MATERIALS = Object.keys(MATERIAL_LABELS);

const ROOM_PRESETS = {
    recording_studio: { label: 'Studio', dims: [4, 3, 2.8], materials: { floor: 'carpet_thick', walls: 'acoustic_foam', ceiling: 'acoustic_panel' }, source: [1.5, 1.2, 1.4], listener: [2.5, 1.2, 1.4], rays: 80000 },
    living_room: { label: 'Living Room', dims: [6, 4, 2.6], materials: { floor: 'carpet_thin', walls: 'drywall', ceiling: 'plaster' }, source: [1.5, 1.0, 2.0], listener: [4.5, 1.0, 2.0], rays: 80000 },
    concert_hall: { label: 'Concert Hall', dims: [30, 15, 12], materials: { floor: 'wood_floor', walls: 'concrete', ceiling: 'acoustic_panel' }, source: [5, 1.5, 7.5], listener: [20, 1.5, 7.5], rays: 150000 },
    bathroom: { label: 'Bathroom', dims: [2.5, 2.5, 2.4], materials: { floor: 'glass', walls: 'glass', ceiling: 'plaster' }, source: [0.8, 1.5, 1.2], listener: [1.7, 1.5, 1.2], rays: 50000 },
    cathedral: { label: 'Cathedral', dims: [40, 20, 18], materials: { floor: 'concrete', walls: 'brick', ceiling: 'concrete' }, source: [10, 1.5, 10], listener: [30, 1.5, 10], rays: 150000 },
    lecture_hall: { label: 'Lecture Hall', dims: [15, 6, 10], materials: { floor: 'carpet_thin', walls: 'drywall', ceiling: 'acoustic_panel' }, source: [2, 1.5, 5], listener: [12, 1.5, 5], rays: 100000 },
};

// ── DOM refs ───────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ── Init ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    scene = new RoomScene($('#viewport'));
    populateMaterialSelects();
    populatePresets();
    bindEvents();
    syncSceneFromUI();
});

function populateMaterialSelects() {
    for (const sel of $$('.mat-select')) {
        sel.innerHTML = MATERIALS.map(m =>
            `<option value="${m}">${MATERIAL_LABELS[m]}</option>`
        ).join('');
    }
    // defaults
    $('#mat-floor').value = 'concrete';
    $('#mat-walls').value = 'concrete';
    $('#mat-ceiling').value = 'concrete';
}

function populatePresets() {
    const grid = $('#preset-grid');
    grid.innerHTML = '';
    for (const [key, p] of Object.entries(ROOM_PRESETS)) {
        const btn = document.createElement('button');
        btn.className = 'preset-btn';
        btn.dataset.preset = key;
        btn.textContent = p.label;
        btn.addEventListener('click', () => applyPreset(key));
        grid.appendChild(btn);
    }
}

function applyPreset(key) {
    const p = ROOM_PRESETS[key];
    if (!p) return;

    // Highlight active preset
    for (const btn of $$('.preset-btn')) {
        btn.classList.toggle('active', btn.dataset.preset === key);
    }

    // Room type = shoebox
    setRoomType('shoebox');

    // Dims
    $('#dim-x').value = p.dims[0];
    $('#dim-y').value = p.dims[1];
    $('#dim-z').value = p.dims[2];

    // Materials
    if (p.materials) {
        $('#mat-floor').value = p.materials.floor || 'concrete';
        $('#mat-walls').value = p.materials.walls || 'concrete';
        $('#mat-ceiling').value = p.materials.ceiling || 'concrete';
    }

    // Positions
    $('#src-x').value = p.source[0];
    $('#src-y').value = p.source[1];
    $('#src-z').value = p.source[2];
    $('#lis-x').value = p.listener[0];
    $('#lis-y').value = p.listener[1];
    $('#lis-z').value = p.listener[2];

    // Rays
    if (p.rays) {
        $('#ray-count').value = p.rays;
        $('#ray-count-val').textContent = formatRayCount(p.rays);
    }

    syncSceneFromUI();
}

function setRoomType(type) {
    for (const btn of $$('.seg-btn')) {
        btn.classList.toggle('active', btn.dataset.room === type);
    }
    $('#dims-row').classList.toggle('hidden', type === 'dome');
    $('#radius-row').classList.toggle('hidden', type !== 'dome');
    $('#materials-section').style.display = type === 'dome' ? 'none' : '';
}

function getRoomType() {
    const active = $('.seg-btn.active');
    return active ? active.dataset.room : 'shoebox';
}

function getDims() {
    if (getRoomType() === 'dome') {
        const r = parseFloat($('#dome-radius').value) || 10;
        return [r, r, r];
    }
    return [
        parseFloat($('#dim-x').value) || 10,
        parseFloat($('#dim-y').value) || 5,
        parseFloat($('#dim-z').value) || 8,
    ];
}

function getMaterials() {
    return {
        floor: $('#mat-floor').value,
        walls: $('#mat-walls').value,
        ceiling: $('#mat-ceiling').value,
    };
}

function getSource() {
    return [
        parseFloat($('#src-x').value) || 2,
        parseFloat($('#src-y').value) || 1.5,
        parseFloat($('#src-z').value) || 1.5,
    ];
}

function getListener() {
    return [
        parseFloat($('#lis-x').value) || 8,
        parseFloat($('#lis-y').value) || 1.5,
        parseFloat($('#lis-z').value) || 1.5,
    ];
}

function formatRayCount(n) {
    return n >= 1000 ? (n / 1000) + 'k' : n;
}

// ── Event binding ──────────────────────────────────────────────
function bindEvents() {
    // Room type toggle
    for (const btn of $$('.seg-btn')) {
        btn.addEventListener('click', () => {
            setRoomType(btn.dataset.room);
            clearPresetHighlight();
            syncSceneFromUI();
        });
    }

    // Dims, positions, materials — live update the 3D view
    const liveInputs = [
        '#dim-x', '#dim-y', '#dim-z', '#dome-radius',
        '#src-x', '#src-y', '#src-z',
        '#lis-x', '#lis-y', '#lis-z',
        '#mat-floor', '#mat-walls', '#mat-ceiling',
    ];
    for (const sel of liveInputs) {
        $(sel).addEventListener('input', () => {
            clearPresetHighlight();
            syncSceneFromUI();
        });
    }

    // Sliders
    $('#ray-count').addEventListener('input', (e) => {
        $('#ray-count-val').textContent = formatRayCount(parseInt(e.target.value));
    });
    $('#ir-len').addEventListener('input', (e) => {
        $('#ir-len-val').textContent = (parseInt(e.target.value) / 1000).toFixed(1) + 's';
    });
    $('#scattering').addEventListener('input', (e) => {
        $('#scattering-val').textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Camera reset
    $('#reset-cam').addEventListener('click', () => scene.resetCamera());

    // Simulate
    $('#simulate-btn').addEventListener('click', () => runSimulation());

    // Audio
    $('#play-ir').addEventListener('click', () => {
        if (currentIRBuffer) playIR(currentIRBuffer);
    });
    $('#play-convolved').addEventListener('click', () => {
        if (currentIRBuffer) playConvolved(currentIRBuffer);
    });
    $('#play-user').addEventListener('click', () => {
        if (currentIRBuffer && userAudioBuffer) playUserConvolved(userAudioBuffer, currentIRBuffer);
    });

    // Audio upload
    $('#audio-upload').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        try {
            const arrayBuffer = await file.arrayBuffer();
            userAudioBuffer = await decodeAudioFile(arrayBuffer);
            const name = file.name.length > 12 ? file.name.slice(0, 10) + '..' : file.name;
            $('#user-audio-name').textContent = name;
            $('#play-user').style.display = '';
            if (currentIRBuffer) $('#play-user').disabled = false;
        } catch (err) {
            console.error('Audio decode failed:', err);
            alert('Could not decode audio file. Try a WAV or MP3.');
        }
    });
}

function clearPresetHighlight() {
    for (const btn of $$('.preset-btn')) btn.classList.remove('active');
}

function syncSceneFromUI() {
    const type = getRoomType();
    const dims = getDims();
    const mats = getMaterials();
    scene.updateRoom(type, dims, mats);
    scene.setSource(getSource());
    scene.setListener(getListener());
}

// ── Simulation ─────────────────────────────────────────────────
async function runSimulation() {
    if (simulating) return;
    simulating = true;

    const btn = $('#simulate-btn');
    btn.disabled = true;
    btn.textContent = 'Simulating...';
    $('#sim-status').classList.remove('hidden');

    const type = getRoomType();
    const dims = getDims();

    const body = {
        room_type: type,
        dims,
        source: getSource(),
        listener: getListener(),
        rays: parseInt($('#ray-count').value),
        scattering: parseFloat($('#scattering').value),
        sr: 44100,
        ir_len_ms: parseInt($('#ir-len').value),
        debug_rays: true,
    };

    // Per-surface materials for shoebox
    if (type === 'shoebox') {
        body.materials = getMaterials();
    } else {
        body.absorption = 0.1;
    }

    try {
        const resp = await fetch(API_BASE + '/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'Simulation failed');
        }

        const data = await resp.json();
        displayResults(data);
    } catch (err) {
        console.error('Simulation error:', err);
        alert('Simulation failed: ' + err.message);
    } finally {
        simulating = false;
        btn.disabled = false;
        btn.textContent = 'Simulate';
        $('#sim-status').classList.add('hidden');
    }
}

async function displayResults(data) {
    // Metrics
    const { rt60, edt, c50 } = data.metrics;
    animateMetric('#m-rt60', rt60, 3);
    animateMetric('#m-edt', edt, 3);
    animateMetric('#m-c50', c50, 2);

    // Show results strip
    $('#results-strip').classList.remove('hidden');

    // Waveform
    drawWaveform(data.waveform);

    // Ray visualization
    if (data.ray_paths && data.ray_paths.length > 0) {
        scene.showRays(data.ray_paths);
    }

    // Audio
    try {
        currentIRBuffer = await decodeWav(data.wav_base64);
        $('#play-ir').disabled = false;
        $('#play-convolved').disabled = false;
        if (userAudioBuffer) $('#play-user').disabled = false;
        $('#upload-audio-label').classList.remove('disabled');

        // Download link
        const blob = base64ToBlob(data.wav_base64, 'audio/wav');
        const url = URL.createObjectURL(blob);
        const dl = $('#download-ir');
        dl.href = url;
        dl.download = 'impulse_response.wav';
    } catch (e) {
        console.warn('Audio decode failed:', e);
    }
}

function animateMetric(selector, value, decimals) {
    const el = $(selector);
    if (value === null || value === undefined) {
        el.textContent = 'N/A';
        return;
    }
    // Animate from 0 to value
    const target = parseFloat(value);
    const duration = 600;
    const start = performance.now();
    const tick = (now) => {
        const t = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - t, 3); // ease-out cubic
        el.textContent = (target * ease).toFixed(decimals);
        if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}

function drawWaveform(samples) {
    const canvas = $('#waveform');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    // Size canvas properly
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const mid = h / 2;

    ctx.clearRect(0, 0, w, h);

    if (!samples || samples.length === 0) return;

    const max = Math.max(...samples.map(Math.abs), 0.001);
    const step = w / samples.length;

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, 'rgba(59, 130, 246, 0.6)');
    grad.addColorStop(0.5, 'rgba(59, 130, 246, 0.3)');
    grad.addColorStop(1, 'rgba(59, 130, 246, 0.1)');

    ctx.beginPath();
    ctx.moveTo(0, mid);
    for (let i = 0; i < samples.length; i++) {
        const x = i * step;
        const y = mid - (samples[i] / max) * (mid - 4);
        ctx.lineTo(x, y);
    }
    // Mirror below
    for (let i = samples.length - 1; i >= 0; i--) {
        const x = i * step;
        const y = mid + (samples[i] / max) * (mid - 4);
        ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Center line
    ctx.beginPath();
    ctx.moveTo(0, mid);
    for (let i = 0; i < samples.length; i++) {
        ctx.lineTo(i * step, mid - (samples[i] / max) * (mid - 4));
    }
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
    ctx.lineWidth = 1;
    ctx.stroke();
}

function base64ToBlob(base64, mime) {
    const binary = atob(base64);
    const arr = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) arr[i] = binary.charCodeAt(i);
    return new Blob([arr], { type: mime });
}
