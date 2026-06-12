// Audio Contexts
let audioCtx = null;
let inputAudioCtx = null;
let analyser = null;
let nextPlayTime = 0;

// WebSockets
let ttsWs = null;
let s2sWs = null;

// State
let currentMode = 'tts'; // 'tts' or 's2s'
let isRecording = false;
let mediaStream = null;
let scriptProcessor = null;
let micSource = null;

// Latency History for Averages
let latencyHistory = {
    vad: [],
    stt: [],
    llm: [],
    tts: [],
    total: []
};

function resetLatencyHistory() {
    latencyHistory.vad = [];
    latencyHistory.stt = [];
    latencyHistory.llm = [];
    latencyHistory.tts = [];
    latencyHistory.total = [];
    updateLatencyUI();
}

function calculateAverage(arr) {
    if (!arr || arr.length === 0) return 0;
    const sum = arr.reduce((a, b) => a + b, 0);
    return Math.round(sum / arr.length);
}

function updateLatencyUI() {
    const avgVad = calculateAverage(latencyHistory.vad);
    const avgStt = calculateAverage(latencyHistory.stt);
    const avgLlm = calculateAverage(latencyHistory.llm);
    const avgTts = calculateAverage(latencyHistory.tts);
    const avgTotal = calculateAverage(latencyHistory.total);
    
    metricVad.textContent = avgVad > 0 ? `${avgVad} ms` : "0 ms";
    metricStt.textContent = avgStt > 0 ? `${avgStt} ms` : "0 ms";
    metricTts.textContent = avgTts > 0 ? `${avgTts} ms` : "0 ms";
    
    const metricLlm = document.getElementById('metric-llm');
    if (metricLlm) {
        metricLlm.textContent = avgLlm > 0 ? `${avgLlm} ms` : "0 ms";
    }
    
    metricTotal.textContent = avgTotal > 0 ? `${avgTotal} ms` : "0 ms";
}

// UI Elements
const voiceSelect = document.getElementById('voice-select');
const speedRange = document.getElementById('speed-range');
const speedVal = document.getElementById('speed-val');
const ttsInput = document.getElementById('tts-input');
const logsContainer = document.getElementById('logs-container');
const languageSelect = document.getElementById('language-select');
const taskSelect = document.getElementById('task-select');
const sttEngineSelect = document.getElementById('stt-engine-select');
const ttsEngineSelect = document.getElementById('tts-engine-select');
const apiKeyGroup = document.getElementById('api-key-group');
const cloudEngines = ['elevenlabs', 'deepgram', 'murf', 'openai', 'cartesia', 'fishaudio', 'cosyvoice', 'claude'];

function getApiKeys() {
    const keys = {};
    cloudEngines.forEach(engine => {
        const input = document.getElementById(`api-key-${engine}`);
        if (input) {
            keys[engine] = input.value;
        }
    });
    return keys;
}

// Latency elements
const metricVad = document.getElementById('metric-vad');
const metricStt = document.getElementById('metric-stt');
const metricTts = document.getElementById('metric-tts');
const metricTotal = document.getElementById('metric-total');

// Canvas Visualizer
const canvas = document.getElementById('waveform');
const canvasCtx = canvas.getContext('2d');

// Resize canvas to fill container
function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Initialize Visualizer Loop
function drawVisualizer() {
    requestAnimationFrame(drawVisualizer);
    
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas with subtle transparency for tail effect
    canvasCtx.fillStyle = 'rgba(11, 10, 19, 0.25)';
    canvasCtx.fillRect(0, 0, width, height);
    
    if (!analyser) {
        // Draw idle wave
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = 'rgba(139, 92, 246, 0.15)';
        canvasCtx.beginPath();
        const sliceWidth = width / 100;
        let x = 0;
        for (let i = 0; i < 100; i++) {
            const y = height / 2 + Math.sin(i * 0.15 + Date.now() * 0.005) * 5;
            if (i === 0) canvasCtx.moveTo(x, y);
            else canvasCtx.lineTo(x, y);
            x += sliceWidth;
        }
        canvasCtx.stroke();
        return;
    }
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);
    
    canvasCtx.lineWidth = 3;
    
    // Gradient outline
    const gradient = canvasCtx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, '#8b5cf6'); // Purple
    gradient.addColorStop(0.5, '#ec4899'); // Pink
    gradient.addColorStop(1, '#f97316'); // Orange
    canvasCtx.strokeStyle = gradient;
    
    canvasCtx.beginPath();
    
    const sliceWidth = width / bufferLength;
    let x = 0;
    
    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * height / 2;
        
        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }
        
        x += sliceWidth;
    }
    
    canvasCtx.lineTo(width, height / 2);
    canvasCtx.stroke();
}
drawVisualizer();

// Web Audio API Playback setup
function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048;
        analyser.connect(audioCtx.destination);
        nextPlayTime = audioCtx.currentTime;
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
}

// Queue and play audio float32 PCM chunks
function playAudioChunk(float32Array) {
    initAudio();
    const buffer = audioCtx.createBuffer(1, float32Array.length, 24000); // Kokoro plays at 24000Hz
    buffer.getChannelData(0).set(float32Array);
    
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(analyser);
    
    const now = audioCtx.currentTime;
    if (nextPlayTime < now) {
        nextPlayTime = now;
    }
    
    source.start(nextPlayTime);
    nextPlayTime += buffer.duration;
}

// Update Range Label
function updateSpeedValue(val) {
    speedVal.textContent = val + 'x';
}

function toggleAiConversation(shouldSendConfig = true) {
    const aiConvToggle = document.getElementById('ai-conv-toggle');
    const aiConvSettings = document.getElementById('ai-conv-settings');
    const metricLlmRow = document.getElementById('metric-llm-row');
    
    const isEnabled = aiConvToggle ? aiConvToggle.checked : false;
    
    if (aiConvSettings) {
        aiConvSettings.style.display = isEnabled ? 'grid' : 'none';
    }
    if (metricLlmRow) {
        metricLlmRow.style.display = isEnabled ? 'flex' : 'none';
    }
    
    localStorage.setItem('ai_conversation', isEnabled);
    
    if (isEnabled) {
        fetchClaudeModels();
    }
    
    if (shouldSendConfig) {
        sendConfigUpdate();
    }
}

async function fetchClaudeModels() {
    const apiKeyInput = document.getElementById('api-key-claude');
    const apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
    if (!apiKey) {
        populateDefaultClaudeModels();
        return;
    }
    
    const select = document.getElementById('claude-model-select');
    if (!select) return;
    
    const prevValue = select.value;
    select.innerHTML = '<option value="">Loading models...</option>';
    
    try {
        const response = await fetch('/api/claude/models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ api_key: apiKey })
        });
        const data = await response.json();
        if (data.status === 'success' && data.models && data.models.length > 0) {
            select.innerHTML = '';
            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.id;
                opt.textContent = m.name;
                select.appendChild(opt);
            });
            if (data.models.some(m => m.id === prevValue)) {
                select.value = prevValue;
            } else {
                const sonnetModel = data.models.find(m => m.id.includes('sonnet'));
                if (sonnetModel) {
                    select.value = sonnetModel.id;
                } else {
                    select.value = data.models[0].id;
                }
            }
            localStorage.setItem('claude_model', select.value);
            sendConfigUpdate();
        } else {
            console.error("Failed to fetch models:", data.message);
            populateDefaultClaudeModels();
            if (apiKey.length > 5) {
                addLog('system', 'Failed to fetch Claude models. Using defaults.');
            }
        }
    } catch (e) {
        console.error("Error fetching Claude models:", e);
        populateDefaultClaudeModels();
    }
}

function populateDefaultClaudeModels() {
    const select = document.getElementById('claude-model-select');
    if (!select) return;
    const prevValue = select.value;
    select.innerHTML = `
        <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
        <option value="claude-3-7-sonnet-20250219">Claude 3.7 Sonnet</option>
        <option value="claude-3-5-haiku-20241022">Claude 3.5 Haiku</option>
        <option value="claude-3-opus-20240229">Claude 3 Opus</option>
    `;
    if (prevValue) {
        select.value = prevValue;
    }
}

function toggleApiKeyInput() {
    const sttEngine = sttEngineSelect.value;
    const ttsEngine = ttsEngineSelect.value;
    let anyCloud = false;
    
    cloudEngines.forEach(engine => {
        const container = document.getElementById(`api-key-${engine}-container`);
        if (container) {
            const isNeeded = (sttEngine === engine) || (ttsEngine === engine);
            if (isNeeded) {
                container.style.display = 'block';
                anyCloud = true;
            } else {
                container.style.display = 'none';
            }
        }
    });
    
    if (anyCloud) {
        apiKeyGroup.style.display = 'flex';
        apiKeyGroup.style.flexDirection = 'column';
    } else {
        apiKeyGroup.style.display = 'none';
    }
}

function toggleVoices() {
    const engine = ttsEngineSelect.value;
    const optgroups = voiceSelect.getElementsByTagName('optgroup');
    
    // Hide all first
    for (let i = 0; i < optgroups.length; i++) {
        optgroups[i].style.display = 'none';
    }
    
    if (optgroups.length >= 8) {
        if (engine === 'elevenlabs') {
            optgroups[1].style.display = 'block';
            voiceSelect.value = "21m00Tcm4TlvDq8ikWAM"; // default Rachel
        } else if (engine === 'murf') {
            optgroups[2].style.display = 'block';
            voiceSelect.value = "en-US-nate"; // default Nate
        } else if (engine === 'openai') {
            optgroups[3].style.display = 'block';
            voiceSelect.value = "alloy"; // default Alloy
        } else if (engine === 'deepgram') {
            optgroups[4].style.display = 'block';
            voiceSelect.value = "aura-asteria-en"; // default Asteria
        } else if (engine === 'cartesia') {
            optgroups[5].style.display = 'block';
            voiceSelect.value = "a0e99841-438c-4a64-b679-ae501e7d6091"; // default Barista
        } else if (engine === 'fishaudio') {
            optgroups[6].style.display = 'block';
            voiceSelect.value = "9f51c1d912ea4a9ba14c45c276329fc5"; // default Fish Audio Voice 1
        } else if (engine === 'cosyvoice' || engine === 'local_cosyvoice') {
            optgroups[7].style.display = 'block';
            voiceSelect.value = "FunAudioLLM/CosyVoice2-0.5B:alex"; // default CosyVoice2
        } else {
            optgroups[0].style.display = 'block';
            voiceSelect.value = "af_nicole"; // default Nicole
        }
    }
    sendConfigUpdate();
}

// Send config update over WS dynamically
function sendConfigUpdate() {
    const aiConvToggle = document.getElementById('ai-conv-toggle');
    const claudeModelSelect = document.getElementById('claude-model-select');
    
    const configMsg = JSON.stringify({
        voice: voiceSelect.value,
        speed: parseFloat(speedRange.value),
        language: languageSelect.value,
        task: taskSelect.value,
        stt_engine: sttEngineSelect.value,
        tts_engine: ttsEngineSelect.value,
        api_key: getApiKeys(),
        ai_conversation: aiConvToggle ? aiConvToggle.checked : false,
        claude_model: claudeModelSelect ? claudeModelSelect.value : "claude-3-5-sonnet-20241022"
    });
    
    if (s2sWs && s2sWs.readyState === WebSocket.OPEN) {
        s2sWs.send(configMsg);
    }
}

// Persist API Keys
document.addEventListener('DOMContentLoaded', () => {
    cloudEngines.forEach(engine => {
        const input = document.getElementById(`api-key-${engine}`);
        if (input) {
            const savedKey = localStorage.getItem(`apiKey_${engine}`);
            if (savedKey) {
                input.value = savedKey;
            }
            input.addEventListener('input', (e) => {
                localStorage.setItem(`apiKey_${engine}`, e.target.value);
                sendConfigUpdate();
                if (engine === 'claude') {
                    fetchClaudeModels();
                }
            });
        }
    });
    toggleApiKeyInput();

    // Load saved AI Conversation Toggle State
    const savedAiConv = localStorage.getItem('ai_conversation') === 'true';
    const aiConvToggle = document.getElementById('ai-conv-toggle');
    if (aiConvToggle) {
        aiConvToggle.checked = savedAiConv;
        toggleAiConversation(false);
    }

    // Load saved Claude model
    const savedClaudeModel = localStorage.getItem('claude_model');
    const claudeModelSelect = document.getElementById('claude-model-select');
    if (claudeModelSelect) {
        if (savedClaudeModel) {
            claudeModelSelect.value = savedClaudeModel;
        }
        claudeModelSelect.addEventListener('change', (e) => {
            localStorage.setItem('claude_model', e.target.value);
            sendConfigUpdate();
        });
    }

    // Initial fetch of Claude models if key exists
    const claudeKeyInput = document.getElementById('api-key-claude');
    if (claudeKeyInput && claudeKeyInput.value) {
        fetchClaudeModels();
    }
});

// Logs helper
function addLog(speaker, text) {
    const log = document.createElement('div');
    log.className = `log-entry ${speaker}`;
    log.textContent = text;
    logsContainer.appendChild(log);
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// Tab Switching
function switchMode(mode) {
    if (mode === currentMode) return;
    
    currentMode = mode;
    document.getElementById('tab-tts').classList.toggle('active', mode === 'tts');
    document.getElementById('tab-s2s').classList.toggle('active', mode === 's2s');
    document.getElementById('panel-tts').style.display = mode === 'tts' ? 'flex' : 'none';
    document.getElementById('panel-s2s').style.display = mode === 's2s' ? 'flex' : 'none';
    
    // Stop recording if switching modes
    if (mode === 'tts' && isRecording) {
        stopSpeechToSpeech();
    }
    
    resetLatencyHistory();
}

// Generate TTS from written Text
function generateTTS() {
    const text = ttsInput.value.trim();
    if (!text) return;
    
    initAudio();
    nextPlayTime = audioCtx.currentTime; // Reset queue alignment
    addLog('user', text);
    
    if (ttsWs) {
        ttsWs.close();
    }
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ttsWs = new WebSocket(`${protocol}//${window.location.host}/ws/tts`);
    ttsWs.binaryType = 'arraybuffer';
    
    ttsWs.onopen = () => {
        const aiConvToggle = document.getElementById('ai-conv-toggle');
        const claudeModelSelect = document.getElementById('claude-model-select');
        ttsWs.send(JSON.stringify({
            text: text,
            voice: voiceSelect.value,
            speed: parseFloat(speedRange.value),
            stt_engine: sttEngineSelect.value,
            tts_engine: ttsEngineSelect.value,
            api_key: getApiKeys(),
            ai_conversation: aiConvToggle ? aiConvToggle.checked : false,
            claude_model: claudeModelSelect ? claudeModelSelect.value : "claude-3-5-sonnet-20241022"
        }));
    };
    
    ttsWs.onmessage = (event) => {
        if (typeof event.data === 'string') {
            const data = JSON.parse(event.data);
            if (data.type === 'ai_transcript') {
                addLog('ai', data.text);
            } else if (data.type === 'latency') {
                latencyHistory.vad.push(0);
                latencyHistory.stt.push(0);
                if (data.tts_gen_ms) latencyHistory.tts.push(data.tts_gen_ms);
                if (data.llm_latency_ms) latencyHistory.llm.push(data.llm_latency_ms);
                latencyHistory.total.push(data.tts_gen_ms + (data.llm_latency_ms || 0));
                updateLatencyUI();
            } else if (data.type === 'done') {
                addLog('system', 'TTS playback finished.');
                ttsWs.close();
            }
        } else {
            // Receive raw float32 bytes
            const float32Array = new Float32Array(event.data);
            playAudioChunk(float32Array);
        }
    };
    
    ttsWs.onerror = (e) => {
        addLog('system', 'Error connecting to TTS service');
        console.error(e);
    };
}

// Speech to Speech Functions
async function startSpeechToSpeech() {
    initAudio();
    nextPlayTime = audioCtx.currentTime;
    resetLatencyHistory();
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    s2sWs = new WebSocket(`${protocol}//${window.location.host}/ws/s2s`);
    s2sWs.binaryType = 'arraybuffer';
    
    s2sWs.onopen = async () => {
        const aiConvToggle = document.getElementById('ai-conv-toggle');
        const claudeModelSelect = document.getElementById('claude-model-select');
        // Send initial configs
        s2sWs.send(JSON.stringify({
            voice: voiceSelect.value,
            speed: parseFloat(speedRange.value),
            language: languageSelect.value,
            task: taskSelect.value,
            stt_engine: sttEngineSelect.value,
            tts_engine: ttsEngineSelect.value,
            api_key: getApiKeys(),
            ai_conversation: aiConvToggle ? aiConvToggle.checked : false,
            claude_model: claudeModelSelect ? claudeModelSelect.value : "claude-3-5-sonnet-20241022"
        }));
        
        try {
            // Request microphone at 16000Hz (auto-resampled by browser!)
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            inputAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            micSource = inputAudioCtx.createMediaStreamSource(mediaStream);
            
            // ScriptProcessorNode with bufferSize = 512 to feed VAD exactly what it expects
            scriptProcessor = inputAudioCtx.createScriptProcessor(512, 1, 1);
            
            scriptProcessor.onaudioprocess = (e) => {
                const inputBuffer = e.inputBuffer.getChannelData(0);
                if (s2sWs && s2sWs.readyState === WebSocket.OPEN) {
                    s2sWs.send(inputBuffer.buffer); // Send float32 buffer directly
                }
            };
            
            micSource.connect(scriptProcessor);
            scriptProcessor.connect(inputAudioCtx.destination);
            
            isRecording = true;
            document.getElementById('mic-btn').classList.add('active');
            document.getElementById('mic-status-label').textContent = 'Listening (Speak now...)';
            document.getElementById('mic-status-label').classList.add('recording');
            addLog('system', 'Listening started. Speak into your microphone...');
            
        } catch (err) {
            addLog('system', 'Failed to get microphone permissions.');
            console.error(err);
            s2sWs.close();
        }
    };
    
    s2sWs.onmessage = (event) => {
        if (typeof event.data === 'string') {
            const data = JSON.parse(event.data);
            if (data.type === 'status') {
                if (data.status === 'speech_detected') {
                    document.getElementById('mic-status-label').textContent = 'Recording Speech...';
                } else if (data.status === 'processing') {
                    document.getElementById('mic-status-label').textContent = 'Processing Voice...';
                }
            } else if (data.type === 'transcript') {
                addLog('user', data.text);
            } else if (data.type === 'ai_transcript') {
                addLog('ai', data.text);
            } else if (data.type === 'latency') {
                latencyHistory.vad.push(300);
                if (data.stt_latency_ms) latencyHistory.stt.push(data.stt_latency_ms);
                if (data.tts_gen_ms) latencyHistory.tts.push(data.tts_gen_ms);
                if (data.llm_latency_ms) latencyHistory.llm.push(data.llm_latency_ms);
                if (data.total_latency_ms) latencyHistory.total.push(data.total_latency_ms);
                updateLatencyUI();
            } else if (data.type === 'done') {
                document.getElementById('mic-status-label').textContent = 'Listening (Speak now...)';
            }
        } else {
            // Receive voice response chunk
            const float32Array = new Float32Array(event.data);
            playAudioChunk(float32Array);
        }
    };
    
    s2sWs.onclose = () => {
        stopSpeechToSpeech();
    };
}

function stopSpeechToSpeech() {
    isRecording = false;
    document.getElementById('mic-btn').classList.remove('active');
    document.getElementById('mic-status-label').textContent = 'Click to Start Listening';
    document.getElementById('mic-status-label').classList.remove('recording');
    
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    if (micSource) {
        micSource.disconnect();
        micSource = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (inputAudioCtx) {
        inputAudioCtx.close();
        inputAudioCtx = null;
    }
    if (s2sWs) {
        s2sWs.close();
        s2sWs = null;
    }
    addLog('system', 'Connection closed.');
}

function toggleMic() {
    if (isRecording) {
        stopSpeechToSpeech();
    } else {
        startSpeechToSpeech();
    }
}
