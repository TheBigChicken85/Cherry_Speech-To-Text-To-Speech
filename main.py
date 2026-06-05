import multiprocessing
multiprocessing.freeze_support()

import tkinter as tk
from tkinter import ttk
import threading
import pyttsx3
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import torch
import requests
import time
import io
import queue
import subprocess
import sys
import os
from tkinter import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FP16   = torch.cuda.is_available()          # fp16 only on CUDA — fixes silent crash on CPU

VOICEBOX_URL = 'http://127.0.0.1:17493'
VOICEBOX_PREFIX = '[VB] '                    # prefix shown in voice dropdown

# ── TTS model list ────────────────────────────────────────────────────────────
# Display names shown in the UI → maps to the API id sent in generate payloads.
# The API is queried at runtime; this dict is the fallback if the endpoint is
# unavailable or returns no recognisable id fields.

VB_TTS_MODELS_FALLBACK = {
    'Qwen3-TTS 1.7B':        'Qwen3-TTS 1.7B',
    'Qwen3-TTS 0.6B':        'Qwen3-TTS 0.6B',
    'Qwen CustomVoice 1.7B': 'Qwen CustomVoice 1.7B',
    'Qwen CustomVoice 0.6B': 'Qwen CustomVoice 0.6B',
    'LuxTTS':                'LuxTTS',
    'Chatterbox':            'Chatterbox',
    'Chatterbox Turbo':      'Chatterbox Turbo',
    'TADA 1B':               'TADA 1B',
    'TADA 3B Multilingual':  'TADA 3B Multilingual',
    'Kokoro 82M':            'Kokoro 82M',
}

# Populated at runtime by _load_vb_model_map(); keyed by display name → api id
_vb_model_map: dict[str, str] = dict(VB_TTS_MODELS_FALLBACK)


def _load_vb_model_map() -> dict[str, str]:
    """
    Query GET /models and build a display_name → api_id mapping.
    Each entry may look like:
        {"id": "kokoro_82m", "name": "Kokoro 82M", ...}
    Falls back to VB_TTS_MODELS_FALLBACK on any error or missing fields.
    """
    try:
        r = requests.get(f'{VOICEBOX_URL}/models', timeout=4)
        r.raise_for_status()
        data = r.json()
        models = data if isinstance(data, list) else data.get('models', [])
        result = {}
        for m in models:
            if isinstance(m, dict):
                name = m.get('name') or m.get('display_name') or str(m.get('id', ''))
                mid  = m.get('id')   or m.get('model_id')    or name
                if name:
                    result[name] = mid
            else:
                s = str(m)
                result[s] = s
        return result if result else dict(VB_TTS_MODELS_FALLBACK)
    except Exception:
        return dict(VB_TTS_MODELS_FALLBACK)

# ── Launch Voicebox silently in background, minimized to tray ─────────────────

def launch_voicebox_silent():
    """
    Try to launch Voicebox without a visible window, minimized to tray.
    First tries --minimized flag; then schedules a PowerShell fallback to
    minimize whatever window appears after startup.
    """
    candidates = [
        r'C:\Program Files\Voicebox\Voicebox.exe',
        r'C:\Program Files (x86)\Voicebox\Voicebox.exe',
        os.path.expanduser(r'~\AppData\Local\Voicebox\Voicebox.exe'),
        os.path.expanduser(r'~\AppData\Local\Programs\Voicebox\Voicebox.exe'),
        'voicebox',   # if it's on PATH
    ]

    creation_flags = 0
    if sys.platform == 'win32':
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS

    for path in candidates:
        try:
            subprocess.Popen(
                [path, '--minimized'],          # ask Voicebox to start minimized/tray
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
            # Fallback: after 3 s, minimize via PowerShell in case --minimized isn't supported
            if sys.platform == 'win32':
                threading.Timer(3.0, _minimize_voicebox_window).start()
            return True
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return False


def _minimize_voicebox_window():
    """
    Fallback: use PowerShell + Win32 ShowWindow(SW_MINIMIZE) to collapse
    any visible Voicebox window. Runs silently; fails gracefully.
    """
    if sys.platform != 'win32':
        return
    # Inline C# injected via Add-Type so we can call ShowWindow without an external DLL
    script = r"""
$proc = Get-Process -Name 'Voicebox' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($proc -and $proc.MainWindowHandle -ne 0) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WinApi {
    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}
"@
    [WinApi]::ShowWindow($proc.MainWindowHandle, 2) | Out-Null
}
"""
    try:
        subprocess.Popen(
            ['powershell', '-NonInteractive', '-WindowStyle', 'Hidden', '-Command', script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception:
        pass


# ── Connect-or-launch logic ───────────────────────────────────────────────────

def _voicebox_api_alive() -> bool:
    """Return True if the Voicebox HTTP backend is already responding."""
    try:
        r = requests.get(f'{VOICEBOX_URL}/profiles', timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def connect_or_launch_voicebox() -> str:
    """
    If the Voicebox API is already reachable (window already open), attach to it.
    Otherwise launch it silently and minimized.
    Returns 'connected' | 'launched' | 'not_found'.
    """
    if _voicebox_api_alive():
        return 'connected'       # already running — just use it
    if launch_voicebox_silent():
        return 'launched'
    return 'not_found'


# Run at import time — connect to existing instance or launch a new one
_voicebox_status = connect_or_launch_voicebox()  # 'connected' | 'launched' | 'not_found'

# ── pyttsx3 TTS engine ────────────────────────────────────────────────────────

_pyttsx3_engine = pyttsx3.init()

def get_pyttsx3_voices():
    return [(v.name, v.id) for v in _pyttsx3_engine.getProperty('voices')]

def speak_pyttsx3(text, voice_id, log_cb):
    try:
        _pyttsx3_engine.setProperty('voice', voice_id)
        _pyttsx3_engine.say(text)
        _pyttsx3_engine.runAndWait()
    except Exception as e:
        log_cb(f'pyttsx3 error: {e}')

# ── Voicebox API helpers ───────────────────────────────────────────────────────

def voicebox_running():
    """Return True if Voicebox backend is reachable."""
    try:
        r = requests.get(f'{VOICEBOX_URL}/profiles', timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def get_voicebox_profiles():
    """Return list of (display_name, profile_id) for all Voicebox voice profiles."""
    try:
        r = requests.get(f'{VOICEBOX_URL}/profiles', timeout=4)
        r.raise_for_status()
        data = r.json()
        profiles = data if isinstance(data, list) else data.get('profiles', [])
        return [(f"{VOICEBOX_PREFIX}{p['name']}", p['id']) for p in profiles]
    except Exception:
        return []

def get_voicebox_tts_models() -> list[str]:
    """Return display names from the live model map (or fallback)."""
    return list(_vb_model_map.keys())


def get_vb_model_id(display_name: str) -> str:
    """Resolve a display name to the API id Voicebox expects."""
    return _vb_model_map.get(display_name, display_name)

# tracks log line tags for VB generation messages — cleared on stop
_vb_log_tags: list[str] = []
_vb_log_lock = threading.Lock()


# ── Voicebox output folder (user-configurable) ───────────────────────────────

VOICEBOX_OUTPUT_DIR: str = os.path.expanduser(
    r'~\AppData\Roaming\sh.voicebox.app\generations' 
)

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a'}


def _play_audio_file(path: str, out_device_idx):
    """Read an audio file from disk and play it on the chosen output device."""
    audio_data, samplerate = sf.read(path, dtype='float32')
    if audio_data.ndim == 1:
        audio_data = audio_data[:, np.newaxis]
    sd.play(audio_data, samplerate=samplerate, device=out_device_idx)
    sd.wait()


def _play_audio_bytes(data: bytes, out_device_idx):
    """Decode and play raw audio bytes."""
    audio_data, samplerate = sf.read(io.BytesIO(data), dtype='float32')
    if audio_data.ndim == 1:
        audio_data = audio_data[:, np.newaxis]
    sd.play(audio_data, samplerate=samplerate, device=out_device_idx)
    sd.wait()


def speak_voicebox(text, profile_id, tts_model, out_device_idx, log_cb):
    """
    1. Snapshot existing files in VOICEBOX_OUTPUT_DIR.
    2. POST /generate to kick off generation.
    3. Watch the folder — as soon as a new audio file appears, play it and stop watching.
    """
    try:
        # ── 1. snapshot ──────────────────────────────────────────────────────
        watch_dir = VOICEBOX_OUTPUT_DIR
        if not os.path.isdir(watch_dir):
            log_cb(f'Voicebox: output folder not found: {watch_dir}', vb=True)
            log_cb('Voicebox: set the correct folder in the OUTPUT FOLDER field.', vb=True)
            return

        before = {
            f for f in os.listdir(watch_dir)
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
        }

        # ── 2. fire the request (don't wait for the body — generation is async) ─
        payload = {'text': text, 'profile_id': profile_id, 'language': 'en'}
        if tts_model:
            payload['model'] = get_vb_model_id(tts_model)

        log_cb('Voicebox: sending request…', vb=True)
        r = requests.post(f'{VOICEBOX_URL}/generate', json=payload, timeout=300)
        r.raise_for_status()

        # ── 3. watch for new file ─────────────────────────────────────────────
        log_cb('Voicebox: waiting for audio file…', vb=True)
        deadline = time.time() + 300   # 5-minute ceiling
        while time.time() < deadline:
            time.sleep(0.25)
            after = {
                f for f in os.listdir(watch_dir)
                if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
            }
            new_files = after - before
            if not new_files:
                continue

            # Pick the newest of any new files (handles rare simultaneous writes)
            new_path = max(
                (os.path.join(watch_dir, f) for f in new_files),
                key=os.path.getmtime
            )

            # Wait briefly for the file to finish writing
            prev_size = -1
            for _ in range(20):   # up to 2 s
                time.sleep(0.1)
                try:
                    cur_size = os.path.getsize(new_path)
                except OSError:
                    cur_size = 0
                if cur_size == prev_size and cur_size > 0:
                    break
                prev_size = cur_size

            log_cb(f'Voicebox: playing {os.path.basename(new_path)}', vb=True)
            _play_audio_file(new_path, out_device_idx)
            return

        log_cb('Voicebox: timed out waiting for audio file', vb=True)

    except Exception as e:
        log_cb(f'Voicebox error: {e}', vb=True)

    except Exception as e:
        log_cb(f'Voicebox error: {e}', vb=True)

# ── unified speak dispatcher ──────────────────────────────────────────────────

def speak(text, voice_id, tts_model, out_device_idx, log_cb):
    """Route to Voicebox or pyttsx3 depending on voice_id prefix."""
    if voice_id.startswith(VOICEBOX_PREFIX):
        profile_id = voice_id[len(VOICEBOX_PREFIX):]
        speak_voicebox(text, profile_id, tts_model, out_device_idx, log_cb)
    else:
        speak_pyttsx3(text, voice_id, log_cb)

# ── audio devices ─────────────────────────────────────────────────────────────

def get_input_devices():
    return [(f"{i}: {d['name']}", i)
            for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]

def get_output_devices():
    return [(f"{i}: {d['name']}", i)
            for i, d in enumerate(sd.query_devices()) if d['max_output_channels'] > 0]

# ── Whisper VAD recorder ──────────────────────────────────────────────────────

running       = False
SAMPLE_RATE   = 16000
BLOCK_SIZE    = 1600          # 100 ms blocks
SILENCE_RATIO = 0.007
SPEECH_BLOCKS = 3
END_SILENCE   = 12
MAX_RECORD_S  = 15

def recorder_loop(voice_id, tts_model, out_device_idx, in_device_idx, model_name, log_cb):
    global running

    log_cb(f'Loading Whisper "{model_name}" on {DEVICE}…')
    try:
        model = whisper.load_model(model_name, device=DEVICE)
    except Exception as e:
        log_cb(f'Whisper load error: {e}')
        return

    log_cb('Listening… speak now.')
    last_text = [None]

    pre_roll   = []
    recording  = []
    speech_cnt = 0
    silence_cnt= 0
    in_speech  = False

    def process_utterance(frames):
        audio = np.concatenate(frames).flatten()
        if len(audio) < SAMPLE_RATE * 0.3:
            return
        try:
            result = model.transcribe(audio, fp16=FP16, language=None)
            text   = result['text'].strip()
            lang   = result.get('language', '?')
            if not text or text == last_text[0]:
                return
            last_text[0] = text
            log_cb(f'[{lang}] "{text}"')
            threading.Thread(
                target=speak,
                args=(text, voice_id, tts_model, out_device_idx, log_cb),
                daemon=True
            ).start()
        except Exception as e:
            log_cb(f'Transcribe error: {e}')

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=BLOCK_SIZE,
            device=in_device_idx,
        ) as stream:
            while running:
                block, overflowed = stream.read(BLOCK_SIZE)
                if overflowed:
                    pass

                rms = float(np.sqrt(np.mean(block ** 2)))
                is_speech = rms > SILENCE_RATIO

                if not in_speech:
                    pre_roll.append(block.copy())
                    if len(pre_roll) > SPEECH_BLOCKS + 2:
                        pre_roll.pop(0)

                    if is_speech:
                        speech_cnt += 1
                        if speech_cnt >= SPEECH_BLOCKS:
                            in_speech   = True
                            silence_cnt = 0
                            recording   = list(pre_roll)
                            pre_roll.clear()
                            speech_cnt  = 0
                    else:
                        speech_cnt = max(0, speech_cnt - 1)
                else:
                    recording.append(block.copy())
                    total_s = len(recording) * BLOCK_SIZE / SAMPLE_RATE

                    if is_speech:
                        silence_cnt = 0
                    else:
                        silence_cnt += 1

                    if silence_cnt >= END_SILENCE or total_s >= MAX_RECORD_S:
                        in_speech = False
                        threading.Thread(
                            target=process_utterance,
                            args=(recording,),
                            daemon=True
                        ).start()
                        recording   = []
                        silence_cnt = 0

    except Exception as e:
        if running:
            log_cb(f'Recording error: {e}')

    log_cb('Stopped.')

# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Cherry STTTS')
        self.configure(bg='#0f0f0f')
        self.minsize(620, 660)
        self._build_ui()

    # ── voice list helpers ────────────────────────────────────────────────────

    def _load_voices(self):
        voices = []
        for name, vid in get_pyttsx3_voices():
            voices.append((name, vid))
        for name, pid in get_voicebox_profiles():
            voices.append((name, VOICEBOX_PREFIX + pid))
        return voices

    def _refresh_voices(self):
        self._voices = self._load_voices()
        self._voice_map = {v[0]: v[1] for v in self._voices}
        names = [v[0] for v in self._voices]
        self.voice_cb['values'] = names
        if names:
            self.voice_cb.current(0)
        vb_count = sum(1 for n in names if n.startswith(VOICEBOX_PREFIX))
        if vb_count:
            prefix = 'Already running' if _voicebox_status == 'connected' else 'Voicebox'
            self.log(f'{prefix} connected — {vb_count} profile(s) loaded.')
        else:
            if _voicebox_status == 'connected':
                self.log('Voicebox is running but returned no profiles.')
            elif _voicebox_status == 'launched':
                self.log('Voicebox launched in background — waiting for profiles…')
            else:
                self.log('Voicebox not found. Open Voicebox app to use its voices.')

    def _refresh_vb_models(self):
        """Re-query Voicebox for the live model list and reload the dropdown."""
        global _vb_model_map
        _vb_model_map = _load_vb_model_map()
        models = get_voicebox_tts_models()
        self.vb_model_cb['values'] = models
        if models:
            self.vb_model_cb.current(0)

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        PAD        = 16
        BG         = '#0f0f0f'
        CARD       = '#1a1a1a'
        ACC        = "#fca4ff"
        FG         = "#ffc1ff"
        MUTED      = '#555'
        GREEN      = "#b900aa"
        FONT_TITLE = ('Courier New', 20, 'bold')
        FONT_LABEL = ('Courier New', 9)
        FONT_BTN   = ('Courier New', 10, 'bold')
        FONT_LOG   = ('Courier New', 9)
        FONT_INPUT = ('Courier New', 10)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        outer = tk.Frame(self, bg=BG)
        outer.grid(row=0, column=0, sticky='nsew')
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(7, weight=1)

        # title
        tk.Label(outer, text='Cherry STTTS', font=FONT_TITLE,
                 bg=BG, fg=ACC).grid(row=0, column=0, padx=PAD, pady=(PAD, 2), sticky='w')
        tk.Label(outer, text='speech → text → speech  |  whisper + pyttsx3 + voicebox',
                 font=FONT_LABEL, bg=BG, fg=MUTED).grid(row=1, column=0, padx=PAD, sticky='w')
        tk.Frame(outer, bg=ACC, height=1).grid(row=2, column=0, padx=PAD,
                                                pady=(6, PAD), sticky='ew')
        try:
            self.iconbitmap("icon.ico")
        except tk.TclError:
            pass
        # ── settings ──────────────────────────────────────────────────────────
        sf_frame = tk.Frame(outer, bg=BG)
        sf_frame.grid(row=3, column=0, padx=PAD, sticky='ew')
        sf_frame.columnconfigure(1, weight=1)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox',
                        fieldbackground=CARD,
                        background=CARD,
                        foreground='black',
                        selectbackground=CARD,
                        selectforeground='black',
                        bordercolor='#333',
                        arrowcolor=ACC)
        self.option_add('*TCombobox*Listbox.foreground', 'black')
        self.option_add('*TCombobox*Listbox.background', '#e8e8e8')
        self.option_add('*TCombobox*Listbox.selectBackground', ACC)
        self.option_add('*TCombobox*Listbox.selectForeground', '#0f0f0f')

        def lbl_cb(parent, r, label, var, values):
            tk.Label(parent, text=label, font=FONT_LABEL,
                     bg=BG, fg='#777').grid(row=r, column=0, padx=(0, 10), pady=5, sticky='w')
            cb = ttk.Combobox(parent, textvariable=var, values=values, state='readonly')
            cb.grid(row=r, column=1, pady=5, sticky='ew')
            if values:
                cb.current(0)
            return cb

        # VOICE
        self.voice_var = tk.StringVar()
        self._voices   = []
        self._voice_map = {}
        tk.Label(sf_frame, text='VOICE', font=FONT_LABEL,
                 bg=BG, fg='#777').grid(row=0, column=0, padx=(0, 10), pady=5, sticky='w')
        self.voice_cb = ttk.Combobox(sf_frame, textvariable=self.voice_var,
                                     values=[], state='readonly')
        self.voice_cb.grid(row=0, column=1, pady=5, sticky='ew')
        tk.Button(sf_frame, text='↺', font=FONT_BTN,
                  bg=CARD, fg=ACC, activebackground='#2a2a2a',
                  relief='flat', padx=6, pady=2,
                  command=self._refresh_voices).grid(row=0, column=2, padx=(6, 0))

        # MIC IN
        in_devices   = get_input_devices()
        self.in_var  = tk.StringVar()
        self._in_ids = {d[0]: d[1] for d in in_devices}
        lbl_cb(sf_frame, 1, 'MIC IN', self.in_var, [d[0] for d in in_devices])

        # AUDIO OUT
        out_devices   = get_output_devices()
        self.out_var  = tk.StringVar()
        self._out_ids = {d[0]: d[1] for d in out_devices}
        lbl_cb(sf_frame, 2, 'AUDIO OUT', self.out_var, [d[0] for d in out_devices])

        # VOICEBOX OUTPUT FOLDER
        tk.Label(sf_frame, text='VB OUTPUT FOLDER', font=FONT_LABEL,
                 bg=BG, fg='#777').grid(row=3, column=0, padx=(0, 10), pady=5, sticky='w')
        self.vb_dir_var = tk.StringVar(value=VOICEBOX_OUTPUT_DIR)
        vb_dir_entry = tk.Entry(sf_frame, textvariable=self.vb_dir_var,
                                font=FONT_LABEL, bg=CARD, fg=FG,
                                insertbackground=ACC, relief='flat', bd=4)
        vb_dir_entry.grid(row=3, column=1, pady=5, sticky='ew')
        tk.Button(sf_frame, text='…', font=FONT_BTN,
                  bg=CARD, fg=ACC, activebackground='#2a2a2a',
                  relief='flat', padx=6, pady=2,
                  command=self._browse_vb_dir).grid(row=3, column=2, padx=(6, 0))
        self.vb_dir_var.trace_add('write', lambda *_: self._update_vb_dir())

        # ── AI MODELS row ─────────────────────────────────────────────────────
        tk.Label(sf_frame, text='AI MODELS', font=FONT_LABEL,
                 bg=BG, fg='#777').grid(row=4, column=0, padx=(0, 10), pady=5, sticky='w')

        models_frame = tk.Frame(sf_frame, bg=BG)
        models_frame.grid(row=4, column=1, columnspan=2, pady=5, sticky='ew')
        models_frame.columnconfigure(0, weight=1)
        models_frame.columnconfigure(1, weight=1)

        # Whisper (left half)
        whisper_sub = tk.Frame(models_frame, bg=BG)
        whisper_sub.grid(row=0, column=0, sticky='ew', padx=(0, 4))
        whisper_sub.columnconfigure(0, weight=1)
        tk.Label(whisper_sub, text='WHISPER', font=FONT_LABEL,
                 bg=BG, fg='#555').grid(row=0, column=0, sticky='w')
        self.model_var = tk.StringVar(value='base')
        self.whisper_cb = ttk.Combobox(whisper_sub, textvariable=self.model_var,
                                       values=['tiny', 'base', 'small', 'medium', 'large'],
                                       state='readonly')
        self.whisper_cb.grid(row=1, column=0, sticky='ew')
        self.whisper_cb.current(1)

        # Voicebox TTS model (right half)
        vb_sub = tk.Frame(models_frame, bg=BG)
        vb_sub.grid(row=0, column=1, sticky='ew', padx=(4, 0))
        vb_sub.columnconfigure(0, weight=1)
        tk.Label(vb_sub, text='VOICEBOX MODEL', font=FONT_LABEL,
                 bg=BG, fg='#555').grid(row=0, column=0, sticky='w')
        self.vb_model_var = tk.StringVar()
        self.vb_model_cb = ttk.Combobox(vb_sub, textvariable=self.vb_model_var,
                                        values=get_voicebox_tts_models(),
                                        state='readonly')
        self.vb_model_cb.grid(row=1, column=0, sticky='ew')
        self.vb_model_cb.current(0)

        # MIC SENSITIVITY slider
        tk.Label(sf_frame, text='MIC SENSITIVITY', font=FONT_LABEL,
                 bg=BG, fg='#777').grid(row=5, column=0, padx=(0, 10), pady=5, sticky='w')
        self.thresh_var = tk.DoubleVar(value=SILENCE_RATIO)
        thresh_scale = tk.Scale(
            sf_frame, variable=self.thresh_var,
            from_=0.002, to=0.05, resolution=0.001,
            orient='horizontal',
            bg=BG, fg=GREEN,
            highlightthickness=0,
            troughcolor="#ffb8fb",
            activebackground=GREEN,
            sliderrelief='flat',
            command=lambda _: self._update_threshold()
        )
        try:
            thresh_scale.configure(highlightbackground=GREEN)
        except tk.TclError:
            pass
        thresh_scale.grid(row=5, column=1, pady=5, sticky='ew')

        tk.Frame(outer, bg='#222', height=1).grid(row=4, column=0, padx=PAD,
                                                   pady=(PAD, 8), sticky='ew')

        # ── type-to-speak ──────────────────────────────────────────────────────
        inp = tk.Frame(outer, bg=BG)
        inp.grid(row=5, column=0, padx=PAD, pady=(0, 8), sticky='ew')
        inp.columnconfigure(0, weight=1)
        tk.Label(inp, text='TYPE TO SPEAK', font=FONT_LABEL,
                 bg=BG, fg='#777').grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 4))
        self.text_input = tk.Entry(inp, font=FONT_INPUT, bg=CARD, fg=FG,
                                   insertbackground=ACC, relief='flat', bd=6)
        self.text_input.grid(row=1, column=0, sticky='ew', padx=(0, 8))
        self.text_input.bind('<Return>', lambda e: self.speak_typed())
        tk.Button(inp, text='SPEAK', font=FONT_BTN,
                  bg=ACC, fg='#0f0f0f', activebackground='#aadd00',
                  relief='flat', padx=12, pady=4,
                  command=self.speak_typed).grid(row=1, column=1)

        tk.Frame(outer, bg='#222', height=1).grid(row=6, column=0, padx=PAD,
                                                   pady=(8, 8), sticky='ew')

        # ── log ───────────────────────────────────────────────────────────────
        log_frame = tk.Frame(outer, bg=BG)
        log_frame.grid(row=7, column=0, padx=PAD, pady=(0, 8), sticky='nsew')
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)
        tk.Label(log_frame, text='LOG', font=FONT_LABEL, bg=BG, fg='#777').grid(
            row=0, column=0, sticky='w')
        sb = tk.Scrollbar(log_frame, bg=BG, troughcolor=CARD, relief='flat', width=10)
        sb.grid(row=1, column=1, sticky='ns')
        self.log_box = tk.Text(log_frame, font=FONT_LOG, bg=CARD, fg=ACC,
                               insertbackground=ACC, relief='flat', bd=0,
                               state='disabled', wrap='word', yscrollcommand=sb.set)
        self.log_box.grid(row=1, column=0, sticky='nsew')
        sb.config(command=self.log_box.yview)

        # ── control buttons ───────────────────────────────────────────────────
        btn_frame = tk.Frame(outer, bg=BG)
        btn_frame.grid(row=8, column=0, padx=PAD, pady=(0, PAD), sticky='w')
        self.start_btn = tk.Button(
            btn_frame, text='▶  START', font=FONT_BTN,
            bg=ACC, fg='#0f0f0f', activebackground='#aadd00',
            relief='flat', padx=20, pady=10, command=self.start)
        self.start_btn.pack(side='left', padx=(0, 10))
        self.stop_btn = tk.Button(
            btn_frame, text='■  STOP', font=FONT_BTN,
            bg=CARD, fg=MUTED, activebackground='#2a2a2a',
            relief='flat', padx=20, pady=10, state='disabled', command=self.stop)
        self.stop_btn.pack(side='left')

        self.after(100, self._refresh_voices)
        self.after(3000, self._refresh_vb_models)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _update_threshold(self):
        global SILENCE_RATIO
        SILENCE_RATIO = self.thresh_var.get()

    def _browse_vb_dir(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title='Select Voicebox output folder',
                                    initialdir=self.vb_dir_var.get())
        if d:
            self.vb_dir_var.set(d)

    def _update_vb_dir(self):
        global VOICEBOX_OUTPUT_DIR
        VOICEBOX_OUTPUT_DIR = self.vb_dir_var.get()

    def _lock_settings(self, lock):
        state = 'disabled' if lock else 'readonly'
        for w in self.voice_cb.master.winfo_children():
            if isinstance(w, ttk.Combobox):
                w.config(state=state)
        for cb in (self.whisper_cb, self.vb_model_cb):
            cb.config(state=state)
        self.text_input.config(state='disabled' if lock else 'normal')

    def log(self, message, vb=False):
        def _append():
            self.log_box.config(state='normal')
            if vb:
                tag = f'vb_{id(message)}_{time.time()}'
                self.log_box.insert('end', '> ' + message + '\n', tag)
                with _vb_log_lock:
                    _vb_log_tags.append(tag)
            else:
                self.log_box.insert('end', '> ' + message + '\n')
            self.log_box.see('end')
            self.log_box.config(state='disabled')
        self.after(0, _append)

    def speak_typed(self):
        text = self.text_input.get().strip()
        if not text:
            return
        self.text_input.delete(0, 'end')
        self.log(f'Text: "{text}"')
        voice_id    = self._voice_map.get(self.voice_var.get(), '')
        tts_model   = self.vb_model_var.get() or None
        out_dev_idx = self._out_ids.get(self.out_var.get(), None)
        threading.Thread(
            target=speak, args=(text, voice_id, tts_model, out_dev_idx, self.log),
            daemon=True).start()

    def start(self):
        global running
        running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal', fg='#f0f0f0')
        self._lock_settings(True)

        voice_id    = self._voice_map.get(self.voice_var.get(), '')
        tts_model   = self.vb_model_var.get() or None
        in_idx      = self._in_ids.get(self.in_var.get(), None)
        out_idx     = self._out_ids.get(self.out_var.get(), None)
        model       = self.model_var.get()

        threading.Thread(
            target=recorder_loop,
            args=(voice_id, tts_model, out_idx, in_idx, model, self.log),
            daemon=True
        ).start()

    def stop(self):
        global running
        running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled', fg='#555')
        self._lock_settings(False)
        self.log('Stopping…')
        self.after(100, self._clear_vb_log_lines)

    def _clear_vb_log_lines(self):
        """Delete all log lines that were tagged as Voicebox generation messages."""
        with _vb_log_lock:
            tags = list(_vb_log_tags)
            _vb_log_tags.clear()
        self.log_box.config(state='normal')
        for tag in tags:
            ranges = self.log_box.tag_ranges(tag)
            if ranges:
                self.log_box.delete(ranges[0], ranges[1])
        self.log_box.config(state='disabled')


if __name__ == '__main__':
    app = App()
    app.mainloop()
