# Research Report: Willow Voice & Push-to-Talk Dictation Architecture

## Table of Contents

1. [What is Willow Voice?](#1-what-is-willow-voice)
2. [Push-to-Talk / Hotkey Pattern](#2-push-to-talk--hotkey-pattern)
3. [Dictation into Any App](#3-dictation-into-any-app)
4. [Architecture: Audio Capture -> Transcription -> Text Injection](#4-architecture-audio-capture---transcription---text-injection)
5. [macOS Global Hotkeys in Python](#5-macos-global-hotkeys-in-python)
6. [Text Injection into Focused App](#6-text-injection-into-focused-app)
7. [Similar Tools Comparison](#7-similar-tools-comparison)
8. [Accessibility Permissions](#8-accessibility-permissions)
9. [Recommended Architecture for quill](#9-recommended-architecture-for-quill)

---

## 1. What is Willow Voice?

**Willow Voice** (https://willowvoice.com) is a **commercial, closed-source** AI dictation application for macOS (and soon Windows/iPhone). It is a Y Combinator-backed product that converts speech to text in any application.

**Important disambiguation**: There is a *separate* open-source project called "Willow" (https://github.com/HeyWillow/willow) which is a voice assistant alternative to Alexa/Google Home for smart home use. This is NOT the same product. The dictation tool "Willow Voice" is closed-source.

### Key Features
- **Hold-to-talk dictation**: Press and hold a hotkey (default: Fn key), speak, release to transcribe
- **No Hands Mode**: Double-tap the hotkey to toggle continuous listening
- **Context-aware transcription**: Reads screen context to improve accuracy on names, technical terms
- **Smart formatting**: Automatic punctuation, paragraph structure, email formatting
- **Custom dictionaries**: User-defined words/terms for correct spelling
- **Sub-1-second latency**: Cloud-based processing with no voice data storage
- **50+ language support**
- **40%+ more accurate** than built-in macOS dictation (per their claims)

### Pricing
- Free tier available (no credit card required)
- Paid plans for advanced features

### Architecture (inferred)
Willow Voice is cloud-based - audio is sent to their servers for transcription and formatting, then the text is injected at the cursor position. They claim privacy-first design with no voice data storage.

---

## 2. Push-to-Talk / Hotkey Pattern

### Willow Voice's Hotkey System

**Default key**: Function (Fn) key

**Modes**:
1. **Hold-to-talk**: Press and hold Fn -> speak -> release Fn -> text appears
2. **No Hands Mode**: Double-tap Fn -> speak freely -> tap Fn again to stop

**Customization**:
- Up to 4 different hotkeys can be configured
- Each can be a single key or a combination (e.g., Fn+Ctrl, Cmd+D, Fn+D)
- Combination hotkeys require at least one modifier key
- Popular alternatives: Option, Command (left or right variants)
- Separate hotkeys for: Dictation, Assistant mode, Hands-Free mode

**UX Flow**:
1. User clicks into any text field (Slack, Gmail, Google Docs, etc.)
2. Press and hold hotkey
3. A small floating bar appears indicating recording is active
4. Speak naturally
5. Release hotkey
6. Text is formatted and placed at cursor position

### Common Hotkey Patterns Across Tools

| Tool | Default Key | Hold-to-talk | Toggle Mode |
|------|------------|--------------|-------------|
| Willow Voice | Fn | Yes (hold) | Double-tap for No Hands |
| SuperWhisper | Configurable | Yes (mouse hold) | Quick click toggles |
| open-wispr | Globe key (keyCode 63) | Yes (hold) | No |
| push-to-talk-dictate | Left Option | Yes (hold) | No |
| whisper-dictation (foges) | Cmd+Option | No | Toggle on/off |
| Wispr Flow | Configurable | Yes | Yes |

---

## 3. Dictation into Any App

All these tools share a common pattern for typing transcribed text into the currently focused application:

### Approach 1: Clipboard + Paste (Most Common)
1. Copy transcribed text to system clipboard
2. Simulate Cmd+V keystroke
3. Optionally restore previous clipboard contents

**Pros**: Works in virtually all apps, handles Unicode/special characters well
**Cons**: Clobbers clipboard contents, may trigger paste formatting in some apps

### Approach 2: Simulated Keystrokes (Character-by-character)
1. Use pynput or CGEventCreateKeyboardEvent to type each character
2. Simulate as if the user typed it

**Pros**: Doesn't touch clipboard, more natural
**Cons**: Slow for long text, may miss special characters, can be interrupted

### Approach 3: AppleScript/System Events
```bash
osascript -e 'tell application "System Events" to keystroke "Hello World"'
```

**Pros**: Simple, no additional dependencies
**Cons**: Limited character support, requires Accessibility permission

### What Real Projects Use

- **push-to-talk-dictate**: pynput Controller.type() + pyperclip as fallback
- **open-wispr**: Native Swift helper for "fast paste" (clipboard + Cmd+V)
- **whisper-dictation (foges)**: Likely pynput or pyautogui (not explicitly documented)
- **Willow Voice**: Proprietary implementation (closed source)
- **SuperWhisper**: Native macOS app with direct text injection

---

## 4. Architecture: Audio Capture -> Transcription -> Text Injection

### Generic Pipeline

```
[Hotkey Press Detected]
        |
        v
[Start Audio Capture] --> [Buffer audio in memory or temp file]
        |
[Hotkey Release Detected]
        |
        v
[Stop Audio Capture]
        |
        v
[Transcribe Audio] --> Whisper (local) or Cloud API
        |
        v
[Optional: LLM Post-Processing] --> Grammar fix, formatting, filler removal
        |
        v
[Inject Text] --> Clipboard+Paste or Keystroke simulation
```

### Latency Breakdown (typical local pipeline)

| Stage | Duration | Notes |
|-------|----------|-------|
| Audio capture | Real-time | No latency, streams during hold |
| Audio encoding | ~50ms | WAV/PCM, minimal overhead |
| Whisper transcription | 200ms-3s | Depends on model size and hardware |
| LLM post-processing | 200ms-1s | Optional, adds significant latency |
| Text injection | ~50ms | Clipboard+paste is nearly instant |
| **Total** | **300ms-4s** | Excluding speech duration |

### push-to-talk-dictate (Rasala) - Reference Architecture

```
Python 3.11+ on macOS Apple Silicon

Audio:      sounddevice + numpy + scipy
Hotkey:     pynput (Listener for key press/release)
ASR:        mlx-whisper (MLX-optimized Whisper Large V3)
LLM:        mlx-lm (Qwen2.5 or Phi-3-Mini for grammar cleanup)
Output:     pynput Controller.type() / pyperclip
Config:     Command-line args
```

Key files:
- `app.py` - Desktop app entry point, hotkey listener
- `transcribe.py` - Whisper transcription pipeline
- `audio.py` - Audio capture management

### open-wispr - Reference Architecture

```
Swift + whisper.cpp on macOS Apple Silicon

Audio:      Native macOS audio capture
Hotkey:     Native Swift key listener (Globe key, keyCode 63)
ASR:        whisper.cpp with Metal acceleration
Output:     Native Swift "fast paste" helper
Config:     ~/.config/open-wispr/config.json
Service:    Homebrew background service + menu bar icon
```

---

## 5. macOS Global Hotkeys in Python

### Option A: pynput (Recommended for Python)

**Library**: `pynput` (https://pypi.org/project/pynput/)

**Global Hotkey Registration**:
```python
from pynput import keyboard

def on_activate():
    print("Hotkey activated!")

# Method 1: GlobalHotKeys convenience class
with keyboard.GlobalHotKeys({
    '<cmd>+<shift>+d': on_activate,
}) as h:
    h.join()

# Method 2: Listener with manual key tracking (better for hold-to-talk)
from pynput.keyboard import Key, Listener

recording = False

def on_press(key):
    global recording
    if key == Key.alt_l:  # Left Option key
        if not recording:
            recording = True
            start_recording()

def on_release(key):
    global recording
    if key == Key.alt_l:
        if recording:
            recording = False
            stop_and_transcribe()

with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
```

**macOS Caveats**:
- Ctrl and Alt modifier combos may not work as global hotkeys on macOS
- Cmd-based combos work best
- Single modifier keys (Option, Fn) work well for hold-to-talk
- Requires Accessibility permission for the running process (Terminal, Python, or packaged app)
- On macOS, pynput uses Quartz event taps under the hood

### Option B: Quartz CGEventTap (Low-level, macOS-only)

```python
import Quartz
from Quartz import (
    CGEventTapCreate, kCGSessionEventTap, kCGHeadInsertEventTap,
    kCGEventKeyDown, kCGEventKeyUp, CGEventGetIntegerValueField,
    kCGKeyboardEventKeycode, CFRunLoopGetCurrent, CFRunLoopRun,
    CGEventTapEnable, CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource, kCFRunLoopCommonModes
)

def callback(proxy, event_type, event, refcon):
    keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
    if event_type == kCGEventKeyDown:
        print(f"Key down: {keycode}")
    elif event_type == kCGEventKeyUp:
        print(f"Key up: {keycode}")
    return event

event_mask = (1 << kCGEventKeyDown) | (1 << kCGEventKeyUp)
tap = CGEventTapCreate(
    kCGSessionEventTap,
    kCGHeadInsertEventTap,
    0,  # kCGEventTapOptionDefault
    event_mask,
    callback,
    None
)

source = CFMachPortCreateRunLoopSource(None, tap, 0)
CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes)
CGEventTapEnable(tap, True)
CFRunLoopRun()
```

**Pros**: Full control, can intercept any key including Fn and Globe
**Cons**: Complex, macOS-only, requires pyobjc or Quartz bindings

### Option C: NSEvent.addGlobalMonitorForEvents (Cocoa, macOS-only)

```python
from AppKit import NSEvent, NSKeyDownMask, NSKeyUpMask

def handler(event):
    print(f"Key: {event.keyCode()}")

NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
    NSKeyDownMask | NSKeyUpMask,
    handler
)

# Requires a running NSApplication / CFRunLoop
from PyObjCTools import AppHelper
AppHelper.runConsoleEventLoop()
```

**Pros**: Clean Cocoa API, good for packaged apps
**Cons**: Requires pyobjc, needs a run loop

### Recommendation

**Use pynput for Python projects**. It wraps Quartz event taps on macOS and provides a clean, cross-platform API. For hold-to-talk, use `Listener` with `on_press`/`on_release` rather than `GlobalHotKeys` (which only fires on press, not release).

For detecting the Globe/Fn key specifically, you may need the lower-level Quartz approach since pynput may not capture it reliably. The Globe key is keyCode 63 on macOS.

---

## 6. Text Injection into Focused App

### Method 1: Clipboard + Cmd+V (Recommended)

```python
import subprocess
from pynput.keyboard import Key, Controller

def inject_text(text: str):
    """Inject text into focused app via clipboard + paste."""
    # Save current clipboard (optional)
    try:
        old_clipboard = subprocess.run(
            ["pbpaste"], capture_output=True, text=True
        ).stdout
    except Exception:
        old_clipboard = None

    # Set clipboard to transcribed text
    subprocess.run(["pbcopy"], input=text, text=True)

    # Simulate Cmd+V
    keyboard = Controller()
    keyboard.press(Key.cmd)
    keyboard.press('v')
    keyboard.release('v')
    keyboard.release(Key.cmd)

    # Restore clipboard (optional, with small delay)
    # import time; time.sleep(0.1)
    # if old_clipboard is not None:
    #     subprocess.run(["pbcopy"], input=old_clipboard, text=True)
```

**Alternative using pyperclip**:
```python
import pyperclip
from pynput.keyboard import Key, Controller

def inject_text(text: str):
    pyperclip.copy(text)
    kb = Controller()
    kb.press(Key.cmd)
    kb.press('v')
    kb.release('v')
    kb.release(Key.cmd)
```

### Method 2: pynput Controller.type()

```python
from pynput.keyboard import Controller

def inject_text(text: str):
    keyboard = Controller()
    keyboard.type(text)  # Types character by character
```

**Pros**: No clipboard involvement
**Cons**: Slow for long text, may have issues with Unicode, can be interrupted by user input

### Method 3: AppleScript via osascript

```python
import subprocess

def inject_text(text: str):
    # Escape text for AppleScript
    escaped = text.replace('\\', '\\\\').replace('"', '\\"')
    subprocess.run([
        "osascript", "-e",
        f'tell application "System Events" to keystroke "{escaped}"'
    ])
```

**Pros**: Simple, no Python dependencies
**Cons**: Character escaping issues, limited Unicode support, slow for long text

### Method 4: AppleScript clipboard + paste

```python
import subprocess

def inject_text(text: str):
    subprocess.run([
        "osascript", "-e",
        f'set the clipboard to "{text}"',
        "-e",
        'tell application "System Events" to keystroke "v" using command down'
    ])
```

### Method 5: CGEventCreateKeyboardEvent (Low-level)

```python
from Quartz import (
    CGEventCreateKeyboardEvent, CGEventPost, kCGHIDEventTap,
    CGEventSetFlags, kCGEventFlagMaskCommand
)

def paste_from_clipboard():
    """Simulate Cmd+V at the OS level."""
    # Key code 9 = 'v'
    event_down = CGEventCreateKeyboardEvent(None, 9, True)
    event_up = CGEventCreateKeyboardEvent(None, 9, False)
    CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
    CGEventSetFlags(event_up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, event_down)
    CGEventPost(kCGHIDEventTap, event_up)
```

### Recommendation

**Use clipboard + Cmd+V paste** (Method 1). This is what most real-world tools use:
- Works in all apps (native, Electron, web browsers)
- Handles Unicode, emoji, multi-line text correctly
- Fast regardless of text length
- The only downside is clobbering the clipboard, which can be mitigated by save/restore

---

## 7. Similar Tools Comparison

### Commercial Tools

| Tool | Price | Local/Cloud | Platform | Key Feature |
|------|-------|-------------|----------|-------------|
| **Willow Voice** | Free + paid | Cloud | macOS, Windows, iOS | Context-aware, smart formatting |
| **SuperWhisper** | $8.49/mo | Local | macOS | Multiple Whisper model sizes |
| **Wispr Flow** | $8.25/mo | Cloud | macOS | Context-aware, AI command mode |
| **macOS Dictation** | Free | Cloud/Local | macOS | Built-in, no setup needed |

### Open-Source Tools

| Tool | Language | ASR Engine | Hotkey Lib | Text Injection |
|------|----------|-----------|------------|----------------|
| **push-to-talk-dictate** | Python | mlx-whisper | pynput | pynput + pyperclip |
| **open-wispr** | Swift | whisper.cpp | Native Swift | Native paste |
| **whisper-dictation (foges)** | Python | OpenAI Whisper | pynput/keyboard | pynput |
| **whisper-dictation (ashwin-pc)** | Python | OpenAI Whisper | Globe key | Paste at cursor |
| **whisper-writer** | Python | OpenAI Whisper | keyboard lib | pyautogui |
| **OpenWhispr** | Python | Whisper/Parakeet | Global hotkey | Cross-platform |
| **mlx-whisper-dictation** | Python | mlx-whisper | Configurable | Clipboard paste |

### Key Architectural Patterns

All tools follow the same fundamental pattern:
1. **Global hotkey listener** running in background
2. **Audio capture** triggered by hotkey press (hold) or toggle
3. **ASR engine** for transcription (Whisper variants)
4. **Optional LLM** for formatting/cleanup
5. **Text injection** via clipboard+paste or keystroke simulation

---

## 8. Accessibility Permissions

### Required macOS Permissions

For a push-to-talk dictation app on macOS, you need:

#### 1. Accessibility Permission
- **What**: Allows the app to monitor keyboard events globally and simulate keystrokes
- **Where**: System Settings > Privacy & Security > Accessibility
- **Required for**: pynput Listener, CGEventTap, keystroke simulation, Cmd+V paste
- **Granularity**: Per-application (Terminal.app, Python.app, or your packaged .app)

#### 2. Input Monitoring Permission
- **What**: Allows reading keyboard input from any application
- **Where**: System Settings > Privacy & Security > Input Monitoring
- **Required for**: Global key event monitoring (macOS Big Sur+)
- **Note**: On Big Sur+, this can override/supplement Accessibility permissions

#### 3. Microphone Permission
- **What**: Allows access to the microphone for audio capture
- **Where**: System Settings > Privacy & Security > Microphone
- **Required for**: sounddevice, pyaudio, or any audio capture library
- **Behavior**: macOS prompts automatically on first access

### How Apps Request Permissions

macOS does not provide a programmatic API to request Accessibility or Input Monitoring permissions. The approach is:

1. **Detect permission status**: Check if the app has permission
   ```python
   import subprocess
   # No direct API; detect by attempting to create an event tap
   # If it returns None, permission is not granted
   from Quartz import CGEventTapCreate, kCGSessionEventTap, kCGHeadInsertEventTap
   tap = CGEventTapCreate(kCGSessionEventTap, kCGHeadInsertEventTap, 0, 0, lambda *a: None, None)
   has_permission = tap is not None
   ```

2. **Guide the user**: Show a dialog explaining which permissions to enable
   ```python
   import subprocess
   subprocess.run([
       "osascript", "-e",
       'display dialog "Please enable Accessibility permission for this app in System Settings > Privacy & Security > Accessibility" buttons {"Open Settings", "OK"}'
   ])
   # Optionally open the settings pane:
   subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"])
   ```

3. **For packaged apps**: The app bundle identifier is what appears in the permission list. For unpackaged Python scripts, the Terminal or Python interpreter process is what needs permission.

### Gotchas

- **pynput silently fails** if Accessibility permission is not granted - no error, just no events
- **Removing and re-adding** the app in Accessibility settings may be needed after updates
- **Code signing**: Unsigned apps may need to be re-authorized after each build
- **SIP (System Integrity Protection)**: Cannot be bypassed; event taps respect SIP
- **Sandboxed apps**: Cannot use CGEventTap; App Store apps have limited access
- **Testing tip**: Run `tccutil reset Accessibility` (requires SIP disable) to reset all permissions

---

## 9. Recommended Architecture for quill

Based on this research, here is the recommended architecture for implementing push-to-talk dictation in quill:

### Technology Stack

```
Hotkey:         pynput.keyboard.Listener (on_press / on_release)
Audio Capture:  sounddevice (already likely in use)
Transcription:  faster-whisper or mlx-whisper (local)
Text Injection: pyperclip + pynput Cmd+V paste
UI Feedback:    Floating overlay window (tkinter or rumps menu bar)
```

### Hold-to-Talk Flow

```python
# Pseudocode for the core loop

from pynput.keyboard import Key, Listener, Controller
import sounddevice as sd
import pyperclip
import threading

class PushToTalkDictation:
    def __init__(self, transcriber):
        self.transcriber = transcriber
        self.recording = False
        self.audio_buffer = []
        self.keyboard = Controller()

    def on_press(self, key):
        if key == Key.alt_l and not self.recording:
            self.recording = True
            self.start_audio_capture()

    def on_release(self, key):
        if key == Key.alt_l and self.recording:
            self.recording = False
            audio = self.stop_audio_capture()
            # Run transcription in background thread
            threading.Thread(target=self.transcribe_and_inject, args=(audio,)).start()

    def start_audio_capture(self):
        self.audio_buffer = []
        self.stream = sd.InputStream(callback=self._audio_callback, samplerate=16000, channels=1)
        self.stream.start()

    def _audio_callback(self, indata, frames, time, status):
        self.audio_buffer.append(indata.copy())

    def stop_audio_capture(self):
        self.stream.stop()
        self.stream.close()
        return np.concatenate(self.audio_buffer)

    def transcribe_and_inject(self, audio):
        text = self.transcriber.transcribe(audio)
        if text.strip():
            pyperclip.copy(text)
            self.keyboard.press(Key.cmd)
            self.keyboard.press('v')
            self.keyboard.release('v')
            self.keyboard.release(Key.cmd)

    def run(self):
        with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()
```

### Key Design Decisions

1. **pynput over CGEventTap**: Higher-level API, cross-platform potential, well-maintained
2. **Clipboard+paste over keystroke simulation**: Reliable across all apps, handles Unicode
3. **sounddevice over pyaudio**: Simpler API, better maintained, numpy integration
4. **Local transcription**: Privacy-first, no API keys needed, faster for short utterances
5. **Threading for transcription**: Keep the hotkey listener responsive during transcription

### Required Permissions Checklist

- [ ] Accessibility (System Settings > Privacy & Security > Accessibility)
- [ ] Input Monitoring (System Settings > Privacy & Security > Input Monitoring)
- [ ] Microphone (auto-prompted on first use)

### Edge Cases to Handle

1. **Key repeat**: macOS sends repeated key-down events when holding a key; debounce with a `self.recording` flag
2. **App focus changes**: If user switches apps while recording, text should still inject into the newly focused app (clipboard+paste handles this naturally)
3. **Empty transcription**: Don't paste if Whisper returns empty/whitespace
4. **Long recordings**: Consider chunked transcription or a maximum recording duration
5. **Clipboard restoration**: Optionally save/restore the clipboard after a short delay
6. **Audio device changes**: Handle microphone disconnection gracefully
7. **Multiple hotkey presses**: Ignore press events while already recording

---

## Sources

- Willow Voice: https://willowvoice.com
- Willow Voice Help - Hotkey Settings: https://help.willowvoice.com/en/articles/10876257-hotkey-settings
- Willow Voice Help - Dictating: https://help.willowvoice.com/en/articles/10876920-dictating-with-willow-voice
- SuperWhisper: https://superwhisper.com
- SuperWhisper Keyboard Shortcuts: https://superwhisper.com/docs/get-started/settings-shortcuts
- Wispr Flow: https://wisprflow.ai
- push-to-talk-dictate (Rasala): https://github.com/Rasala/push-to-talk-dictate
- open-wispr: https://github.com/human37/open-wispr
- whisper-dictation (foges): https://github.com/foges/whisper-dictation
- whisper-dictation (ashwin-pc): https://github.com/ashwin-pc/whisper-dictation
- whisper-writer: https://github.com/savbell/whisper-writer
- OpenWhispr: https://github.com/OpenWhispr/openwhispr
- pynput documentation: https://pynput.readthedocs.io/en/latest/keyboard.html
- pynput platform limitations: https://pynput.readthedocs.io/en/latest/limitations.html
- Apple CGEventCreateKeyboardEvent: https://developer.apple.com/documentation/coregraphics/1456564-cgeventcreatekeyboardevent
- macOS Input Monitoring: https://support.apple.com/guide/mac-help/control-access-to-input-monitoring-on-mac-mchl4cedafb6/mac
- HeyWillow (different project - voice assistant): https://github.com/HeyWillow/willow
