# Dictation Mode Manual Test Script

These tests require real hardware interaction and macOS permissions.
They cover scenarios that cannot be automated in CI.

---

## Prerequisites

- macOS with Accessibility and Input Monitoring permissions granted for Terminal/Python
- Microphone available and working
- quill installed and running from menu bar
- A text editor (TextEdit) open for verifying text injection

---

## M1: Real hotkey detection

**Steps:**
1. Launch quill from the menu bar
2. Click "Enable Dictation" from the menu
3. Verify: menu bar title shows "Dictation (hold alt_r to speak)"
4. Press and hold the Right Alt key
5. Verify: menu bar title changes to "Dictation: listening..."
6. Release the Right Alt key
7. Verify: menu bar title shows "Dictation: transcribing..." briefly, then returns to "Dictation (hold alt_r to speak)"

**Pass criteria:** Title transitions match all three states (idle, listening, transcribing).

---

## M2: Real audio capture and transcription

**Steps:**
1. Open TextEdit and place cursor in a new document
2. Enable dictation from the quill menu
3. Press and hold Right Alt
4. Speak clearly: "Hello world"
5. Release Right Alt
6. Wait for transcription to complete

**Pass criteria:** "Hello world" (or close approximation) appears at the cursor position in TextEdit.

---

## M3: Text injection into different apps

**Steps:**
1. Open TextEdit, place cursor in document
2. Enable dictation, hold hotkey, speak "First test", release
3. Verify: text appears in TextEdit
4. Switch to Safari, click in the address bar or a text field on a webpage
5. Hold hotkey, speak "Second test", release
6. Verify: text appears in Safari's text field
7. Switch to Terminal, hold hotkey, speak "Third test", release
8. Verify: text appears at the Terminal prompt

**Pass criteria:** Transcribed text appears in each target application.

---

## M4: Accessibility permission denied

**Steps:**
1. Go to System Settings > Privacy & Security > Accessibility
2. Remove the quill app (or Terminal/Python) from the allowed list
3. Launch quill and click "Enable Dictation"
4. Observe behavior:
   - **Outcome A (error detected):** A notification appears with title "Dictation Permission Required" and instructions to enable Accessibility
   - **Outcome B (silent failure):** Dictation appears to enable (title shows "Dictation (hold alt_r to speak)") but pressing the hotkey does nothing — no recording starts
5. If Outcome B: this is a known limitation (pynput silently fails without permissions). Report as expected behavior.
6. Re-add the app to Accessibility permissions to continue testing

**Pass criteria:** Either a notification is shown (Outcome A) or silent failure is observed and documented (Outcome B). The app should not crash.

---

## M5: Disable dictation stops hotkey

**Steps:**
1. Enable dictation, verify hotkey works (hold -> "listening..." title appears)
2. Click "Disable Dictation" from the menu
3. Verify: title returns to "Quill"
4. Press and hold the Right Alt key
5. Verify: nothing happens (no title change, no recording)

**Pass criteria:** After disabling, hotkey has no effect.

---

## M6: Conflict with Record Note mode

*Requires `ENABLE_TRANSCRIPTION=true` to show recording menu items.*

**Steps:**
1. From idle state, click "Start Recording" to begin a note recording
2. Open the menu and check "Enable Dictation"
3. Verify: "Enable Dictation" is greyed out / non-clickable
4. Click "Stop Recording" to end the recording
5. Wait for processing to complete (state returns to idle)
6. Verify: "Enable Dictation" is now clickable again

**Steps (reverse direction):**
7. Enable dictation (state shows "Dictation (hold alt_r to speak)")
8. Check "Start Recording" in the menu
9. Verify: "Start Recording" is greyed out / non-clickable
10. Click "Disable Dictation"
11. Verify: "Start Recording" is now clickable again

**Pass criteria:** Dictation and Record Note modes are mutually exclusive.

---

## M7: Max duration auto-stop

**Steps:**
1. Set environment variable: `export DICTATION_MAX_SECONDS=5`
2. Launch quill (or restart if already running)
3. Enable dictation
4. Press and hold the hotkey
5. Do NOT release — hold for more than 5 seconds
6. Observe: after ~5 seconds, recording should stop automatically
7. Verify: title transitions from "listening..." to "transcribing..." without releasing the key
8. If you were speaking, verify that transcribed text appears at cursor

**Pass criteria:** Recording auto-stops after the configured max duration.

---

## Notes

- If any test fails due to an app bug (not a test issue), report to the team lead with:
  - Which manual test failed (M1-M7)
  - Expected vs actual behavior
  - Any error messages or logs
- These tests complement the 24 automated tests in `tests/test_e2e_dictation.py`
