#!/bin/bash
#
# Build a self-contained Quill.app macOS bundle.
#
# Usage:  ./scripts/build-app.sh
#
# The resulting .app in dist/ bundles:
#   - Python venv with all dependencies
#   - Whisper model (base.pt)
#   - faster-whisper model (base)
#   - ffmpeg binary
#
# The .app requires NO external dependencies — just double-click to run.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="Quill"
BUNDLE_ID="com.quill.app"
VERSION="0.1.0"
DIST_DIR="$PROJECT_DIR/dist"
APP_DIR="$DIST_DIR/$APP_NAME.app"
CONTENTS="$APP_DIR/Contents"
RESOURCES="$CONTENTS/Resources"

echo "==> Building self-contained $APP_NAME.app ..."

# Clean previous build
rm -rf "$APP_DIR"

# Create bundle structure
mkdir -p "$CONTENTS/MacOS"
mkdir -p "$RESOURCES/bin"
mkdir -p "$RESOURCES/models/whisper"
mkdir -p "$RESOURCES/models/faster-whisper"

# --- Info.plist ---
cat > "$CONTENTS/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleName</key>
	<string>Quill</string>
	<key>CFBundleDisplayName</key>
	<string>Quill</string>
	<key>CFBundleIdentifier</key>
	<string>${BUNDLE_ID}</string>
	<key>CFBundleVersion</key>
	<string>${VERSION}</string>
	<key>CFBundleShortVersionString</key>
	<string>${VERSION}</string>
	<key>CFBundleExecutable</key>
	<string>quill-launcher</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>LSUIElement</key>
	<true/>
	<key>LSMinimumSystemVersion</key>
	<string>13.0</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>LSApplicationCategoryType</key>
	<string>public.app-category.productivity</string>
	<key>NSMicrophoneUsageDescription</key>
	<string>Quill needs microphone access to record audio for transcription.</string>
	<key>NSAppleEventsUsageDescription</key>
	<string>Quill needs Accessibility access for global keyboard shortcuts.</string>
</dict>
</plist>
PLIST

# --- Compile native launcher ---
echo "==> Compiling native launcher ..."
clang -o "$CONTENTS/MacOS/quill-launcher" \
    "$PROJECT_DIR/launcher.c" \
    -mmacosx-version-min=13.0 \
    -arch arm64

# --- Bundle Python venv + project source ---
echo "==> Bundling Python environment ..."
BUNDLED_PROJECT="$RESOURCES/quill"
mkdir -p "$BUNDLED_PROJECT"

# Copy project source
cp -R "$PROJECT_DIR/quill" "$BUNDLED_PROJECT/quill"
cp "$PROJECT_DIR/pyproject.toml" "$BUNDLED_PROJECT/"

# Copy the entire venv
cp -R "$PROJECT_DIR/.venv" "$BUNDLED_PROJECT/.venv"

# Fix the venv shebang/symlinks to use relative paths:
# The venv's python is a symlink to the Homebrew Python framework.
# We need to copy the actual Python binary + framework into the bundle
# so it's self-contained.
echo "==> Embedding Python runtime ..."
VENV_BIN="$BUNDLED_PROJECT/.venv/bin"
PYTHON_REAL="$(readlink -f "$PROJECT_DIR/.venv/bin/python")"
PYTHON_FRAMEWORK_DIR="$(dirname "$(dirname "$PYTHON_REAL")")"

# Replace the symlink with the real binary
rm -f "$VENV_BIN/python" "$VENV_BIN/python3" "$VENV_BIN/python3.11"
cp "$PYTHON_REAL" "$VENV_BIN/python"
ln -sf python "$VENV_BIN/python3"
ln -sf python "$VENV_BIN/python3.11"

# Copy the Python standard library into the bundle
PYTHON_LIB="$PYTHON_FRAMEWORK_DIR/lib/python3.11"
BUNDLED_LIB="$BUNDLED_PROJECT/.venv/lib/python3.11"
# Copy stdlib modules that aren't already present (site-packages is already there)
rsync -a --ignore-existing "$PYTHON_LIB/" "$BUNDLED_LIB/" \
    --exclude='site-packages' --exclude='test' --exclude='__pycache__' \
    --exclude='ensurepip' --exclude='idlelib' --exclude='tkinter' \
    --exclude='turtle*' --exclude='turtledemo'

# --- Bundle ffmpeg ---
echo "==> Bundling ffmpeg ..."
FFMPEG_PATH="$(command -v ffmpeg || true)"
if [ -z "$FFMPEG_PATH" ]; then
    echo "ERROR: ffmpeg not found. Install with: brew install ffmpeg"
    exit 1
fi
# Resolve symlink
FFMPEG_REAL="$(readlink -f "$FFMPEG_PATH")"
cp "$FFMPEG_REAL" "$RESOURCES/bin/ffmpeg"
chmod +x "$RESOURCES/bin/ffmpeg"

# Bundle ffmpeg's dylibs
echo "==> Bundling ffmpeg dynamic libraries ..."
FFMPEG_LIB_DIR="$(dirname "$FFMPEG_REAL")/../lib"
if [ -d "$FFMPEG_LIB_DIR" ]; then
    mkdir -p "$RESOURCES/bin/lib"
    # Copy all ffmpeg/av* dylibs
    for dylib in "$FFMPEG_LIB_DIR"/lib{avcodec,avdevice,avfilter,avformat,avutil,swresample,swscale,postproc}*.dylib; do
        if [ -f "$dylib" ] && [ ! -L "$dylib" ]; then
            cp "$dylib" "$RESOURCES/bin/lib/"
        fi
    done
    # Rewrite ffmpeg's dylib paths to use @loader_path/lib/
    for dylib in "$RESOURCES/bin/lib/"*.dylib; do
        basename_dylib="$(basename "$dylib")"
        install_name_tool -id "@loader_path/lib/$basename_dylib" "$dylib" 2>/dev/null || true
    done
    for dylib_ref in $(otool -L "$RESOURCES/bin/ffmpeg" | grep '/opt/homebrew' | awk '{print $1}'); do
        basename_ref="$(basename "$dylib_ref")"
        if [ -f "$RESOURCES/bin/lib/$basename_ref" ]; then
            install_name_tool -change "$dylib_ref" "@loader_path/lib/$basename_ref" "$RESOURCES/bin/ffmpeg" 2>/dev/null || true
        fi
    done
    # Also fix inter-dylib references
    for dylib in "$RESOURCES/bin/lib/"*.dylib; do
        for dylib_ref in $(otool -L "$dylib" | grep '/opt/homebrew' | awk '{print $1}'); do
            basename_ref="$(basename "$dylib_ref")"
            if [ -f "$RESOURCES/bin/lib/$basename_ref" ]; then
                install_name_tool -change "$dylib_ref" "@loader_path/$basename_ref" "$dylib" 2>/dev/null || true
            fi
        done
    done
fi

# --- Bundle Whisper model ---
echo "==> Bundling Whisper model (base) ..."
WHISPER_MODEL="$HOME/.cache/whisper/base.pt"
if [ -f "$WHISPER_MODEL" ]; then
    cp "$WHISPER_MODEL" "$RESOURCES/models/whisper/base.pt"
else
    echo "WARNING: Whisper base model not found at $WHISPER_MODEL"
    echo "         Downloading it now ..."
    "$PROJECT_DIR/.venv/bin/python" -c "import whisper; whisper.load_model('base')"
    cp "$HOME/.cache/whisper/base.pt" "$RESOURCES/models/whisper/base.pt"
fi

# --- Bundle faster-whisper model ---
echo "==> Bundling faster-whisper model (base) ..."
FW_MODEL_DIR="$HOME/.cache/huggingface/hub/models--Systran--faster-whisper-base"
if [ -d "$FW_MODEL_DIR" ]; then
    cp -R "$FW_MODEL_DIR" "$RESOURCES/models/faster-whisper/models--Systran--faster-whisper-base"
else
    echo "WARNING: faster-whisper base model not found at $FW_MODEL_DIR"
    echo "         Downloading it now ..."
    "$PROJECT_DIR/.venv/bin/python" -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
    cp -R "$FW_MODEL_DIR" "$RESOURCES/models/faster-whisper/models--Systran--faster-whisper-base"
fi

# --- Prune large unnecessary files from venv to reduce bundle size ---
echo "==> Pruning unnecessary files ..."
find "$BUNDLED_PROJECT/.venv" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLED_PROJECT/.venv" -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLED_PROJECT/.venv" -type d -name 'test' -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLED_PROJECT/.venv" -name '*.pyc' -delete 2>/dev/null || true
# Remove CUDA-specific torch files (we only need CPU on macOS)
find "$BUNDLED_PROJECT/.venv" -path '*/torch/lib/libcudnn*' -delete 2>/dev/null || true
find "$BUNDLED_PROJECT/.venv" -path '*/torch/lib/libnccl*' -delete 2>/dev/null || true
find "$BUNDLED_PROJECT/.venv" -path '*/torch/lib/libcublas*' -delete 2>/dev/null || true
find "$BUNDLED_PROJECT/.venv" -path '*/torch/lib/libcusparse*' -delete 2>/dev/null || true
find "$BUNDLED_PROJECT/.venv" -path '*/torch/lib/libcusolver*' -delete 2>/dev/null || true
find "$BUNDLED_PROJECT/.venv" -path '*/torch/lib/libcufft*' -delete 2>/dev/null || true
find "$BUNDLED_PROJECT/.venv" -path '*/torch/lib/libcurand*' -delete 2>/dev/null || true
# Remove nvidia-specific packages
find "$BUNDLED_PROJECT/.venv" -type d -name 'nvidia' -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLED_PROJECT/.venv" -type d -name 'triton' -exec rm -rf {} + 2>/dev/null || true

# --- Code sign ---
echo "==> Code signing ..."
# Sign all .so and .dylib files first (deep signing)
find "$APP_DIR" -name '*.so' -o -name '*.dylib' | while read -r lib; do
    codesign --force --sign - "$lib" 2>/dev/null || true
done
codesign --force --sign - "$CONTENTS/MacOS/quill-launcher"
codesign --force --sign - --deep "$APP_DIR"

# Verify
codesign --verify "$APP_DIR" 2>&1 && echo "==> Code signature valid" || echo "==> Warning: code signature issue (expected for ad-hoc)"

# --- Remove quarantine so Gatekeeper doesn't block the app ---
echo "==> Removing quarantine attributes ..."
xattr -r -d com.apple.quarantine "$APP_DIR" 2>/dev/null || true

# --- Create DMG with drag-to-Applications layout ---
echo "==> Creating DMG ..."
DMG_PATH="$DIST_DIR/Quill-${VERSION}.dmg"
DMG_STAGING="$DIST_DIR/dmg-staging"
rm -f "$DMG_PATH"
rm -rf "$DMG_STAGING"
mkdir -p "$DMG_STAGING"
cp -R "$APP_DIR" "$DMG_STAGING/"
ln -s /Applications "$DMG_STAGING/Applications"

hdiutil create -volname "Quill" \
    -srcfolder "$DMG_STAGING" \
    -ov -format UDZO \
    "$DMG_PATH"
rm -rf "$DMG_STAGING"

# --- Summary ---
APP_SIZE="$(du -sh "$APP_DIR" | cut -f1)"
DMG_SIZE="$(du -sh "$DMG_PATH" | cut -f1)"

echo ""
echo "============================================"
echo "  Build complete!"
echo "============================================"
echo ""
echo "  App:  $APP_DIR ($APP_SIZE)"
echo "  DMG:  $DMG_PATH ($DMG_SIZE)"
echo ""
echo "  To test:  open '$APP_DIR'"
echo "  To distribute: share the .dmg file"
echo ""
echo "  The app is fully self-contained:"
echo "    - Python runtime embedded"
echo "    - Whisper base model included"
echo "    - faster-whisper base model included"
echo "    - ffmpeg included"
echo ""
echo "  Users just need:"
echo "    - macOS 13+"
echo "    - Ollama (optional, for summarization)"
echo ""
