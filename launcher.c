/*
 * Native launcher for Quill macOS menu bar app.
 *
 * This thin Mach-O binary serves as the CFBundleExecutable for the .app bundle.
 * It finds the project's Python venv and exec's into it to run the app.
 *
 * Why a native binary instead of a shell script?
 * - macOS requires a Mach-O binary for proper app identity
 * - Permissions (Microphone, Accessibility) are granted to the .app bundle
 * - LaunchServices properly registers the app
 *
 * The process name in Activity Monitor will show as "Quill" because
 * we set argv[0] to "Quill" before exec.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mach-o/dyld.h>

int main(int argc, char *argv[]) {
    /* Find the path to this executable */
    char exe_path[4096];
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) != 0) {
        fprintf(stderr, "Quill: could not determine executable path\n");
        return 1;
    }

    char real_path[4096];
    if (realpath(exe_path, real_path) == NULL) {
        fprintf(stderr, "Quill: could not resolve path\n");
        return 1;
    }

    /*
     * Navigate from:
     *   .../Quill.app/Contents/MacOS/quill-launcher
     * to:
     *   .../Quill.app/Contents/
     */
    /* Find last /MacOS/ and truncate */
    char *macos = strstr(real_path, "/Contents/MacOS/");
    if (macos == NULL) {
        fprintf(stderr, "Quill: unexpected bundle structure\n");
        return 1;
    }
    /* Point to /Contents */
    char contents_dir[4096];
    size_t contents_len = (macos - real_path) + strlen("/Contents");
    strncpy(contents_dir, real_path, contents_len);
    contents_dir[contents_len] = '\0';

    /* Try embedded venv first: Contents/Resources/quill/.venv/bin/python */
    char python_path[4096];
    char project_dir[4096];

    snprintf(project_dir, sizeof(project_dir), "%s/Resources/quill", contents_dir);
    snprintf(python_path, sizeof(python_path), "%s/.venv/bin/python", project_dir);

    if (access(python_path, X_OK) != 0) {
        /* Fallback: use the project in ~/workspace/quill */
        const char *home = getenv("HOME");
        if (home == NULL) {
            fprintf(stderr, "Quill: HOME not set\n");
            return 1;
        }
        snprintf(project_dir, sizeof(project_dir), "%s/workspace/quill", home);
        snprintf(python_path, sizeof(python_path), "%s/.venv/bin/python", project_dir);
    }

    if (access(python_path, X_OK) != 0) {
        fprintf(stderr, "Quill: Python not found at %s\n", python_path);
        fprintf(stderr, "Please ensure the quill venv exists.\n");
        return 1;
    }

    /* Build the inline Python command */
    char pycmd[4096];
    snprintf(pycmd, sizeof(pycmd),
        "import sys; sys.path.insert(0, '%s'); "
        "from quill.app import main; main()",
        project_dir);

    /* Set PYTHONPATH */
    setenv("PYTHONPATH", project_dir, 1);

    /* Compute bundle dir (parent of Contents) for env var */
    char bundle_dir[4096];
    snprintf(bundle_dir, sizeof(bundle_dir), "%s/..", contents_dir);
    char resolved_bundle[4096];
    if (realpath(bundle_dir, resolved_bundle) != NULL) {
        setenv("QUILL_APP_BUNDLE", resolved_bundle, 1);
    }

    /*
     * Prepend bundled bin/ (contains ffmpeg) to PATH so whisper
     * and the app can find ffmpeg without a system install.
     */
    char bundled_bin[4096];
    snprintf(bundled_bin, sizeof(bundled_bin), "%s/Resources/bin", contents_dir);
    if (access(bundled_bin, F_OK) == 0) {
        const char *old_path = getenv("PATH");
        char new_path[8192];
        if (old_path != NULL) {
            snprintf(new_path, sizeof(new_path), "%s:%s", bundled_bin, old_path);
        } else {
            snprintf(new_path, sizeof(new_path), "%s", bundled_bin);
        }
        setenv("PATH", new_path, 1);
    }

    /*
     * Prevent Python from showing its own Dock icon.
     * LSBackgroundOnly tells macOS this process is a background agent.
     * rumps/PyObjC will handle the menu bar presence.
     */
    setenv("LSBackgroundOnly", "1", 1);

    /*
     * exec into Python with argv[0] = "Quill"
     * This makes Activity Monitor show "Quill" as the process name.
     */
    char *new_argv[] = {
        "Quill",
        "-c",
        pycmd,
        NULL
    };

    execv(python_path, new_argv);

    /* If we get here, execv failed */
    perror("Quill: failed to launch Python");
    return 1;
}
