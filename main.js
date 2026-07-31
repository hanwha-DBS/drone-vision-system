const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const { spawn, execFile } = require('child_process');
const fs = require('fs').promises;
const fsSync = require('fs');
const path = require('path');
const zmq = require('zeromq');

const ENGINE_PORT = 5555;

let mainWindow;
let pythonProcess = null;
let engineReady = false;
// Last error-ish line seen on the engine's stdout/stderr, surfaced in the
// engine-crashed event so the UI can show *why* it died (e.g. port in use).
let lastEngineError = '';

// ===== Watch folder state =====
const VIDEO_EXTS = /\.(mp4|avi|mov|mkv|webm)$/i;
const WATCH_STABILITY_MS = 2000;
const WATCH_POLL_MS = 1500;

let watchInterval = null;
let watchFolder = null;
const watchSeenFiles = new Set();
const watchStability = new Map();

async function pollWatchFolder() {
    if (!watchFolder) return;
    try {
        const entries = await fs.readdir(watchFolder, { withFileTypes: true });
        const now = Date.now();

        for (const entry of entries) {
            if (!entry.isFile() || !VIDEO_EXTS.test(entry.name)) continue;
            const full = path.join(watchFolder, entry.name);
            if (watchSeenFiles.has(full)) continue;

            let stat;
            try { stat = await fs.stat(full); } catch { continue; }

            const prev = watchStability.get(full);
            if (!prev || prev.size !== stat.size) {
                watchStability.set(full, { size: stat.size, since: now });
                continue;
            }

            if (now - prev.since >= WATCH_STABILITY_MS) {
                watchSeenFiles.add(full);
                watchStability.delete(full);
                if (mainWindow && !mainWindow.isDestroyed()) {
                    mainWindow.webContents.send('watched-file-ready', {
                        path: full,
                        size: stat.size,
                        timestamp: new Date().toISOString(),
                    });
                }
            }
        }
    } catch (err) {
        console.error('[Watcher] poll error:', err.message);
    }
}

function resolvePythonPath() {
    // Prefer the project venv (pinned ultralytics/transformers versions);
    // fall back to whatever `python` is on PATH.
    const venvPython = path.join(__dirname, '.venv', 'Scripts', 'python.exe');
    return fsSync.existsSync(venvPython) ? venvPython : 'python';
}

function freeEnginePort() {
    // Best-effort cleanup of an orphaned engine. If a previous engine process
    // didn't exit cleanly (app force-closed / crashed), it keeps port 5555
    // bound and the fresh engine crashes at bind() with "Address in use" —
    // before it can print READY — which the UI shows as an engine error.
    // Find the (single) process owning the port and, if it's a python, kill
    // it so the new engine can bind. Never rejects — the spawn proceeds either
    // way, and engine.py now reports a clear message if the port is still held.
    return new Promise((resolve) => {
        if (process.platform === 'win32') {
            const ps = [
                `$conns = Get-NetTCPConnection -LocalPort ${ENGINE_PORT} -State Listen -ErrorAction SilentlyContinue;`,
                `foreach ($procId in ($conns.OwningProcess | Select-Object -Unique)) {`,
                `  $p = Get-Process -Id $procId -ErrorAction SilentlyContinue;`,
                `  if ($p -and $p.ProcessName -like 'python*') {`,
                `    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue;`,
                `    Write-Output "freed engine port from python PID $procId";`,
                `  }`,
                `}`,
            ].join(' ');
            execFile(
                'powershell.exe',
                ['-NoProfile', '-NonInteractive', '-Command', ps],
                (err, stdout) => {
                    if (stdout && stdout.trim()) console.log('[Main]', stdout.trim());
                    // Give Windows a moment to release the port after the kill.
                    setTimeout(resolve, 400);
                }
            );
        } else {
            // mac/linux: lsof gives the listener pid; kill it best-effort.
            execFile(
                '/bin/sh',
                ['-c', `lsof -ti tcp:${ENGINE_PORT} -sTCP:LISTEN | xargs -r kill -9`],
                () => setTimeout(resolve, 200)
            );
        }
    });
}

async function startPythonEngine() {
    // Clear any orphaned engine holding the port before spawning a fresh one.
    await freeEnginePort();

    return new Promise((resolve) => {
        const pythonPath = resolvePythonPath();
        console.log('[Main] starting python engine:', pythonPath);
        lastEngineError = '';

        pythonProcess = spawn(pythonPath, ['engine.py'], {
            cwd: __dirname,
            env: {
                ...process.env,
                PYTHONUNBUFFERED: '1',
            },
        });

        pythonProcess.stdout.on('data', (data) => {
            const logs = data.toString().split('\n');

            logs.forEach((line) => {
                const log = line.trim();
                if (!log) return;

                console.log('[Python]', log);

                // engine.py prints fatal startup problems (e.g. port bind
                // failure) to stdout with this prefix — remember the last one.
                if (log.includes('[AI Engine ERR]')) {
                    lastEngineError = log;
                }

                if (log.includes('READY')) {
                    engineReady = true;
                    if (mainWindow && !mainWindow.isDestroyed()) {
                        mainWindow.webContents.send('engine-ready');
                    }
                    resolve();
                }
            });
        });

        pythonProcess.stderr.on('data', (data) => {
            const text = data.toString();
            const lastLine = text.trim().split('\n').pop();
            if (lastLine) lastEngineError = lastLine;
            console.error('[Python ERR]', text);
        });

        pythonProcess.on('close', (code) => {
            engineReady = false;
            // The engine's close event also fires during app shutdown, after
            // the BrowserWindow has been destroyed — sending to it then throws.
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('engine-crashed', { code, detail: lastEngineError });
            }
        });
    });
}

async function sendZmqRequest(payload, timeoutMs = 120000) {
    const sock = new zmq.Request();

    try {
        sock.connect('tcp://127.0.0.1:5555');
        await sock.send(JSON.stringify(payload));

        const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('TIMEOUT')), timeoutMs)
        );
        const result = await Promise.race([sock.receive(), timeoutPromise]);
        return JSON.parse(result.toString());
    } finally {
        sock.close();
    }
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1520,
        height: 980,
        minWidth: 1280,
        minHeight: 860,
        backgroundColor: '#071017',
        icon: path.join(__dirname, 'Resources', 'hanwha.ico'),
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        },
    });

    mainWindow.loadFile('index.html');
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// Single-instance lock: a second launch must not spawn a second engine that
// would collide on port 5555. Hand focus to the existing window instead.
const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
    app.quit();
} else {
    app.on('second-instance', () => {
        if (mainWindow) {
            if (mainWindow.isMinimized()) mainWindow.restore();
            mainWindow.focus();
        }
    });

    app.whenReady().then(async () => {
        createWindow();
        await startPythonEngine();
    });
}

app.on('will-quit', () => {
    if (watchInterval) {
        clearInterval(watchInterval);
        watchInterval = null;
    }
    if (pythonProcess) {
        console.log('[Main] stopping python engine');
        pythonProcess.kill();
    }
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

ipcMain.handle('check-engine', () => ({ ready: engineReady }));

ipcMain.handle('check-engine-status', async () => {
    try {
        // Short timeout on purpose: this is the liveness poll. With the default
        // 120s the renderer's error path could never fire for a hung engine —
        // the request just hung and the UI kept showing "loading" forever.
        return await sendZmqRequest({ type: 'status' }, 6000);
    } catch {
        return {
            status: 'error',
            model_loaded: false,
        };
    }
});

ipcMain.handle('start-video-analysis', async (_, options = {}) => {
    if (!engineReady) {
        return { status: 'error', msg: 'python engine is not ready yet' };
    }

    const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
        properties: ['openFile'],
        filters: [{ name: 'Videos', extensions: ['mp4', 'avi', 'mov', 'mkv', 'webm'] }],
    });

    if (canceled || filePaths.length === 0) {
        return null;
    }

    try {
        const analysisInterval = Number(options.analysis_interval_seconds || 3);
        const analysisProfile = String(options.analysis_profile || 'balanced');
        const response = await sendZmqRequest(
            {
                type: 'start_video',
                video_path: filePaths[0],
                analysis_interval_seconds: analysisInterval,
                analysis_profile: analysisProfile,
                pause_at_end: true,
            },
            10000
        );

        return {
            ...response,
            selected_path: filePaths[0],
            analysis_interval_seconds: analysisInterval,
            analysis_profile: analysisProfile,
        };
    } catch (err) {
        if (err.message === 'TIMEOUT') {
            return { status: 'error', msg: 'engine start request timed out' };
        }

        return { status: 'error', msg: `communication error: ${err.message}` };
    }
});

ipcMain.handle('get-engine-settings', async () => {
    try {
        return await sendZmqRequest({ type: 'get_settings' }, 10000);
    } catch (err) {
        return { status: 'error', msg: err.message };
    }
});

ipcMain.handle('save-engine-settings', async (_, settings = {}) => {
    try {
        return await sendZmqRequest({ type: 'update_settings', settings }, 10000);
    } catch (err) {
        return { status: 'error', msg: err.message };
    }
});

ipcMain.handle('get-video-analysis-status', async () => {
    try {
        return await sendZmqRequest({ type: 'video_status' }, 10000);
    } catch (err) {
        return { status: 'error', message: err.message };
    }
});

ipcMain.handle('stop-video-analysis', async () => {
    try {
        return await sendZmqRequest({ type: 'stop_video' }, 10000);
    } catch (err) {
        return { status: 'error', msg: err.message };
    }
});

ipcMain.handle('pause-video-analysis', async (_, paused = true) => {
    try {
        return await sendZmqRequest({ type: 'pause_video', paused: !!paused }, 10000);
    } catch (err) {
        return { status: 'error', msg: err.message };
    }
});

ipcMain.handle('seek-video-analysis', async (_, fraction = 0) => {
    try {
        return await sendZmqRequest({ type: 'seek_video', fraction: Number(fraction) || 0 }, 10000);
    } catch (err) {
        return { status: 'error', msg: err.message };
    }
});

ipcMain.handle('capture-frame-analysis', async () => {
    try {
        return await sendZmqRequest({ type: 'capture_frame' }, 10000);
    } catch (err) {
        return { status: 'error', msg: err.message };
    }
});

ipcMain.handle('quit-app', async () => {
    app.quit();
    return { status: 'ok' };
});

ipcMain.handle('select-image', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
        properties: ['openFile'],
        filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp'] }],
    });
    if (canceled || filePaths.length === 0) return null;
    return { path: filePaths[0] };
});

ipcMain.handle('analyze-image', async (_, imagePath, analysisProfile = 'balanced') => {
    if (!imagePath) return { status: 'error', msg: 'image path required' };
    try {
        // CPU full-stack analysis of one image can take a while — allow time.
        return await sendZmqRequest(
            { type: 'analyze_image', image_path: imagePath, analysis_profile: String(analysisProfile || 'balanced') },
            180000
        );
    } catch (err) {
        if (err.message === 'TIMEOUT') return { status: 'error', msg: 'analysis timed out' };
        return { status: 'error', msg: err.message };
    }
});

ipcMain.handle('open-path', async (_, targetPath) => {
    if (!targetPath) {
        return { status: 'error', msg: 'path is required' };
    }

    const error = await shell.openPath(targetPath);
    if (error) {
        return { status: 'error', msg: error };
    }

    return { status: 'ok' };
});

// ===== Watch folder IPC =====
ipcMain.handle('select-folder', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory'],
        title: '감시할 폴더 선택',
    });
    if (canceled || filePaths.length === 0) return null;
    return filePaths[0];
});

ipcMain.handle('start-folder-watch', async (_, folderPath) => {
    if (!folderPath) return { status: 'error', msg: 'folder path required' };

    if (watchInterval) {
        clearInterval(watchInterval);
        watchInterval = null;
    }
    watchSeenFiles.clear();
    watchStability.clear();
    watchFolder = folderPath;

    try {
        // Don't baseline existing files — they may be unprocessed.
        // The renderer's processedSet (localStorage) handles dedup across sessions.
        const entries = await fs.readdir(folderPath, { withFileTypes: true });
        const existingCount = entries.filter(
            (e) => e.isFile() && VIDEO_EXTS.test(e.name)
        ).length;
        watchInterval = setInterval(pollWatchFolder, WATCH_POLL_MS);
        pollWatchFolder();
        return { status: 'ok', folder: folderPath, existing_count: existingCount };
    } catch (err) {
        watchFolder = null;
        return { status: 'error', msg: err.message };
    }
});

ipcMain.handle('stop-folder-watch', async () => {
    if (watchInterval) {
        clearInterval(watchInterval);
        watchInterval = null;
    }
    watchFolder = null;
    watchSeenFiles.clear();
    watchStability.clear();
    return { status: 'ok' };
});

ipcMain.handle('start-video-from-path', async (_, options = {}) => {
    if (!engineReady) {
        return { status: 'error', msg: 'python engine is not ready yet' };
    }
    if (!options.video_path) {
        return { status: 'error', msg: 'video_path required' };
    }

    try {
        const analysisInterval = Number(options.analysis_interval_seconds || 3);
        const analysisProfile = String(options.analysis_profile || 'balanced');
        const response = await sendZmqRequest(
            {
                type: 'start_video',
                video_path: options.video_path,
                analysis_interval_seconds: analysisInterval,
                analysis_profile: analysisProfile,
            },
            10000
        );

        return {
            ...response,
            selected_path: options.video_path,
            analysis_interval_seconds: analysisInterval,
            analysis_profile: analysisProfile,
        };
    } catch (err) {
        if (err.message === 'TIMEOUT') {
            return { status: 'error', msg: 'engine start request timed out' };
        }
        return { status: 'error', msg: `communication error: ${err.message}` };
    }
});
