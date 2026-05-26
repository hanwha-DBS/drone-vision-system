const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const zmq = require('zeromq');

let mainWindow;
let pythonProcess = null;
let engineReady = false;

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
                if (mainWindow) {
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

function startPythonEngine() {
    return new Promise((resolve) => {
        console.log('[Main] starting python engine');

        pythonProcess = spawn('python', ['engine.py'], {
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

                if (log.includes('READY')) {
                    engineReady = true;
                    if (mainWindow) {
                        mainWindow.webContents.send('engine-ready');
                    }
                    resolve();
                }
            });
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error('[Python ERR]', data.toString());
        });

        pythonProcess.on('close', (code) => {
            engineReady = false;
            if (mainWindow) {
                mainWindow.webContents.send('engine-crashed', code);
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
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        },
    });

    mainWindow.loadFile('index.html');
}

app.whenReady().then(async () => {
    createWindow();
    await startPythonEngine();
});

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
        return await sendZmqRequest({ type: 'status' });
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
