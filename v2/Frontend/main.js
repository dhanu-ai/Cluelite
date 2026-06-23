const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fetch = require('node-fetch');

let mainWindow;
let hiddenWindow;
let suggestionTimeout = null;
let retryDelay = 2000;
let abortController = new AbortController();
let isInteractive = false;

// ===== Create Electron Window =====
function createWindow() {
  hiddenWindow = new BrowserWindow({
    show: false,
    width: 1,
    height: 1,
    frame: false,
    transparent: true
  });

  mainWindow = new BrowserWindow({
    parent: hiddenWindow,
    width: 760,
    height: 520,
    transparent: true,
    backgroundColor: '#00000000',
    frame: false,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    hasShadow: false,
    focusable: true,
    useContentSize: true,
    type: 'toolbar',
    center: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'render.html'));

  mainWindow.on('will-resize', e => e.preventDefault());
  mainWindow.on('bounds-changing', (e, bounds) => {
    bounds.width = 760;
    bounds.height = 520;
  });

  mainWindow.webContents.on('did-finish-load', () => {
    mainWindow.setIgnoreMouseEvents(true, { forward: true });
    mainWindow.setContentProtection(true);
    isInteractive = false;
  });

  mainWindow.on('focus', () => {
    if (!isInteractive) mainWindow.setIgnoreMouseEvents(true, { forward: true });
  });
}

// ===== Wait for Backend =====
async function waitForBackend(retries = 20, delay = 1000) {
  return new Promise((resolve, reject) => {
    const attempt = async count => {
      try {
        const res = await fetch('http://localhost:8000/ping');
        if (res.ok) return resolve();
      } catch {}
      if (count <= 0) return reject('Backend not responding');
      setTimeout(() => attempt(count - 1), delay);
    };
    attempt(retries);
  });
}

// ===== App Events =====
app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// ===== IPC Handlers =====
ipcMain.on('get-position', e => {
  e.returnValue = mainWindow.getBounds();
});

ipcMain.on('set-position', (_, pos) => {
  mainWindow.setBounds({ x: pos.x, y: pos.y, width: 760, height: 520 });
});

ipcMain.on('set-interactive', (_, enable) => {
  if (mainWindow) {
    isInteractive = enable;
    mainWindow.setIgnoreMouseEvents(!enable, { forward: true });
    if (enable) {
      mainWindow.focus();
      mainWindow.webContents.focus();
    }
  }
});

ipcMain.on('minimize', () => mainWindow?.minimize());
ipcMain.on('close', () => {
  if (mainWindow) mainWindow.close();
  if (hiddenWindow) hiddenWindow.close();
});

ipcMain.handle('pick-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'PDF Files', extensions: ['pdf'] }]
  });
  return result.filePaths;
});

ipcMain.handle('ask-cluelite', async (_, query) => {
  try {
    const res = await fetch('http://localhost:8000/askcluelite', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    return await res.json();
  } catch {
    return { response: '⚠️ Error contacting backend. Please make sure the backend server is running.' };
  }
});

ipcMain.on('start-cluelite', async () => {
  try {
    abortController = new AbortController();
    await waitForBackend();

    await fetch('http://localhost:8000/start', { method: 'POST', signal: abortController.signal });

    const fetchSuggestion = async () => {
      if (suggestionTimeout) {
        clearTimeout(suggestionTimeout);
        suggestionTimeout = null;
      }

      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 4000);
        const res = await fetch('http://localhost:8000/suggestion', { signal: controller.signal });
        clearTimeout(timeout);

        if (res.ok) {
          const json = await res.json();
          if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('ai-suggestion', json.suggestion);
          retryDelay = 2000;
        }
      } catch {
        retryDelay = Math.min(retryDelay * 2, 30000);
      }

      if (!abortController.signal.aborted) suggestionTimeout = setTimeout(fetchSuggestion, retryDelay);
    };

    fetchSuggestion();
  } catch {
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('ai-suggestion', '⚠️ Failed to connect to backend');
  }
});

ipcMain.on('stop-cluelite', async () => {
  try {
    abortController.abort();
    if (suggestionTimeout) {
      clearTimeout(suggestionTimeout);
      suggestionTimeout = null;
    }
    await fetch('http://localhost:8000/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
    retryDelay = 2000;
  } catch {}
});