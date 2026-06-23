const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fetch = require('node-fetch');

let mainWindow;
let hiddenWindow;
let suggestionTimeout = null;
let retryDelay = 2000;
let abortController = new AbortController();

function createWindow() {
  // Create invisible dummy parent
  hiddenWindow = new BrowserWindow({
    show: false,
    width: 1,
    height: 1,
    frame: false,
    transparent: true
  });

  // Create the overlay window
  mainWindow = new BrowserWindow({
    parent: hiddenWindow,
    width: 400,
    height: 600,
    transparent: true,
    frame: false,
    resizable: false,           // Disable OS resize handles
    alwaysOnTop: true,
    skipTaskbar: true,
    hasShadow: false,
    focusable: true,            // Needed for typing
    useContentSize: true,
    type: 'toolbar',
    center: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    }
  });

  // Prevent ANY resize attempts
  mainWindow.on('will-resize', (e) => {
    e.preventDefault();
  });

  // Extra safeguard against bounds changes during drag
  mainWindow.on('bounds-changing', (e, bounds) => {
    bounds.width = 400;
    bounds.height = 600;
  });

  mainWindow.loadFile('render.html');

  mainWindow.webContents.on('did-finish-load', () => {
    mainWindow.setIgnoreMouseEvents(true, { forward: true });
    mainWindow.setContentProtection(true);
    mainWindow.webContents.send('init-hover-detection');
  });
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// Position handling
ipcMain.on('get-position', (event) => {
  const pos = mainWindow.getBounds();
  event.returnValue = { x: pos.x, y: pos.y };
});

ipcMain.on('set-position', (_, position) => {
  const bounds = mainWindow.getBounds();
  mainWindow.setBounds({
    x: position.x,
    y: position.y,
    width: 400,
    height: 600
  });
});

// Click-through logic
ipcMain.on('set-interactive', (_, enable) => {
  mainWindow.setIgnoreMouseEvents(!enable, { forward: true });
});

// Window controls
ipcMain.on('minimize', () => mainWindow?.minimize());
ipcMain.on('close', () => {
  if (mainWindow) mainWindow.close();
  if (hiddenWindow) hiddenWindow.close();
});

// Backend communication
async function waitForBackend(retries = 20, delay = 1000) {
  return new Promise((resolve, reject) => {
    const attempt = async (count) => {
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

ipcMain.on('start-cluelite', async () => {
  try {
    abortController = new AbortController();
    const signal = abortController.signal;
    
    await waitForBackend();
    await fetch('http://localhost:8000/start', { 
      method: 'POST',
      signal
    });

    const fetchSuggestion = async () => {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 4000);
        const res = await fetch('http://localhost:8000/suggestion', {
          signal: controller.signal
        });
        clearTimeout(timeout);
        const json = await res.json();
        mainWindow.webContents.send('ai-suggestion', json.suggestion);
        retryDelay = 2000;
        suggestionTimeout = setTimeout(fetchSuggestion, 5000);
      } catch {
        suggestionTimeout = setTimeout(fetchSuggestion, retryDelay);
        retryDelay = Math.min(retryDelay * 2, 30000);
      }
    };

    fetchSuggestion();
  } catch (err) {
    console.error('[Start Failed]', err);
    mainWindow.webContents.send('ai-suggestion', '⚠️ Failed to connect to backend');
  }
});

ipcMain.on('stop-cluelite', async () => {
  try {
    abortController.abort();
    
    if (suggestionTimeout) {
      clearTimeout(suggestionTimeout);
      suggestionTimeout = null;
    }
    
    const email = await mainWindow.webContents.executeJavaScript(
      "document.getElementById('user-email').value"
    );
    await fetch('http://localhost:8000/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: email || 'test@cluelite.ai' }),
    });
    retryDelay = 2000;
  } catch (err) {
    console.error('Stop failed:', err);
  }
});
