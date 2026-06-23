const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  minimize: () => ipcRenderer.send('minimize'),
  close: () => ipcRenderer.send('close'),
  startCluelite: () => ipcRenderer.send('start-cluelite'),
  stopCluelite: () => ipcRenderer.send('stop-cluelite'),
  setInteractive: (enabled) => ipcRenderer.send('set-interactive', enabled),
  getPosition: () => ipcRenderer.sendSync('get-position'),
  setPosition: (position) => ipcRenderer.send('set-position', position),
  onSuggestion: (cb) => ipcRenderer.on('ai-suggestion', (_, data) => cb(data)),
  onHoverInit: (cb) => ipcRenderer.on('init-hover-detection', cb),
});