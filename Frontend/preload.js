const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  minimize: () => ipcRenderer.send('minimize'),
  close: () => ipcRenderer.send('close'),
  pickFile: () => ipcRenderer.invoke('pick-file'),
  askCluelite: (query) => ipcRenderer.invoke('ask-cluelite', query),
  startCluelite: () => ipcRenderer.send('start-cluelite'),
  stopCluelite: () => ipcRenderer.send('stop-cluelite'),
  onSuggestion: (cb) => ipcRenderer.on('ai-suggestion', (_, data) => cb(data)),
  onHoverInit: (cb) => ipcRenderer.on('init-hover-detection', cb),
  setInteractive: (enabled) => ipcRenderer.send('set-interactive', enabled),
});