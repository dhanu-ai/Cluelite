const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  startCluelite: () => ipcRenderer.send('start-cluelite'),
  stopCluelite: () => ipcRenderer.send('stop-cluelite'),
  askCluelite: query => ipcRenderer.invoke('ask-cluelite', query),
  pickFile: () => ipcRenderer.invoke('pick-file'),
  close: () => ipcRenderer.send('close'),
  onSuggestion: callback => ipcRenderer.on('ai-suggestion', (_, suggestion) => callback(suggestion)),
  enableInteraction: () => ipcRenderer.send('enable-interaction'),
  disableInteraction: () => ipcRenderer.send('disable-interaction')
});
