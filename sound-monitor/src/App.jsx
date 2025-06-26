import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import SoundMonitor from './components/SoundMonitor'


function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <SoundMonitor />
    </div>
  )
}

export default App
