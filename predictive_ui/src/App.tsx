import { useState } from 'react';
import { Layers, Activity, Users, Terminal, MessageSquare, Home } from 'lucide-react';
import OASIS3D from './components/layers/OASIS3D';
import BoardroomDebate from './components/layers/BoardroomDebate';
import IngestorGraph from './components/layers/IngestorGraph';
import AssemblyMatrix from './components/layers/AssemblyMatrix';
import HandoffTerminal from './components/layers/HandoffTerminal';
import LandingPage from './components/LandingPage';
import InputSetupPage from './components/InputSetupPage';
import { useWebSocket } from './hooks/useWebSocket';
import { usePipelineStore } from './store/usePipelineStore';

function App() {
  const ws = useWebSocket();
  const { isConnected, simulationStatus, simulationTitle, simulationProgress } = usePipelineStore();
  const [activeLayer, setActiveLayer] = useState(0); // 0 = landing page

  const navItems = [
    { id: 0, name: 'Home', icon: Home },
    { id: 1, name: 'Ingestion', icon: Layers },
    { id: 3, name: 'Personas', icon: Users },
    { id: 5, name: 'OASIS Sim', icon: Activity },
    { id: 6, name: 'Boardroom', icon: MessageSquare },
    { id: 8, name: 'Handoff', icon: Terminal },
  ];

  // Landing page gets its own full-screen layout (no dashboard chrome)
  if (activeLayer === 0) {
    return (
      <div className="min-h-screen w-screen bg-background text-textLight relative">
        {/* Brutalist top nav for landing */}
        <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b-8 border-brand px-6 py-4 flex items-center justify-between shadow-neo-black">
          <div className="flex items-center gap-3 font-black text-lg md:text-2xl text-black uppercase tracking-tighter">
            <Activity className="text-brand w-6 h-6 md:w-8 md:h-8" strokeWidth={3} />
            <span>PREDICTIVE<span className="text-brand">REALITY</span>ENGINE</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveLayer(1)}
              className="px-4 py-2 md:px-6 md:py-3 bg-white text-black font-black text-xs md:text-sm tracking-widest uppercase border-4 border-black shadow-neo-black transition-all hover:translate-x-1 hover:translate-y-1 hover:shadow-neo-pressed"
            >
              Preview Dashboard
            </button>
            <button
              onClick={() => setActiveLayer(0.5)}
              className="px-4 py-2 md:px-6 md:py-3 bg-black text-white font-black text-xs md:text-sm tracking-widest uppercase border-4 border-black shadow-neo-white transition-all hover:translate-x-1 hover:translate-y-1 hover:shadow-neo-pressed"
            >
              Launch Engine
            </button>
          </div>
        </nav>

        <div className="pt-32">
          <LandingPage onStart={() => setActiveLayer(0.5)} />
        </div>
      </div>
    );
  }

  const handleStartSimulation = async (inputs: any) => {
    try {
      const response = await fetch('http://localhost:8000/api/upload_text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(inputs)
      });
      if (response.ok) {
        const data = await response.json();
        // Go to Ingestion layer immediately when simulation starts
        setActiveLayer(1);
        
        // Trigger the backend pipeline via websocket
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'config', files: data.files }));
        } else {
          console.warn("WebSocket not open, backend will not start. Dev fallback.");
        }
      } else {
        console.error("Failed to upload inputs", await response.text());
      }
    } catch (e) {
      console.error("API error:", e);
      // fallback for dev if backend isn't up
      setActiveLayer(1);
    }
  };

  if (activeLayer === 0.5) {
    return <InputSetupPage 
      onStartSimulation={handleStartSimulation} 
      onSkip={() => setActiveLayer(1)}
    />;
  }

  return (
    <div className="h-screen w-screen bg-white flex flex-col overflow-hidden">
      
      {/* ── Top Nav Bar ─────────────────────────────────── */}
      <header
        className="fixed top-0 left-0 right-0 z-50 bg-white border-b-8 border-black flex items-stretch"
        style={{ height: '72px' }}
      >
        {/* Logo area */}
        <div className="flex items-center gap-3 px-6 border-r-8 border-black bg-black text-white">
          <Activity className="text-brand w-5 h-5" strokeWidth={3} />
          <span className="font-black text-base uppercase tracking-tighter whitespace-nowrap">
            PREDICTIVE<span className="text-brand">REALITY</span>ENGINE
          </span>
        </div>

        {/* Pipeline layer tabs */}
        <nav className="flex-1 flex items-stretch overflow-x-auto">
          {navItems.map((item, i) => (
            <button
              key={item.id}
              onClick={() => setActiveLayer(item.id)}
              className={`flex items-center gap-2 px-5 font-black text-xs uppercase tracking-widest border-r-4 border-black cursor-pointer transition-colors whitespace-nowrap
                ${activeLayer === item.id
                  ? 'bg-brand text-black border-b-0'
                  : 'bg-white text-black hover:bg-black/5'
                }
              `}
            >
              <item.icon className="w-4 h-4" strokeWidth={3} />
              {item.id === 0 ? 'Home' : `L${item.id}: ${item.name}`}
            </button>
          ))}
        </nav>

        {/* Status bar */}
        <div className="flex items-center gap-4 px-6 border-l-8 border-black">
          {simulationStatus !== 'idle' && (
            <div className="flex items-center gap-2 border-4 border-black px-3 py-1">
              {simulationStatus === 'running' ? (
                <>
                  <span className="w-2 h-2 bg-brand animate-pulse" />
                  <span className="font-black text-xs uppercase tracking-widest text-brand">{simulationTitle}</span>
                  <span className="font-black text-xs text-black/50">{simulationProgress?.percent ?? 0}%</span>
                </>
              ) : (
                <>
                  <span className="w-2 h-2 bg-green-600" />
                  <span className="font-black text-xs uppercase tracking-widest text-green-600">Complete</span>
                </>
              )}
            </div>
          )}
          <div className="flex items-center gap-2">
            <span className={`w-3 h-3 border-2 border-black ${isConnected ? 'bg-brand animate-pulse' : 'bg-red-500'}`} />
            <span className="font-black text-xs uppercase tracking-widest">{isConnected ? 'Live' : 'Offline'}</span>
          </div>
          <button
            onClick={() => setActiveLayer(0)}
            className="px-4 py-2 bg-black text-white border-4 border-black font-black text-xs uppercase tracking-widest cursor-pointer transition-all hover:translate-x-1 hover:translate-y-1 hover:bg-brand hover:text-black"
          >
            Home
          </button>
        </div>
      </header>

      {/* ── Main Content ─────────────────────────────────── */}
      <main className="flex-1 w-full h-full relative overflow-hidden" style={{ marginTop: '72px' }}>
        {activeLayer === 1 && <IngestorGraph />}
        {activeLayer === 3 && <AssemblyMatrix />}
        {activeLayer === 5 && <OASIS3D />}
        {activeLayer === 6 && <BoardroomDebate />}
        {activeLayer === 8 && <HandoffTerminal />}
        {/* Fallback empty state when jumping via Preview */}
        {![1, 3, 5, 6, 8].includes(activeLayer) && activeLayer !== 0 && activeLayer !== 0.5 && (
          <div className="w-full h-full flex flex-col items-center justify-center gap-6">
            <div className="w-20 h-20 bg-brand border-4 border-black flex items-center justify-center animate-pulse shadow-neo-black">
              <Activity size={36} className="text-black" strokeWidth={3} />
            </div>
            <div className="text-center">
              <h2 className="font-black text-3xl uppercase mb-2">Select a Layer</h2>
              <p className="font-bold text-black/50 text-sm uppercase tracking-widest">Use the navigation above to explore the pipeline</p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;

