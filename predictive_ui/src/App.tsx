import { useState, useEffect, useRef } from 'react';
import { Layers, Activity, Users, Terminal, MessageSquare, Home, FileText, Square, Brain } from 'lucide-react';
import OASIS3D from './components/layers/OASIS3D';
import BoardroomDebate from './components/layers/BoardroomDebate';
import IngestorGraph from './components/layers/IngestorGraph';
import AssemblyMatrix from './components/layers/AssemblyMatrix';
import HandoffTerminal from './components/layers/HandoffTerminal';
import SeedReviewPage from './components/layers/SeedReviewPage';
import MemoryQueryPage from './components/layers/MemoryQueryPage';
import LandingPage from './components/LandingPage';
import InputSetupPage from './components/InputSetupPage';
import { useWebSocket } from './hooks/useWebSocket';
import { usePipelineStore } from './store/usePipelineStore';
import { API_BASE_URL } from './config';
import { BackendTerminal } from './components/ui/BackendTerminal';

function formatUserFriendlyMessage(rawMessage: string): string | null {
  if (!rawMessage) return null;
  const lowerMsg = rawMessage.toLowerCase();

  // Layer 1: Ingestion
  if (lowerMsg.includes('ingest') || lowerMsg.includes('chunking') || lowerMsg.includes('loading') || lowerMsg.includes('extract') || lowerMsg.includes('normalize')) {
    const match = rawMessage.match(/([a-zA-Z0-9_\-\s]+\.(?:json|csv|md|txt|pdf|docx))/i) || rawMessage.match(/(?:file|document|chunking)\s+([a-zA-Z0-9_\-\s]+)/i);
    const filename = match ? match[1].trim() : 'documents';
    return `Ingesting file ${filename}... Building semantic knowledge graphs in LightRAG.`;
  }

  // Layer 2: Feature Discovery
  if (lowerMsg.includes('gap') || lowerMsg.includes('unarticulated') || lowerMsg.includes('feature discovery') || lowerMsg.includes('market')) {
    return "Analyzing market gaps and uncovering unarticulated user needs...";
  }

  // Layer 3: Personas
  if (lowerMsg.includes('persona') || lowerMsg.includes('psychological') || lowerMsg.includes('trait')) {
    return "Synthesizing psychological profiles... Creating rich user personas (This usually takes ~10-15 minutes depending on LLM latency).";
  }

  // Layer 5 Post-Simulation (Clustering/Alignment happens after sim)
  if (lowerMsg.includes('cluster') || lowerMsg.includes('alignment') || lowerMsg.includes('aai')) {
    return "Clustering semantic beliefs and computing the Agent Alignment Index (AAI)...";
  }

  // Layer 4 / 5: Simulation
  if (lowerMsg.includes('simulat') || lowerMsg.includes('agent') || lowerMsg.includes('action') || lowerMsg.includes('step')) {
    return "Simulation running. Generating autonomous agent interactions and tracking behavioral shifts...";
  }

  // Layer 6: Memory
  if (lowerMsg.includes('memor') || lowerMsg.includes('metric') || lowerMsg.includes('store') || lowerMsg.includes('hindsight')) {
    return "Storing simulation memory and evaluating core metrics...";
  }

  // Layer 7: Final Synthesis
  if (lowerMsg.includes('report') || lowerMsg.includes('executive') || lowerMsg.includes('synthesis')) {
    return "Generating the final Predictive Reality Executive Report...";
  }

  return null;
}

const LiveStatusMessage = () => {
  const [displayMessage, setDisplayMessage] = useState<string>('');
  const latestLog = usePipelineStore(state =>
    state.systemLogs.length > 0 ? state.systemLogs[state.systemLogs.length - 1] : ''
  );

  useEffect(() => {
    if (latestLog) {
      const friendlyMsg = formatUserFriendlyMessage(latestLog);
      if (friendlyMsg) {
        setDisplayMessage(friendlyMsg);
      }
    }
  }, [latestLog]);

  if (!displayMessage) return null;

  return (
    <div className="text-[10px] font-bold text-black/70 truncate uppercase tracking-wider">
      {displayMessage}
    </div>
  );
};

function App() {
  const ws = useWebSocket();
  const evalWsRef = useRef<WebSocket | null>(null);
  const { isConnected, simulationStatus, simulationTitle, simulationProgress, pendingAction, sessionId, simulationReport, isBootstrapped, stopSimulation } = usePipelineStore();
  const [activeLayer, setActiveLayer] = useState(0.5); // 0.5 = Input Setup Page
  const [statusWidth, setStatusWidth] = useState(450);
  const [hasAutoSwitched, setHasAutoSwitched] = useState(false);
  const [showTerminal, setShowTerminal] = useState(false);

  // Clean up evaluation WebSocket on unmount
  useEffect(() => {
    return () => {
      if (evalWsRef.current) {
        evalWsRef.current.close();
      }
    };
  }, []);

  // Auto-switch to Seeds tab when review is needed
  useEffect(() => {
    if (pendingAction?.action === 'review_seeds') {
      setActiveLayer(4);
    }
  }, [pendingAction]);

  // Auto-switch to dashboard on load if a simulation already exists
  useEffect(() => {
    if (isBootstrapped && sessionId && activeLayer === 0.5 && !hasAutoSwitched) {
      if (simulationReport) {
        setActiveLayer(8); // Handoff
      } else if (pendingAction?.action === 'review_seeds') {
        setActiveLayer(4); // Seeds
      } else {
        // If simulation is running (or recently started), show the OASIS Sim layer, 
        // since the user wants the latest simulation to be shown.
        // Wait, if it's completely empty we'd show ingestion, but if it has a sessionId it's running!
        setActiveLayer(5); // OASIS Sim
      }
      setHasAutoSwitched(true);
    }
  }, [isBootstrapped, sessionId, activeLayer, hasAutoSwitched, simulationReport, pendingAction]);

  const navItems = [
    { id: 0.5, name: 'Home', icon: Home },
    { id: 1, name: 'Ingestion', icon: Layers },
    { id: 3, name: 'Personas', icon: Users },
    { id: 4, name: 'Seeds', icon: FileText },
    { id: 5, name: 'OASIS Sim', icon: Activity },
    { id: 6, name: 'Boardroom', icon: MessageSquare },
    { id: 8, name: 'Handoff', icon: Terminal },
    { id: 9, name: 'Memory', icon: Brain },
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

  // InputSetupPage handles the /api/upload call and passes back server-side file paths.
  // We switch to Layer 1 (Ingestion) and trigger the backend simulation via FastAPI's ws/evaluate endpoint.
  const handleStartSimulation = async (filePaths: Record<string, string>, boardroomOnly: boolean = false) => {
    // Switch to Ingestion layer immediately
    setActiveLayer(1);

    // Close any previous evaluation socket connection
    if (evalWsRef.current) {
      try {
        evalWsRef.current.close();
      } catch (e) { }
    }

    const wsBaseUrl = import.meta.env.VITE_WS_URL || (API_BASE_URL
      ? API_BASE_URL.replace(/^http/, 'ws')
      : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`);
    const evalUrl = `${wsBaseUrl}/ws/evaluate`;

    console.log(`[FastAPI WS] Connecting to ${evalUrl}...`);
    const socket = new WebSocket(evalUrl);
    evalWsRef.current = socket;

    socket.onopen = () => {
      console.log("[FastAPI WS] Connected! Sending configuration...");
      socket.send(JSON.stringify({
        files: filePaths,
        boardroom_only: boardroomOnly
      }));
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("[FastAPI WS] Received event:", data);

        // Handle rejection from the backend (e.g., simulation already running)
        if (data.type === 'error') {
          alert(`Simulation Error: ${data.message}`);
          setActiveLayer(0.5); // Redirect back to Home
        }
      } catch (e) {
        console.error("[FastAPI WS] Failed to parse message:", e);
      }
    };

    socket.onerror = (err) => {
      console.error("[FastAPI WS] Error occurred:", err);
    };

    socket.onclose = (event) => {
      console.log("[FastAPI WS] Connection closed:", event.code, event.reason);
      if (evalWsRef.current === socket) {
        evalWsRef.current = null;
      }
    };
  };

  // Signal the backend to stop the simulation gracefully (proceed to next step).
  // We rely on the backend's WebSocket to update the frontend state to 'idle'
  // once the final metrics are fully aggregated.
  const handleStopSimulation = async () => {
    try {
      await fetch(`${API_BASE_URL}/api/simulation/stop`, { method: 'POST' });
    } catch (e) {
      console.warn('Stop request failed — backend may already be done:', e);
    }
  };

  // Hard abort the simulation entirely at any layer.
  const handleAbortSimulation = async () => {
    try {
      await fetch(`${API_BASE_URL}/api/simulation/abort`, { method: 'POST' });
    } catch (e) {
      console.warn('Abort request failed:', e);
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
        <div className="flex items-center gap-3 px-6 border-r-8 border-black bg-black text-white shrink-0">
          <Activity className="text-brand w-5 h-5" strokeWidth={3} />
          <span className="font-black text-base uppercase tracking-tighter whitespace-nowrap hidden sm:inline">
            PREDICTIVE<span className="text-brand">REALITY</span>ENGINE
          </span>
        </div>

        {/* Pipeline layer tabs */}
        <nav className="flex-1 flex items-stretch overflow-x-auto min-w-0 scrollbar-hide">
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
              {item.id === 0.5 ? 'Home' : `L${item.id}: ${item.name}`}
            </button>
          ))}
        </nav>

        {/* Status bar */}
        <div
          className="flex items-center gap-4 px-6 border-l-8 border-black shrink-0 relative"
          style={{ width: statusWidth, minWidth: '350px', maxWidth: '80vw' }}
        >
          {/* Custom resize handle over the black border */}
          <div
            className="absolute -left-2 top-0 bottom-0 w-4 cursor-col-resize z-50"
            onMouseDown={(e) => {
              e.preventDefault();
              const startX = e.clientX;
              const startWidth = statusWidth;

              const onMouseMove = (moveEvent: MouseEvent) => {
                setStatusWidth(Math.max(350, Math.min(window.innerWidth * 0.8, startWidth - (moveEvent.clientX - startX))));
              };

              const onMouseUp = () => {
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
              };

              document.addEventListener('mousemove', onMouseMove);
              document.addEventListener('mouseup', onMouseUp);
            }}
          />

          {simulationStatus !== 'idle' && (
            <div className="flex items-center gap-2 border-4 border-black px-3 py-1 flex-1 min-w-0 overflow-hidden">
              {simulationStatus === 'running' ? (
                <>
                  <span className="w-2 h-2 bg-brand animate-pulse shrink-0" />
                  <div className="flex flex-col flex-1 min-w-0 justify-center">
                    <div className="font-black text-xs uppercase tracking-widest text-brand overflow-x-auto whitespace-nowrap scrollbar-hide">
                      {simulationTitle}
                    </div>
                    <LiveStatusMessage />
                  </div>
                  <span className="font-black text-xs text-black/50 shrink-0">{simulationProgress?.percent ?? 0}%</span>
                </>
              ) : (
                <>
                  <span className="w-2 h-2 bg-green-600 shrink-0" />
                  <span className="font-black text-xs uppercase tracking-widest text-green-600 shrink-0">Complete</span>
                </>
              )}
            </div>
          )}
          <div className="flex items-center gap-2 shrink-0">
            <span className={`w-3 h-3 border-2 border-black ${isConnected ? 'bg-brand animate-pulse' : 'bg-red-500'}`} />
            <span className="font-black text-xs uppercase tracking-widest">{isConnected ? 'Live' : 'Offline'}</span>
          </div>

          {/* Terminal Toggle Button */}
          <button
            onClick={() => setShowTerminal(!showTerminal)}
            className={`flex items-center gap-2 px-4 py-2 border-l-4 border-black font-black text-xs uppercase tracking-widest cursor-pointer shrink-0 transition-colors ${showTerminal ? 'bg-black text-white' : 'bg-white text-black hover:bg-black/5'
              }`}
          >
            <Terminal className="w-4 h-4" strokeWidth={3} />
            <span className="hidden sm:inline">Terminal</span>
          </button>

          {simulationStatus !== 'idle' && (
            <button
              onClick={handleAbortSimulation}
              className="flex items-center gap-1.5 px-4 py-2 border-l-4 border-black bg-red-600 text-white font-black text-xs uppercase tracking-widest cursor-pointer hover:bg-red-700 transition-colors shrink-0"
              title="Immediately abort the pipeline at any stage"
            >
              <Square className="w-3 h-3 fill-white" strokeWidth={4} />
              <span className="hidden sm:inline">Abort Simulation</span>
            </button>
          )}



          <button
            onClick={() => setActiveLayer(0.5)}
            className="shrink-0 px-4 py-2 bg-black text-white border-4 border-black font-black text-xs uppercase tracking-widest cursor-pointer transition-all hover:translate-x-1 hover:translate-y-1 hover:bg-brand hover:text-black"
          >
            Home
          </button>
        </div>
      </header>

      {/* ── Main Content ─────────────────────────────────── */}
      <main className="flex-1 w-full h-full relative overflow-hidden" style={{ marginTop: '72px' }}>
        {activeLayer === 1 && <IngestorGraph />}
        {activeLayer === 3 && <AssemblyMatrix />}
        {activeLayer === 4 && <SeedReviewPage />}
        {activeLayer === 5 && <OASIS3D />}
        {activeLayer === 6 && <BoardroomDebate />}
        {activeLayer === 8 && <HandoffTerminal />}
        {activeLayer === 9 && <MemoryQueryPage />}
        {/* Fallback empty state when jumping via Preview */}
        {![1, 3, 4, 5, 6, 8, 9].includes(activeLayer) && activeLayer !== 0 && activeLayer !== 0.5 && (
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

        {/* Floating Backend Terminal */}
        <BackendTerminal
          isOpen={showTerminal}
          onClose={() => setShowTerminal(false)}
        />
      </main>
    </div>
  );
}

export default App;

