import React, { useState, useEffect } from 'react';
import { usePipelineStore } from '../../store/usePipelineStore';

export const SeedReviewPage = () => {
  const { pendingAction, setPendingAction, simulationConfig } = usePipelineStore();
  const [instruction, setInstruction] = useState('');
  const [isRefining, setIsRefining] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [glitchText, setGlitchText] = useState('HUMAN-IN-THE-LOOP');
  
  if (!pendingAction || pendingAction.action !== 'review_seeds') {
    return null;
  }
  
  const payload = pendingAction.payload;
  // Make a local copy of seeds to show
  const [currentSeeds, setCurrentSeeds] = useState<string[]>(payload.seeds || []);

  // Cyberpunk glitch effect on title
  useEffect(() => {
    const interval = setInterval(() => {
      if (Math.random() > 0.8) {
        setGlitchText('H_MAN-IN-T#E-L00P');
        setTimeout(() => setGlitchText('HUMAN-IN-THE-LOOP'), 150);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleRefine = async () => {
    if (!instruction.trim()) return;
    setIsRefining(true);
    try {
      const res = await fetch('http://localhost:8000/api/simulation/refine_seeds', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          seeds: currentSeeds,
          instruction: instruction,
          provider: simulationConfig?.llm_model ? 'ollama' : undefined,
          model: simulationConfig?.llm_model
        })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.seeds && data.seeds.length > 0) {
          setCurrentSeeds(data.seeds);
          setInstruction('');
        }
      } else {
        console.error("Failed to refine seeds", await res.text());
      }
    } catch (e) {
      console.error("Refine error:", e);
    } finally {
      setIsRefining(false);
    }
  };

  const handleApprove = async () => {
    setIsSubmitting(true);
    try {
      const sessionId = simulationConfig?.simulation_id || 'default';
      const res = await fetch(`http://localhost:8000/api/simulation/${sessionId}/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'action_response',
          action: 'review_seeds',
          data: { seeds: currentSeeds }
        })
      });
      if (res.ok) {
        setPendingAction(null);
      } else {
        console.error("Failed to send approval", await res.text());
        setIsSubmitting(false);
      }
    } catch (e) {
      console.error("Approve error:", e);
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex flex-col p-4 md:p-8 overflow-y-auto bg-pink-50 selection:bg-cyan-400 selection:text-black" style={{
      backgroundImage: 'radial-gradient(#000 1px, transparent 1px)',
      backgroundSize: '24px 24px'
    }}>
      <div className="max-w-6xl mx-auto w-full flex flex-col lg:flex-row gap-8">
        
        {/* Left Column: Context & Refinement */}
        <div className="w-full lg:w-1/3 flex flex-col gap-6 sticky top-8 h-fit">
          <div className="border-4 border-black bg-white p-6 shadow-[8px_8px_0_0_#000]">
            <div className="inline-block bg-black text-cyan-400 font-mono font-black text-xs px-2 py-1 mb-4 uppercase tracking-widest">
              SYSTEM OVERRIDE // ACTIVE
            </div>
            <h2 className="font-black text-4xl uppercase tracking-tighter leading-none mb-2 text-pink-600">
              {glitchText}
            </h2>
            <p className="font-bold text-black uppercase tracking-widest text-sm border-l-4 border-cyan-400 pl-3 mt-4">
              Awaiting operator authorization. Review generated seeds before simulation lock.
            </p>
          </div>

          <div className="flex flex-col gap-0 border-4 border-black shadow-[8px_8px_0_0_#ec4899] bg-white">
            <div className="bg-black text-white p-3 font-black text-sm uppercase tracking-widest flex justify-between items-center">
              <span>Refinement Terminal</span>
              <span className="text-pink-500 animate-pulse">_</span>
            </div>
            <div className="p-4 flex flex-col gap-4">
              <p className="text-xs font-mono font-bold text-gray-500 uppercase">
                Input persona directives, chain-of-thought logic, or styling constraints below.
              </p>
              <textarea 
                className="w-full min-h-[150px] border-2 border-black p-4 font-mono text-sm outline-none focus:border-cyan-400 focus:bg-cyan-50 transition-none resize-y"
                placeholder="> INJECT SKILL: Make posts sound highly critical, use cynical tone..."
                value={instruction}
                onChange={e => setInstruction(e.target.value)}
                disabled={isRefining || isSubmitting}
              />
              <button 
                onClick={handleRefine}
                disabled={isRefining || isSubmitting || !instruction.trim()}
                className="w-full py-4 bg-cyan-400 border-2 border-black font-black uppercase tracking-widest text-black hover:bg-pink-500 hover:text-white transition-none disabled:opacity-50 active:translate-y-1 active:shadow-none"
              >
                {isRefining ? 'EXECUTING...' : 'APPLY MUTATION'}
              </button>
            </div>
          </div>

          <button 
            onClick={handleApprove}
            disabled={isSubmitting || isRefining}
            className="w-full py-6 bg-brand border-4 border-black font-black uppercase tracking-widest text-xl hover:bg-black hover:text-brand transition-none shadow-[8px_8px_0_0_#000] active:shadow-none active:translate-y-2 active:translate-x-2 disabled:opacity-50"
          >
            {isSubmitting ? 'INITIALIZING...' : 'AUTHORIZE SIMULATION'}
          </button>
        </div>

        {/* Right Column: Seed Cards */}
        <div className="w-full lg:w-2/3 flex flex-col gap-6">
          <div className="flex justify-between items-end border-b-4 border-black pb-2">
            <h3 className="font-black text-2xl uppercase tracking-tighter">Generated Seeds</h3>
            <span className="font-mono font-bold text-sm bg-black text-white px-2 py-1">TOTAL: {currentSeeds.length}</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {currentSeeds.map((seed, idx) => (
              <div 
                key={idx} 
                className="group flex flex-col border-4 border-black bg-white shadow-[4px_4px_0_0_#000] hover:shadow-[8px_8px_0_0_#06b6d4] transition-none"
              >
                <div className="border-b-4 border-black bg-gray-100 px-3 py-2 flex justify-between items-center">
                  <span className="font-black text-lg">POST_{String(idx + 1).padStart(2, '0')}</span>
                  <div className="w-3 h-3 rounded-full bg-pink-500 border-2 border-black" />
                </div>
                <div className="p-5 flex-1 relative">
                  <div className="absolute top-0 right-0 w-8 h-8 border-l-4 border-b-4 border-black opacity-0 group-hover:opacity-100 transition-none" />
                  <p className="font-medium leading-relaxed text-gray-800 text-sm md:text-base">
                    {seed}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
};

export default SeedReviewPage;
