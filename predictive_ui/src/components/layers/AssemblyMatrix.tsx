import React, { useMemo, useState } from 'react';
import { Users, Shield, Lock, Briefcase, UserSquare2, ChevronRight, Loader2, Zap, Cpu, Brain, Database, Globe, AlertTriangle, X } from 'lucide-react';
import { usePipelineStore } from '../../store/usePipelineStore';
import { cleanPersonaName } from '../../utils/nameHelper';


const OCEAN_LABELS: Record<string, string> = {
  openness: 'O', conscientiousness: 'C', extraversion: 'E', agreeableness: 'A', neuroticism: 'N'
};
const OCEAN_COLORS: Record<string, string> = {
  openness: '#3B82F6', conscientiousness: '#22C55E', extraversion: '#FF4500',
  agreeableness: '#A855F7', neuroticism: '#EF4444'
};
const JOURNEY_COLORS: Record<string, string> = {
  awareness: 'bg-blue-100 text-blue-700',
  consideration: 'bg-yellow-100 text-yellow-700',
  decision: 'bg-green-100 text-green-700',
  retention: 'bg-purple-100 text-purple-700',
  advocacy: 'bg-brand/10 text-brand',
};

function OceanBars({ scores }: { scores: Record<string, number> }) {
  const keys = Object.keys(scores);
  if (keys.length === 0) return null;
  return (
    <div className="space-y-1.5 mt-3">
      <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-2">OCEAN Profile</div>
      {keys.map((k) => {
        const label = OCEAN_LABELS[k] ?? k[0].toUpperCase();
        const color = OCEAN_COLORS[k] ?? '#888';
        const pct = Math.min(100, Math.max(0, (scores[k] ?? 0) * 100));
        return (
          <div key={k} className="flex items-center gap-2">
            <span className="font-black text-xs w-4 text-black/50">{label}</span>
            <div className="flex-1 h-2 bg-black/10 border border-black overflow-hidden">
              <div
                className="h-full transition-all duration-700"
                style={{ width: `${pct}%`, backgroundColor: color }}
              />
            </div>
            <span className="font-black text-xs w-8 text-right" style={{ color }}>{pct.toFixed(0)}</span>
          </div>
        );
      })}
    </div>
  );
}

function SimConfigPanel() {
  const { simulationConfig } = usePipelineStore();
  if (!simulationConfig) return null;

  const cfg = simulationConfig;
  const rows = [
    { label: 'Platform', value: cfg.platform_type || '—', icon: Globe },
    { label: 'LLM Model', value: cfg.llm_model || '—', icon: Cpu },
    { label: 'Timesteps', value: String(cfg.num_timesteps), icon: Database },
    { label: 'Declared Pop.', value: cfg.declared_population.toLocaleString(), icon: Users },
    { label: 'Active Cohort', value: cfg.llm_active_cohort.toLocaleString(), icon: Brain },
    { label: 'Shadow Agents', value: cfg.shadow_agents.toLocaleString(), icon: Users },
    { label: 'Hindsight', value: cfg.hindsight_available ? 'Active' : 'Off', icon: cfg.hindsight_available ? Brain : AlertTriangle },
    { label: 'Focus Group', value: cfg.interview_phase_enabled ? 'Enabled' : 'Disabled', icon: Users },
  ];

  return (
    <div className="border-b-8 border-black">
      <div className="bg-black text-white px-8 py-4 flex items-center gap-3 border-b-4 border-brand">
        <Cpu className="w-5 h-5 text-brand" strokeWidth={3} />
        <h2 className="font-black text-sm uppercase tracking-widest flex-1">Simulation Configuration</h2>
        <span className="text-xs font-black text-brand uppercase tracking-widest border-2 border-brand px-2 py-0.5">
          ID: {cfg.simulation_id.slice(0, 16)}…
        </span>
      </div>
      <div className="grid grid-cols-4 md:grid-cols-8 divide-x-4 divide-black">
        {rows.map(({ label, value, icon: Icon }) => (
          <div key={label} className="p-4 flex flex-col gap-1">
            <div className="flex items-center gap-1 mb-1">
              <Icon className="w-3 h-3 text-black/40" strokeWidth={3} />
              <span className="text-xs font-black uppercase tracking-widest text-black/40">{label}</span>
            </div>
            <span className={`font-black text-sm uppercase ${value === 'Active' ? 'text-green-600' : value === 'Off' || value === 'Disabled' ? 'text-black/40' : 'text-black'}`}>
              {value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AssemblyMatrix() {
  const { personas, boardroomPersonas, pipelineStages, spawnedAgents, simulationConfig } = usePipelineStore();
  const [selectedPersona, setSelectedPersona] = useState<any | null>(null);

  const stageStatus = pipelineStages.layer3;
  const isWaiting = stageStatus === 'waiting';
  const isRunning = stageStatus === 'running';

  // Merge spawned agent psychological data into personas list
  const enrichedPersonas = useMemo(() => {
    return personas.map((p) => {
      const spawned = Object.values(spawnedAgents).find(
        (s) => s.agent_name === p.name || s.agent_id === p.id
      );
      const cleanName = cleanPersonaName(p.name);
      return { 
        ...p, 
        name: cleanName,
        mbti: spawned?.mbti ?? '', 
        ocean_scores: spawned?.ocean_scores ?? {}, 
        buyer_journey: spawned?.buyer_journey ?? '',
        bio: p.bio ?? spawned?.bio ?? ''
      };
    });
  }, [personas, spawnedAgents]);

  const spawnedList = useMemo(() => {
    return Object.values(spawnedAgents).map(agent => {
      const cleanName = cleanPersonaName(agent.agent_name);
      return {
        ...agent,
        agent_name: cleanName,
        bio: agent.bio ?? ''
      };
    });
  }, [spawnedAgents]);
  
  const totalAgents = spawnedList.length || enrichedPersonas.length;

  const dynamicBoardroom = useMemo(() => {
    let list = boardroomPersonas.map(p => {
      const cleanName = cleanPersonaName(p.name);
      return {
        name: cleanName,
        role: p.role,
        role_short: p.role_short || '',
        traits: p.traits || [],
        accent: 'bg-brand',
        icon: Briefcase,
        bio: p.bio || '',
      };
    });

    if (list.length === 0) {
      list = [
        { name: 'CEO', role: 'Chief Executive Officer', role_short: 'CEO', traits: ['Market Strategy', 'Growth', 'Fundraising'], accent: 'bg-brand', icon: Briefcase },
        { name: 'CTO', role: 'Chief Technology Officer', role_short: 'CTO', traits: ['Architecture', 'Tech Debt', 'Scalability'], accent: 'bg-black', icon: Cpu },
        { name: 'CISO', role: 'Chief Information Security Officer', role_short: 'CISO', traits: ['Data Privacy', 'Compliance', 'Threat Modeling'], accent: 'bg-brand', icon: Shield },
        { name: 'CMO', role: 'Chief Marketing Officer', role_short: 'CMO', traits: ['Brand Safety', 'Customer Outcomes', 'Quality'], accent: 'bg-black', icon: Globe },
        { name: 'CFO', role: 'Chief Financial Officer', role_short: 'CFO', traits: ['Unit Economics', 'Burn Rate', 'ROI Modeling'], accent: 'bg-brand', icon: Briefcase },
        { name: 'CPO', role: 'Chief Product Officer', role_short: 'CPO', traits: ['Product-Market Fit', 'UX Strategy', 'Roadmap Prioritization'], accent: 'bg-black', icon: Brain },
        { name: 'Legal', role: 'General Counsel', role_short: 'Legal', traits: ['Regulatory Compliance', 'Liability', 'Contract Risk'], accent: 'bg-brand', icon: Lock },
        { name: 'Data', role: 'Head of Data & ML', role_short: 'Data', traits: ['ML Accuracy', 'Data Quality', 'Model Bias'], accent: 'bg-black', icon: Database },
        { name: 'Sales', role: 'Head of Sales', role_short: 'Sales', traits: ['Revenue Pipeline', 'Customer Acquisition', 'Competitive Win Rate'], accent: 'bg-brand', icon: Zap },
        { name: 'CS', role: 'Head of Customer Success', role_short: 'CS', traits: ['Onboarding', 'Retention', 'Change Management'], accent: 'bg-black', icon: Users },
      ];
    }

    const getRoleIcon = (roleShort: string) => {
      switch (roleShort.toUpperCase()) {
        case 'CEO': return Briefcase;
        case 'CTO': return Cpu;
        case 'CISO': return Shield;
        case 'CMO': return Globe;
        case 'CFO': return Briefcase;
        case 'CPO': return Brain;
        case 'LEGAL': return Lock;
        case 'DATA': return Database;
        case 'SALES': return Zap;
        case 'CS': return Users;
        default: return Briefcase;
      }
    };

    return list.map((item, idx) => ({
      ...item,
      accent: idx % 2 === 0 ? 'bg-brand' : 'bg-black',
      icon: getRoleIcon(item.role_short || item.name),
    }));
  }, [boardroomPersonas]);

  return (
    <div className="w-full h-full flex flex-col bg-white overflow-y-auto" style={{ paddingTop: '72px' }}>

      {/* ── Simulation Config Band ──────────────────────── */}
      <SimConfigPanel />

      {/* ── Boardroom Assembly ─────────────────────────── */}
      <div className="border-b-8 border-black">
        <div className="bg-black text-white px-8 py-5 flex items-center gap-3 border-b-4 border-brand">
          <Briefcase className="w-5 h-5 text-brand" strokeWidth={3} />
          <h2 className="font-black text-lg uppercase tracking-widest">Layer 4: Boardroom Assembly</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-[4px] bg-black border-b-4 border-black">
          {dynamicBoardroom.map((exec) => (
            <div key={exec.name} onClick={() => setSelectedPersona({ ...exec, isBoardroom: true })} className="p-6 flex flex-col gap-4 bg-white group cursor-pointer hover:bg-brand/10 transition-colors duration-200">
              <div className={`w-12 h-12 ${exec.accent} border-4 border-black flex items-center justify-center shadow-neo-black transition-all duration-200 group-hover:shadow-none group-hover:translate-x-1 group-hover:translate-y-1`}>
                <exec.icon className={`w-6 h-6 ${exec.accent === 'bg-brand' ? 'text-black' : 'text-white'}`} strokeWidth={3} />
              </div>
              <div>
                <h3 className="font-black text-lg uppercase truncate" title={exec.name}>{exec.name}</h3>
                <p className="text-xs font-black text-brand uppercase tracking-wider truncate" title={exec.role}>{exec.role}</p>
                <div className="mt-3 pt-3 border-t-2 border-black">
                  <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-1 flex items-center gap-1">
                    <Lock className="w-3 h-3" strokeWidth={3} /> Core Directives
                  </div>
                  <p className="text-xs font-bold leading-snug text-black/70 line-clamp-3" title={exec.traits.join(', ')}>
                    {exec.traits.length > 0 ? exec.traits.join(', ') : 'Aligned with general boardroom mandate.'}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Persona Generation ─────────────────────────── */}
      <div className="flex-1 relative">
        <div className="bg-white px-8 py-5 border-b-4 border-black flex items-center gap-3">
          <Users className="w-5 h-5 text-brand" strokeWidth={3} />
          <h2 className="font-black text-lg uppercase tracking-widest flex-1">Layer 3: Synthetic Persona Generation</h2>
          {totalAgents > 0 && (
            <span className="border-4 border-brand bg-brand px-3 py-1 font-black text-sm uppercase tracking-widest">
              {totalAgents} Agents
            </span>
          )}
        </div>

        {/* Pending / Running Overlay */}
        {(isWaiting || isRunning) && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-white/90" style={{ top: '60px' }}>
            <div className="border-8 border-black px-12 py-10 flex flex-col items-center gap-4 text-center shadow-neo-black max-w-sm">
              {isRunning ? (
                <div className="w-12 h-12 bg-brand border-4 border-black flex items-center justify-center">
                  <Loader2 className="w-6 h-6 text-black animate-spin" strokeWidth={3} />
                </div>
              ) : (
                <div className="w-12 h-12 bg-white border-4 border-black flex items-center justify-center">
                  <div className="w-3 h-3 bg-black animate-pulse" />
                </div>
              )}
              <div>
                <p className="font-black text-sm uppercase tracking-widest text-brand mb-2">
                  {isRunning ? 'Layer 3 — Generating Personas' : 'Stage Pending'}
                </p>
                <p className="text-sm font-bold text-black/60">
                  {isRunning ? 'LLM is synthesising behavioural profiles…' : 'Waiting for previous stages to complete.'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Persona cards — enriched with OCEAN + MBTI + buyer journey */}
        {enrichedPersonas.length > 0 ? (
          <div className="grid grid-cols-3 divide-x-4 divide-black border-b-4 border-black">
            {enrichedPersonas.map((persona) => (
              <div key={persona.id} onClick={() => setSelectedPersona({ ...persona, isBoardroom: false })} className="p-6 flex flex-col gap-4 group cursor-pointer hover:bg-brand/5 transition-colors duration-200">
                <div className="flex justify-between items-start">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-black text-lg uppercase truncate">{persona.name}</h3>
                    <p className="text-sm font-black text-brand uppercase tracking-wider">{persona.role}</p>
                    {persona.buyer_journey && (
                      <span className={`inline-block mt-2 text-xs font-black uppercase tracking-widest px-2 py-0.5 ${JOURNEY_COLORS[persona.buyer_journey] ?? 'bg-black/10 text-black/60'}`}>
                        {persona.buyer_journey}
                      </span>
                    )}
                  </div>
                  <div className="text-right flex-none ml-4">
                    <div className="text-xs font-black uppercase tracking-widest text-black/40">Impact</div>
                    <div className={`text-3xl font-black ${persona.impact > 80 ? 'text-brand' : 'text-black'}`}>
                      {persona.impact}
                    </div>
                    {persona.mbti && (
                      <div className="mt-1 text-xs font-black border-2 border-black px-2 py-0.5 uppercase tracking-widest">
                        {persona.mbti}
                      </div>
                    )}
                  </div>
                </div>

                <div className="border-t-2 border-black pt-4">
                  <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-2">Pain Points</div>
                  <div className="flex flex-wrap gap-2">
                    {(persona.traits.length > 0 ? persona.traits : ['Analysed']).map((trait, i) => (
                      <span key={i} className="text-xs px-2 py-1 bg-black text-white font-bold uppercase">
                        {trait}
                      </span>
                    ))}
                  </div>
                </div>

                {/* OCEAN bars — only if data available from spawn event */}
                <OceanBars scores={persona.ocean_scores} />

                <button 
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedPersona({ ...persona, isBoardroom: false });
                  }}
                  className="mt-auto py-2 px-4 bg-white border-4 border-black shadow-neo-black font-black text-sm uppercase tracking-widest flex items-center justify-between cursor-pointer transition-all duration-200 hover:translate-x-1 hover:translate-y-1 hover:shadow-none"
                >
                  View Profile <ChevronRight className="w-4 h-4" strokeWidth={3} />
                </button>
              </div>
            ))}
          </div>
        ) : spawnedList.length > 0 ? (
          /* Fallback: show spawned agents from simulation when persona sync hasn't arrived */
          <div className="grid grid-cols-3 divide-x-4 divide-black border-b-4 border-black">
            {spawnedList.slice(0, 9).map((agent) => (
              <div key={agent.agent_id} onClick={() => setSelectedPersona({ ...agent, isBoardroom: false })} className="p-6 flex flex-col gap-4 group cursor-pointer hover:bg-brand/5 transition-colors duration-200">
                <div className="flex justify-between items-start">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-black text-lg uppercase truncate">{agent.agent_name}</h3>
                    <p className="text-sm font-black text-brand uppercase tracking-wider">{agent.role}</p>
                    {agent.buyer_journey && (
                      <span className={`inline-block mt-2 text-xs font-black uppercase tracking-widest px-2 py-0.5 ${JOURNEY_COLORS[agent.buyer_journey] ?? 'bg-black/10 text-black/60'}`}>
                        {agent.buyer_journey}
                      </span>
                    )}
                  </div>
                  <div className="text-right flex-none ml-4">
                    <div className="text-xs font-black uppercase tracking-widest text-black/40">Impact</div>
                    <div className="text-3xl font-black text-black">{agent.impact}</div>
                    {agent.mbti && (
                      <div className="mt-1 text-xs font-black border-2 border-black px-2 py-0.5 uppercase tracking-widest">{agent.mbti}</div>
                    )}
                  </div>
                </div>
                <div className="border-t-2 border-black pt-4">
                  <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-2">Traits</div>
                  <div className="flex flex-wrap gap-2">
                    {(agent.traits.length > 0 ? agent.traits : ['Agent']).map((t, i) => (
                      <span key={i} className="text-xs px-2 py-1 bg-black text-white font-bold uppercase">{t}</span>
                    ))}
                  </div>
                </div>
                <OceanBars scores={agent.ocean_scores ?? {}} />
                
                <button 
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedPersona({ ...agent, isBoardroom: false });
                  }}
                  className="mt-auto py-2 px-4 bg-white border-4 border-black shadow-neo-black font-black text-sm uppercase tracking-widest flex items-center justify-between cursor-pointer transition-all duration-200 hover:translate-x-1 hover:translate-y-1 hover:shadow-none"
                >
                  View Profile <ChevronRight className="w-4 h-4" strokeWidth={3} />
                </button>
              </div>
            ))}
          </div>
        ) : !isWaiting && !isRunning ? (
          <div className="flex flex-col items-center justify-center h-48 gap-4">
            <div className="w-12 h-12 bg-brand border-4 border-black flex items-center justify-center animate-pulse">
              <Zap size={20} className="text-black" strokeWidth={3} />
            </div>
            <p className="font-black text-sm uppercase tracking-widest text-black/40">Awaiting persona data…</p>
          </div>
        ) : null}
      </div>

      {/* ── Detail Modal ────────────────────────────── */}
      {selectedPersona && (
        <div 
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedPersona(null)}
        >
          <div 
            className="bg-white border-8 border-black shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] max-w-2xl w-full flex flex-col max-h-[85vh] overflow-hidden animate-in fade-in zoom-in-95 duration-150"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header Band */}
            <div className="bg-black text-white px-8 py-5 flex items-center justify-between border-b-4 border-brand flex-none">
              <div className="flex items-center gap-3">
                {selectedPersona.isBoardroom ? (
                  <Briefcase className="w-5 h-5 text-brand" strokeWidth={3} />
                ) : (
                  <Users className="w-5 h-5 text-brand" strokeWidth={3} />
                )}
                <h3 className="font-black text-lg uppercase tracking-widest">
                  {selectedPersona.isBoardroom ? 'Boardroom Executive' : 'Synthetic User Profile'}
                </h3>
              </div>
              <button 
                onClick={() => setSelectedPersona(null)}
                className="w-10 h-10 bg-white text-black border-4 border-black flex items-center justify-center font-black hover:bg-brand hover:translate-x-[2px] hover:translate-y-[2px] transition-all"
              >
                <X className="w-5 h-5" strokeWidth={3} />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-8 overflow-y-auto flex flex-col gap-6 text-black flex-1">
              {/* Profile Intro Banner */}
              <div className="flex flex-col sm:flex-row gap-6 items-start pb-6 border-b-4 border-black">
                <div className={`w-20 h-20 flex-none border-4 border-black flex items-center justify-center shadow-neo-black ${selectedPersona.accent || 'bg-brand'}`}>
                  {selectedPersona.icon ? (
                    <selectedPersona.icon className={`w-10 h-10 ${selectedPersona.accent === 'bg-black' ? 'text-white' : 'text-black'}`} strokeWidth={3} />
                  ) : (
                    <Users className="w-10 h-10 text-black" strokeWidth={3} />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="font-black text-2xl uppercase leading-none mb-1 break-words">{selectedPersona.name}</h4>
                  <p className="text-sm font-black text-brand uppercase tracking-wider mb-3 break-words">{selectedPersona.role}</p>
                  
                  {/* Badges */}
                  <div className="flex flex-wrap gap-2">
                    {selectedPersona.buyer_journey && (
                      <span className={`text-xs font-black uppercase tracking-widest px-2.5 py-1 ${JOURNEY_COLORS[selectedPersona.buyer_journey] || 'bg-black/10 text-black/60'}`}>
                        Journey: {selectedPersona.buyer_journey}
                      </span>
                    )}
                    {selectedPersona.mbti && (
                      <span className="text-xs font-black border-2 border-black px-2.5 py-0.5 uppercase tracking-widest bg-white">
                        MBTI: {selectedPersona.mbti}
                      </span>
                    )}
                    {selectedPersona.impact !== undefined && (
                      <span className="text-xs font-black bg-black text-white px-2.5 py-1 uppercase tracking-widest">
                        Impact: {selectedPersona.impact}%
                      </span>
                    )}
                    {selectedPersona.isBoardroom && (
                      <span className="text-xs font-black bg-brand px-2.5 py-1 text-black uppercase tracking-widest border-2 border-black">
                        Role Short: {selectedPersona.role_short || selectedPersona.name}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Bio / Psychological Profile */}
              <div>
                <h5 className="font-black text-xs uppercase tracking-widest text-black/40 mb-2 font-mono">Psychological Profile & Directives</h5>
                <div className="bg-black/5 border-4 border-black p-5 font-bold text-black/85 leading-relaxed text-sm whitespace-pre-line">
                  {selectedPersona.bio || 'This persona behaves in accordance with their designated role and company domain requirements. No custom narrative bio has been generated yet.'}
                </div>
              </div>

              {/* Traits / Expertise Tags */}
              {selectedPersona.traits && selectedPersona.traits.length > 0 && (
                <div>
                  <h5 className="font-black text-xs uppercase tracking-widest text-black/40 mb-3 font-mono">Key Directives & Traits</h5>
                  <div className="flex flex-wrap gap-2">
                    {selectedPersona.traits.map((trait: string, idx: number) => (
                      <span key={idx} className="text-xs px-3 py-1.5 bg-black text-white font-black uppercase tracking-wider border-2 border-black">
                        {trait}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* OCEAN Metrics */}
              {selectedPersona.ocean_scores && Object.keys(selectedPersona.ocean_scores).length > 0 && (
                <div className="pt-4 border-t-4 border-black">
                  <OceanBars scores={selectedPersona.ocean_scores} />
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="bg-black/5 p-6 border-t-4 border-black flex justify-end flex-none">
              <button 
                onClick={() => setSelectedPersona(null)}
                className="py-3 px-8 bg-brand text-black border-4 border-black font-black uppercase tracking-widest shadow-neo-black hover:translate-x-1 hover:translate-y-1 hover:shadow-none transition-all duration-200"
              >
                Close Profile
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
