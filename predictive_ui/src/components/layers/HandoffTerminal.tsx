import React, { useState } from 'react';
import { Terminal as TerminalIcon, FileCode2, GitMerge, Loader2, CheckCircle2, TrendingUp, TrendingDown, AlertTriangle, BarChart3, Zap, Users, DollarSign, Activity, ChevronDown, ChevronUp, Target, Shield, BookOpen } from 'lucide-react';
import { usePipelineStore } from '../../store/usePipelineStore';

function Sparkline({ data, color }: { data: number[]; color: string }) {
  if (!data || data.length < 2) return null;
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const w = 140; const h = 40;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return (
    <svg width={w} height={h}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="3" strokeLinecap="square" strokeLinejoin="miter" />
    </svg>
  );
}

function RiskBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="mb-3">
      <div className="flex justify-between text-xs font-black mb-1 uppercase tracking-widest">
        <span className="text-black/50">{label}</span>
        <span style={{ color }}>{(value * 100).toFixed(0)}%</span>
      </div>
      <div className="h-3 bg-black/10 border-2 border-black overflow-hidden">
        <div className="h-full transition-all duration-700" style={{ width: `${value * 100}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function MiniBar({ value, max = 1, color }: { value: number; max?: number; color: string }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="w-16 h-2 bg-black/10 border border-black overflow-hidden">
      <div className="h-full" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  );
}

function CollapsibleSection({ title, icon: Icon, badge, defaultOpen = false, children }: {
  title: string; icon: React.ElementType; badge?: string; defaultOpen?: boolean; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-4 border-black shadow-neo-black">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-2 px-5 py-3 bg-black text-white cursor-pointer hover:bg-black/80 transition-colors duration-200"
      >
        <Icon className="w-4 h-4 text-brand" strokeWidth={3} />
        <span className="font-black text-xs uppercase tracking-widest flex-1 text-left">{title}</span>
        {badge && <span className="text-xs font-black text-brand border border-brand px-2 py-0.5">{badge}</span>}
        {open ? <ChevronUp className="w-3 h-3 text-white/60" strokeWidth={3} /> : <ChevronDown className="w-3 h-3 text-white/60" strokeWidth={3} />}
      </button>
      {open && <div className="p-5">{children}</div>}
    </div>
  );
}

export default function HandoffTerminal() {
  const {
    simulationStatus, simulationTitle, simulationProgress, simulationReport, debateMessages,
    finalRecommendation, focusGroupResults, populationStats, agentJournals,
  } = usePipelineStore();

  const isRunning = simulationStatus === 'running';
  const isComplete = simulationStatus === 'complete';
  const isIdle = simulationStatus === 'idle';
  const progressPercent = simulationProgress?.percent ?? 0;

  return (
    <div className="w-full h-full flex bg-white">

      {/* ── Left: Simulation Report ──────────────────────── */}
      <div className="flex-[2] flex flex-col overflow-hidden border-r-8 border-black">
        <div className="bg-black text-white px-6 py-4 border-b-4 border-brand flex items-center gap-3 flex-none">
          <FileCode2 className="w-4 h-4 text-brand" strokeWidth={3} />
          <h2 className="font-black text-sm uppercase tracking-widest flex-1">Layer 7–8: Spec &amp; Handoff</h2>
          {isRunning && (
            <span className="text-xs font-black text-brand uppercase flex items-center gap-1 animate-pulse">
              <span className="w-2 h-2 bg-brand" /> Simulation Running
            </span>
          )}
          {isComplete && (
            <span className="text-xs font-black text-black bg-brand uppercase flex items-center gap-1 px-2 py-1">
              <CheckCircle2 className="w-3.5 h-3.5" strokeWidth={3} /> Complete
            </span>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {isIdle && (
            <div className="flex flex-col items-center justify-center h-full gap-4">
              <div className="w-16 h-16 bg-brand border-4 border-black flex items-center justify-center animate-pulse">
                <BarChart3 className="w-8 h-8 text-black" strokeWidth={3} />
              </div>
              <p className="font-black text-sm uppercase tracking-widest text-black/40">Awaiting simulation start…</p>
            </div>
          )}

          {(isRunning || isComplete) && (
            <>
              {/* Progress block */}
              <div className="border-4 border-black shadow-neo-black p-5">
                <div className="font-black text-lg uppercase mb-3">{simulationTitle || 'OASIS Simulation'}</div>
                <div className="flex justify-between text-xs font-black mb-2 uppercase tracking-widest text-black/50">
                  <span>Timestep {simulationProgress?.timestep ?? 0} / {simulationProgress?.total ?? 'N/A'}</span>
                  <span>{progressPercent}%</span>
                </div>
                <div className="h-4 bg-black/10 border-2 border-black overflow-hidden">
                  <div className="h-full bg-brand transition-all duration-700" style={{ width: `${progressPercent}%` }} />
                </div>
              </div>

              {/* Live sentiment */}
              {simulationProgress && (
                <div className="grid grid-cols-3 divide-x-4 divide-black border-4 border-black shadow-neo-black">
                  {[
                    { key: 'satisfaction', label: 'Satisfaction', color: '#22C55E', value: simulationProgress.satisfaction },
                    { key: 'frustration', label: 'Frustration', color: '#EF4444', value: simulationProgress.frustration },
                    { key: 'trust', label: 'Trust', color: '#3B82F6', value: simulationProgress.trust },
                  ].map(m => (
                    <div key={m.key} className="p-4 text-center">
                      <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-1">{m.label}</div>
                      <div className="text-3xl font-black" style={{ color: m.color }}>{(m.value * 100).toFixed(0)}%</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Full simulation report */}
              {isComplete && simulationReport && (
                <div className="space-y-5">
                  <div className="font-black text-sm uppercase tracking-widest text-brand border-b-4 border-black pb-2">
                    Prediction Report — {simulationReport.feature_title}
                  </div>

                  {/* Executive Summary */}
                  {simulationReport.executive_summary && (
                    <div className="border-4 border-black shadow-neo-black p-5 bg-black text-white relative">
                      <div className="absolute -top-3 -left-3 w-6 h-6 bg-brand border-2 border-black flex items-center justify-center">
                        <FileCode2 className="w-3 h-3 text-black" strokeWidth={3} />
                      </div>
                      <div className="text-xs font-black uppercase tracking-widest text-brand mb-3 ml-2">Executive Summary</div>
                      <p className="text-sm font-bold text-white/90 whitespace-pre-wrap leading-relaxed">{simulationReport.executive_summary}</p>
                    </div>
                  )}
                  <div className="grid grid-cols-3 divide-x-4 divide-black border-4 border-black shadow-neo-black">
                    <div className="p-4 text-center">
                      <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-1">Net Promoter</div>
                      <div className={`text-3xl font-black ${simulationReport.nps >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {simulationReport.nps >= 0 ? '+' : ''}{simulationReport.nps}
                      </div>
                    </div>
                    <div className="p-4 text-center">
                      <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-1">Churn Vel.</div>
                      <div className={`text-2xl font-black flex items-center justify-center gap-1 ${simulationReport.churn_velocity > 0 ? 'text-red-600' : 'text-green-600'}`}>
                        {simulationReport.churn_velocity > 0 ? <TrendingUp className="w-5 h-5" strokeWidth={3} /> : <TrendingDown className="w-5 h-5" strokeWidth={3} />}
                        {(simulationReport.churn_velocity * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-4 text-center">
                      <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-1">Adoption</div>
                      <div className={`text-2xl font-black ${simulationReport.adoption_momentum >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {simulationReport.adoption_momentum >= 0 ? '+' : ''}{(simulationReport.adoption_momentum * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="border-4 border-black shadow-neo-black p-5">
                    <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-4">Risk Distribution</div>
                    {Object.entries(simulationReport.risk_distribution).map(([k, v]) => (
                      <RiskBar key={k} label={k.replace('_', ' ')} value={v}
                        color={k === 'HIGH_RISK' ? '#EF4444' : k === 'MODERATE' ? '#FF4500' : '#22C55E'} />
                    ))}
                  </div>

                  <div className="grid grid-cols-3 gap-0 divide-x-4 divide-black border-4 border-black shadow-neo-black">
                    <div className="p-4">
                      <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-3">Satisfaction Curve</div>
                      <Sparkline data={simulationReport.satisfaction_curve} color="#22C55E" />
                    </div>
                    <div className="p-4">
                      <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-3">Frustration Curve</div>
                      <Sparkline data={simulationReport.frustration_curve} color="#EF4444" />
                    </div>
                    <div className="p-4">
                      <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-3">Trust Curve</div>
                      <Sparkline data={simulationReport.trust_curve} color="#3B82F6" />
                    </div>
                  </div>

                  {simulationReport.top_risk_factors.length > 0 && (
                    <div className="border-4 border-black shadow-neo-black p-5">
                      <div className="text-xs font-black uppercase tracking-widest mb-4 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-brand" strokeWidth={3} /> Top Risk Factors
                      </div>
                      {simulationReport.top_risk_factors.slice(0, 5).map((f, i) => (
                        <div key={i} className="flex justify-between text-sm font-bold mb-3 pb-3 border-b-2 border-black/10 last:border-0">
                          <span className="uppercase tracking-wide">{f.factor.replace(/_/g, ' ')}</span>
                          <span className="font-black text-brand">{(f.frequency * 100).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* G2: Focus Group Results */}
                  {focusGroupResults && (
                    <CollapsibleSection title="Focus Group Results" icon={Users} badge={`${focusGroupResults.participants} Participants`} defaultOpen>
                      <div className="grid grid-cols-2 divide-x-4 divide-black border-4 border-black mb-4">
                        <div className="p-4 text-center">
                          <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-1">Avg WTP</div>
                          <div className="text-2xl font-black text-green-600">${focusGroupResults.aggregate.avg_wtp_usd.toFixed(0)}</div>
                        </div>
                        <div className="p-4 text-center">
                          <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-1">Adoption Intent</div>
                          <div className="text-2xl font-black text-brand">{focusGroupResults.aggregate.adoption_intent_pct.toFixed(1)}%</div>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 divide-x-4 divide-black border-4 border-black mb-4">
                        <div className="p-4 text-center">
                          <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-1">Churn Risk Δ</div>
                          <div className={`text-2xl font-black ${focusGroupResults.aggregate.churn_risk_delta > 0 ? 'text-red-600' : 'text-green-600'}`}>
                            {focusGroupResults.aggregate.churn_risk_delta > 0 ? '+' : ''}{(focusGroupResults.aggregate.churn_risk_delta * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="p-4">
                          <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-2">Top Objections</div>
                          <ul className="space-y-1">
                            {focusGroupResults.aggregate.top_objections.slice(0, 3).map((obj, i) => (
                              <li key={i} className="text-xs font-bold text-black/70 flex gap-2">
                                <span className="text-brand font-black">{i + 1}.</span> {obj}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </CollapsibleSection>
                  )}

                  {/* G3: Population Stats */}
                  {populationStats && (
                    <CollapsibleSection title="Population Statistics" icon={Activity} badge={populationStats.statistical_confidence} defaultOpen>
                      <div className="grid grid-cols-3 divide-x-4 divide-black border-4 border-black mb-4">
                        <div className="p-3 text-center">
                          <div className="text-xs font-black uppercase text-black/40 mb-1">Extrapolated NPS</div>
                          <div className={`text-xl font-black ${populationStats.extrapolated_nps >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {populationStats.extrapolated_nps >= 0 ? '+' : ''}{populationStats.extrapolated_nps}
                          </div>
                        </div>
                        <div className="p-3 text-center">
                          <div className="text-xs font-black uppercase text-black/40 mb-1">Churn Count</div>
                          <div className="text-xl font-black text-red-600">{populationStats.extrapolated_churn_count.toLocaleString()}</div>
                        </div>
                        <div className="p-3 text-center">
                          <div className="text-xs font-black uppercase text-black/40 mb-1">Champions</div>
                          <div className="text-xl font-black text-green-600">{populationStats.extrapolated_champion_count.toLocaleString()}</div>
                        </div>
                      </div>
                      <div className="space-y-2 text-xs font-bold">
                        <div className="flex justify-between border-b border-black/10 pb-1">
                          <span className="text-black/50 uppercase tracking-widest">High-Risk %</span>
                          <span className="text-red-600 font-black">{populationStats.extrapolated_high_risk_pct.toFixed(1)}% <span className="text-black/40">({populationStats.extrapolated_high_risk_ci})</span></span>
                        </div>
                        <div className="flex justify-between border-b border-black/10 pb-1">
                          <span className="text-black/50 uppercase tracking-widest">Declared Pop.</span>
                          <span className="font-black">{populationStats.declared_population.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between border-b border-black/10 pb-1">
                          <span className="text-black/50 uppercase tracking-widest">Active Cohort</span>
                          <span className="font-black">{populationStats.llm_active_cohort.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-black/50 uppercase tracking-widest">Margin of Error</span>
                          <span className="font-black text-brand">{populationStats.margin_of_error}</span>
                        </div>
                      </div>
                    </CollapsibleSection>
                  )}

                  {/* G1: Agent Journals */}
                  {agentJournals && agentJournals.journals.length > 0 && (
                    <CollapsibleSection title="Agent Behavioral Journals" icon={BookOpen} badge={`${agentJournals.count} Agents`}>
                      <div className="space-y-3 max-h-72 overflow-y-auto">
                        {agentJournals.journals.slice(0, 10).map((j, i) => (
                          <div key={i} className="border-2 border-black p-3 space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="font-black text-xs uppercase">{j.agent_name}</span>
                              <span className="text-xs text-black/40 font-black uppercase tracking-widest">{j.segment_source}</span>
                            </div>
                            <div className="grid grid-cols-5 gap-1">
                              {[
                                { label: 'Sat', value: j.satisfaction, color: '#22C55E' },
                                { label: 'Fru', value: j.frustration, color: '#EF4444' },
                                { label: 'Tru', value: j.trust, color: '#3B82F6' },
                                { label: 'Urg', value: j.urgency, color: '#FF4500' },
                                { label: 'Adv', value: j.advocacy, color: '#A855F7' },
                              ].map(m => (
                                <div key={m.label} className="text-center">
                                  <div className="text-xs font-black text-black/40 mb-1">{m.label}</div>
                                  <div className="text-xs font-black" style={{ color: m.color }}>{(m.value * 100).toFixed(0)}</div>
                                  <MiniBar value={m.value} color={m.color} />
                                </div>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </CollapsibleSection>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* ── Right: Final Recommendation + Handoff ─────── */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="bg-black text-white px-6 py-4 border-b-4 border-brand flex items-center gap-3 flex-none">
          <TerminalIcon className="w-4 h-4 text-brand" strokeWidth={3} />
          <h2 className="font-black text-sm uppercase tracking-widest">Layer 8: Handoff</h2>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {!isComplete ? (
            <div className="flex flex-col justify-center items-center h-full gap-6">
              <div className={`w-20 h-20 border-4 border-black flex items-center justify-center ${isRunning ? 'bg-brand' : 'bg-white'} shadow-neo-black`}>
                {isRunning ? (
                  <Loader2 className="w-10 h-10 text-black animate-spin" strokeWidth={3} />
                ) : (
                  <Zap className="w-10 h-10 text-black/20" strokeWidth={3} />
                )}
              </div>
              <div className="text-center">
                <h3 className="font-black text-2xl uppercase mb-2">
                  {isIdle ? 'Awaiting Simulation' : 'Compiling Artifacts'}
                </h3>
                <p className="text-sm font-bold text-black/50 max-w-xs mx-auto">
                  {isIdle
                    ? 'Run the OASIS simulation to generate prediction reports and deployment artifacts.'
                    : `Generating spec, monitoring plan, and Jira tickets. (${progressPercent}% done)`}
                </p>
              </div>
            </div>
          ) : (
            <>
              <div className="w-20 h-20 bg-green-500 border-4 border-black flex items-center justify-center shadow-neo-black mx-auto">
                <CheckCircle2 className="w-10 h-10 text-white" strokeWidth={3} />
              </div>

              {/* G10: Full final recommendation */}
              {finalRecommendation && (
                <div className="space-y-4">
                  <div className={`border-4 border-black shadow-neo-black p-5 ${finalRecommendation.final_verdict === 'APPROVE' ? 'bg-green-50' : finalRecommendation.final_verdict === 'REJECT' ? 'bg-red-50' : 'bg-yellow-50'}`}>
                    <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-2">Final Verdict</div>
                    <div className={`text-3xl font-black uppercase ${finalRecommendation.final_verdict === 'APPROVE' ? 'text-green-600' : finalRecommendation.final_verdict === 'REJECT' ? 'text-red-600' : 'text-yellow-600'}`}>
                      {finalRecommendation.final_verdict}
                    </div>
                    <div className="flex items-center gap-2 mt-2">
                      <div className="flex-1 h-2 bg-black/10 border border-black overflow-hidden">
                        <div className="h-full bg-green-500 transition-all duration-700"
                          style={{ width: `${finalRecommendation.overall_confidence * 100}%` }} />
                      </div>
                      <span className="font-black text-xs text-green-600">{(finalRecommendation.overall_confidence * 100).toFixed(0)}% confidence</span>
                    </div>
                    <p className="text-sm font-bold text-black/70 mt-3 leading-relaxed">{finalRecommendation.summary_for_leadership}</p>
                    <div className="text-xs text-black/30 font-black uppercase tracking-widest mt-2">
                      {finalRecommendation.total_time_minutes.toFixed(1)} min total pipeline time
                    </div>
                  </div>

                  {/* Top risks */}
                  {finalRecommendation.top_risks.length > 0 && (
                    <CollapsibleSection title="Top Risks" icon={AlertTriangle} badge={String(finalRecommendation.top_risks.length)} defaultOpen>
                      <div className="space-y-2">
                        {finalRecommendation.top_risks.map((r: any, i: number) => (
                          <div key={i} className="flex gap-3 border-b border-black/10 pb-2 last:border-0">
                            <span className="font-black text-xs text-red-500 w-5 flex-none">{i + 1}.</span>
                            <div>
                              <div className="font-black text-xs uppercase">{r.risk_factor ?? r.factor ?? 'Risk'}</div>
                              {r.description && <p className="text-xs text-black/60 mt-0.5">{r.description}</p>}
                            </div>
                          </div>
                        ))}
                      </div>
                    </CollapsibleSection>
                  )}

                  {/* Next steps */}
                  {finalRecommendation.next_steps.length > 0 && (
                    <CollapsibleSection title="Next Steps" icon={Target} defaultOpen>
                      <ol className="space-y-2">
                        {finalRecommendation.next_steps.map((s: any, i: number) => (
                          <li key={i} className="flex gap-3 text-xs font-bold">
                            <span className="font-black text-brand w-5 flex-none">{i + 1}.</span>
                            <span className="text-black/70">{s.action ?? s.step ?? String(s)}</span>
                          </li>
                        ))}
                      </ol>
                    </CollapsibleSection>
                  )}

                  {/* Stakeholder approvals */}
                  {finalRecommendation.stakeholder_approvals.length > 0 && (
                    <CollapsibleSection title="Stakeholder Approvals" icon={Shield}>
                      <div className="space-y-2">
                        {finalRecommendation.stakeholder_approvals.map((a: any, i: number) => (
                          <div key={i} className="flex justify-between items-center border-b border-black/10 pb-2 last:border-0">
                            <span className="font-black text-xs uppercase">{a.stakeholder ?? a.name ?? `Stakeholder ${i + 1}`}</span>
                            <span className={`text-xs font-black uppercase px-2 py-0.5 ${a.approved || a.verdict === 'APPROVE' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                              {a.verdict ?? (a.approved ? 'Approved' : 'Rejected')}
                            </span>
                          </div>
                        ))}
                      </div>
                    </CollapsibleSection>
                  )}
                </div>
              )}

              <p className="text-sm font-bold text-black/50 max-w-xs mx-auto text-center">
                NPS: {simulationReport?.nps ?? 'N/A'} &middot; {debateMessages.length} boardroom signals captured
              </p>
              <button className="w-full py-5 bg-black text-white border-4 border-black shadow-neo-brand font-black text-lg uppercase tracking-widest flex items-center justify-center gap-3 cursor-pointer transition-all duration-200 hover:translate-x-1 hover:translate-y-1 hover:shadow-none">
                <GitMerge className="w-5 h-5" strokeWidth={3} /> Deploy to Engineering
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
