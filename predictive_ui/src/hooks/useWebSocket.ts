import { useEffect, useRef } from 'react';
import { usePipelineStore } from '../store/usePipelineStore';
import { cleanPersonaName } from '../utils/nameHelper';

// Port 8080 = the active OASIS file-watch bridge (websocket.js)
// Port 8000 = FastAPI orchestrator (not consumed by this hook — see audit)
const defaultUrl = typeof window !== 'undefined'
  ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:8080`
  : 'ws://localhost:8080';

export function useWebSocket(url: string = defaultUrl) {
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const {
    addAction,
    addDebateMessage,
    setActiveSpeaker,
    setConnected,
    setIngestionNodes,
    setPersonas,
    setBoardroomPersonas,
    addSpawnedAgent,
    setSimulationStatus,
    setSimulationProgress,
    setSimulationReport,
    resetForNewSimulation,
    startSimulationStage,
    setPipelineStage,
    resetPipelineStages,
    setFinalRecommendation,
    setSqliteData,
    // G4
    setNetworkTopology,
    // G8
    setSimulationConfig,
    // G12
    setSeedPosts,
    // G2
    setFocusGroupResults,
    // G3
    setPopulationStats,
    // G1
    setAgentJournals,
    // G5
    setConsensusResult,
    // G9
    addSycophancyAlert,
    // G7
    addEagleEyeResult,
  } = usePipelineStore();

  useEffect(() => {
    let isMounted = true;

    function connect() {
      if (!isMounted) return;
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        console.log('[WS] Connected to OASIS Real-Time Bridge');
        setConnected(true);
        if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // ── Standard OASIS agent action (includes INTERVIEW_RESPONSE, FOCUS_GROUP) ──
          if (data.agent_id && data.action_type) {
            // G7: Eagle's Eye interview response — route to dedicated store field too
            if (data.action_type === 'INTERVIEW_RESPONSE') {
              addEagleEyeResult({
                agent_id: data.agent_id,
                agent_name: data.agent_name,
                content: data.content,
                timestep: data.timestep ?? 0,
              });
            }
            addAction(data);
            return;
          }

          // ── Route by type ─────────────────────────────────────────────────
          switch (data.type) {

            // Layer 1: Ingestion
            case 'ingestion_sync':
              setIngestionNodes(data.nodes ?? []);
              setPipelineStage('layer1', 'done');
              break;

             // Layer 3: Personas
            case 'persona_sync':
              setPersonas(data.personas ?? []);
              setPipelineStage('layer3', 'done');
              break;

            // Layer 4: Boardroom Personas
            case 'boardroom_persona_sync': {
              const cleanedPersonas = (data.personas ?? []).map((p: any) => {
                const rawName = p.name || '';
                const cleanName = cleanPersonaName(rawName);
                return {
                  ...p,
                  name: cleanName
                };
              });
              setBoardroomPersonas(cleanedPersonas);
              break;
            }

            // Pipeline stage progress events (running/done per layer)
            case 'pipeline_progress': {
              const layerKey = `layer${data.layer}` as 'layer1' | 'layer3' | 'layer5';
              if (layerKey === 'layer1' || layerKey === 'layer3' || layerKey === 'layer5') {
                setPipelineStage(layerKey, data.status === 'running' ? 'running' : 'done');
              }
              if (data.layer === 5 && data.status === 'running') {
                setPipelineStage('layer5', 'running');
              }
              break;
            }

            // Pipeline reset — new simulation is starting, clear all layers to waiting and wipe old graph data
            case 'pipeline_reset':
              resetPipelineStages();
              resetForNewSimulation('Initializing New Simulation...');
              break;

            // Layer 5: Agent spawn into graph
            case 'agent_spawn':
              addSpawnedAgent({
                agent_id: data.agent_id,
                agent_name: data.agent_name,
                agent_type: data.agent_type,
                role: data.role,
                traits: data.traits ?? [],
                impact: data.impact ?? 50,
                // Core profile
                mbti: data.mbti ?? '',
                mbti_description: data.mbti_description ?? '',
                ocean_scores: data.ocean_scores ?? {},
                buyer_journey: data.buyer_journey ?? '',
                buyer_journey_detail: data.buyer_journey_detail ?? undefined,
                bio: data.bio ?? '',
                // Structured psychological fields
                emotional_triggers: data.emotional_triggers ?? undefined,
                communication_style: data.communication_style ?? undefined,
                decision_pattern: data.decision_pattern ?? undefined,
                predicted_stance: data.predicted_stance ?? undefined,
                questions_they_will_ask: data.questions_they_will_ask ?? [],
                // Persona metadata
                domain_expertise: data.domain_expertise ?? [],
                profile_confidence: data.profile_confidence ?? 0,
                grounding_quality: data.grounding_quality ?? 1,
                persona_type: data.persona_type ?? 'INTERNAL',
                network_position_hint: data.network_position_hint ?? 'peripheral',
                influence_strength: data.influence_strength ?? 0.5,
                receptiveness: data.receptiveness ?? 0.5,
                // External persona context
                market_context: data.market_context ?? undefined,
                evidence_sources: data.evidence_sources ?? [],
              });
              break;

            // Layer 5: Simulation lifecycle
            case 'simulation_start':
              startSimulationStage(data.feature_title ?? 'Simulation');
              break;
            case 'progress':
              setSimulationProgress({
                timestep: data.timestep,
                total: data.total,
                percent: data.percent,
                satisfaction: data.satisfaction,
                frustration: data.frustration,
                trust: data.trust,
              });
              break;

            // G8: Simulation configuration + scale metadata (emitted right after sampler init)
            case 'simulation_config':
              setSimulationConfig({
                simulation_id: data.simulation_id,
                hindsight_available: data.hindsight_available ?? false,
                llm_model: data.llm_model ?? '',
                platform_type: data.platform_type ?? '',
                num_timesteps: data.num_timesteps ?? 0,
                declared_population: data.declared_population ?? 0,
                llm_active_cohort: data.llm_active_cohort ?? 0,
                shadow_agents: data.shadow_agents ?? 0,
                interview_phase_enabled: data.interview_phase_enabled ?? false,
              });
              break;

            // G4: Social network topology (emitted after network build)
            case 'network_topology':
              setNetworkTopology({
                simulation_id: data.simulation_id,
                hub_agent_id: data.hub_agent_id,
                total_edges: data.total_edges ?? 0,
                avg_degree: data.avg_degree ?? 0,
                edges: data.edges ?? [],
              });
              break;

            // G12: Seed posts that started the debate
            case 'seed_posts':
              setSeedPosts(data.seeds ?? []);
              break;

            // Layer 8: Handoff / prediction report (all fields now forwarded by websocket.js)
            case 'simulation_report':
              setSimulationStatus('complete');
              setSimulationReport({
                feature_title: data.feature_title,
                nps: data.nps,
                churn_velocity: data.churn_velocity,
                adoption_momentum: data.adoption_momentum,
                population_size: data.population_size ?? 0,
                risk_distribution: data.risk_distribution ?? {},
                top_risk_factors: data.top_risk_factors ?? [],
                satisfaction_curve: data.satisfaction_curve ?? [],
                frustration_curve: data.frustration_curve ?? [],
                trust_curve: data.trust_curve ?? [],
                segments: data.segments ?? [],
                decision_events: data.decision_events ?? [],
                focus_group_insights: data.focus_group_insights ?? {},
                executive_summary: data.executive_summary ?? '',
              });
              break;

            // G10: Final pipeline recommendation (full — verdict + risks + next steps + approvals)
            case 'final_recommendation':
              setFinalRecommendation({
                feature_name: data.feature_name,
                final_verdict: data.final_verdict,
                overall_confidence: data.overall_confidence,
                summary_for_leadership: data.summary_for_leadership,
                top_risks: data.top_risks ?? [],
                next_steps: data.next_steps ?? [],
                stakeholder_approvals: data.stakeholder_approvals ?? [],
                total_time_minutes: data.total_time_minutes ?? 0,
              });
              break;

            // G5: Boardroom consensus result (full vote breakdown)
            case 'consensus_result':
              setConsensusResult({
                feature_name: data.feature_name,
                overall_verdict: data.overall_verdict,
                approval_confidence: data.approval_confidence ?? 0,
                stakeholder_verdicts: data.stakeholder_verdicts ?? {},
                approvals: data.approvals ?? [],
                debate_rounds_count: data.debate_rounds_count ?? 0,
                phase_1: data.phase_1 ?? {},
                phase_2_gate: data.phase_2_gate ?? null,
                mitigations: data.mitigations ?? [],
                next_steps: data.next_steps ?? [],
                simulation_key_quotes: data.simulation_key_quotes ?? [],
                behavioral_insights: data.behavioral_insights ?? [],
                tension_shifts: data.tension_shifts ?? {},
              });
              break;

            // G2: Focus group results (WTP, adoption intent, churn risk, objections)
            case 'focus_group_results':
              setFocusGroupResults({
                simulation_id: data.simulation_id,
                participants: data.participants ?? 0,
                metrics: data.metrics ?? [],
                aggregate: data.aggregate ?? {
                  avg_wtp_usd: 0,
                  adoption_intent_pct: 0,
                  churn_risk_delta: 0,
                  top_objections: [],
                },
              });
              break;

            // G3: Population-scale statistics with confidence intervals
            case 'population_stats':
              setPopulationStats({
                simulation_id: data.simulation_id,
                declared_population: data.declared_population ?? 0,
                llm_active_cohort: data.llm_active_cohort ?? 0,
                shadow_agents: data.shadow_agents ?? 0,
                extrapolated_high_risk_pct: data.extrapolated_high_risk_pct ?? 0,
                extrapolated_high_risk_ci: data.extrapolated_high_risk_ci ?? '',
                extrapolated_nps: data.extrapolated_nps ?? 0,
                extrapolated_churn_count: data.extrapolated_churn_count ?? 0,
                extrapolated_champion_count: data.extrapolated_champion_count ?? 0,
                statistical_confidence: data.statistical_confidence ?? '',
                margin_of_error: data.margin_of_error ?? '',
              });
              break;

            // G1: Per-agent behavioral decision journals (richest signal)
            case 'agent_journals':
              setAgentJournals({
                simulation_id: data.simulation_id,
                count: data.count ?? 0,
                journals: data.journals ?? [],
              });
              break;

            // G9: Sycophancy alert — accumulate all instances, show data-validity warning
            case 'sycophancy_alert':
              addSycophancyAlert({
                agent_id: data.agent_id,
                agent_name: data.agent_name,
                timestep: data.timestep,
                pattern: data.pattern,
                frustration_at_collapse: data.frustration_at_collapse,
                trust_at_collapse: data.trust_at_collapse,
                data_validity_warning: data.data_validity_warning ?? true,
                triggering_content: data.triggering_content ?? '',
                signal_history: data.signal_history ?? [],
              });
              console.warn(
                `[WS] ⚠️ Sycophancy collapse — ${data.agent_name} ` +
                `(fru=${data.frustration_at_collapse}, trust=${data.trust_at_collapse}): ` +
                `"${(data.triggering_content ?? '').slice(0, 120)}..."`
              );
              break;

            // G3: SQLite rich social data (users, posts, comments from simulation DB)
            case 'sqlite_data':
              setSqliteData({
                users: data.users ?? [],
                posts: data.posts ?? [],
                comments: data.comments ?? [],
              });
              if (data.posts && data.posts.length > 0) {
                const mappedSeeds = data.posts.map((p: any, i: number) => ({
                  index: p.post_id || i + 1,
                  content: p.content || '',
                }));
                // Only overwrite if we haven't already received seed_posts explicitly
                const currentSeeds = usePipelineStore.getState().seedPosts;
                if (!currentSeeds || currentSeeds.length === 0) {
                  setSeedPosts(mappedSeeds);
                }
              }
              break;

            // Layer 6: Boardroom
            case 'debate_message': {
              const rawSender = data.message?.sender;
              const cleanSender = rawSender ? cleanPersonaName(rawSender) : '';
              const cleanMessage = {
                ...data.message,
                sender: cleanSender || 'Agent'
              };
              addDebateMessage(cleanMessage);
              setActiveSpeaker(cleanSender || null);
              break;
            }

            default:
              // Ignore unknown message types silently
              break;
          }
        } catch (e) {
          console.error('[WS] Failed to parse message:', e);
        }
      };

      ws.current.onclose = () => {
        console.log('[WS] Disconnected — will attempt reconnect in 3s');
        setConnected(false);
        if (isMounted) {
          reconnectTimer.current = setTimeout(connect, 3000);
        }
      };

      ws.current.onerror = (err) => {
        console.warn('[WS] Connection error:', err);
      };
    }

    connect();

    return () => {
      isMounted = false;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (ws.current) ws.current.close();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]); // Only reconnect if URL changes — all store functions are stable refs from Zustand

  return ws.current;
}
