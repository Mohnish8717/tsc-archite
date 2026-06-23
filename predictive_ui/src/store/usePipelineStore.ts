import { create } from 'zustand';

// --- Type Definitions based on backend OASIS output ---
export interface AgentAction {
  timestamp: string;
  agent_id: string;
  agent_name: string;
  timestep: number;
  action_type: 'spawn' | 'comment' | 'upvote' | 'downvote' | 'follow' | string;
  content: string;
  platform: string;
  metadata?: {
    target_id?: string | null;
    confidence?: number;
    signal_type?: string;
    impact?: number;
    // G6: expanded signal forensics
    all_signals?: string[];
    signal_factors?: string[];
    signal_quote?: string;
    raw_intensity?: number;
  };
}

export interface BoardroomMessage {
  id: string;
  sender: string;
  text: string;
  type: 'normal' | 'challenge' | 'system';
}

export interface IngestionNode {
  id: string;
  label: string;
  type: 'input' | 'process' | 'output';
  status: 'active' | 'pending' | 'error';
}

export interface KGNode {
  id: string;
  label: string;
  entityType: string;
  mentions: number;
}

export interface KGEdge {
  source: string;
  target: string;
  relationshipType: string;
  weight: number;
}

// ── Psychological sub-model types (mirror of Python models) ──────────────────
export interface PersonaEmotionalTriggers {
  excited_by: string[];
  frustrated_by: string[];
  scared_of: string[];
}
export interface PersonaCommunicationStyle {
  default: string;
  formality: string;
  conflict_handling: string;
  preferred_channels: string[];
}
export interface PersonaDecisionPattern {
  speed: string;
  preference: string;
  influencers: string[];
  justification: string;
  risk_tolerance: string;
}
export interface PersonaPredictedStance {
  feature: string;
  prediction: string;          // 'APPROVE' | 'REJECT' | 'CONDITIONAL_APPROVE'
  confidence: number;          // 0-1
  likely_conditions: string[];
  potential_objections: string[];
}
export interface PersonaBuyerJourneyDetail {
  awareness_channel: string;
  evaluation_trigger: string;
  key_proof_points: string[];
  deal_breakers: string[];
  success_metric: string;
  roi_threshold_months?: number;
  willingness_to_pay_band?: string;
}
export interface PersonaMarketContext {
  company_size_band: string;
  buyer_role: string;
  annual_solution_budget_usd: number;
  pricing_sensitivity: string;
  sales_cycle_weeks: number;
  deployment_preference: string;
  industry_vertical: string;
  regulatory_burden: string;
}

export interface SyntheticPersona {
  id: string;
  name: string;
  role: string;
  traits: string[];
  impact: number;
  bio?: string;
  role_short?: string;
  // ── Rich psychological profile ────────────────────────────────────────────
  mbti?: string;
  mbti_description?: string;
  key_traits?: string[];
  emotional_triggers?: PersonaEmotionalTriggers;
  communication_style?: PersonaCommunicationStyle;
  decision_pattern?: PersonaDecisionPattern;
  predicted_stance?: PersonaPredictedStance;
  questions_they_will_ask?: string[];
  // ── FinalPersona metadata ─────────────────────────────────────────────────
  domain_expertise?: string[];
  profile_confidence?: number;
  grounding_quality?: number;
  persona_type?: string;
  network_position_hint?: string;
  influence_strength?: number;
  receptiveness?: number;
  evidence_sources?: string[];
  // ── Buyer Journey & Market (external personas) ────────────────────────────
  buyer_journey?: string;               // stage string for indicator
  buyer_journey_detail?: PersonaBuyerJourneyDetail;
  market_context?: PersonaMarketContext;
}

export interface SpawnedAgent {
  agent_id: string;
  agent_name: string;
  agent_type: string;
  role: string;
  traits: string[];
  impact: number;
  // ── Rich psychological profile ────────────────────────────────────────────
  mbti?: string;
  mbti_description?: string;
  ocean_scores?: Record<string, number>;
  buyer_journey?: string;               // stage string for indicator
  buyer_journey_detail?: PersonaBuyerJourneyDetail;
  bio?: string;
  emotional_triggers?: PersonaEmotionalTriggers;
  communication_style?: PersonaCommunicationStyle;
  decision_pattern?: PersonaDecisionPattern;
  predicted_stance?: PersonaPredictedStance;
  questions_they_will_ask?: string[];
  domain_expertise?: string[];
  profile_confidence?: number;
  grounding_quality?: number;
  persona_type?: string;
  network_position_hint?: string;
  influence_strength?: number;
  receptiveness?: number;
  market_context?: PersonaMarketContext;
  evidence_sources?: string[];
}

export interface SimulationProgress {
  timestep: number;
  total: number;
  percent: number;
  satisfaction: number;
  frustration: number;
  trust: number;
}

export interface SimulationReport {
  feature_title: string;
  nps: number;
  churn_velocity: number;
  adoption_momentum: number;
  population_size: number;
  risk_distribution: Record<string, number>;
  top_risk_factors: Array<{ factor: string; frequency: number }>;
  satisfaction_curve: number[];
  frustration_curve: number[];
  trust_curve: number[];
  segments: Array<Record<string, unknown>>;
  decision_events: Array<Record<string, unknown>>;
  focus_group_insights: Record<string, unknown>;
  executive_summary: string;
}

// G3: rich social platform data from SQLite DB
export interface SqliteData {
  users: Array<Record<string, unknown>>;
  comments: Array<Record<string, unknown>>;
  posts: Array<Record<string, unknown>>;
}

// G4: Social network topology edges
export interface NetworkTopology {
  simulation_id: string;
  hub_agent_id: string;
  total_edges: number;
  avg_degree?: number;
  density?: number;
  clustering_coefficient?: number;
  avg_betweenness_centrality?: number;
  edges: Array<{ from: string; to: string }>;
}

// G8: Simulation configuration + scale metadata
export interface SimulationConfig {
  simulation_id: string;
  hindsight_available: boolean;
  llm_model: string;
  platform_type: string;
  num_timesteps: number;
  declared_population: number;
  llm_active_cohort: number;
  shadow_agents: number;
  interview_phase_enabled: boolean;
  feature_title?: string;
  feature_description?: string;
}

// G12: Seed posts that bootstrapped the debate
export interface SeedPost {
  index: number;
  content: string;
}

// G2: Focus group per-agent 14-metric extraction
export interface FocusGroupResult {
  simulation_id: string;
  participants: number;
  metrics: Array<Record<string, unknown>>;
  aggregate: {
    avg_wtp_usd: number;
    adoption_intent_pct: number;
    churn_risk_delta: number;
    top_objections: string[];
  };
}

// G3: Population-scale statistical confidence
export interface PopulationStats {
  simulation_id: string;
  declared_population: number;
  llm_active_cohort: number;
  shadow_agents: number;
  extrapolated_high_risk_pct: number;
  extrapolated_high_risk_ci: string;
  extrapolated_nps: number;
  extrapolated_churn_count: number;
  extrapolated_champion_count: number;
  statistical_confidence: string;
  margin_of_error: string;
}

// G1: Per-agent decision journal (final behavioral state)
export interface AgentJournalEntry {
  agent_id: string;
  agent_name: string;
  segment_source: string;
  satisfaction: number;
  frustration: number;
  trust: number;
  urgency: number;
  advocacy: number;
  decisions: Array<Record<string, unknown>>;
  signals: Array<Record<string, unknown>>;
}
export interface AgentJournals {
  simulation_id: string;
  count: number;
  journals: AgentJournalEntry[];
}

// G5: Boardroom consensus result
export interface ConsensusResult {
  feature_name: string;
  overall_verdict: string;
  approval_confidence: number;
  stakeholder_verdicts: Record<string, string>;
  approvals: Array<Record<string, unknown>>;
  debate_rounds_count: number;
  phase_1: Record<string, unknown>;
  phase_2_gate: Record<string, unknown> | null;
  mitigations: string[];
  next_steps: string[];
  simulation_key_quotes: string[];
  behavioral_insights: string[];
  tension_shifts: Record<string, number>;
}

// G9: Sycophancy alert with triggering content
export interface SycophancyAlert {
  agent_id: string;
  agent_name: string;
  timestep: number;
  pattern: string;
  frustration_at_collapse: number;
  trust_at_collapse: number;
  data_validity_warning: boolean;
  triggering_content: string;
  signal_history: string[];
}

// G7: Eagle's Eye mid-simulation interview result
export interface EagleEyeResult {
  agent_id: string;
  agent_name: string;
  content: string;
  timestep: number;
}

export type PipelineStageStatus = 'waiting' | 'running' | 'done';

interface PipelineState {
  // Layer 1: Ingestion
  ingestionNodes: IngestionNode[];
  setIngestionNodes: (nodes: IngestionNode[]) => void;
  updateIngestionNode: (nodeUpdate: Partial<IngestionNode> & { id: string }) => void;

  // Live Knowledge Graph
  kgNodes: KGNode[];
  kgEdges: KGEdge[];
  setKnowledgeGraph: (nodes: KGNode[], edges: KGEdge[]) => void;

  // Layer 3: Personas
  personas: SyntheticPersona[];
  setPersonas: (personas: SyntheticPersona[]) => void;

  // Layer 4: Boardroom Personas
  boardroomPersonas: SyntheticPersona[];
  setBoardroomPersonas: (personas: SyntheticPersona[]) => void;

  // Pipeline Stage Status (drives "waiting" overlays in Layer 1, 3, 5 components)
  pipelineStages: { layer1: PipelineStageStatus; layer3: PipelineStageStatus; layer5: PipelineStageStatus };
  setPipelineStage: (layer: 'layer1' | 'layer3' | 'layer5', status: PipelineStageStatus) => void;
  resetPipelineStages: () => void;

  // Interactive Backend Calls
  pendingAction: { action: string; payload: any } | null;
  setPendingAction: (actionData: { action: string; payload: any } | null) => void;

  // Layer 5: OASIS Simulation State
  actions: AgentAction[];
  activeAgents: number;
  hotScoreAvg: number;
  tensionStatus: 'Normal' | 'Elevated' | 'Critical';
  addAction: (action: AgentAction) => void;

  // Agent Registry (from spawn events) — Record, NOT Map, for Zustand reactivity
  spawnedAgents: Record<string, SpawnedAgent>;
  addSpawnedAgent: (agent: SpawnedAgent) => void;

  // Simulation Lifecycle
  simulationStatus: 'idle' | 'running' | 'complete';
  simulationTitle: string;
  simulationProgress: SimulationProgress | null;
  simulationReport: SimulationReport | null;
  setSimulationStatus: (status: 'idle' | 'running' | 'complete') => void;
  setSimulationTitle: (title: string) => void;
  setSimulationProgress: (progress: SimulationProgress) => void;
  setSimulationReport: (report: SimulationReport) => void;
  resetForNewSimulation: (title: string) => void;
  startSimulationStage: (title: string) => void;
  // Stop mid-run — resets lifecycle fields, preserves layer data
  stopSimulation: () => void;

  // Layer 6: Boardroom Debate State
  debateMessages: BoardroomMessage[];
  activeSpeaker: string | null;
  addDebateMessage: (msg: BoardroomMessage) => void;
  setActiveSpeaker: (speaker: string | null) => void;

  // Layer 8 / Pipeline final verdict
  finalRecommendation: {
    feature_name: string;
    final_verdict: string;
    overall_confidence: number;
    summary_for_leadership: string;
    // G10: rich fields
    top_risks: Array<Record<string, unknown>>;
    next_steps: Array<Record<string, unknown>>;
    stakeholder_approvals: Array<Record<string, unknown>>;
    total_time_minutes: number;
  } | null;
  setFinalRecommendation: (rec: PipelineState['finalRecommendation']) => void;

  // G3: rich social platform data from SQLite (agents, posts, comments)
  sqliteData: SqliteData | null;
  setSqliteData: (data: SqliteData) => void;

  // G4: Social network topology
  networkTopology: NetworkTopology | null;
  setNetworkTopology: (topo: NetworkTopology) => void;

  // G8: Simulation configuration + scale metadata
  simulationConfig: SimulationConfig | null;
  setSimulationConfig: (cfg: SimulationConfig) => void;

  // G12: Seed posts
  seedPosts: SeedPost[];
  setSeedPosts: (posts: SeedPost[]) => void;

  // G2: Focus group results
  focusGroupResults: FocusGroupResult | null;
  setFocusGroupResults: (results: FocusGroupResult) => void;

  // G3: Population statistics with confidence intervals
  populationStats: PopulationStats | null;
  setPopulationStats: (stats: PopulationStats) => void;

  // G1: Per-agent decision journals
  agentJournals: AgentJournals | null;
  setAgentJournals: (journals: AgentJournals) => void;

  // G5: Boardroom consensus result
  consensusResult: ConsensusResult | null;
  setConsensusResult: (result: ConsensusResult) => void;

  // G9: Sycophancy alerts list
  sycophancyAlerts: SycophancyAlert[];
  addSycophancyAlert: (alert: SycophancyAlert) => void;

  // G7: Eagle's Eye interview results
  eagleEyeResults: EagleEyeResult[];
  addEagleEyeResult: (result: EagleEyeResult) => void;

  // Connection State
  isConnected: boolean;
  setConnected: (status: boolean) => void;
  isBootstrapped: boolean;
  setIsBootstrapped: (status: boolean) => void;
  
  // Session tracking
  sessionId: string | null;
  setSessionId: (id: string | null) => void;

  // System Logs
  systemLogs: string[];
  addSystemLog: (log: string) => void;

  // Upvoting state
  upvotedItems: Record<string, number>;
  upvoteItem: (id: string) => void;
}

export const usePipelineStore = create<PipelineState>((set) => ({
  // Ingestion
  ingestionNodes: [],
  setIngestionNodes: (nodes) => set({ ingestionNodes: nodes }),
  updateIngestionNode: (nodeUpdate) => set((state) => ({
    ingestionNodes: state.ingestionNodes.map((n) =>
      n.id === nodeUpdate.id ? { ...n, ...nodeUpdate } : n
    ),
  })),

  // Live Knowledge Graph
  kgNodes: [],
  kgEdges: [],
  setKnowledgeGraph: (nodes, edges) => set({ kgNodes: nodes, kgEdges: edges }),

  // Personas
  personas: [],
  setPersonas: (personas) => set({ personas }),

  // Boardroom Personas
  boardroomPersonas: [],
  setBoardroomPersonas: (personas) => set({ boardroomPersonas: personas }),

  // Pipeline stages — start all as 'waiting' until backend emits progress
  pipelineStages: { layer1: 'waiting', layer3: 'waiting', layer5: 'waiting' },
  setPipelineStage: (layer, status) => set((state) => ({
    pipelineStages: { ...state.pipelineStages, [layer]: status },
  })),
  resetPipelineStages: () => set({
    pipelineStages: { layer1: 'waiting', layer3: 'waiting', layer5: 'waiting' },
    pendingAction: null,
  }),

  pendingAction: null,
  setPendingAction: (actionData) => set({ pendingAction: actionData }),

  // OASIS actions
  actions: [],
  activeAgents: 0,
  hotScoreAvg: 5.0,
  tensionStatus: 'Normal',
  addAction: (action) => set((state) => {
    // Gracefully map raw backend action types and signal categories to frontend expected keys ('upvote' | 'downvote' | 'comment')
    const typeUpper = (action.action_type || '').toUpperCase();
    const sigType = (action.metadata?.signal_type || '').toLowerCase();
    
    let mappedType = action.action_type;
    const isLike = typeUpper.includes('LIKE') || typeUpper === 'UPVOTE';
    const isDislike = typeUpper.includes('DISLIKE') || typeUpper === 'DOWNVOTE';
    const isPost = typeUpper.includes('POST') || typeUpper.includes('SPAWN');
    const isComment = typeUpper.includes('COMMENT');

    if (isLike || isDislike) {
      // Determine the sentiment of the post they are reacting to
      let targetSentiment = 'positive'; // default to trusting the product
      
      if (action.metadata?.target_id) {
        // Find the target agent's last meaningful action to infer sentiment
        const targetActions = state.actions.filter(a => String(a.agent_id) === String(action.metadata!.target_id));
        if (targetActions.length > 0) {
          const lastAction = targetActions[targetActions.length - 1];
          if (lastAction.action_type === 'downvote') targetSentiment = 'negative';
          if (lastAction.action_type === 'upvote') targetSentiment = 'positive';
        }
      }

      // If they LIKE a negative post, they are agreeing with the friction (downvote).
      // If they DISLIKE a negative post, they are disagreeing with friction (upvote).
      if (isLike) {
        mappedType = targetSentiment === 'negative' ? 'downvote' : 'upvote';
      } else {
        mappedType = targetSentiment === 'negative' ? 'upvote' : 'downvote';
      }
    } else {
      // Standard mapping for comments/posts
      if (sigType === 'positive') {
        mappedType = 'upvote';
      } else if (sigType === 'negative') {
        mappedType = 'downvote';
      } else if (isPost) {
        mappedType = 'post';
      } else if (isComment) {
        mappedType = 'comment';
      }
    }

    const mappedAction = {
      ...action,
      action_type: mappedType
    };

    const newActions = [...state.actions.slice(-500), mappedAction];
    const newActiveAgents = new Set(newActions.map(a => a.agent_id)).size;
    const recentUpvotes = newActions.filter(a => a.action_type === 'upvote').length;
    const recentDownvotes = newActions.filter(a => a.action_type === 'downvote').length;

    // Dynamically calculate tension/heat score based on volume and trust/friction ratio
    const totalSentiment = recentUpvotes + recentDownvotes;
    const frictionRatio = totalSentiment > 0 ? recentDownvotes / totalSentiment : 0.0;
    const hotScore = Math.min(10, Math.max(0, 5 + (recentDownvotes - recentUpvotes) * 0.15 + frictionRatio * 3));
    const tension: 'Normal' | 'Elevated' | 'Critical' = hotScore > 8 ? 'Critical' : (hotScore > 6 ? 'Elevated' : 'Normal');

    return { actions: newActions, activeAgents: newActiveAgents, hotScoreAvg: Number(hotScore.toFixed(1)), tensionStatus: tension };
  }),


  // Agent registry — plain Record for Zustand reactivity (Map doesn't trigger re-renders)
  spawnedAgents: {},
  addSpawnedAgent: (agent) => set((state) => ({
    spawnedAgents: { ...state.spawnedAgents, [agent.agent_id]: agent }
  })),

  // Simulation lifecycle
  simulationStatus: 'idle',
  simulationTitle: '',
  simulationProgress: null,
  simulationReport: null,
  setSimulationStatus: (status) => set({ simulationStatus: status }),
  setSimulationTitle: (title) => set({ simulationTitle: title }),
  setSimulationProgress: (progress) => set({ simulationProgress: progress }),
  setSimulationReport: (report) => set({ simulationReport: report }),
  // Reset everything when a brand-new simulation run starts
  resetForNewSimulation: (title) => set({
    actions: [],
    activeAgents: 0,
    hotScoreAvg: 5.0,
    tensionStatus: 'Normal',
    spawnedAgents: {},
    simulationStatus: 'running',
    simulationTitle: title,
    simulationProgress: null,
    simulationReport: null,
    debateMessages: [],
    activeSpeaker: null,
    ingestionNodes: [],
    personas: [],
    boardroomPersonas: [],
    pipelineStages: { layer1: 'waiting', layer3: 'waiting', layer5: 'waiting' },
    networkTopology: null,
    simulationConfig: null,
    seedPosts: [],
    focusGroupResults: null,
    populationStats: null,
    agentJournals: null,
    consensusResult: null,
    sycophancyAlerts: [],
    eagleEyeResults: [],
  }),
  startSimulationStage: (title) => set((state) => ({
    actions: [],
    activeAgents: 0,
    hotScoreAvg: 5.0,
    tensionStatus: 'Normal',
    simulationStatus: 'running',
    simulationTitle: title,
    simulationProgress: null,
    pipelineStages: { ...state.pipelineStages, layer5: 'running' },
  })),
  // Stop mid-run: reset lifecycle only — layer data (agents, chunks, debate) stays visible
  stopSimulation: () => set({
    simulationStatus: 'idle',
    simulationProgress: null,
    simulationTitle: '',
  }),

  // Boardroom
  debateMessages: [],
  activeSpeaker: null,
  addDebateMessage: (msg) => set((state) => ({
    debateMessages: [...state.debateMessages.slice(-100), msg]
  })),
  setActiveSpeaker: (speaker) => set({ activeSpeaker: speaker }),

  // Layer 8 final recommendation
  finalRecommendation: null,
  setFinalRecommendation: (rec) => set({ finalRecommendation: rec }),

  // G3: SQLite rich data
  sqliteData: null,
  setSqliteData: (data) => set({ sqliteData: data }),

  // G4: Network topology
  networkTopology: null,
  setNetworkTopology: (topo) => set({ networkTopology: topo }),

  // G8: Simulation config
  simulationConfig: null,
  setSimulationConfig: (cfg) => set({ simulationConfig: cfg }),

  // G12: Seed posts
  seedPosts: [],
  setSeedPosts: (posts) => set({ seedPosts: posts }),

  // G2: Focus group results
  focusGroupResults: null,
  setFocusGroupResults: (results) => set({ focusGroupResults: results }),

  // G3: Population stats
  populationStats: null,
  setPopulationStats: (stats) => set({ populationStats: stats }),

  // G1: Agent journals
  agentJournals: null,
  setAgentJournals: (journals) => set({ agentJournals: journals }),

  // G5: Consensus result
  consensusResult: null,
  setConsensusResult: (result) => set({ consensusResult: result }),

  // G9: Sycophancy alerts (accumulate all, don't replace)
  sycophancyAlerts: [],
  addSycophancyAlert: (alert) => set((state) => ({
    sycophancyAlerts: [...state.sycophancyAlerts, alert]
  })),

  // G7: Eagle's Eye results
  eagleEyeResults: [],
  addEagleEyeResult: (result) => set((state) => ({
    eagleEyeResults: [...state.eagleEyeResults, result]
  })),

  // Connection
  isConnected: false,
  setConnected: (status) => set({ isConnected: status }),
  isBootstrapped: false,
  setIsBootstrapped: (status) => set({ isBootstrapped: status }),

  sessionId: null,
  setSessionId: (id) => set({ sessionId: id }),

  // System Logs
  systemLogs: [],
  addSystemLog: (log) => set((state) => ({
    systemLogs: [...state.systemLogs.slice(-999), log]
  })),

  // Upvoting state
  upvotedItems: {},
  upvoteItem: (id) => set((state) => {
    const currentCount = state.upvotedItems[id] || 0;
    return {
      upvotedItems: {
        ...state.upvotedItems,
        [id]: currentCount + 1
      }
    };
  })
}));
