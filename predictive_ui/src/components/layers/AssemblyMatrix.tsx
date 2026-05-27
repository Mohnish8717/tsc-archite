import { useMemo, useState } from 'react';
import { Users, Shield, Lock, Briefcase, ChevronRight, Loader2, Zap, Cpu, Brain, Database, Globe, AlertTriangle, X, Activity, Fingerprint, ListTodo, CheckCircle2, XCircle, HelpCircle, TrendingUp, MessageSquare, Target, Building2, DollarSign, Clock, Wifi, BarChart2, ChevronDown, ChevronUp } from 'lucide-react';
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

const MBTI_DESCRIPTIONS: Record<string, { title: string; style: string; desc: string }> = {
  INTJ: { title: 'Architect', style: 'Analytical & Strategic', desc: 'Driven by logic and system efficiency. Skeptical of unsupported claims, values long-term roadmap integration.' },
  INTP: { title: 'Logician', style: 'Objective & Inventive', desc: 'Enjoys building complex technical solutions. Values elegant APIs, hates redundant processes and UI fluff.' },
  ENTJ: { title: 'Commander', style: 'Decisive & Efficient', desc: 'Focused on ROI, team velocity, and organizational scalability. Demands clear commitments and timelines.' },
  ENTP: { title: 'Debater', style: 'Innovative & Out-of-the-box', desc: 'Curious explorer of raw capabilities. Suggests creative workarounds, likes discussing architectural paradigm shifts.' },
  INFJ: { title: 'Advocate', style: 'Idealistic & Visionary', desc: 'Values long-term trust, transparency, and product alignment. Highly sensitive to silent changelog removals.' },
  INFP: { title: 'Mediator', style: 'Empathetic & Value-driven', desc: 'Prefers products that align with personal developer happiness. Values intuitive onboarding and helpful documentation.' },
  ENFJ: { title: 'Protagonist', style: 'Inspiring & Collaborative', desc: 'Focuses on team satisfaction and adoption momentum. Advocates for peer consensus and clear training plans.' },
  ENFP: { title: 'Campaigner', style: 'Enthusiastic & Creative', desc: 'Early adopter of cutting-edge features. Energetic promoter who amplifies excitement across social channels.' },
  ISTJ: { title: 'Inspector', style: 'Methodical & Fact-based', desc: 'Strictly reads full documentation before engaging. Demands stability, compliance, and regression-free upgrades.' },
  ISFJ: { title: 'Defender', style: 'Reliable & Support-oriented', desc: 'Prefers familiar, stable workflows. Sensitive to dashboard slowness and sudden UI or schema redesigns.' },
  ESTJ: { title: 'Executive', style: 'Structured & Task-focused', desc: 'Values standard operating procedures, security protocols, and cost-per-seat metrics. Highly professional.' },
  ESFJ: { title: 'Consul', style: 'Harmonious & Practical', desc: 'Supports team consistency and community success. Values high-touch customer support and interactive training.' },
  ISTP: { title: 'Virtuoso', style: 'Pragmatic & Technical', desc: 'Hands-on problem solver. Improvised workarounds are their specialty; prefers self-serve tools and fast load times.' },
  ISFP: { title: 'Adventurer', style: 'Flexible & Aesthetic', desc: 'Appreciates premium visual aesthetics, responsive interfaces, and seamless, fluid user experiences.' },
  ESTP: { title: 'Entrepreneur', style: 'Action-oriented & Bold', desc: 'Demands instant speed and high performance. Churns quickly if friction blocks core daily transactions.' },
  ESFP: { title: 'Entertainer', style: 'Expressive & Engaging', desc: 'Highly vocal on community forums. Amplifies both extreme delight and catastrophic frustrations publicly.' },
};

const BUYER_JOURNEY_STAGES = [
  { key: 'awareness', label: 'Awareness', desc: 'Discovering product capabilities and potential fit' },
  { key: 'trigger', label: 'Trigger', desc: 'Experiencing an immediate paint-point or growth requirement' },
  { key: 'evaluation', label: 'Evaluation', desc: 'Comparing against competitors and testing workflows' },
  { key: 'proof', label: 'Proof', desc: 'Validating performance, security, and ROI metrics' },
  { key: 'decision', label: 'Decision', desc: 'Finalizing procurement, expansion, or churn resolution' }
];

const getBuyerJourneyIndex = (journey: string): number => {
  const normalized = (journey || '').toLowerCase();
  if (normalized.includes('aware')) return 0;
  if (normalized.includes('trig')) return 1;
  if (normalized.includes('eval')) return 2;
  if (normalized.includes('proof') || normalized.includes('consider')) return 3;
  if (normalized.includes('decis') || normalized.includes('commit') || normalized.includes('retent')) return 4;
  return -1;
};

interface BioSections {
  identityAnchor: string;
  behavioralRules: string;
  communicationFingerprint: string;
  emotionalTriggers: string;
  currentPosition: string;
  rawBio: string;
}

export const normalizeBio = (bio: any): string => {
  if (!bio) return '';
  
  let parsed = bio;
  if (typeof bio === 'string') {
    const trimmed = bio.trim();
    if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
      try {
        parsed = JSON.parse(trimmed);
      } catch (e) {
        // Not valid JSON, keep as string
      }
    }
  }

  if (typeof parsed === 'object' && parsed !== null) {
    return Object.entries(parsed)
      .map(([key, val]) => {
        const valueStr = typeof val === 'object' ? JSON.stringify(val) : String(val);
        if (key.startsWith('[')) {
          return `${key} ${valueStr}`;
        }
        // convert key from camelCase or snake_case to upper case spaces
        const formattedKey = key
          .replace(/([A-Z])/g, ' $1')
          .replace(/[_\s]+/g, ' ')
          .trim()
          .toUpperCase();
        return `[${formattedKey}] ${valueStr}`;
      })
      .join('\n');
  }

  return String(bio);
};

const parseBioSections = (bioText: any = ''): BioSections => {
  const bioStr = normalizeBio(bioText);
  const sections = {
    identityAnchor: '',
    behavioralRules: '',
    communicationFingerprint: '',
    emotionalTriggers: '',
    currentPosition: '',
    rawBio: bioStr,
  };

  if (!bioStr) return sections;

  const tags = [
    { key: 'identityAnchor', label: '[IDENTITY ANCHOR]' },
    { key: 'behavioralRules', label: '[BEHAVIORAL RULES]' },
    { key: 'communicationFingerprint', label: '[COMMUNICATION FINGERPRINT]' },
    { key: 'emotionalTriggers', label: '[EMOTIONAL TRIGGERS]' },
    { key: 'currentPosition', label: '[CURRENT POSITION]' }
  ];

  const matches: { key: keyof Omit<BioSections, 'rawBio'>; index: number; label: string }[] = [];
  tags.forEach(tag => {
    const idx = bioStr.indexOf(tag.label);
    if (idx !== -1) {
      matches.push({ key: tag.key as any, index: idx, label: tag.label });
    }
  });

  matches.sort((a, b) => a.index - b.index);

  if (matches.length === 0) {
    sections.identityAnchor = bioStr;
    return sections;
  }

  for (let i = 0; i < matches.length; i++) {
    const startIdx = matches[i].index + matches[i].label.length;
    const endIdx = i + 1 < matches.length ? matches[i + 1].index : bioStr.length;
    sections[matches[i].key] = bioStr.substring(startIdx, endIdx).trim();
  }

  return sections;
};

interface Demographics {
  age: string;
  gender: string;
  occupation: string;
  location: string;
  techLiteracy: string;
}

const parseDemographics = (bioText: string = ''): Demographics => {
  const parsed = parseBioSections(bioText);
  const text = parsed.identityAnchor || bioText;
  
  const extract = (pattern: RegExp, defaultVal: string = 'N/A'): string => {
    const match = text.match(pattern);
    return match ? match[1].trim() : defaultVal;
  };

  let age = extract(/(?:age|years\s+old)[:\s-]+(\d+)/i);
  if (age === 'N/A') {
    const fallbackMatch = text.match(/\b(\d{2})\b-year-old/i) || text.match(/aged\s+(\d{2})\b/i);
    if (fallbackMatch) age = fallbackMatch[1];
  }

  let gender = extract(/(?:gender|sex)[:\s-]+(male|female|non-binary|other)/i);
  if (gender === 'N/A') {
    const fallbackMatch = text.match(/\b(female|male|non-binary)\b/i);
    if (fallbackMatch) gender = fallbackMatch[1];
  }

  let occupation = extract(/(?:occupation|job|role|title)[:\s-]+([^\n.,;]+)/i);
  if (occupation === 'N/A') {
    const fallbackMatch = text.match(/(?:works\s+as\s+a|is\s+a|is\s+an)\s+([^\n.,;]+)/i);
    if (fallbackMatch) occupation = fallbackMatch[1];
  }

  let location = extract(/(?:location|city|lives\s+in)[:\s-]+([^\n.,;]+)/i);
  if (location === 'N/A') {
    const fallbackMatch = text.match(/(?:lives\s+in|based\s+in)\s+([^\n.,;]+)/i);
    if (fallbackMatch) location = fallbackMatch[1];
  }

  let techLiteracy = extract(/(?:tech\s+literacy|literacy|tech\s+savvy)[:\s-]+([^\n.,;]+)/i);
  if (techLiteracy === 'N/A') {
    const fallbackMatch = text.match(/(high|medium|low|expert|basic)\s+tech\s+literacy/i) || text.match(/tech\s+literacy\s+is\s+(high|medium|low|expert|basic)/i);
    if (fallbackMatch) techLiteracy = fallbackMatch[1];
  }

  return { age, gender, occupation, location, techLiteracy };
};

interface EmotionalTriggers {
  excited: string[];
  frustrated: string[];
  scared: string[];
  raw: string;
}

const parseEmotionalTriggersList = (text: string = ''): EmotionalTriggers => {
  if (!text) return { excited: [], frustrated: [], scared: [], raw: '' };

  const excitedMatch = text.match(/(?:excited|excited\s+by|motivations?|loves?|desires?)[:\s-]+([\s\S]*?)(?=(?:frustrated|frustrated\s+by|scared|scared\s+of|fears|anxieties|blockers|$))/i);
  const frustratedMatch = text.match(/(?:frustrated|frustrated\s+by|annoyed|blockers|hates?)[:\s-]+([\s\S]*?)(?=(?:excited|excited\s+by|scared|scared\s+of|fears|anxieties|$))/i);
  const scaredMatch = text.match(/(?:scared|scared\s+of|anxieties|fears|worries?)[:\s-]+([\s\S]*?)(?=(?:excited|excited\s+by|frustrated|frustrated\s+by|$))/i);

  const parseBullets = (str: string = ''): string[] => {
    if (!str) return [];
    return str.split('\n')
      .map(l => l.replace(/^[-*+\d.\s]+/, '').trim())
      .filter(l => l.length > 2);
  };

  return {
    excited: parseBullets(excitedMatch ? excitedMatch[1] : ''),
    frustrated: parseBullets(frustratedMatch ? frustratedMatch[1] : ''),
    scared: parseBullets(scaredMatch ? scaredMatch[1] : ''),
    raw: text
  };
};

const renderTelemetryTab = (persona: any, simulationConfig: any, simulationStatus: string) => {
  const demo = parseDemographics(persona.bio);

  const statusText = simulationStatus === 'running'
    ? 'Simulation Thread Active — Real-time telemetry streaming'
    : simulationStatus === 'completed'
      ? 'Simulation Session Sealed — Historical telemetry stored'
      : 'Simulation Thread Awaiting Run — Target configuration queued';

  const statusColor = simulationStatus === 'running'
    ? 'bg-green-100 text-green-800 border-green-500'
    : simulationStatus === 'completed'
      ? 'bg-blue-100 text-blue-800 border-blue-500'
      : 'bg-yellow-100 text-yellow-800 border-yellow-500';

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Simulation Status */}
      <div className={`border-4 border-black p-4 flex items-center gap-3 font-mono font-black uppercase text-xs tracking-wider shadow-neo-black ${statusColor}`}>
        <Activity className={`w-4 h-4 ${simulationStatus === 'running' ? 'animate-pulse' : ''}`} strokeWidth={3} />
        <span>{statusText}</span>
      </div>

      {/* Telemetry Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 font-mono">
        <div className="border-4 border-black p-4 bg-white shadow-neo-black flex flex-col gap-1">
          <div className="flex items-center gap-1.5 mb-1">
            <Fingerprint className="w-4 h-4 text-black/50" strokeWidth={3} />
            <span className="text-[10px] font-black uppercase tracking-widest text-black/50">Agent Spawning ID</span>
          </div>
          <span className="font-black text-sm uppercase break-all">{persona.agent_id || 'AWAITING_SPAWN'}</span>
        </div>

        <div className="border-4 border-black p-4 bg-white shadow-neo-black flex flex-col gap-1">
          <div className="flex items-center gap-1.5 mb-1">
            <Database className="w-4 h-4 text-black/50" strokeWidth={3} />
            <span className="text-[10px] font-black uppercase tracking-widest text-black/50">Parent Simulation ID</span>
          </div>
          <span className="font-black text-sm uppercase break-all">{simulationConfig?.simulation_id ? `${simulationConfig.simulation_id.slice(0, 16)}…` : 'N/A'}</span>
        </div>

        <div className="border-4 border-black p-4 bg-white shadow-neo-black flex flex-col gap-1 font-sans">
          <div className="flex items-center gap-1.5 mb-1 font-mono">
            <Cpu className="w-4 h-4 text-black/50" strokeWidth={3} />
            <span className="text-[10px] font-black uppercase tracking-widest text-black/50">LLM Platform / Model</span>
          </div>
          <span className="font-black text-sm uppercase">{simulationConfig ? `${simulationConfig.platform_type || 'STITCH'} / ${simulationConfig.llm_model || 'DEFAULT'}` : 'N/A'}</span>
        </div>

        <div className="border-4 border-black p-4 bg-white shadow-neo-black flex flex-col gap-1 font-sans">
          <div className="flex items-center gap-1.5 mb-1 font-mono">
            <Activity className="w-4 h-4 text-black/50" strokeWidth={3} />
            <span className="text-[10px] font-black uppercase tracking-widest text-black/50">Cohort Scale & shadow</span>
          </div>
          <span className="font-black text-sm uppercase">
            {simulationConfig ? `${simulationConfig.llm_active_cohort} active / ${simulationConfig.shadow_agents} shadow` : 'N/A'}
          </span>
        </div>
      </div>

      {/* Demographics Card */}
      <div className="border-4 border-black p-6 bg-brand/5 shadow-neo-black font-sans">
        <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/50 mb-4 flex items-center gap-2 border-b-2 border-black pb-2">
          <Users className="w-4 h-4 text-black" strokeWidth={3} /> Demographic Anchors
        </h5>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-y-4 gap-x-6">
          <div className="flex flex-col">
            <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider">Age</span>
            <span className="font-black text-sm uppercase text-black">{demo.age}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider">Gender</span>
            <span className="font-black text-sm uppercase text-black">{demo.gender}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider">Location</span>
            <span className="font-black text-sm uppercase text-black truncate" title={demo.location}>{demo.location}</span>
          </div>
          <div className="flex flex-col col-span-2 sm:col-span-1">
            <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider">Tech Literacy</span>
            <span className="font-black text-sm uppercase text-brand">{demo.techLiteracy}</span>
          </div>
          <div className="flex flex-col col-span-2">
            <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider">Occupation</span>
            <span className="font-black text-sm uppercase text-black truncate" title={demo.occupation}>{demo.occupation}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const STANCE_CONFIG: Record<string, { label: string; bg: string; border: string; text: string; icon: any }> = {
  APPROVE: { label: 'APPROVE', bg: 'bg-green-50', border: 'border-green-600', text: 'text-green-700', icon: CheckCircle2 },
  REJECT: { label: 'REJECT', bg: 'bg-red-50', border: 'border-red-600', text: 'text-red-700', icon: XCircle },
  CONDITIONAL_APPROVE: { label: 'CONDITIONAL', bg: 'bg-amber-50', border: 'border-amber-500', text: 'text-amber-700', icon: HelpCircle },
};

const renderPsychologyTab = (persona: any) => {
  const parsedBio = parseBioSections(persona.bio);
  const mbtiCode = (persona.mbti || '').toUpperCase();
  const mbtiInfo = MBTI_DESCRIPTIONS[mbtiCode] || {
    title: 'Custom Agent',
    style: 'Adaptive Decision Maker',
    desc: 'Behaves with bespoke characteristics mapped dynamically from simulation data parameters.',
  };

  const stance = persona.predicted_stance;
  const stancePrediction = (stance?.prediction || '').toUpperCase();
  const stanceCfg = STANCE_CONFIG[stancePrediction];

  const dp = persona.decision_pattern;
  const cs = persona.communication_style;
  const profileConf = persona.profile_confidence ?? 0;
  const groundingQ = persona.grounding_quality ?? 1;

  return (
    <div className="space-y-6 animate-in fade-in duration-200">

      {/* ── Predicted Stance Banner ──────────────────────────────────────── */}
      {stanceCfg && stancePrediction && (
        <div className={`border-4 ${stanceCfg.border} p-5 ${stanceCfg.bg} shadow-neo-black`}>
          <div className="flex items-start gap-4">
            <div className={`flex-none w-12 h-12 border-4 border-black flex items-center justify-center ${stanceCfg.bg}`}>
              <stanceCfg.icon className={`w-6 h-6 ${stanceCfg.text}`} strokeWidth={3} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <span className={`font-mono font-black text-xs uppercase tracking-widest px-3 py-1 border-2 border-black ${stanceCfg.text} bg-white`}>
                  {stanceCfg.label}
                </span>
                <span className="font-mono text-[10px] font-black uppercase tracking-widest text-black/50">
                  Confidence: {Math.round((stance?.confidence ?? 0) * 100)}%
                </span>
                <div className="flex-1 h-2 bg-black/10 border border-black overflow-hidden max-w-24">
                  <div
                    className="h-full transition-all duration-700"
                    style={{ width: `${Math.round((stance?.confidence ?? 0) * 100)}%`, backgroundColor: stancePrediction === 'APPROVE' ? '#16a34a' : stancePrediction === 'REJECT' ? '#dc2626' : '#d97706' }}
                  />
                </div>
              </div>
              {stance?.likely_conditions?.length > 0 && (
                <div className="mb-2">
                  <span className="font-mono font-black text-[10px] uppercase tracking-widest text-black/50 block mb-1">Likely Conditions</span>
                  <ul className="space-y-1">
                    {stance.likely_conditions.map((c: string, i: number) => (
                      <li key={i} className="flex items-start gap-1.5 text-xs font-bold text-black/80">
                        <span className="text-green-600 mt-0.5 flex-none">✓</span>{c}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {stance?.potential_objections?.length > 0 && (
                <div>
                  <span className="font-mono font-black text-[10px] uppercase tracking-widest text-black/50 block mb-1">Potential Objections</span>
                  <ul className="space-y-1">
                    {stance.potential_objections.map((o: string, i: number) => (
                      <li key={i} className="flex items-start gap-1.5 text-xs font-bold text-black/80">
                        <span className="text-red-500 mt-0.5 flex-none">✗</span>{o}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── MBTI Dashboard Card ──────────────────────────────────────────── */}
      <div className="border-4 border-black p-6 bg-white shadow-neo-black flex flex-col sm:flex-row gap-6">
        <div className="flex-none flex flex-col items-center">
          <div className="w-20 h-20 bg-brand border-4 border-black flex items-center justify-center font-black text-2xl uppercase tracking-widest shadow-neo-black mb-2">
            {mbtiCode || 'SPEC'}
          </div>
          <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider">Cognitive Profile</span>
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-mono font-black text-xs uppercase bg-black text-brand px-2 py-0.5">{mbtiInfo.title}</span>
            <span className="text-xs font-black uppercase tracking-widest text-black/50">{mbtiInfo.style}</span>
          </div>
          {/* Prefer LLM-generated description over hardcoded fallback */}
          <p className="text-sm font-bold leading-relaxed text-black/75">
            {persona.mbti_description || mbtiInfo.desc}
          </p>
        </div>
      </div>

      {/* ── Key Traits ──────────────────────────────────────────────────── */}
      {(persona.key_traits?.length > 0 || persona.traits?.length > 0) && (
        <div>
          <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-2 flex items-center gap-2">
            <Target className="w-3.5 h-3.5" strokeWidth={3} /> Key Traits
          </h5>
          <div className="flex flex-wrap gap-2">
            {(persona.key_traits?.length > 0 ? persona.key_traits : persona.traits).map((t: string, i: number) => (
              <span key={i} className="text-xs px-3 py-1.5 bg-brand/10 border-2 border-black font-black uppercase tracking-wider">{t}</span>
            ))}
          </div>
        </div>
      )}

      {/* ── OCEAN Scores ─────────────────────────────────────────────────── */}
      {persona.ocean_scores && Object.keys(persona.ocean_scores).length > 0 && (
        <div className="border-4 border-black p-6 bg-white shadow-neo-black">
          <OceanBars scores={persona.ocean_scores} />
        </div>
      )}

      {/* ── Decision Pattern ────────────────────────────────────────────── */}
      {dp && (dp.speed || dp.preference || dp.risk_tolerance) && (
        <div className="border-4 border-black p-5 bg-white shadow-neo-black">
          <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-4 flex items-center gap-2">
            <TrendingUp className="w-3.5 h-3.5" strokeWidth={3} /> Decision Pattern
          </h5>
          <div className="grid grid-cols-3 gap-3 mb-4">
            {[['Speed', dp.speed], ['Preference', dp.preference], ['Risk Tolerance', dp.risk_tolerance]].map(([label, val]) => (
              val ? (
                <div key={label} className="flex flex-col">
                  <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider">{label}</span>
                  <span className="font-black text-sm uppercase text-brand">{val}</span>
                </div>
              ) : null
            ))}
          </div>
          {dp.justification && (
            <div className="border-t-2 border-black/10 pt-3">
              <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider block mb-1">Internal Justification Framework</span>
              <p className="text-xs font-bold text-black/80 leading-relaxed">{dp.justification}</p>
            </div>
          )}
          {dp.influencers?.length > 0 && (
            <div className="border-t-2 border-black/10 pt-3 mt-3">
              <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider block mb-2">Who They Listen To</span>
              <div className="flex flex-wrap gap-1.5">
                {dp.influencers.map((inf: string, i: number) => (
                  <span key={i} className="text-[10px] px-2 py-1 bg-black text-white font-black uppercase">{inf}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Questions They Will Ask ──────────────────────────────────────── */}
      {persona.questions_they_will_ask?.length > 0 && (
        <div>
          <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-3 flex items-center gap-2">
            <MessageSquare className="w-3.5 h-3.5" strokeWidth={3} /> Questions They Will Ask
          </h5>
          <div className="space-y-2">
            {persona.questions_they_will_ask.map((q: string, i: number) => (
              <div key={i} className="flex items-start gap-3 border-2 border-black p-3 bg-black/5">
                <span className="flex-none w-5 h-5 bg-black text-brand font-black text-[10px] flex items-center justify-center">Q{i + 1}</span>
                <p className="text-xs font-bold text-black/85 leading-relaxed">{q}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Domain Expertise ────────────────────────────────────────────── */}
      {persona.domain_expertise?.length > 0 && (
        <div>
          <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-2">Domain Expertise</h5>
          <div className="flex flex-wrap gap-2">
            {persona.domain_expertise.map((d: string, i: number) => (
              <span key={i} className="text-xs px-2.5 py-1 border-2 border-black font-black uppercase tracking-wider bg-white">{d}</span>
            ))}
          </div>
        </div>
      )}

      {/* ── Identity Anchor Narrative ────────────────────────────────────── */}
      <div>
        <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-2">Identity & Narrative Anchors</h5>
        <div className="bg-black/5 border-4 border-black p-5 font-bold text-black/85 leading-relaxed text-sm whitespace-pre-line">
          {parsedBio.identityAnchor || parsedBio.rawBio || 'No narrative identity anchor generated yet.'}
        </div>
      </div>

      {/* ── Behavioral Rules ─────────────────────────────────────────────── */}
      {parsedBio.behavioralRules && (
        <div>
          <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-2">Behavioral Decision Matrix</h5>
          <div className="bg-black/5 border-4 border-black p-5 font-mono text-xs font-bold text-black/80 leading-relaxed whitespace-pre-line">
            {parsedBio.behavioralRules}
          </div>
        </div>
      )}

      {/* ── Communication Fingerprint from bio ───────────────────────────── */}
      {parsedBio.communicationFingerprint && (
        <div>
          <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-2">Communication Fingerprint (Narrative)</h5>
          <div className="bg-black/5 border-4 border-black p-5 font-bold text-black/85 leading-relaxed text-sm whitespace-pre-line">
            {parsedBio.communicationFingerprint}
          </div>
        </div>
      )}

      {/* ── Profile Quality Meters ───────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-4">
        <div className="border-4 border-black p-4 bg-white shadow-neo-black">
          <span className="text-[10px] font-mono font-black uppercase tracking-widest text-black/40 block mb-2">Profile Confidence</span>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2.5 bg-black/10 border border-black overflow-hidden">
              <div className="h-full bg-brand transition-all duration-700" style={{ width: `${Math.round(profileConf * 100)}%` }} />
            </div>
            <span className="font-black text-sm text-brand">{Math.round(profileConf * 100)}%</span>
          </div>
        </div>
        <div className="border-4 border-black p-4 bg-white shadow-neo-black">
          <span className="text-[10px] font-mono font-black uppercase tracking-widest text-black/40 block mb-2">Grounding Quality</span>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2.5 bg-black/10 border border-black overflow-hidden">
              <div className="h-full bg-green-500 transition-all duration-700" style={{ width: `${Math.round(groundingQ * 100)}%` }} />
            </div>
            <span className="font-black text-sm text-green-600">{Math.round(groundingQ * 100)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const renderTriggersTab = (persona: any) => {
  const parsedBio = parseBioSections(persona.bio);
  const activeIndex = getBuyerJourneyIndex(typeof persona.buyer_journey === 'string' ? persona.buyer_journey : '');

  // Prefer structured arrays from backend; fall back to bio-text parsing
  const structuredET = persona.emotional_triggers;
  const hasStructuredTriggers = structuredET &&
    (structuredET.excited_by?.length > 0 || structuredET.frustrated_by?.length > 0 || structuredET.scared_of?.length > 0);

  const bioTriggers = parseEmotionalTriggersList(parsedBio.emotionalTriggers);
  const hasBioTriggers = bioTriggers.excited.length > 0 || bioTriggers.frustrated.length > 0 || bioTriggers.scared.length > 0;

  const cs = persona.communication_style;
  const bj = persona.buyer_journey_detail;
  const mc = persona.market_context;

  return (
    <div className="space-y-6 animate-in fade-in duration-200">

      {/* ── Buyer Journey Stage Indicator ───────────────────────────────── */}
      <div className="border-4 border-black p-6 bg-white shadow-neo-black">
        <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-4 flex items-center gap-2">
          <ListTodo className="w-4 h-4 text-black" strokeWidth={3} /> Buyer Journey Alignment
        </h5>
        <div className="flex flex-col md:flex-row gap-4 items-stretch md:items-center justify-between">
          {BUYER_JOURNEY_STAGES.map((stage, idx) => {
            const isActive = idx === activeIndex;
            const isCompleted = idx < activeIndex;
            let borderStyle = 'border-2 border-black/20 text-black/40 bg-black/[0.02]';
            if (isActive) borderStyle = 'border-4 border-black bg-brand text-black shadow-neo-black font-black scale-105 z-10';
            else if (isCompleted) borderStyle = 'border-4 border-black bg-black/5 text-black/70 font-black';
            return (
              <div key={stage.key} className={`flex-1 p-3 flex flex-col justify-between transition-all duration-200 ${borderStyle}`}>
                <div>
                  <div className="flex items-center gap-1.5 justify-between">
                    <span className="font-mono text-[10px] font-black uppercase tracking-widest opacity-60">Stage {idx + 1}</span>
                    {isCompleted && <span className="text-[10px] text-green-600 font-mono font-black">● DONE</span>}
                    {isActive && <span className="text-[10px] text-black font-mono font-black animate-pulse">● ACTIVE</span>}
                  </div>
                  <h6 className="font-black text-sm uppercase mt-1">{stage.label}</h6>
                </div>
                {isActive && <p className="text-[10px] font-bold mt-2 leading-tight text-black/70 border-t-2 border-black pt-1.5">{stage.desc}</p>}
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Structured Emotional Triggers ───────────────────────────────── */}
      <div className="space-y-4">
        <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40">Emotional Triggers & Sentiment Anchors</h5>
        {hasStructuredTriggers ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border-4 border-black p-4 bg-green-50/50 shadow-neo-black flex flex-col gap-2">
              <span className="font-mono font-black text-[10px] uppercase tracking-widest text-green-700 bg-green-100 border-2 border-green-700 px-2 py-0.5 self-start">Excited By</span>
              {structuredET.excited_by.length > 0 ? (
                <ul className="list-disc pl-4 space-y-1.5 text-xs font-bold text-black/85 leading-snug">
                  {structuredET.excited_by.map((item: string, i: number) => <li key={i}>{item}</li>)}
                </ul>
              ) : <p className="text-xs text-black/40 font-bold italic">No positive triggers noted.</p>}
            </div>
            <div className="border-4 border-black p-4 bg-red-50/50 shadow-neo-black flex flex-col gap-2">
              <span className="font-mono font-black text-[10px] uppercase tracking-widest text-red-700 bg-red-100 border-2 border-red-700 px-2 py-0.5 self-start">Frustrated By</span>
              {structuredET.frustrated_by.length > 0 ? (
                <ul className="list-disc pl-4 space-y-1.5 text-xs font-bold text-black/85 leading-snug">
                  {structuredET.frustrated_by.map((item: string, i: number) => <li key={i}>{item}</li>)}
                </ul>
              ) : <p className="text-xs text-black/40 font-bold italic">No friction points noted.</p>}
            </div>
            <div className="border-4 border-black p-4 bg-yellow-50/50 shadow-neo-black flex flex-col gap-2">
              <span className="font-mono font-black text-[10px] uppercase tracking-widest text-yellow-800 bg-yellow-100 border-2 border-yellow-800 px-2 py-0.5 self-start">Scared Of / Risk</span>
              {structuredET.scared_of.length > 0 ? (
                <ul className="list-disc pl-4 space-y-1.5 text-xs font-bold text-black/85 leading-snug">
                  {structuredET.scared_of.map((item: string, i: number) => <li key={i}>{item}</li>)}
                </ul>
              ) : <p className="text-xs text-black/40 font-bold italic">No severe anxieties logged.</p>}
            </div>
          </div>
        ) : hasBioTriggers ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border-4 border-black p-4 bg-green-50/50 shadow-neo-black flex flex-col gap-2">
              <span className="font-mono font-black text-[10px] uppercase tracking-widest text-green-700 bg-green-100 border-2 border-green-700 px-2 py-0.5 self-start">Excited By</span>
              <ul className="list-disc pl-4 space-y-1.5 text-xs font-bold text-black/85 leading-snug">
                {bioTriggers.excited.map((item, i) => <li key={i}>{item}</li>)}
              </ul>
            </div>
            <div className="border-4 border-black p-4 bg-red-50/50 shadow-neo-black flex flex-col gap-2">
              <span className="font-mono font-black text-[10px] uppercase tracking-widest text-red-700 bg-red-100 border-2 border-red-700 px-2 py-0.5 self-start">Frustrated By</span>
              <ul className="list-disc pl-4 space-y-1.5 text-xs font-bold text-black/85 leading-snug">
                {bioTriggers.frustrated.map((item, i) => <li key={i}>{item}</li>)}
              </ul>
            </div>
            <div className="border-4 border-black p-4 bg-yellow-50/50 shadow-neo-black flex flex-col gap-2">
              <span className="font-mono font-black text-[10px] uppercase tracking-widest text-yellow-800 bg-yellow-100 border-2 border-yellow-800 px-2 py-0.5 self-start">Scared Of / Risk</span>
              <ul className="list-disc pl-4 space-y-1.5 text-xs font-bold text-black/85 leading-snug">
                {bioTriggers.scared.map((item, i) => <li key={i}>{item}</li>)}
              </ul>
            </div>
          </div>
        ) : (
          <div className="border-4 border-black p-5 bg-black/5 text-sm font-bold leading-relaxed whitespace-pre-line font-sans">
            {parsedBio.emotionalTriggers || 'No customized emotional triggers generated yet.'}
          </div>
        )}
      </div>

      {/* ── Communication Style Card ─────────────────────────────────────── */}
      {cs && (cs.default || cs.formality || cs.conflict_handling) && (
        <div className="border-4 border-black p-5 bg-white shadow-neo-black">
          <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-4 flex items-center gap-2">
            <MessageSquare className="w-3.5 h-3.5" strokeWidth={3} /> Communication Style
          </h5>
          <div className="grid grid-cols-3 gap-3 mb-3">
            {[['Default Style', cs.default], ['Formality', cs.formality], ['Conflict Handling', cs.conflict_handling]].map(([label, val]) => (
              val ? (
                <div key={label} className="flex flex-col">
                  <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider">{label}</span>
                  <span className="font-black text-sm uppercase text-black">{val}</span>
                </div>
              ) : null
            ))}
          </div>
          {cs.preferred_channels?.length > 0 && (
            <div className="border-t-2 border-black/10 pt-3">
              <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider block mb-2">Preferred Channels</span>
              <div className="flex flex-wrap gap-1.5">
                {cs.preferred_channels.map((ch: string, i: number) => (
                  <span key={i} className="text-[10px] px-2 py-1 border-2 border-black font-black uppercase bg-white">{ch}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Buyer Journey Detail (external personas) ─────────────────────── */}
      {bj && (
        <div className="border-4 border-black p-5 bg-white shadow-neo-black">
          <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-4 flex items-center gap-2">
            <Target className="w-3.5 h-3.5" strokeWidth={3} /> Buyer Journey Detail
          </h5>
          {bj.evaluation_trigger && (
            <div className="mb-3">
              <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider block mb-1">Evaluation Trigger</span>
              <p className="text-sm font-bold text-black/85">{bj.evaluation_trigger}</p>
            </div>
          )}
          <div className="grid grid-cols-3 gap-3 mb-3">
            {bj.roi_threshold_months && (
              <div className="flex flex-col">
                <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider">ROI Threshold</span>
                <span className="font-black text-sm uppercase text-brand">{bj.roi_threshold_months}mo</span>
              </div>
            )}
            {bj.willingness_to_pay_band && (
              <div className="flex flex-col">
                <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider">WTP Band</span>
                <span className="font-black text-sm uppercase text-black">{bj.willingness_to_pay_band}</span>
              </div>
            )}
            {bj.success_metric && (
              <div className="flex flex-col col-span-1">
                <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider">Success Metric</span>
                <span className="font-black text-xs uppercase text-black leading-tight">{bj.success_metric}</span>
              </div>
            )}
          </div>
          {bj.key_proof_points?.length > 0 && (
            <div className="mb-3">
              <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider block mb-1.5">Key Proof Points Needed</span>
              <ul className="space-y-1">
                {bj.key_proof_points.map((pt: string, i: number) => (
                  <li key={i} className="flex items-start gap-1.5 text-xs font-bold text-black/80">
                    <span className="text-green-600 mt-0.5 flex-none">✓</span>{pt}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {bj.deal_breakers?.length > 0 && (
            <div className="border-4 border-red-300 p-3 bg-red-50">
              <span className="text-[10px] font-mono font-black uppercase text-red-700 tracking-wider block mb-1.5">Deal Breakers</span>
              <ul className="space-y-1">
                {bj.deal_breakers.map((db: string, i: number) => (
                  <li key={i} className="flex items-start gap-1.5 text-xs font-bold text-red-800">
                    <span className="mt-0.5 flex-none">✗</span>{db}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* ── Market Context (external personas) ──────────────────────────── */}
      {mc && (
        <div className="border-4 border-black p-5 bg-white shadow-neo-black">
          <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-4 flex items-center gap-2">
            <Building2 className="w-3.5 h-3.5" strokeWidth={3} /> Market Context
          </h5>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              ['Company Size', mc.company_size_band],
              ['Buyer Role', mc.buyer_role],
              ['Industry', mc.industry_vertical],
              ['Pricing Sensitivity', mc.pricing_sensitivity],
              ['Deployment', mc.deployment_preference],
              ['Regulatory Burden', mc.regulatory_burden],
              ['Sales Cycle', mc.sales_cycle_weeks ? `${mc.sales_cycle_weeks} wks` : null],
              ['Annual Budget', mc.annual_solution_budget_usd ? `$${mc.annual_solution_budget_usd.toLocaleString()}` : null],
            ].filter(([, val]) => val).map(([label, val]) => (
              <div key={label as string} className="flex flex-col">
                <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider">{label}</span>
                <span className="font-black text-xs uppercase text-black mt-0.5">{val}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Network & Influence Metrics ──────────────────────────────────── */}
      {(persona.network_position_hint || persona.influence_strength !== undefined || persona.receptiveness !== undefined) && (
        <div className="border-4 border-black p-5 bg-white shadow-neo-black">
          <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-4 flex items-center gap-2">
            <BarChart2 className="w-3.5 h-3.5" strokeWidth={3} /> Network & Influence Metrics
          </h5>
          <div className="space-y-3">
            {persona.network_position_hint && (
              <div className="flex items-center gap-3">
                <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider w-28">Position</span>
                <span className="text-xs font-black uppercase border-2 border-black px-2 py-0.5">{persona.network_position_hint}</span>
              </div>
            )}
            {persona.influence_strength !== undefined && (
              <div className="flex items-center gap-3">
                <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider w-28">Influence</span>
                <div className="flex-1 h-2.5 bg-black/10 border border-black overflow-hidden max-w-40">
                  <div className="h-full bg-brand transition-all duration-700" style={{ width: `${Math.round((persona.influence_strength ?? 0) * 100)}%` }} />
                </div>
                <span className="font-black text-xs text-brand">{Math.round((persona.influence_strength ?? 0) * 100)}%</span>
              </div>
            )}
            {persona.receptiveness !== undefined && (
              <div className="flex items-center gap-3">
                <span className="text-[10px] font-mono font-black uppercase text-black/40 tracking-wider w-28">Receptiveness</span>
                <div className="flex-1 h-2.5 bg-black/10 border border-black overflow-hidden max-w-40">
                  <div className="h-full bg-blue-500 transition-all duration-700" style={{ width: `${Math.round((persona.receptiveness ?? 0) * 100)}%` }} />
                </div>
                <span className="font-black text-xs text-blue-600">{Math.round((persona.receptiveness ?? 0) * 100)}%</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Evidence Sources ─────────────────────────────────────────────── */}
      {persona.evidence_sources?.length > 0 && (
        <div>
          <h5 className="font-mono font-black text-xs uppercase tracking-widest text-black/40 mb-2 flex items-center gap-2">
            <Database className="w-3.5 h-3.5" strokeWidth={3} /> Evidence Sources
          </h5>
          <div className="border-4 border-black p-4 bg-black/5 space-y-1.5">
            {persona.evidence_sources.map((src: string, i: number) => (
              <div key={i} className="flex items-start gap-2 text-xs font-bold text-black/75">
                <span className="text-[10px] font-mono bg-black text-brand px-1.5 py-0.5 flex-none mt-0.5">{i + 1}</span>
                {src}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Active Stance / Sycophancy Shield ───────────────────────────── */}
      {parsedBio.currentPosition && (
        <div className="border-4 border-black p-5 bg-brand/5 shadow-neo-black flex items-start gap-4">
          <div className="w-10 h-10 bg-black text-brand border-2 border-black flex items-center justify-center flex-none shadow-neo-black">
            <Shield className="w-5 h-5" strokeWidth={3} />
          </div>
          <div>
            <h6 className="font-mono font-black text-xs uppercase tracking-widest text-black/50 mb-1">Active Stance / Sycophancy Shield</h6>
            <p className="text-sm font-bold leading-relaxed text-black/80">{parsedBio.currentPosition}</p>
          </div>
        </div>
      )}
    </div>
  );
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
  const { personas, boardroomPersonas, pipelineStages, spawnedAgents, simulationConfig, simulationStatus } = usePipelineStore();
  const [selectedPersona, setSelectedPersona] = useState<any | null>(null);
  const [activeTab, setActiveTab] = useState<'telemetry' | 'psychology' | 'triggers'>('telemetry');

  const handleSelectPersona = (persona: any) => {
    setSelectedPersona(persona);
    setActiveTab('telemetry');
  };

  const stageStatus = pipelineStages.layer3;
  const isWaiting = stageStatus === 'waiting';
  const isRunning = stageStatus === 'running';

  // Merge spawned agent psychological data into personas list — spread ALL rich fields
  const enrichedPersonas = useMemo(() => {
    return personas.map((p) => {
      const spawned = Object.values(spawnedAgents).find(
        (s) => s.agent_name === p.name || s.agent_id === p.id
      );
      const cleanName = cleanPersonaName(p.name);
      return {
        ...p,
        name: cleanName,
        // Core profile — persona_sync wins over agent_spawn for these
        bio: p.bio ?? spawned?.bio ?? '',
        mbti: p.mbti ?? spawned?.mbti ?? '',
        mbti_description: p.mbti_description ?? spawned?.mbti_description ?? '',
        key_traits: p.key_traits ?? spawned?.traits ?? [],
        // OCEAN comes from spawn event only
        ocean_scores: spawned?.ocean_scores ?? {},
        // Buyer journey: prefer persona_sync structured data
        buyer_journey: p.buyer_journey ?? spawned?.buyer_journey ?? '',
        buyer_journey_detail: p.buyer_journey_detail ?? spawned?.buyer_journey_detail,
        // Structured psychological sub-models
        emotional_triggers: p.emotional_triggers ?? spawned?.emotional_triggers,
        communication_style: p.communication_style ?? spawned?.communication_style,
        decision_pattern: p.decision_pattern ?? spawned?.decision_pattern,
        predicted_stance: p.predicted_stance ?? spawned?.predicted_stance,
        questions_they_will_ask: p.questions_they_will_ask ?? spawned?.questions_they_will_ask ?? [],
        // Persona metadata
        domain_expertise: p.domain_expertise ?? spawned?.domain_expertise ?? [],
        profile_confidence: p.profile_confidence ?? spawned?.profile_confidence ?? 0,
        grounding_quality: p.grounding_quality ?? spawned?.grounding_quality ?? 1,
        persona_type: p.persona_type ?? spawned?.persona_type ?? 'INTERNAL',
        network_position_hint: p.network_position_hint ?? spawned?.network_position_hint ?? 'peripheral',
        influence_strength: p.influence_strength ?? spawned?.influence_strength ?? 0.5,
        receptiveness: p.receptiveness ?? spawned?.receptiveness ?? 0.5,
        // External context
        market_context: p.market_context ?? spawned?.market_context,
        evidence_sources: p.evidence_sources ?? spawned?.evidence_sources ?? [],
        // Spawn identity
        agent_id: spawned?.agent_id ?? p.id,
        agent_type: spawned?.agent_type ?? '',
      };
    });
  }, [personas, spawnedAgents]);

  const spawnedList = useMemo(() => {
    return Object.values(spawnedAgents).map(agent => {
      const cleanName = cleanPersonaName(agent.agent_name);
      return {
        ...agent,
        name: cleanName,
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
        { name: 'CEO', role: 'Chief Executive Officer', role_short: 'CEO', traits: ['Market Strategy', 'Growth', 'Fundraising'], accent: 'bg-brand', icon: Briefcase, bio: '' },
        { name: 'CTO', role: 'Chief Technology Officer', role_short: 'CTO', traits: ['Architecture', 'Tech Debt', 'Scalability'], accent: 'bg-black', icon: Cpu, bio: '' },
        { name: 'CISO', role: 'Chief Information Security Officer', role_short: 'CISO', traits: ['Data Privacy', 'Compliance', 'Threat Modeling'], accent: 'bg-brand', icon: Shield, bio: '' },
        { name: 'CMO', role: 'Chief Marketing Officer', role_short: 'CMO', traits: ['Brand Safety', 'Customer Outcomes', 'Quality'], accent: 'bg-black', icon: Globe, bio: '' },
        { name: 'CFO', role: 'Chief Financial Officer', role_short: 'CFO', traits: ['Unit Economics', 'Burn Rate', 'ROI Modeling'], accent: 'bg-brand', icon: Briefcase, bio: '' },
        { name: 'CPO', role: 'Chief Product Officer', role_short: 'CPO', traits: ['Product-Market Fit', 'UX Strategy', 'Roadmap Prioritization'], accent: 'bg-black', icon: Brain, bio: '' },
        { name: 'Legal', role: 'General Counsel', role_short: 'Legal', traits: ['Regulatory Compliance', 'Liability', 'Contract Risk'], accent: 'bg-brand', icon: Lock, bio: '' },
        { name: 'Data', role: 'Head of Data & ML', role_short: 'Data', traits: ['ML Accuracy', 'Data Quality', 'Model Bias'], accent: 'bg-black', icon: Database, bio: '' },
        { name: 'Sales', role: 'Head of Sales', role_short: 'Sales', traits: ['Revenue Pipeline', 'Customer Acquisition', 'Competitive Win Rate'], accent: 'bg-brand', icon: Zap, bio: '' },
        { name: 'CS', role: 'Head of Customer Success', role_short: 'CS', traits: ['Onboarding', 'Retention', 'Change Management'], accent: 'bg-black', icon: Users, bio: '' },
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
            <div key={exec.name} onClick={() => handleSelectPersona({ ...exec, isBoardroom: true })} className="p-6 flex flex-col gap-4 bg-white group cursor-pointer hover:bg-brand/10 transition-colors duration-200">
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
              <div key={persona.id} onClick={() => handleSelectPersona({ ...persona, isBoardroom: false })} className="p-6 flex flex-col gap-4 group cursor-pointer hover:bg-brand/5 transition-colors duration-200">
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
                    handleSelectPersona({ ...persona, isBoardroom: false });
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
              <div key={agent.agent_id} onClick={() => handleSelectPersona({ ...agent, isBoardroom: false })} className="p-6 flex flex-col gap-4 group cursor-pointer hover:bg-brand/5 transition-colors duration-200">
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
                    handleSelectPersona({ ...agent, isBoardroom: false });
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
                  </div>
                </div>
              </div>

              {selectedPersona.isBoardroom ? (
                <>
                  {/* Bio / Psychological Profile */}
                  <div>
                    <h5 className="font-black text-xs uppercase tracking-widest text-black/40 mb-2 font-mono">Psychological Profile & Directives</h5>
                    <div className="bg-black/5 border-4 border-black p-5 font-bold text-black/85 leading-relaxed text-sm whitespace-pre-line">
                      {normalizeBio(selectedPersona.bio) || 'This executive behaves in accordance with their designated role and company domain requirements.'}
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
                </>
              ) : (
                <>
                  {/* Tab Navigation */}
                  <div className="flex border-4 border-black font-mono font-black text-xs uppercase tracking-widest flex-none shadow-neo-black">
                    <button
                      onClick={() => setActiveTab('telemetry')}
                      className={`flex-1 py-3 text-center border-r-4 border-black last:border-r-0 cursor-pointer transition-all ${
                        activeTab === 'telemetry' ? 'bg-brand text-black font-black' : 'bg-white hover:bg-brand/10 text-black/60'
                      }`}
                    >
                      Telemetry
                    </button>
                    <button
                      onClick={() => setActiveTab('psychology')}
                      className={`flex-1 py-3 text-center border-r-4 border-black last:border-r-0 cursor-pointer transition-all ${
                        activeTab === 'psychology' ? 'bg-brand text-black font-black' : 'bg-white hover:bg-brand/10 text-black/60'
                      }`}
                    >
                      Psychology
                    </button>
                    <button
                      onClick={() => setActiveTab('triggers')}
                      className={`flex-1 py-3 text-center border-r-4 border-black last:border-r-0 cursor-pointer transition-all ${
                        activeTab === 'triggers' ? 'bg-brand text-black font-black' : 'bg-white hover:bg-brand/10 text-black/60'
                      }`}
                    >
                      Triggers
                    </button>
                  </div>

                  {/* Tab Content */}
                  <div className="flex-1 mt-2">
                    {activeTab === 'telemetry' && renderTelemetryTab(selectedPersona, simulationConfig, simulationStatus)}
                    {activeTab === 'psychology' && renderPsychologyTab(selectedPersona)}
                    {activeTab === 'triggers' && renderTriggersTab(selectedPersona)}
                  </div>
                </>
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
