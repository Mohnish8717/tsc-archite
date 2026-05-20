"use client";
import React from "react";
import { ContainerScroll } from "@/components/ui/container-scroll-animation";
import AnimatedTextCycle from "@/components/ui/animated-text-cycle";
import RadialOrbitalTimeline from "@/components/ui/radial-orbital-timeline";
import {
  Layers,
  Search,
  Users,
  Briefcase,
  Globe,
  Swords,
  FileText,
  Package,
  Zap,
  Shield,
  Brain,
  ArrowRight,
  Activity,
  BarChart3,
  CheckCircle2,
} from "lucide-react";

/* ─── Pipeline data for the orbital timeline ─── */
const pipelineData = [
  {
    id: 1,
    title: "Ingestion",
    date: "Layer 1",
    content:
      "RAG-enhanced parsing of Zendesk tickets, Slack logs, and customer interviews into core semantic signals.",
    category: "Data",
    icon: Layers,
    relatedIds: [2],
    status: "completed" as const,
    energy: 100,
  },
  {
    id: 2,
    title: "Discovery",
    date: "Layer 2",
    content:
      "Dynamically clusters pain points into Tension Clusters and drafts Feature Proposals addressing real friction.",
    category: "Analysis",
    icon: Search,
    relatedIds: [1, 3],
    status: "completed" as const,
    energy: 92,
  },
  {
    id: 3,
    title: "Personas",
    date: "Layer 3",
    content:
      "Builds deep psychological user personas with cognitive biases, social interaction patterns, and value systems.",
    category: "Generation",
    icon: Users,
    relatedIds: [2, 4],
    status: "completed" as const,
    energy: 88,
  },
  {
    id: 4,
    title: "Boardroom",
    date: "Layer 4",
    content:
      "Initializes the Autonomous Executive Suite - CEO, CTO, CISO, Legal, Product - each with private intelligence briefs.",
    category: "Assembly",
    icon: Briefcase,
    relatedIds: [3, 5],
    status: "completed" as const,
    energy: 80,
  },
  {
    id: 5,
    title: "OASIS Sim",
    date: "Layer 5",
    content:
      "Spins up hundreds of synthetic users in CAMEL-AI OASIS. They interact, argue, post - generating predictive behavioral data.",
    category: "Simulation",
    icon: Globe,
    relatedIds: [4, 6],
    status: "in-progress" as const,
    energy: 65,
  },
  {
    id: 6,
    title: "Debate",
    date: "Layer 6",
    content:
      "AG2 adversarial boardroom debate with anti-sycophancy measures. Every claim verified against Hindsight Memory Bank.",
    category: "Validation",
    icon: Swords,
    relatedIds: [5, 7],
    status: "in-progress" as const,
    energy: 55,
  },
  {
    id: 7,
    title: "Spec Gen",
    date: "Layer 7",
    content:
      "Auto-compiles debate consensus into a high-fidelity PRD - UI changes, data model updates, and prioritized tasks.",
    category: "Output",
    icon: FileText,
    relatedIds: [6, 8],
    status: "pending" as const,
    energy: 35,
  },
  {
    id: 8,
    title: "Handoff",
    date: "Layer 8",
    content:
      "Generates engineering-ready artifacts: integration tests, monitoring plans, and deployment configs.",
    category: "Delivery",
    icon: Package,
    relatedIds: [7],
    status: "pending" as const,
    energy: 15,
  },
];

/* ─── Feature cards data ─── */
const features = [
  {
    icon: Brain,
    title: "Predictive Intelligence",
    description:
      "Simulate thousands of synthetic users before writing a single line of code. Predict adoption, churn, and regulatory friction.",
  },
  {
    icon: Swords,
    title: "Anti-Sycophancy Debate",
    description:
      "Autonomous executives with logit-bias manipulation force genuine adversarial challenge. No echo chambers allowed.",
  },
  {
    icon: Shield,
    title: "Zero-Hallucination Grounding",
    description:
      "Every boardroom claim is verified against the Hindsight Memory Bank - actual simulation data, not fabrication.",
  },
  {
    icon: Activity,
    title: "Real-Time Observation",
    description:
      "Watch synthetic social dynamics unfold live - sentiment shifts, coalition formation, and backlash prediction in real-time.",
  },
  {
    icon: BarChart3,
    title: "Automated Analytics",
    description:
      "Market Sentiment Series, Decision Journals, and Tension Ledgers generated automatically from every simulation run.",
  },
  {
    icon: FileText,
    title: "Auto-Generated PRDs",
    description:
      "Engineering-ready specifications compiled directly from boardroom consensus - UI changes, data models, and test plans.",
  },
];

const MarqueeStripe = () => (
  <div className="w-full h-14 bg-brand border-y-8 border-black flex items-center overflow-hidden">
    <div className="flex animate-[marquee_100s_linear_infinite] whitespace-nowrap">
      {[...Array(10)].map((_, i) => (
        <span key={i} className="px-8 font-black text-black italic uppercase tracking-tighter text-sm">
          PREDICTING FRICTION BEFORE IT HAPPENS // AUTONOMOUS SOFTWARE GENESIS // ADVERSARIAL BOARDROOM SIMULATION // HINDSIGHT MEMORY PROTOCOL v3.0 // VALIDATING MARKET REALITY IN SILICON //
        </span>
      ))}
    </div>
  </div>
);

/* ─── Types ─── */
interface LandingPageProps {
  onStart?: () => void;
}

export default function LandingPage({ onStart }: LandingPageProps) {
  return (
    <div className="min-h-screen bg-background text-black font-sans selection:bg-black selection:text-white overflow-x-hidden">
      <MarqueeStripe />
      {/* ═══ HERO with ContainerScroll ═══ */}
      <div className="relative pt-10">
        <ContainerScroll
          titleComponent={
            <div className="flex flex-col items-center">
              {/* Badge */}
              <div className="inline-flex items-center gap-2 px-6 py-2 bg-black border-4 border-black text-white font-mono text-sm tracking-[0.2em] font-bold uppercase mb-12 shadow-neo-white transform -rotate-2">
                <span className="w-2 h-2 bg-background animate-pulse" />
                Autonomous Software Factory
              </div>

              <h1 className="text-5xl md:text-[7rem] font-black uppercase tracking-tighter text-black leading-[0.9]">
                Simulate The<br />
                <div className="bg-brand text-black px-6 pb-2 inline-block border-8 border-black transform rotate-1 mt-4 mb-4">
                  <AnimatedTextCycle 
                      words={[
                          "Future",
                          "Market",
                          "Users",
                          "Backlash",
                          "Friction",
                          "Adoption"
                      ]}
                      interval={3000}
                      className="text-black" 
                  />
                </div>
                <br />
                Before You Build
              </h1>

              <p className="mt-12 text-xl md:text-2xl text-black font-bold max-w-3xl leading-snug border-l-8 border-black pl-8 text-left">
                Validate product-market fit, predict regulatory friction, and
                generate engineering-ready specs - using autonomous AI simulation
                and adversarial boardroom debate.
              </p>

              <div className="flex gap-6 mt-16 flex-wrap justify-center relative z-10">
                <button 
                  onClick={onStart}
                  className="inline-flex items-center gap-3 px-8 py-4 bg-black text-white font-black text-lg tracking-widest uppercase border-4 border-black shadow-neo-white transition-all hover:translate-x-1 hover:translate-y-1 hover:shadow-neo-pressed"
                >
                  <Zap size={24} />
                  Define Scenario
                </button>
                <button className="inline-flex items-center gap-3 px-8 py-4 bg-white text-black font-black text-lg tracking-widest uppercase border-4 border-black shadow-neo-black transition-all hover:translate-x-1 hover:translate-y-1 hover:shadow-neo-pressed">
                  <ArrowRight size={24} />
                  See How It Works
                </button>
              </div>
            </div>
          }
        >
          {/* High-Fidelity Neo-Brutalist Dashboard Mockup Built in React */}
          <div className="w-full h-full bg-black flex flex-col border-8 border-black shadow-neo-white overflow-hidden font-mono mx-auto">
            {/* TOP BAR */}
            <div className="h-12 bg-brand border-b-4 border-black flex items-center justify-between px-4 shrink-0">
              <div className="font-black tracking-widest text-black flex items-center gap-2 text-sm">
                <Zap size={16} /> PREDICTIVE REALITY ENGINE // TELEMETRY
              </div>
              <div className="flex gap-2">
                <div className="px-3 py-1 bg-green-400 border-2 border-black text-xs font-black text-black uppercase tracking-widest">Sim Active</div>
                <div className="px-3 py-1 bg-white border-2 border-black text-xs font-black text-black uppercase tracking-widest">Synced</div>
              </div>
            </div>

            {/* MAIN CONTENT */}
            <div className="flex-1 flex overflow-hidden">
              {/* SIDEBAR */}
              <div className="w-64 bg-[#f8f9fa] border-r-4 border-black p-5 flex flex-col gap-5 hidden md:flex shrink-0">
                <div className="border-4 border-black bg-white p-4 shadow-neo-sm hover:-translate-y-1 hover:shadow-neo-pressed transition-all">
                  <div className="text-[10px] text-gray-500 font-black mb-1 tracking-[0.2em]">NETWORK HEAT</div>
                  <div className="text-5xl font-black text-[#FF4500]">9.8</div>
                </div>
                <div className="border-4 border-black bg-white p-4 shadow-neo-sm hover:-translate-y-1 hover:shadow-neo-pressed transition-all">
                  <div className="text-[10px] text-gray-500 font-black mb-1 tracking-[0.2em]">ACTIVE NODES</div>
                  <div className="text-4xl font-black">2,048</div>
                </div>
                <div className="border-4 border-black bg-white p-4 shadow-neo-sm hover:-translate-y-1 hover:shadow-neo-pressed transition-all">
                  <div className="text-[10px] text-gray-500 font-black mb-1 tracking-[0.2em]">TENSION STATUS</div>
                  <div className="text-xl font-black text-red-600 animate-pulse tracking-widest mt-1">CRITICAL</div>
                </div>
                <div className="flex-1" />
                <div className="text-[10px] text-gray-400 font-black tracking-[0.2em] uppercase mb-2">Live System Log</div>
                <div className="text-[10px] text-gray-600 font-bold border-l-4 border-brand pl-3 py-1 leading-relaxed bg-gray-100">
                  [SYS] Agents init...<br/>
                  [SYS] Context injected.<br/>
                  <span className="text-[#FF4500]">[WARN] Product friction.</span><br/>
                  [SYS] Graph mapped.<br/>
                  <span className="text-green-600">[OK] Stream synced.</span>
                </div>
              </div>

              {/* IMAGES GRID (SCROLLABLE) */}
              <div className="flex-1 flex flex-col bg-black overflow-y-auto custom-scrollbar relative" style={{ WebkitFontSmoothing: 'antialiased' }}>
                {/* BOARDROOM PANE */}
                <div className="w-full flex flex-col relative border-b-4 border-black group shrink-0">
                   <div className="h-10 bg-[#1a1a1a] border-b-2 border-black flex items-center px-4 justify-between sticky top-0 z-10 shadow-md">
                     <span className="text-white font-black text-[11px] tracking-widest uppercase">Layer 4 // Boardroom Debate</span>
                     <span className="text-green-400 font-black text-[10px] animate-pulse flex items-center gap-2">
                       <span className="w-2 h-2 rounded-full bg-green-400 block"></span> LIVE FEED
                     </span>
                   </div>
                   <div className="w-full bg-gray-900 relative">
                     <img src="/boardroom_shot.png" alt="Boardroom Simulation" className="w-full h-auto object-cover opacity-95 group-hover:opacity-100 transition-all duration-500" draggable={false} style={{ imageRendering: 'high-quality', transform: 'translateZ(0)' }} />
                   </div>
                </div>
                
                {/* OASIS PANE */}
                <div className="w-full flex flex-col relative group shrink-0">
                   <div className="h-10 bg-[#1a1a1a] border-b-2 border-black flex items-center px-4 justify-between sticky top-0 z-10 shadow-md">
                     <span className="text-white font-black text-[11px] tracking-widest uppercase">Layer 5 // OASIS Sim</span>
                     <span className="text-[#FF4500] font-black text-[10px] flex items-center gap-2">
                       <span className="w-2 h-2 rounded-full bg-[#FF4500] block animate-ping"></span> SYNCED
                     </span>
                   </div>
                   <div className="w-full bg-gray-900 relative">
                     <img src="/oasis_shot.png" alt="OASIS Graph" className="w-full h-auto object-cover opacity-95 group-hover:opacity-100 transition-all duration-500" draggable={false} style={{ imageRendering: 'high-quality', transform: 'translateZ(0)' }} />
                   </div>
                </div>
              </div>
            </div>
          </div>
        </ContainerScroll>
      </div>

      {/* ═══ STATS BAR ═══ */}
      <div className="relative z-10 -mt-16 md:-mt-32">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-0 border-4 border-black bg-white shadow-neo-black">
            {[
              { value: "1000+", label: "Synthetic Agents" },
              { value: "8", label: "Autonomous Layers" },
              { value: "< 5 min", label: "Time to Prediction" },
              { value: "89%", label: "Prediction Accuracy" },
            ].map((stat, i) => (
              <div key={stat.label} className={`p-8 text-center ${i !== 0 ? 'md:border-l-4 md:border-black' : ''} ${i === 1 ? 'border-l-4 border-black' : ''} ${i === 2 ? 'border-t-4 md:border-t-0 border-black' : ''} ${i === 3 ? 'border-t-4 border-l-4 md:border-t-0 md:border-l-4 border-black' : ''}`}>
                <div className="text-3xl md:text-5xl font-black text-black">
                  {stat.value}
                </div>
                <div className="text-xs md:text-sm text-black font-bold uppercase tracking-widest mt-2">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ═══ FEATURES GRID ═══ */}
      <section className="py-24 md:py-32 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-20">
            <div className="inline-flex items-center gap-2 text-black font-black text-sm tracking-[4px] uppercase mb-6 bg-brand px-4 py-2 border-4 border-black shadow-neo-black transform rotate-1">
              Core Capabilities
            </div>
            <h2 className="text-5xl md:text-7xl font-black uppercase tracking-tighter text-black leading-none">
              Intelligence<br />At Every Layer
            </h2>
            <p className="mt-8 text-black font-bold text-xl max-w-2xl mx-auto leading-snug border-l-8 border-black pl-6 text-left">
              From raw data ingestion to executive-grade decision artifacts —
              every layer is autonomous, adversarial, and grounded in real
              behavior.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature) => (
              <div
                key={feature.title}
                className="group p-8 bg-black text-white border-4 border-black shadow-neo-white transition-all hover:translate-x-1 hover:translate-y-1 hover:shadow-neo-pressed cursor-pointer"
              >
                <div className="w-16 h-16 bg-brand text-black border-4 border-black flex items-center justify-center mb-8 transform -rotate-3 group-hover:rotate-0 transition-transform">
                  <feature.icon size={32} strokeWidth={3} />
                </div>
                <h3 className="font-black text-white text-2xl uppercase tracking-tight mb-4">
                  {feature.title}
                </h3>
                <p className="text-lg text-white font-semibold leading-snug">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ ORBITAL PIPELINE ═══ */}
      <section className="relative py-24 md:py-32 border-t-8 border-black">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 text-white bg-black font-black text-sm tracking-[4px] uppercase mb-6 px-4 py-2 border-4 border-black shadow-neo-white transform -rotate-1">
              The 8-Layer Pipeline
            </div>
            <h2 className="text-5xl md:text-7xl font-black uppercase tracking-tighter text-black leading-none">
              Explore The<br />Architecture
            </h2>
            <p className="mt-8 text-black font-bold text-xl max-w-2xl mx-auto leading-snug border-l-8 border-black pl-6 text-left">
              Click any node to explore the pipeline. Connected layers pulse to
              show data flow.
            </p>
          </div>
        </div>

        <RadialOrbitalTimeline timelineData={pipelineData} />
        
        <div className="mt-20">
          <MarqueeStripe />
        </div>
      </section>

      {/* ═══ SOCIAL PROOF ═══ */}
      <section className="py-24 md:py-32 px-6 bg-white border-y-8 border-brand">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-20">
            <div className="inline-flex items-center gap-2 text-black font-black text-sm tracking-[4px] uppercase mb-6 bg-brand px-4 py-2 border-4 border-black shadow-neo-black transform rotate-2">
              Validation
            </div>
            <h2 className="text-5xl md:text-7xl font-black uppercase tracking-tighter text-black leading-none">
              Built On<br />Cutting-Edge Research
            </h2>
          </div>

          {/* Tech logos */}
          <div className="flex justify-center items-center gap-8 md:gap-16 flex-wrap mb-24 opacity-80">
            {["AG2 AutoGen", "CAMEL-AI OASIS", "Hindsight Memory", "Groq", "OpenAI"].map(
              (name) => (
                <span
                  key={name}
                  className="font-black text-xl md:text-3xl tracking-tighter uppercase text-black"
                >
                  {name}
                </span>
              )
            )}
          </div>

          {/* Testimonial cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                quote:
                  "The boardroom debate caught a compliance risk our legal team had missed entirely. The simulation predicted user backlash with 89% accuracy.",
                name: "Sarah Chen",
                role: "VP of Product, Enterprise SaaS",
                initials: "SC",
              },
              {
                quote:
                  "We ran the Slack AI controversy through the engine. The synthetic personas generated more nuanced criticism than our actual beta testers.",
                name: "Marcus Rodriguez",
                role: "CTO, AI Platform Startup",
                initials: "MR",
              },
              {
                quote:
                  "From raw customer data to a production-ready PRD in under 10 minutes. The auto-generated specs were more thorough than two-week sprints.",
                name: "Aisha Patel",
                role: "Head of Product, HealthTech",
                initials: "AP",
              },
            ].map((t) => (
              <div
                key={t.name}
                className="p-8 bg-white border-4 border-black shadow-neo-black transition-all hover:translate-x-1 hover:translate-y-1 hover:shadow-neo-pressed cursor-pointer"
              >
                <div className="flex gap-2 mb-6 text-brand">
                  {[...Array(5)].map((_, i) => (
                    <CheckCircle2 key={i} size={24} strokeWidth={3} />
                  ))}
                </div>
                <p className="text-xl text-black font-bold leading-snug mb-8">
                  "{t.quote}"
                </p>
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-white border-4 border-black flex items-center justify-center font-black text-lg text-black">
                    {t.initials}
                  </div>
                  <div>
                    <div className="text-lg font-black text-black uppercase">
                      {t.name}
                    </div>
                    <div className="text-sm font-bold text-black uppercase tracking-widest">{t.role}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ CTA ═══ */}
      <section className="py-32 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="bg-black border-8 border-black shadow-neo-white p-12 md:p-24 text-center">
            <h2 className="text-5xl md:text-[6rem] font-black uppercase tracking-tighter text-white leading-none mb-8">
              Predict<br />Your Future
            </h2>
            <p className="text-white font-bold text-2xl max-w-2xl mx-auto mb-12 leading-snug border-l-8 border-background pl-6 text-left">
              Join the early access program and run your first autonomous
              simulation in under 5 minutes.
            </p>
            <div className="flex gap-6 justify-center flex-wrap">
              <button 
                onClick={onStart}
                className="inline-flex items-center gap-3 px-8 py-5 bg-brand text-black font-black text-xl tracking-widest uppercase border-4 border-black shadow-neo-white transition-all hover:translate-x-1 hover:translate-y-1 hover:shadow-neo-pressed"
              >
                <Zap size={28} />
                Run First Simulation
              </button>
              <button className="inline-flex items-center gap-3 px-8 py-5 bg-white text-black font-black text-xl tracking-widest uppercase border-4 border-black shadow-neo-white transition-all hover:translate-x-1 hover:translate-y-1 hover:shadow-neo-pressed">
                View on GitHub
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ═══ FOOTER ═══ */}
      <footer className="border-t-8 border-black py-16 px-6 bg-white">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="flex items-center gap-3 font-black text-2xl text-black uppercase tracking-tighter">
            <Activity size={32} strokeWidth={3} className="text-brand" />
            Predictive Reality Engine
          </div>
          <div className="text-sm font-bold text-black uppercase tracking-widest">
            &copy; 2026 AGPL v3 Licensed
          </div>
        </div>
      </footer>
    </div>
  );
}
