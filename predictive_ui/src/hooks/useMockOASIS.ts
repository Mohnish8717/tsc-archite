/**
 * useMockOASIS - feeds realistic synthetic persona interactions into the
 * pipeline store so the OASIS3D network can be visualised without a live backend.
 *
 * Call `start()` to begin the simulation loop, `stop()` to halt it.
 * All data is driven by the same AgentAction / SpawnedAgent types the real
 * backend produces, so the visualisation is pixel-perfect.
 */

import { useEffect, useRef } from 'react';
import { usePipelineStore } from '../store/usePipelineStore';

// ─── Seed data ────────────────────────────────────────────────────────────────

const PERSONAS = [
  { id: 'agent_001', name: 'Priya Sharma',    role: 'Early Adopter',      traits: ['Tech-Savvy', 'Optimistic', 'Vocal']          },
  { id: 'agent_002', name: 'Marcus Webb',     role: 'Power User',         traits: ['Analytical', 'Demanding', 'Loyal']            },
  { id: 'agent_003', name: 'Leila Okonkwo',   role: 'Privacy Advocate',   traits: ['Cautious', 'Principled', 'Articulate']        },
  { id: 'agent_004', name: 'Tom Bauer',       role: 'Enterprise Buyer',   traits: ['Risk-Averse', 'ROI-Focused', 'Decisive']      },
  { id: 'agent_005', name: 'Sofia Reyes',     role: 'Influencer',         traits: ['Trendy', 'High-Reach', 'Opinionated']         },
  { id: 'agent_006', name: 'James Nakamura',  role: 'Developer',          traits: ['Pragmatic', 'API-First', 'Blunt']             },
  { id: 'agent_007', name: 'Amara Osei',      role: 'Skeptic',            traits: ['Critical', 'Data-Driven', 'Contrarian']       },
  { id: 'agent_008', name: 'Carlos Mendez',   role: 'SMB Owner',          traits: ['Budget-Conscious', 'Practical', 'Impatient']  },
  { id: 'agent_009', name: 'Yuki Tanaka',     role: 'Product Manager',    traits: ['Strategic', 'Empathetic', 'Detail-Oriented']  },
  { id: 'agent_010', name: 'Fatima Al-Said',  role: 'Compliance Officer', traits: ['Regulatory', 'Thorough', 'Conservative']      },
  { id: 'agent_011', name: 'Ethan Cross',     role: 'Lurker',             traits: ['Passive', 'Observant', 'Late-Adopter']        },
  { id: 'agent_012', name: 'Nina Volkov',     role: 'Churn Risk',         traits: ['Frustrated', 'Price-Sensitive', 'Disengaged'] },
];

const COMMENTS: Record<string, string[]> = {
  upvote: [
    'This is exactly what I needed - real signal, no noise.',
    'Impressive. The prediction accuracy is genuinely useful.',
    'Finally a tool that surfaces friction before it becomes churn.',
    'Solid architecture. The boardroom layer is surprisingly insightful.',
    'NPS went up 18 points after we used this in sprint planning.',
    'The persona diversity feels realistic, not like typical AI outputs.',
    'Integration was painless. Rollout took under a day.',
    'This validated a feature we nearly killed. Massive win.',
  ],
  downvote: [
    'The latency on the simulation layer is unacceptable for production.',
    'Privacy controls are vague - compliance team flagged three gaps.',
    'We already have analytics. Not sure what this adds on top.',
    'Onboarding is rough. Took three attempts to get the first run through.',
    'The export format doesn\'t match our data warehouse schema.',
    'Hallucination risk isn\'t addressed anywhere in the docs.',
    'Too expensive for the SMB tier. We\'re evaluating alternatives.',
    'Support SLA is unclear. That\'s a blocker for enterprise approval.',
  ],
  comment: [
    'Would love a Slack integration for real-time alerts.',
    'Is there a way to weight personas by market segment size?',
    'The boardroom debate captured our exact Q4 stakeholder tensions.',
    'How does this handle multi-language persona generation?',
    'API docs need better examples - the quickstart is too abstract.',
    'Curious how the tension scoring adapts to B2C vs B2B inputs.',
    'Can we export the PRD as a Confluence page automatically?',
    'What\'s the roadmap for the compliance grounding module?',
    'The anti-sycophancy architecture is genuinely differentiated.',
    'We ran this against a failed product - it predicted all the friction points.',
  ],
};

const ACTION_TYPES: ('upvote' | 'downvote' | 'comment')[] = [
  'upvote', 'upvote', 'upvote',     // 3x weight → more positive signal
  'downvote',
  'comment', 'comment', 'comment',  // 3x weight → lots of discussion
];

function randomItem<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function makeTimestamp(): string {
  return new Date().toISOString();
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useMockOASIS() {
  const { addAction, addSpawnedAgent, setSimulationStatus, setConnected, setPipelineStage, setPersonas, resetForNewSimulation } = usePipelineStore();
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stepRef = useRef(0);

  const start = () => {
    if (timerRef.current) return; // already running

    // Seed the store with initial state
    resetForNewSimulation('Slack AI Feature - Predictive Reality Run #1');
    setConnected(true);
    setPipelineStage('layer1', 'done');
    setPipelineStage('layer3', 'done');
    setPipelineStage('layer5', 'running');

    // Set synthetic personas for the Personas layer
    setPersonas(PERSONAS.map(p => ({
      id: p.id,
      name: p.name,
      role: p.role,
      traits: p.traits,
      impact: 60 + Math.floor(Math.random() * 40),
    })));

    // Spawn all agents immediately so pins appear in the network
    PERSONAS.forEach(p => {
      addSpawnedAgent({
        agent_id: p.id,
        agent_name: p.name,
        agent_type: 'synthetic_persona',
        role: p.role,
        traits: p.traits,
        impact: 70 + Math.floor(Math.random() * 30),
      });

      // Emit a spawn action for each agent
      addAction({
        timestamp: makeTimestamp(),
        agent_id: p.id,
        agent_name: p.name,
        timestep: 0,
        action_type: 'spawn',
        content: `${p.name} joined the simulation as ${p.role}.`,
        platform: 'oasis',
        metadata: { signal_type: 'spawn', confidence: 1.0 },
      });
    });

    // Emit live actions every 1.4 seconds
    timerRef.current = setInterval(() => {
      stepRef.current += 1;
      const actor = randomItem(PERSONAS);
      const actionType = randomItem(ACTION_TYPES);

      // Pick a random target (another agent) for social interactions
      const otherAgents = PERSONAS.filter(p => p.id !== actor.id);
      const target = randomItem(otherAgents);

      addAction({
        timestamp: makeTimestamp(),
        agent_id: actor.id,
        agent_name: actor.name,
        timestep: stepRef.current,
        action_type: actionType,
        content: randomItem(COMMENTS[actionType]),
        platform: 'twitter',
        metadata: {
          target_id: Math.random() > 0.4 ? target.id : null,   // 60 % agent-to-agent
          confidence: parseFloat((0.6 + Math.random() * 0.4).toFixed(2)),
          signal_type: actionType === 'upvote' ? 'positive' : actionType === 'downvote' ? 'negative' : 'neutral',
          impact: parseFloat((Math.random() * 10).toFixed(1)),
        },
      });
    }, 1400);
  };

  const stop = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setSimulationStatus('idle');
    setConnected(false);
  };

  // Auto-cleanup on unmount
  useEffect(() => () => { stop(); }, []);

  return { start, stop };
}
