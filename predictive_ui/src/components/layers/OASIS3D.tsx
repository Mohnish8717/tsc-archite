import React, { useRef, useState, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrthographicCamera, OrbitControls, Html, Line } from '@react-three/drei';
import * as THREE from 'three';
import { usePipelineStore } from '../../store/usePipelineStore';
import type { AgentAction } from '../../store/usePipelineStore';
import { cleanPersonaName } from '../../utils/nameHelper';
import { Activity, Zap, X, Network, AlertTriangle, FileText, Eye, Maximize2, Minimize2, ThumbsUp } from 'lucide-react';
import { normalizeBio, parseBioSections } from './AssemblyMatrix';
import { EagleEyeInterrogationModal } from './EagleEyeInterrogationModal';


// ─── Stable hash → position ──────────────────────────────────────────────────
function stablePos(id: string, radius = 20): [number, number, number] {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = id.charCodeAt(i) + ((h << 5) - h);
  const r = radius * (0.4 + Math.abs(Math.sin(h)) * 0.6);
  const theta = Math.abs(Math.cos(h)) * Math.PI * 2;
  return [r * Math.cos(theta), 0, r * Math.sin(theta)];
}

function normalizeId(id: string | number | undefined | null): string {
  if (!id) return '';
  return String(id).replace(/^agent_/, '');
}



// ─── Flat ground ─────────────────────────────────────────────────────────────
function Ground() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.95, 0]} receiveShadow>
      <planeGeometry args={[250, 250]} />
      <meshStandardMaterial color="#FAF9F6" roughness={0.9} metalness={0.1} />
    </mesh>
  );
}

function GridLines() {
  const grid = useMemo(() => new THREE.GridHelper(120, 24, '#CBD5E1', '#E2E8F0'), []);
  return <primitive object={grid} position={[0, -1.94, 0]} receiveShadow />;
}

// ─── Connection edge between two agents ──────────────────────────────────────
function ConnectionEdge({ from, to, active, actionType }: {
  from: [number, number, number];
  to: [number, number, number];
  active: boolean;
  actionType: string;
}) {
  const color = actionType === 'upvote' ? '#10B981' : actionType === 'downvote' ? '#EF4444' : '#F59E0B';
  const opacity = active ? 0.95 : 0.2;

  const points = useMemo(() => {
    // Shift up to connect perfectly to the glossy spheres
    const fromPos: [number, number, number] = [from[0], from[1] + 0.4, from[2]];
    const toPos: [number, number, number] = [to[0], to[1] + 0.4, to[2]];
    
    const mid: [number, number, number] = [
      (fromPos[0] + toPos[0]) / 2,
      2.5 + Math.abs(fromPos[0] - toPos[0]) * 0.15,
      (fromPos[2] + toPos[2]) / 2,
    ];
    const curve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(...fromPos),
      new THREE.Vector3(...mid),
      new THREE.Vector3(...toPos),
    );
    return curve.getPoints(24);
  }, [from, to]);

  return (
    <Line
      points={points}
      color={color}
      lineWidth={active ? 3.0 : 1.2}
      transparent
      opacity={opacity}
    />
  );
}

// ─── Structural network topology edge ─────────────────────────────────────────
function TopologyEdge({ from, to }: {
  from: [number, number, number];
  to: [number, number, number];
}) {
  const pulseRef = useRef<THREE.Mesh>(null);
  const pulseT = useRef<number>(Math.random());
  const pulseSpeed = useRef<number>(0.25 + Math.random() * 0.25);

  const points = useMemo(() => {
    // Shift up to connect perfectly to the glossy spheres
    const fromPos: [number, number, number] = [from[0], from[1] + 0.4, from[2]];
    const toPos: [number, number, number] = [to[0], to[1] + 0.4, to[2]];
    
    const mid: [number, number, number] = [
      (fromPos[0] + toPos[0]) / 2,
      0.8 + Math.abs(fromPos[0] - toPos[0]) * 0.05,
      (fromPos[2] + toPos[2]) / 2,
    ];
    const curve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(...fromPos),
      new THREE.Vector3(...mid),
      new THREE.Vector3(...toPos),
    );
    return { points: curve.getPoints(16), curve };
  }, [from, to]);

  useFrame((_, delta) => {
    if (pulseRef.current) {
      pulseT.current += pulseSpeed.current * delta * 0.4;
      if (pulseT.current > 1) {
        pulseT.current = 0;
      }
      const pos = points.curve.getPointAt(pulseT.current);
      pulseRef.current.position.copy(pos);
    }
  });

  return (
    <group>
      <Line
        points={points.points}
        color="#06B6D4"
        lineWidth={1.5}
        transparent
        opacity={0.35}
      />
      <mesh ref={pulseRef}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial color="#22D3EE" emissive="#06B6D4" emissiveIntensity={1.5} />
      </mesh>
    </group>
  );
}

// ─── Travelling pulse particle ────────────────────────────────────────────────
function Pulse({ from, to, actionType, onDone }: {
  from: [number, number, number];
  to: [number, number, number];
  actionType: string;
  onDone: () => void;
}) {
  const ref = useRef<THREE.Mesh>(null);
  const t = useRef(0);
  
  const curve = useMemo(() => {
    const fromPos: [number, number, number] = [from[0], from[1] + 0.4, from[2]];
    const toPos: [number, number, number] = [to[0], to[1] + 0.4, to[2]];
    return new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(...fromPos),
      new THREE.Vector3((fromPos[0] + toPos[0]) / 2, 3.5 + Math.abs(fromPos[0] - toPos[0]) * 0.15, (fromPos[2] + toPos[2]) / 2),
      new THREE.Vector3(...toPos),
    );
  }, [from, to]);
  
  const color = actionType === 'upvote' ? '#10B981' : actionType === 'downvote' ? '#EF4444' : '#F59E0B';

  useFrame((_, delta) => {
    t.current = Math.min(1, t.current + delta * 0.95);
    if (ref.current) {
      const pt = curve.getPointAt(t.current);
      ref.current.position.copy(pt);
    }
    if (t.current >= 1) onDone();
  });

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[0.3, 16, 16]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.2} />
    </mesh>
  );
}

// ─── Role → skin palette (Modern premium neobrutalist color scale) ────────────
const ROLE_SKIN: Record<string, string> = {
  'Early Adopter': '#EC4899',   // Vivid Pink
  'Power User': '#2563EB',      // Deep Sapphire Blue
  'Privacy Advocate': '#0D9488', // Elegant Teal
  'Enterprise Buyer': '#D97706', // Premium Gold
  'Influencer': '#EC4899',      // Pink
  'Developer': '#2563EB',       // Sapphire
  'Skeptic': '#EF4444',         // Ruby Red
  'SMB Owner': '#D97706',       // Gold
  'Product Manager': '#2563EB',  // PM Sapphire
  'Compliance Officer': '#0D9488', // Compliance Teal
  'Lurker': '#6B7280',          // Slate Gray
  'Churn Risk': '#EF4444',      // Red alert
};

function roleSkin(role: string) {
  return ROLE_SKIN[role] ?? '#8B5CF6'; // Vibrant Purple as premium fallback
}

// ─── Premium Abstract Node Pin (Replaces Blocky humanoid geometry) ──────────
function PersonPin({ agent, isSelected, onClick, onInterrogate }: {
  agent: any; isSelected: boolean; onClick: () => void; onInterrogate?: (id: string) => void;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const headRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  React.useEffect(() => {
    if (!isSelected) {
      setExpandedIndex(null);
    }
  }, [isSelected]);

  const skin = isSelected ? '#F59E0B' : roleSkin(agent.role);

  useFrame(({ clock }) => {
    if (!groupRef.current) return;
    const t = clock.getElapsedTime();
    
    // Smooth premium hover/bobbing float animation
    groupRef.current.position.y = Math.sin(t * 1.3 + agent.pos[0]) * 0.12;
    
    if (headRef.current) {
      headRef.current.rotation.y = t * 0.4;
      headRef.current.rotation.x = Math.sin(t * 0.6) * 0.1;
    }
    
    if (ringRef.current && agent.hot) {
      const s = 1 + Math.sin(t * 4.5) * 0.12;
      ringRef.current.scale.set(s, s, 1);
    }

    const ts = isSelected ? 1.35 : hovered ? 1.15 : 1;
    groupRef.current.scale.lerp(new THREE.Vector3(ts, ts, ts), 0.12);
  });

  const pp = {
    onClick: (e: any) => { e.stopPropagation(); onClick(); },
    onPointerOver: (e: any) => {
      e.stopPropagation();
      setHovered(true);
      if (document.body) document.body.style.cursor = 'pointer';
    },
    onPointerOut: () => {
      setHovered(false);
      if (document.body) document.body.style.cursor = 'auto';
    },
  };

  return (
    <group ref={groupRef} position={agent.pos}>

      {/* Modern Soft Ground Shadow disc */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.94, 0]}>
        <circleGeometry args={[1.4, 32]} />
        <meshStandardMaterial color="#000000" transparent opacity={0.12} roughness={1} />
      </mesh>

      {/* Beveled slate pedestal / Base disc */}
      <mesh position={[0, -1.8, 0]} castShadow receiveShadow {...pp}>
        <cylinderGeometry args={[0.9, 1.1, 0.3, 8]} />
        <meshStandardMaterial color={isSelected ? '#F59E0B' : '#1F2937'} roughness={0.4} metalness={0.8} />
      </mesh>
      <mesh position={[0, -1.8, 0]}>
        <cylinderGeometry args={[0.92, 1.12, 0.31, 8]} />
        <meshBasicMaterial color="#000000" wireframe />
      </mesh>

      {/* High-gloss Chrome Stand / Stem */}
      <mesh position={[0, -1.0, 0]} castShadow {...pp}>
        <cylinderGeometry args={[0.06, 0.06, 1.3, 12]} />
        <meshStandardMaterial color="#E5E7EB" roughness={0.05} metalness={0.95} />
      </mesh>
      <mesh position={[0, -1.0, 0]}>
        <cylinderGeometry args={[0.07, 0.07, 1.31, 12]} />
        <meshBasicMaterial color="#000000" wireframe />
      </mesh>

      {/* Glossy Core Sphere */}
      <mesh ref={headRef} position={[0, 0.4, 0]} castShadow receiveShadow {...pp}>
        <sphereGeometry args={[0.75, 32, 32]} />
        <meshStandardMaterial color={skin} roughness={0.15} metalness={0.4} />
      </mesh>
      <mesh position={[0, 0.4, 0]}>
        <sphereGeometry args={[0.77, 12, 12]} />
        <meshBasicMaterial color="#000000" wireframe transparent opacity={0.35} />
      </mesh>

      {/* Active Aura/Halo Torus */}
      {(agent.hot || isSelected) && (
        <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.4, 0]}>
          <torusGeometry args={[1.15, 0.08, 8, 32]} />
          <meshStandardMaterial 
            color={isSelected ? '#F59E0B' : '#06B6D4'} 
            emissive={isSelected ? '#D97706' : '#0891B2'} 
            emissiveIntensity={1.8} 
          />
        </mesh>
      )}

      {/* Name tag floating billboard */}
      <Html center position={[0, 1.6, 0]} style={{ pointerEvents: 'none' }}>
        <div style={{
          background: isSelected ? '#F59E0B' : '#000000',
          color: isSelected ? '#000000' : '#FFFFFF',
          fontFamily: 'monospace', fontWeight: 900, fontSize: '10px',
          padding: '3px 8px', whiteSpace: 'nowrap',
          letterSpacing: '0.1em', textTransform: 'uppercase',
          border: `3px solid #000000`,
          boxShadow: '3px 3px 0 #000000',
          borderRadius: '4px',
        }}>
          {agent.name.split(' ')[0]}
        </div>
      </Html>

      {/* Role badge on hover */}
      {(hovered || isSelected) && !isSelected && (
        <Html center position={[0, 2.3, 0]} style={{ pointerEvents: 'none' }}>
          <div style={{
            background: '#FFFFFF', color: '#000000',
            fontFamily: 'monospace', fontWeight: 900, fontSize: '8px',
            padding: '2px 6px', whiteSpace: 'nowrap',
            letterSpacing: '0.08em', textTransform: 'uppercase',
            border: '2px solid #000000',
            boxShadow: '2px 2px 0 #000000',
          }}>
            {agent.role}
          </div>
        </Html>
      )}

      {/* Clicked selected profile tooltip (High-fidelity glass-brutalist panel) */}
      {isSelected && (
        <Html position={[2.2, 0.8, 0]} style={{ pointerEvents: 'auto', zIndex: 2147483647 }} zIndexRange={[2147483647, 2147483647]}>
          <div
            onWheel={(e) => e.stopPropagation()}
            onPointerDown={(e) => e.stopPropagation()}
            style={{
              transform: 'translateY(-40%)',
              background: 'rgba(255, 255, 255, 0.90)',
              backdropFilter: 'blur(16px)',
              border: '6px solid #000000',
              boxShadow: '8px 8px 0 0 #000000',
              borderRadius: '16px',
              padding: '20px',
              width: '360px',
              maxHeight: '65vh',
              overflowY: 'auto',
              fontFamily: 'monospace',
              color: '#000000',
              zIndex: 2147483647,
            }}>
            {/* Header */}
            <div style={{ borderBottom: '4px solid #000000', paddingBottom: '12px', marginBottom: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ flex: 1, minWidth: 0, marginRight: '16px' }}>
                <div style={{ fontWeight: 900, fontSize: '16px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{agent.name}</div>
                <div style={{ 
                  fontSize: '9px', 
                  fontWeight: 900, 
                  background: skin, 
                  color: isSelected ? '#000000' : '#FFFFFF', 
                  padding: '4px 8px',
                  border: '2px solid #000000',
                  boxShadow: '2px 2px 0px #000000',
                  textTransform: 'uppercase', 
                  letterSpacing: '0.1em',
                  display: 'inline-block',
                  marginTop: '6px',
                  maxWidth: '100%',
                  wordWrap: 'break-word',
                  whiteSpace: 'normal'
                }}>
                  {agent.role}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
                {onInterrogate && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onInterrogate(agent.id); }}
                    style={{ 
                      background: '#10B981', 
                      color: '#000000', 
                      border: '3px solid #000000', 
                      boxShadow: '2px 2px 0px #000000',
                      padding: '4px 8px', 
                      fontWeight: 900, 
                      cursor: 'pointer', 
                      fontSize: '11px',
                      textTransform: 'uppercase'
                    }}
                  >
                    Interrogate
                  </button>
                )}
                <button
                  onClick={(e) => { e.stopPropagation(); onClick(); }}
                  style={{ 
                    background: '#EF4444', 
                    color: '#FFFFFF', 
                    border: '3px solid #000000', 
                    boxShadow: '2px 2px 0px #000000',
                    padding: '4px 8px', 
                    fontWeight: 900, 
                    cursor: 'pointer', 
                    fontSize: '11px',
                    transition: 'transform 0.1s'
                  }}
                >
                  X
                </button>
              </div>
            </div>

            {/* MBTI & Journey Details */}
            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
              {agent.mbti && (
                <div style={{ flex: 1, border: '3px solid #000000', padding: '6px', background: '#F3F4F6', textAlign: 'center', borderRadius: '4px' }}>
                  <div style={{ fontSize: '8px', fontWeight: 900, color: '#6B7280', textTransform: 'uppercase' }}>MBTI</div>
                  <div style={{ fontSize: '13px', fontWeight: 900, color: '#000000' }}>{agent.mbti}</div>
                </div>
              )}
              <div style={{ flex: agent.mbti ? 1.5 : 1, border: '3px solid #000000', padding: '6px', background: '#F3F4F6', textAlign: 'center', borderRadius: '4px' }}>
                <div style={{ fontSize: '8px', fontWeight: 900, color: '#6B7280', textTransform: 'uppercase' }}>Journey Stage</div>
                <div style={{ fontSize: '11px', fontWeight: 900, color: '#F59E0B', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{agent.buyerJourney || 'UNKNOWN'}</div>
              </div>
            </div>

            {/* Parsed Bio Sections */}
            {(() => {
              const sections = parseBioSections(agent.bio);
              
              const FormatBracketText = ({ text }: { text: string }) => {
                const parts = text.split(/(\[[A-Z0-9\s_&/-]+\])/g);
                return (
                  <>
                    {parts.map((part, i) => {
                      if (part.startsWith('[') && part.endsWith(']')) {
                        return (
                          <span key={i} style={{
                            display: 'inline-block',
                            fontSize: '7px',
                            fontWeight: 900,
                            color: '#000000',
                            backgroundColor: '#FEF08A',
                            padding: '1px 4px',
                            margin: '0 3px',
                            border: '1px solid #000000',
                            borderRadius: '2px',
                            boxShadow: '1px 1px 0 #000000',
                            verticalAlign: 'middle',
                            letterSpacing: '0.05em',
                            textTransform: 'uppercase',
                          }}>
                            {part.substring(1, part.length - 1)}
                          </span>
                        );
                      }
                      return <span key={i}>{part}</span>;
                    })}
                  </>
                );
              };

              return (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '14px' }}>
                  {sections.identityAnchor && (
                    <div>
                      <div style={{ fontSize: '8px', fontWeight: 900, color: '#2563EB', textTransform: 'uppercase', marginBottom: '4px', letterSpacing: '0.05em' }}>Identity Anchor</div>
                      <p style={{ fontSize: '11px', fontWeight: 700, margin: 0, lineHeight: 1.5, color: '#374151' }}>
                        <FormatBracketText text={sections.identityAnchor} />
                      </p>
                    </div>
                  )}
                  {sections.behavioralRules && (
                    <div style={{ border: '2px solid #000000', padding: '8px', background: '#F9FAFB', borderRadius: '4px', boxShadow: '2px 2px 0px #000000' }}>
                      <div style={{ fontSize: '8px', fontWeight: 900, color: '#DC2626', textTransform: 'uppercase', marginBottom: '4px', letterSpacing: '0.05em' }}>Behavioral Rules</div>
                      <p style={{ fontSize: '10px', fontFamily: 'monospace', fontWeight: 700, margin: 0, lineHeight: 1.5, color: '#4B5563', whiteSpace: 'pre-wrap' }}>
                        <FormatBracketText text={sections.behavioralRules} />
                      </p>
                    </div>
                  )}
                  {sections.communicationFingerprint && (
                    <div style={{ border: '2px solid #000000', padding: '8px', background: '#FFFBEB', borderRadius: '4px', boxShadow: '2px 2px 0px #000000' }}>
                      <div style={{ fontSize: '8px', fontWeight: 900, color: '#D97706', textTransform: 'uppercase', marginBottom: '4px', letterSpacing: '0.05em' }}>Communication Style</div>
                      <p style={{ fontSize: '10px', fontWeight: 700, margin: 0, lineHeight: 1.5, color: '#4B5563', whiteSpace: 'pre-wrap' }}>
                        <FormatBracketText text={sections.communicationFingerprint} />
                      </p>
                    </div>
                  )}
                  {!sections.identityAnchor && !sections.behavioralRules && !sections.communicationFingerprint && (
                    <div>
                      <div style={{ fontSize: '8px', fontWeight: 900, color: '#6B7280', textTransform: 'uppercase', marginBottom: '4px' }}>Agent Narrative</div>
                      <p style={{ fontSize: '11px', fontWeight: 700, margin: 0, lineHeight: 1.5, color: '#374151' }}>
                        <FormatBracketText text={sections.rawBio} />
                      </p>
                    </div>
                  )}
                </div>
              );
            })()}

            {/* OCEAN Scores with custom bars */}
            {agent.ocean && agent.ocean.O !== undefined && (
              <div style={{ border: '3px solid #000000', padding: '12px', background: '#FFFFFF', marginBottom: '14px', borderRadius: '8px' }}>
                <div style={{ fontSize: '9px', fontWeight: 900, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#000000', borderBottom: '2px solid #000000', paddingBottom: '4px', marginBottom: '8px' }}>
                  OCEAN Profile
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {[
                    { label: 'O - Openness', val: agent.ocean.O, color: '#EC4899' },
                    { label: 'C - Conscientious', val: agent.ocean.C, color: '#2563EB' },
                    { label: 'E - Extraversion', val: agent.ocean.E, color: '#0D9488' },
                    { label: 'A - Agreeable', val: agent.ocean.A, color: '#F59E0B' },
                    { label: 'N - Neuroticism', val: agent.ocean.N, color: '#EF4444' }
                  ].map((trait, i) => (
                    <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', fontWeight: 900 }}>
                        <span>{trait.label}</span>
                        <span>{trait.val}%</span>
                      </div>
                      <div style={{ height: '8px', border: '2px solid #000000', background: '#E5E7EB', overflow: 'hidden', borderRadius: '2px' }}>
                        <div style={{ height: '100%', width: `${trait.val}%`, background: trait.color }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Vote stats */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0', border: '4px solid #000000', marginBottom: '12px', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ padding: '6px', textAlign: 'center', borderRight: '2px solid #000000', background: '#F0FDF4' }}>
                <div style={{ fontSize: '8px', fontWeight: 900, color: '#16A34A', textTransform: 'uppercase' }}>Trust Upvotes</div>
                <div style={{ fontSize: '18px', fontWeight: 900, color: '#16A34A' }}>{agent.upvotes}</div>
              </div>
              <div style={{ padding: '6px', textAlign: 'center', borderLeft: '2px solid #000000', background: '#FEF2F2' }}>
                <div style={{ fontSize: '8px', fontWeight: 900, color: '#DC2626', textTransform: 'uppercase' }}>Friction Downvotes</div>
                <div style={{ fontSize: '18px', fontWeight: 900, color: '#DC2626' }}>{agent.downvotes}</div>
              </div>
            </div>

            {/* Recent activity */}
            <div style={{ fontSize: '9px', fontWeight: 900, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#6B7280', borderBottom: '2px solid #000000', paddingBottom: '4px', marginBottom: '8px' }}>
              Transmission Telemetry
            </div>
            <div style={{ maxHeight: '120px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {agent.recent.length === 0 && (
                <div style={{ fontSize: '10px', color: '#9CA3AF', fontWeight: 700, fontStyle: 'italic' }}>No active transmissions...</div>
              )}
              {agent.recent.slice(0, 5).map((a: AgentAction, i: number) => {
                const isExpanded = expandedIndex === i;
                return (
                  <div key={i}
                    onClick={(e) => {
                      e.stopPropagation();
                      setExpandedIndex(isExpanded ? null : i);
                    }}
                    style={{
                      border: '2px solid #000000',
                      padding: '6px',
                      background: a.action_type === 'upvote' ? '#F0FDF4' : a.action_type === 'downvote' ? '#FEF2F2' : '#FFF7ED',
                      cursor: 'pointer',
                      borderRadius: '4px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px', alignItems: 'center' }}>
                      <span style={{
                        fontSize: '7px', fontWeight: 900, textTransform: 'uppercase',
                        padding: '1px 4px',
                        border: '1px solid #000000',
                        background: a.action_type === 'upvote' ? '#16A34A' : a.action_type === 'downvote' ? '#DC2626' : '#FF4500',
                        color: '#FFFFFF',
                      }}>{a.action_type}</span>
                      <span style={{ fontSize: '8px', color: '#6B7280', fontWeight: 700 }}>{new Date(a.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <p style={{
                      fontSize: '10px',
                      fontWeight: 700,
                      color: '#1F2937',
                      margin: 0,
                      lineHeight: 1.3,
                      wordBreak: 'break-word',
                      whiteSpace: 'pre-wrap',
                    }}>
                      {isExpanded ? a.content : `${a.content?.slice(0, 60)}${a.content?.length > 60 ? '…' : ''}`}
                    </p>
                    {a.content?.length > 60 && (
                      <div style={{
                        fontSize: '7px',
                        fontWeight: 900,
                        color: '#FF4500',
                        marginTop: '2px',
                        textAlign: 'right',
                        textTransform: 'uppercase',
                      }}>
                        {isExpanded ? 'Collapse ▲' : 'Read More ▼'}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

// ─── Camera ───────────────────────────────────────────────────────────────────
function CamController({ selectedPos }: { selectedPos: [number, number, number] | null }) {
  const { camera } = useThree();
  const ctrlRef = useRef<any>(null);
  useFrame(() => {
    if (!ctrlRef.current) return;
    const target = selectedPos ? new THREE.Vector3(...selectedPos) : new THREE.Vector3(0, 0, 0);
    ctrlRef.current.target.lerp(target, 0.05);
    if (camera instanceof THREE.OrthographicCamera) {
      camera.zoom = THREE.MathUtils.lerp(camera.zoom, selectedPos ? 38 : 16, 0.05);
      camera.updateProjectionMatrix();
    }
    ctrlRef.current.update();
  });
  return (
    <OrbitControls
      ref={ctrlRef}
      autoRotate={!selectedPos}
      autoRotateSpeed={0.15}
      maxPolarAngle={Math.PI / 2.2}
      minPolarAngle={0.2}
      enableZoom
    />
  );
}

// ─── Main scene ───────────────────────────────────────────────────────────────
function NetworkScene({ onSelect, showTopologyLines, onInterrogate }: { onSelect: (pos: [number, number, number] | null) => void; showTopologyLines: boolean; onInterrogate: (id: string) => void }) {
  const { actions, spawnedAgents, networkTopology } = usePipelineStore();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pulses, setPulses] = useState<{ id: string; from: [number, number, number]; to: [number, number, number]; type: string }[]>([]);

  // Build agent map with comprehensive psychological profiles
  const agents = useMemo(() => {
    const map = new Map<string, any>();
    Object.values(spawnedAgents).forEach((s) => {
      const cleanName = cleanPersonaName(s.agent_name);
      const normId = normalizeId(s.agent_id);
      map.set(normId, { 
        id: normId, 
        name: cleanName, 
        pos: stablePos(normId), 
        upvotes: 0, 
        downvotes: 0, 
        recent: [], 
        hot: false,
        role: s.role || 'Lurker',
        bio: s.bio || 'Synthetic agent participating in OASIS swarm analytics.',
        mbti: s.mbti || null,
        traits: s.traits || ['Observer'],
        ocean: (s.ocean_scores && Object.keys(s.ocean_scores).length > 0) ? s.ocean_scores : null,
        buyerJourney: s.buyer_journey || 'Awareness'
      });
    });
    actions.forEach(a => {
      const cleanName = cleanPersonaName(a.agent_name);
      const normId = normalizeId(a.agent_id);
      if (!map.has(normId)) {
        map.set(normId, { 
          id: normId, 
          name: cleanName, 
          pos: stablePos(normId), 
          upvotes: 0, 
          downvotes: 0, 
          recent: [], 
          hot: false,
          role: 'Lurker',
          bio: 'A synthetic participant observing the conversation dynamics.',
          mbti: null,
          traits: ['Observer', 'Analytical'],
          ocean: null,
          buyerJourney: 'Awareness'
        });
      }
      const ag = map.get(normId)!;
      if (a.action_type === 'upvote') ag.upvotes++;
      if (a.action_type === 'downvote') ag.downvotes++;
      if (!ag.recent.find((r: any) => r.timestamp === a.timestamp)) ag.recent.unshift(a);
      if (ag.recent.length > 10) ag.recent.pop();
    });
    // Mark hot agents (active in last 5 actions)
    const recent5 = new Set(actions.slice(-5).map(a => normalizeId(a.agent_id)));
    map.forEach(ag => { ag.hot = recent5.has(ag.id); });
    return Array.from(map.values());
  }, [actions, spawnedAgents]);

  // Edges: pairs that have interacted
  const edges = useMemo(() => {
    const pairs = new Map<string, { from: [number, number, number]; to: [number, number, number]; type: string; active: boolean }>();
    actions.slice(-60).forEach(a => {
      const target = a.metadata?.target_id;
      if (!target) return;
      const normFrom = normalizeId(a.agent_id);
      const normTarget = normalizeId(target);
      const fromAgent = agents.find(ag => ag.id === normFrom);
      const toAgent = agents.find(ag => ag.id === normTarget);
      if (!fromAgent || !toAgent) return;
      const key = [normFrom, normTarget].sort().join('|');
      pairs.set(key, { from: fromAgent.pos, to: toAgent.pos, type: a.action_type, active: actions.slice(-5).some(r => normalizeId(r.agent_id) === normFrom) });
    });
    return Array.from(pairs.values());
  }, [actions, agents]);

  // Structural topology edges
  const topologyEdges = useMemo(() => {
    if (!showTopologyLines || !networkTopology?.edges) return [];
    const list: Array<{ from: [number, number, number]; to: [number, number, number] }> = [];
    networkTopology.edges.forEach(edge => {
      const normFrom = normalizeId(edge.from);
      const normTo = normalizeId(edge.to);
      const fromAgent = agents.find(ag => ag.id === normFrom);
      const toAgent = agents.find(ag => ag.id === normTo);
      if (fromAgent && toAgent) {
        list.push({
          from: fromAgent.pos,
          to: toAgent.pos
        });
      }
    });
    return list;
  }, [showTopologyLines, networkTopology, agents]);

  return (
    <group onPointerMissed={() => { setSelectedId(null); onSelect(null); }}>
      <Ground />
      <GridLines />
      {/* Connection edges */}
      {edges.map((e, i) => (
        <ConnectionEdge key={i} from={e.from} to={e.to} active={e.active} actionType={e.type} />
      ))}

      {/* Structural topology edges */}
      {showTopologyLines && topologyEdges.map((e, i) => (
        <TopologyEdge key={`topo-${i}`} from={e.from} to={e.to} />
      ))}

      {/* Travelling pulses */}
      {pulses.map(p => (
        <Pulse
          key={p.id}
          from={p.from}
          to={p.to}
          actionType={p.type}
          onDone={() => setPulses(prev => prev.filter(x => x.id !== p.id))}
        />
      ))}

      {/* Agent pins */}
      {agents.map(agent => (
        <PersonPin
          key={agent.id}
          agent={agent}
          isSelected={selectedId === agent.id}
          onInterrogate={onInterrogate}
          onClick={() => {
            const closing = selectedId === agent.id;
            setSelectedId(closing ? null : agent.id);
            onSelect(closing ? null : agent.pos);
          }}
        />
      ))}
    </group>
  );
}

// ─── Exported component ───────────────────────────────────────────────────────
export default function OASIS3D() {
  const {
    activeAgents, hotScoreAvg, tensionStatus, actions,
    networkTopology, sycophancyAlerts, eagleEyeResults, seedPosts,
    simulationStatus,
    upvotedItems, upvoteItem, sqliteData
  } = usePipelineStore();
  const [selectedPos, setSelectedPos] = useState<[number, number, number] | null>(null);
  const [showMonitorPanel, setShowMonitorPanel] = useState(true);
  const [showAlerts, setShowAlerts] = useState(false);
  const [showTopologyLines, setShowTopologyLines] = useState(false);
  const [isInterrogationModalOpen, setIsInterrogationModalOpen] = useState(false);
  const [interrogationAgentId, setInterrogationAgentId] = useState<string>('');
  
  // Custom drag reveal width states
  const [panelWidth, setPanelWidth] = useState(400);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ startX: number; startWidth: number } | null>(null);

  const [activeColumnIndex, setActiveColumnIndex] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  const isPanelExpanded = panelWidth > 750;

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    dragStartRef.current = { startX: e.clientX, startWidth: panelWidth };
  };

  React.useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const deltaX = e.clientX - (dragStartRef.current?.startX || 0);
      // Min 300px, max almost full screen
      const newWidth = Math.max(300, Math.min(window.innerWidth - 48, (dragStartRef.current?.startWidth || 400) - deltaX));
      setPanelWidth(newWidth);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      dragStartRef.current = null;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const container = e.currentTarget;
    const index = Math.round(container.scrollLeft / container.clientWidth);
    setActiveColumnIndex(index);
  };

  const scrollToColumn = (index: number) => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        left: index * scrollRef.current.clientWidth,
        behavior: 'smooth'
      });
    }
  };

  const [activityFilter, setActivityFilter] = useState<'combined' | 'posts' | 'comments'>('combined');
  const [expandedActionIds, setExpandedActionIds] = useState<Set<string>>(new Set());

  const toggleActionExpanded = (id: string) => {
    setExpandedActionIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const combinedFeed = useMemo(() => {
    const items: any[] = [];
    
    // Add Seed Posts
    seedPosts.forEach((s, i) => {
      items.push({
        id: `seed_${i}`,
        type: 'post',
        author: 'SYSTEM',
        content: s.content,
        timestamp: new Date(0).toISOString(),
        source: 'seed',
        original: s
      });
    });

    // Add SQLite Posts
    if (sqliteData?.posts) {
      sqliteData.posts.forEach((p: any) => {
        items.push({
          id: p.post_id,
          type: 'post',
          author: cleanPersonaName(p.user_name || p.user_id),
          content: p.content,
          timestamp: p.created_at,
          source: 'sqlite',
          original: p
        });
      });
    }

    // Add SQLite Comments
    if (sqliteData?.comments) {
      sqliteData.comments.forEach((c: any) => {
        items.push({
          id: c.comment_id,
          type: 'comment',
          author: cleanPersonaName(c.user_name || c.user_id),
          content: c.content,
          timestamp: c.created_at,
          targetId: c.post_id,
          source: 'sqlite',
          original: c
        });
      });
    }

    // Add Actions
    actions.forEach((a, i) => {
      const typeLower = (a.action_type || '').toLowerCase();
      let type = 'other';
      if (typeLower === 'post' || typeLower === 'spawn') type = 'post';
      else if (typeLower === 'comment') type = 'comment';
      else if (typeLower === 'upvote' || typeLower === 'downvote' || typeLower === 'like') type = 'interaction';
      
      items.push({
        id: `action_${a.agent_id}_${i}_${a.timestamp}`,
        type: type,
        author: cleanPersonaName(a.agent_name),
        content: a.content,
        timestamp: a.timestamp,
        targetId: a.metadata?.target_id,
        source: 'action',
        original: a
      });
    });

    // Sort by timestamp
    return items.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [actions, sqliteData, seedPosts]);

  const filteredFeed = useMemo(() => {
    if (activityFilter === 'combined') return combinedFeed;
    if (activityFilter === 'posts') return combinedFeed.filter(item => item.type === 'post');
    if (activityFilter === 'comments') return combinedFeed.filter(item => item.type === 'comment');
    return combinedFeed;
  }, [combinedFeed, activityFilter]);

  const getPostContent = (targetId?: string) => {
    if (!targetId) return null;
    const post = combinedFeed.find(item => String(item.id) === String(targetId));
    if (post) return post.content;
    return 'Original post content hidden or deleted.';
  };

  // --- Sub-render blocks for sidebar sections ---
  const debateSeedsBody = (
    <div className="flex-1 divide-y-2 divide-black/10 bg-white overflow-y-auto">
      {seedPosts.map((s, i) => (
        <div key={i} className="flex gap-3 p-3 text-left">
          <span className="font-black text-[10px] border-2 border-black bg-brand px-2 py-0.5 h-fit mt-0.5 flex-none shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]">
            S{s.index + 1}
          </span>
          <p className="text-xs font-bold text-black/80 leading-relaxed">{s.content}</p>
        </div>
      ))}
      {seedPosts.length === 0 && (
        <div className="p-4 text-center font-bold text-xs uppercase tracking-widest text-black/30">
          No active controversy seeds
        </div>
      )}
    </div>
  );

  const eaglesEyeBody = (
    <div className="flex-1 divide-y-2 divide-black/10 bg-white overflow-y-auto">
      {eagleEyeResults.slice().reverse().map((r, i) => {
        const actionKey = `eagle-${i}`;
        const isExpanded = expandedActionIds.has(actionKey);
        return (
          <div
            key={i}
            onClick={() => toggleActionExpanded(actionKey)}
            className="p-3 hover:bg-neutral-50 cursor-pointer transition-colors text-left"
          >
            <div className="flex justify-between items-center mb-1">
              <span className="font-black text-xs uppercase text-black/80">{r.agent_name}</span>
              <span className="text-[10px] font-black text-black/40">INTERCEPTED</span>
            </div>
            <p className="text-xs font-bold text-black/60 leading-relaxed whitespace-pre-wrap">
              {isExpanded ? r.content : `${r.content.slice(0, 100)}${r.content.length > 100 ? '...' : ''}`}
            </p>
          </div>
        );
      })}
      {eagleEyeResults.length === 0 && (
        <div className="p-4 text-center font-bold text-xs uppercase tracking-widest text-black/30">
          Click agent avatar to intercept telemetry
        </div>
      )}
    </div>
  );

  const liveSwarmBody = (
    <div className="flex-1 flex flex-col min-h-0 bg-white">
      {/* Filter Tabs */}
      <div className="flex border-b-4 border-black font-black text-xs uppercase">
        {(['combined', 'posts', 'comments'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActivityFilter(tab)}
            className={`flex-1 py-2 text-center border-r-4 last:border-r-0 border-black cursor-pointer transition-colors ${
              activityFilter === tab ? 'bg-brand text-black font-black' : 'bg-neutral-100 text-black/60 hover:bg-neutral-200'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Activity Feed */}
      <div className="flex-1 overflow-y-auto divide-y-2 divide-black/10">
        {filteredFeed.map((item) => {
          const actionKey = item.id;
          const isExpanded = expandedActionIds.has(actionKey);
          const hasContent = !!item.content;
          const timeStr = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : '';

          let iconColor = 'bg-brand';
          if (item.type === 'post') iconColor = 'bg-blue-500';
          else if (item.type === 'comment') iconColor = 'bg-purple-500';
          else if (item.type === 'interaction') {
             if (item.original.action_type === 'upvote') iconColor = 'bg-green-500';
             else if (item.original.action_type === 'downvote') iconColor = 'bg-red-500';
          }

          const upvotes = upvotedItems[actionKey] || 0;

          return (
            <div
              key={actionKey}
              className={`p-3 transition-colors text-left ${hasContent ? 'hover:bg-neutral-50' : ''}`}
            >
              <div className="flex items-start gap-2.5">
                <span className={`w-2.5 h-2.5 mt-1 border border-black shadow-[1px_1px_0px_0px_rgba(0,0,0,1)] flex-none ${iconColor}`} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-black text-xs uppercase truncate text-black/90">
                      {item.author}
                    </span>
                    <span className="text-[9px] font-black text-black/40 flex-none">
                      {timeStr}
                    </span>
                  </div>
                  
                  {item.type === 'comment' && item.targetId && (
                    <div className="mt-2 mb-2 p-2 bg-neutral-100 border-l-4 border-black/20 text-[10px] font-bold text-black/60 italic rounded-r line-clamp-2">
                      ↳ Replying to: "{getPostContent(item.targetId)}"
                    </div>
                  )}

                  {item.content && (
                    <p 
                      onClick={() => hasContent && toggleActionExpanded(actionKey)}
                      className={`text-xs font-bold text-black/70 leading-relaxed mt-1 border-l-2 border-brand/50 pl-2 ${hasContent ? 'cursor-pointer' : ''}`}
                    >
                      {isExpanded ? item.content : `${item.content.slice(0, 100)}${item.content.length > 100 ? '...' : ''}`}
                    </p>
                  )}

                  {['post', 'comment'].includes(item.type) && (
                    <div className="mt-2 flex items-center gap-2">
                      <button 
                        onClick={(e) => { e.stopPropagation(); upvoteItem(actionKey); }}
                        className={`flex items-center gap-1.5 px-2 py-1 border-2 border-black font-black text-[10px] uppercase transition-colors ${
                          upvotes > 0 ? 'bg-orange-500 text-white' : 'bg-white text-black hover:bg-orange-100'
                        }`}
                      >
                        <ThumbsUp className="w-3 h-3" strokeWidth={3} />
                        {upvotes > 0 ? upvotes : 'Upvote'}
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
        {filteredFeed.length === 0 && (
          <div className="p-8 text-center font-black text-xs uppercase tracking-widest text-black/30">
            Awaiting live activity feed...
          </div>
        )}
      </div>
    </div>
  );

  const tensionBg = tensionStatus === 'Normal' ? 'bg-green-500' : tensionStatus === 'Elevated' ? 'bg-brand' : 'bg-red-500';
  const hasCriticalAlerts = sycophancyAlerts.some(a => a.data_validity_warning);

  return (
    <div className="w-full h-full relative bg-white overflow-hidden font-mono">

      {/* ── Title control bar ─── */}
      <div className="absolute top-0 left-0 right-0 z-20 bg-black border-b-4 border-brand flex items-center justify-between px-6 py-2">
        <div className="flex items-center gap-3">
          <span className="font-black text-xs text-white uppercase tracking-widest">OASIS Simulation Preview</span>
          {simulationStatus === 'running' && (
            <span className="flex items-center gap-1.5 text-xs font-black text-brand uppercase tracking-widest animate-pulse">
              <span className="w-2 h-2 bg-brand rounded-full" /> Live &middot; {actions.length} events
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {/* Network Topology toggle */}
          {networkTopology && (
            <button
              onClick={() => setShowTopologyLines(v => !v)}
              className={`flex items-center gap-1.5 px-3 py-1.5 border-2 font-black text-xs uppercase tracking-widest cursor-pointer transition-colors duration-200 ${
                showTopologyLines 
                  ? 'bg-brand border-brand text-black' 
                  : 'bg-white/10 hover:bg-white/20 border-2 border-white/30 text-white'
              }`}
            >
              <Network className="w-3.5 h-3.5" strokeWidth={3} />
              Topology {showTopologyLines ? 'ON' : 'OFF'}
            </button>
          )}
          {/* Monitor Panel toggle */}
          <button
            onClick={() => setShowMonitorPanel(v => !v)}
            className={`flex items-center gap-1.5 px-3 py-1.5 border-2 font-black text-xs uppercase tracking-widest cursor-pointer transition-colors duration-200 ${
              showMonitorPanel 
                ? 'bg-brand border-brand text-black' 
                : 'bg-white/10 hover:bg-white/20 border-2 border-white/30 text-white'
            }`}
          >
            <Activity className="w-3.5 h-3.5" strokeWidth={3} />
            Monitor Panel {showMonitorPanel ? 'ON' : 'OFF'}
          </button>
          {/* Sycophancy alert toggle */}
          {sycophancyAlerts.length > 0 && (
            <button
              onClick={() => setShowAlerts(v => !v)}
              className={`flex items-center gap-1.5 px-3 py-1.5 border-2 font-black text-xs uppercase tracking-widest cursor-pointer transition-colors duration-200 ${hasCriticalAlerts ? 'bg-red-500 border-red-300 text-white animate-pulse' : 'bg-yellow-500 border-yellow-300 text-black'}`}
            >
              <AlertTriangle className="w-3.5 h-3.5" strokeWidth={3} />
              {sycophancyAlerts.length} Alert{sycophancyAlerts.length > 1 ? 's' : ''}
            </button>
          )}
        </div>
      </div>

      {/* ── Sycophancy Alerts Drawer ─── */}
      {showAlerts && sycophancyAlerts.length > 0 && (
        <div className="absolute top-12 right-0 z-30 w-96 bg-white border-l-8 border-b-8 border-black max-h-72 overflow-y-auto shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
          <div className="bg-red-600 text-white px-4 py-3 flex items-center gap-2 border-b-4 border-black sticky top-0">
            <AlertTriangle className="w-4 h-4" strokeWidth={3} />
            <span className="font-black text-xs uppercase tracking-widest flex-1">Sycophancy Collapses — Data Validity Risk</span>
            <button onClick={() => setShowAlerts(false)} className="cursor-pointer"><X className="w-4 h-4" strokeWidth={3} /></button>
          </div>
          <div className="divide-y-2 divide-black/10">
            {sycophancyAlerts.map((alert, i) => (
              <div key={i} className="px-4 py-3 space-y-1">
                <div className="flex justify-between items-center">
                  <span className="font-black text-xs uppercase">{alert.agent_name}</span>
                  <span className="text-xs font-black text-black/40">T={alert.timestep}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-red-50 border border-red-200 px-2 py-1">
                    <span className="font-black text-red-600">Frustration: {(alert.frustration_at_collapse * 100).toFixed(0)}%</span>
                  </div>
                  <div className="bg-yellow-50 border border-yellow-200 px-2 py-1">
                    <span className="font-black text-yellow-600">Trust: {(alert.trust_at_collapse * 100).toFixed(0)}%</span>
                  </div>
                </div>
                {alert.triggering_content && (
                  <p className="text-xs text-black/60 italic border-l-2 border-brand pl-2">
                    "{alert.triggering_content.slice(0, 120)}{alert.triggering_content.length > 120 ? '…' : ''}"
                  </p>
                )}
                <span className={`inline-block text-xs font-black uppercase px-2 py-0.5 ${alert.data_validity_warning ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'}`}>
                  {alert.pattern}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── 3D Canvas ─── */}
      <div className="absolute inset-0" style={{ top: '44px' }}>
        <Canvas shadows dpr={[1, 1.5]}>
          <OrthographicCamera makeDefault position={[40, 40, 40]} zoom={16} near={-200} far={200} />
          <ambientLight intensity={0.7} />
          <hemisphereLight skyColor="#ffffff" groundColor="#1a1a2e" intensity={0.4} />
          <directionalLight
            position={[20, 40, 20]}
            intensity={1.4}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
            shadow-camera-near={0.5}
            shadow-camera-far={150}
            shadow-camera-left={-40}
            shadow-camera-right={40}
            shadow-camera-top={40}
            shadow-camera-bottom={-40}
            shadow-bias={-0.0005}
          />
          <pointLight position={[-15, 25, -15]} intensity={1.8} color="#F97316" distance={80} decay={1.5} />
          <CamController selectedPos={selectedPos} />
          <NetworkScene 
            onSelect={setSelectedPos} 
            showTopologyLines={showTopologyLines} 
            onInterrogate={(id) => {
              setInterrogationAgentId(id);
              setIsInterrogationModalOpen(true);
            }}
          />
        </Canvas>
      </div>

      {/* ── Swarm Stats HUD (left) ─── */}
      <div className="absolute top-14 left-4 z-10 w-60 border-8 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] bg-white font-mono">
        <div className="bg-black text-white px-4 py-3 border-b-4 border-brand flex items-center gap-2">
          <Activity className="w-4 h-4 text-brand" strokeWidth={3} />
          <span className="font-black text-xs uppercase tracking-widest">Swarm Status</span>
        </div>
        <div className="divide-y-4 divide-black">
          <div className="flex justify-between px-4 py-3">
            <span className="font-black text-xs uppercase tracking-widest text-black/50">Personas Live</span>
            <span className="font-black text-2xl">{activeAgents || 0}</span>
          </div>
          <div className="flex justify-between px-4 py-3">
            <span className="font-black text-xs uppercase tracking-widest text-black/50">Network Heat</span>
            <span className="font-black text-2xl text-brand">{hotScoreAvg ? hotScoreAvg.toFixed(1) : 'N/A'}</span>
          </div>
          <div className="flex justify-between items-center px-4 py-3">
            <span className="font-black text-xs uppercase tracking-widest text-black/50">Tension</span>
            <span className={`font-black text-xs uppercase px-2 py-1 text-white border-4 border-black ${tensionBg}`}>
              {tensionStatus || 'Normal'}
            </span>
          </div>
        </div>
      </div>

      {/* ── Network Topology HUD (bottom-left) ─── */}
      {networkTopology && (
        <div className="absolute bottom-14 left-4 z-10 w-60 border-8 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] bg-white font-mono">
          <button
            onClick={() => setShowTopologyLines(v => !v)}
            className="w-full text-left bg-black hover:bg-neutral-900 text-white px-4 py-3 border-b-4 border-brand flex items-center justify-between gap-2 cursor-pointer transition-colors duration-200"
          >
            <div className="flex items-center gap-2">
              <Network className="w-4 h-4 text-brand" strokeWidth={3} />
              <span className="font-black text-xs uppercase tracking-widest">Network Topology</span>
            </div>
            <span className={`text-[10px] font-black uppercase px-2 py-0.5 border ${
              showTopologyLines 
                ? 'bg-brand text-black border-brand' 
                : 'bg-transparent text-white/50 border-white/20'
            }`}>
              {showTopologyLines ? 'ON' : 'OFF'}
            </span>
          </button>
          <div className="divide-y-4 divide-black">
            <div className="flex justify-between px-4 py-2">
              <span className="font-black text-xs uppercase tracking-widest text-black/50">Total Edges</span>
              <span className="font-black text-xl">{networkTopology.total_edges.toLocaleString()}</span>
            </div>
            {networkTopology.density !== undefined && (
              <div className="flex justify-between px-4 py-2">
                <span className="font-black text-xs uppercase tracking-widest text-black/50">Density</span>
                <span className="font-black text-xl">{(networkTopology.density * 100).toFixed(1)}%</span>
              </div>
            )}
            {networkTopology.clustering_coefficient !== undefined && (
              <div className="flex justify-between px-4 py-2">
                <span className="font-black text-xs uppercase tracking-widest text-black/50">Echo Chamber</span>
                <span className="font-black text-xl">{networkTopology.clustering_coefficient.toFixed(2)}</span>
              </div>
            )}
            {networkTopology.avg_betweenness_centrality !== undefined && (
              <div className="flex justify-between px-4 py-2">
                <span className="font-black text-xs uppercase tracking-widest text-black/50">Info Brokers</span>
                <span className="font-black text-xl">{networkTopology.avg_betweenness_centrality.toFixed(1)}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Consolidated Monitor Sidebar (right) ─── */}
      {showMonitorPanel && (
        <div 
          className={`absolute top-14 right-4 bottom-14 bg-white border-8 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] flex flex-col overflow-hidden font-mono ${
            isDragging ? '' : 'transition-all duration-300'
          }`}
          style={{ width: `${panelWidth}px`, zIndex: 2147483646 }}
        >
          {/* Dynamic Drag Edge Resizer */}
          <div
            onMouseDown={handleMouseDown}
            className="absolute top-0 bottom-0 left-0 w-2 cursor-ew-resize bg-black hover:bg-brand transition-colors z-30 flex items-center justify-center select-none"
            title="Drag to resize panel"
          >
            <div className="w-[1.5px] h-8 bg-white/40 rounded-full" />
          </div>

          <div className="bg-black text-white px-4 py-3 border-b-4 border-brand flex items-center justify-between pl-6">
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-brand" strokeWidth={3} />
              <span className="font-black text-xs uppercase tracking-widest select-none">OASIS Monitor Panel</span>
            </div>
            <div className="flex items-center gap-3">
              <button 
                onClick={() => setPanelWidth(isPanelExpanded ? 400 : 1100)} 
                className="text-white hover:text-brand cursor-pointer transition-colors p-1"
                title={isPanelExpanded ? "Collapse panel to side" : "Expand panel to full-width side-by-side"}
              >
                {isPanelExpanded ? <Minimize2 className="w-5 h-5" strokeWidth={3} /> : <Maximize2 className="w-5 h-5" strokeWidth={3} />}
              </button>
              <button onClick={() => setShowMonitorPanel(false)} className="text-white hover:text-brand cursor-pointer transition-colors p-1">
                <X className="w-5 h-5" strokeWidth={3} />
              </button>
            </div>
          </div>

          {/* Tab bar header switcher shown ONLY when collapsed/single column mode */}
          {!isPanelExpanded && (
            <div className="flex border-b-4 border-black font-black text-[10px] uppercase bg-neutral-100 select-none pl-4">
              {['Seeds', "Eagle's Eye", 'Live Swarm'].map((tabName, index) => (
                <button
                  key={tabName}
                  onClick={() => scrollToColumn(index)}
                  className={`flex-1 py-2.5 text-center border-r-4 last:border-r-0 border-black cursor-pointer transition-all duration-200 ${
                    activeColumnIndex === index 
                      ? 'bg-brand text-black font-black' 
                      : 'bg-neutral-100 text-black/60 hover:bg-neutral-200'
                  }`}
                >
                  {tabName}
                </button>
              ))}
            </div>
          )}

          {/* Horizontally scrollable panel layout if collapsed, side-by-side if expanded */}
          <div 
            ref={scrollRef}
            onScroll={isPanelExpanded ? undefined : handleScroll}
            className={`flex-1 flex flex-row divide-x-8 divide-black select-none ${
              isPanelExpanded 
                ? 'overflow-hidden' 
                : 'overflow-x-auto scrollbar-thin select-text'
            }`}
            onWheel={e => e.stopPropagation()}
          >
            {/* Column 1: Controversy Seeds */}
            <div className={`flex flex-col h-full bg-white pl-2 ${
              isPanelExpanded ? 'flex-1 min-w-0' : 'w-full flex-shrink-0'
            }`}>
              <div className="bg-black text-white px-4 py-2 border-b-4 border-black flex items-center justify-between font-black text-xs uppercase tracking-widest select-none">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-brand" strokeWidth={3} />
                  <span>Controversy Seeds ({seedPosts.length})</span>
                </div>
              </div>
              {debateSeedsBody}
            </div>

            {/* Column 2: Eagle's Eye Insights */}
            <div className={`flex flex-col h-full bg-white ${
              isPanelExpanded ? 'flex-1 min-w-0' : 'w-full flex-shrink-0'
            }`}>
              <div className="bg-black text-white px-4 py-2 border-b-4 border-black flex items-center justify-between font-black text-xs uppercase tracking-widest select-none">
                <div className="flex items-center gap-2">
                  <Eye className="w-4 h-4 text-brand" strokeWidth={3} />
                  <span>Eagle's Eye Insights ({eagleEyeResults.length})</span>
                </div>
                <button 
                  onClick={() => setIsInterrogationModalOpen(true)}
                  className="bg-brand text-black hover:bg-white px-2 py-1 font-bold text-[10px] transition-colors"
                >
                  INTERROGATE
                </button>
              </div>
              {eaglesEyeBody}
            </div>

            {/* Column 3: Live Swarm Activity */}
            <div className={`flex flex-col h-full bg-white ${
              isPanelExpanded ? 'flex-1 min-w-0' : 'w-full flex-shrink-0'
            }`}>
              <div className="bg-black text-white px-4 py-2 border-b-4 border-black flex items-center justify-between font-black text-xs uppercase tracking-widest select-none">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-brand" strokeWidth={3} />
                  <span>Live Swarm Activity ({actions.length})</span>
                </div>
              </div>
              {liveSwarmBody}
            </div>
          </div>
        </div>
      )}

      {/* Drag-to-Reveal Handle when closed */}
      {!showMonitorPanel && (
        <div
          onMouseDown={(e) => {
            setShowMonitorPanel(true);
            setIsDragging(true);
            setPanelWidth(380);
            dragStartRef.current = { startX: e.clientX, startWidth: 380 };
          }}
          className="absolute top-1/2 right-0 -translate-y-1/2 z-20 cursor-ew-resize bg-black hover:bg-neutral-900 text-brand border-4 border-black hover:border-brand px-3 py-6 flex flex-col items-center gap-2 select-none shadow-[-4px_4px_0px_0px_rgba(0,0,0,1)] transition-all"
          style={{ borderRightWidth: '0' }}
        >
          <Activity className="w-5 h-5 animate-pulse" />
          <span className="font-black text-[9px] uppercase tracking-widest [writing-mode:vertical-lr] text-center select-none">
            Drag to Reveal Monitor
          </span>
        </div>
      )}

      {/* ── Controls hint ─── */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 pointer-events-none">
        <div className="bg-black text-white border-4 border-black px-6 py-2 flex items-center gap-3">
          <span className="w-2 h-2 bg-brand animate-pulse" />
          <span className="font-black text-xs uppercase tracking-widest">
            Drag Rotates &middot; Scroll Zooms &middot; Click Avatar &rarr; Intercept Telemetry
          </span>
        </div>
      </div>

      <EagleEyeInterrogationModal 
        isOpen={isInterrogationModalOpen} 
        onClose={() => setIsInterrogationModalOpen(false)} 
        initialAgentId={interrogationAgentId}
      />
    </div>
  );
}
