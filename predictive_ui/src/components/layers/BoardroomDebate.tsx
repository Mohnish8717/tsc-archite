import React, { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useGraph } from '@react-three/fiber';
import { PerspectiveCamera, Environment, OrbitControls, Html, useGLTF, useAnimations } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { SkeletonUtils } from 'three-stdlib';
import { Cpu, ShieldAlert, Zap, AlertCircle, CheckCircle2, XCircle, MinusCircle, Brain, TrendingUp, TrendingDown } from 'lucide-react';
import { usePipelineStore } from '../../store/usePipelineStore';
import { cleanPersonaName } from '../../utils/nameHelper';

// Error Boundary for the 3D Scene
// Global store to prevent agents from overlapping at the same POI
const occupiedPOIs = new Set<string>();
const agentPositions: { [key: string]: THREE.Vector3 } = {}; // For dynamic collision repulsion

class BoardroomErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() { return { hasError: true }; }
  render() {
    if (this.state.hasError) {
      return (
        <div className="w-full h-full flex flex-col items-center justify-center bg-black border-4 border-black font-mono p-8 text-center">
          <AlertCircle className="text-[#FF4500] w-12 h-12 mb-4" />
          <h1 className="text-white font-black uppercase tracking-tighter text-xl mb-2">Sim Engine Failure</h1>
          <p className="text-white/40 text-xs uppercase tracking-widest max-w-xs">Critical WebGL context loss or runtime exception in 3D layer.</p>
          <button onClick={() => window.location.reload()} className="mt-6 px-6 py-2 bg-[#FF4500] text-black font-black uppercase tracking-widest border-2 border-black hover:bg-white transition-all">Reinitialize</button>
        </div>
      );
    }
    return this.props.children;
  }
}

function ImportedAgent({ position: startPos, color, label, speaking, lookAt }: { position: [number, number, number], color: string, label: string, speaking: boolean, lookAt: [number, number, number] }) {
  const group = useRef<THREE.Group>(null);
  const { debateMessages } = usePipelineStore();
  const [currentPos, setCurrentPos] = React.useState(new THREE.Vector3(...startPos));
  const [targetPos, setTargetPos] = React.useState(new THREE.Vector3(...startPos));
  const [path, setPath] = React.useState<THREE.Vector3[]>([]);
  const [isMoving, setIsMoving] = React.useState(false);
  const [isSitting, setIsSitting] = React.useState(false);
  const [idleAnim, setIdleAnim] = React.useState('Idle');
  const [sitAnim, setSitAnim] = React.useState('Sit_Idle');

  // Claim initial position on mount
  useEffect(() => {
    const key = new THREE.Vector3(...startPos).toArray().join(',');
    occupiedPOIs.add(key);
    return () => { occupiedPOIs.delete(key); };
  }, [startPos]);

  // Load the character model
  const { scene, animations } = useGLTF('/models/character.glb');

  const clone = useMemo(() => {
    const clonedScene = SkeletonUtils.clone(scene);
    clonedScene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        mesh.material = (mesh.material as THREE.Material).clone();
        if (mesh.name.toLowerCase().includes('body')) {
          (mesh.material as THREE.MeshStandardMaterial).color.set(color);
          (mesh.material as THREE.MeshStandardMaterial).roughness = 1.0;
          (mesh.material as THREE.MeshStandardMaterial).metalness = 0.0;
        } else if (mesh.name.toLowerCase().includes('cap') || mesh.name.toLowerCase().includes('headphones')) {
          mesh.visible = false;
        }
        mesh.castShadow = true;
        mesh.receiveShadow = true;
      }
    });
    return clonedScene;
  }, [scene, color]);

  const { actions } = useAnimations(animations, group);

  // Exact coordinates from office.glb with manual NavMesh escape routes
  const POIS = useMemo(() => {
    const rawPOIs = [
      { pos: new THREE.Vector3(-3.16, 0, -4.43), type: 'chair', lookAt: new THREE.Vector3(-2.96, 0, -3.45), safePath: [new THREE.Vector3(-2.0, 0, -4.43), new THREE.Vector3(-2.0, 0, 0)] },
      { pos: new THREE.Vector3(-3.71, 0, -2.86), type: 'chair', lookAt: new THREE.Vector3(-2.96, 0, -3.45), safePath: [new THREE.Vector3(-2.0, 0, -2.86), new THREE.Vector3(-2.0, 0, 0)] },
      { pos: new THREE.Vector3(1.08, 0, -1.23), type: 'desk', lookAt: new THREE.Vector3(1.61, 0, -1.05), safePath: [new THREE.Vector3(1.08, 0, 0)] },
      { pos: new THREE.Vector3(1.57, 0, -3.68), type: 'desk', lookAt: new THREE.Vector3(1.39, 0, -3.15), safePath: [new THREE.Vector3(0.5, 0, -3.68), new THREE.Vector3(0.5, 0, 0)] },
      { pos: new THREE.Vector3(3.35, 0, -3.68), type: 'desk', lookAt: new THREE.Vector3(3.16, 0, -3.15), safePath: [new THREE.Vector3(4.2, 0, -3.68), new THREE.Vector3(4.2, 0, 0)] },
      { pos: new THREE.Vector3(3.10, 0, -1.27), type: 'desk', lookAt: new THREE.Vector3(2.57, 0, -1.45), safePath: [new THREE.Vector3(3.10, 0, 0)] },
      { pos: new THREE.Vector3(-4.13, 0, 4.47), type: 'desk', lookAt: new THREE.Vector3(-3.50, 0, 4.47), safePath: [new THREE.Vector3(-2.0, 0, 4.47), new THREE.Vector3(-2.0, 0, 0)] },
      { pos: new THREE.Vector3(0, 0, 0), type: 'standing', lookAt: new THREE.Vector3(0, 0, 1), safePath: [] },
    ];

    const DESK_PULLBACK = 0.1;
    return rawPOIs.map(poi => {
      if (poi.type === 'desk' && poi.lookAt) {
        const dir = poi.pos.clone().sub(poi.lookAt).normalize();
        const adjustedPos = poi.pos.clone().add(dir.multiplyScalar(DESK_PULLBACK));
        return { ...poi, pos: adjustedPos };
      }
      return poi;
    });
  }, []);

  // Compute full obstacle-free path when target changes
  useEffect(() => {
    const startPoi = POIS.find(p => p.pos.distanceTo(currentPos) < 0.1);
    const endPoi = POIS.find(p => p.pos.distanceTo(targetPos) < 0.1);

    if (startPoi && endPoi && startPoi !== endPoi && currentPos.distanceTo(targetPos) > 2.0) {
      const fullPath = [
        ...startPoi.safePath,
        new THREE.Vector3(0, 0, 0),
        ...[...endPoi.safePath].reverse(),
        endPoi.pos
      ];
      setPath(fullPath);
    } else {
      setPath([targetPos]);
    }
  }, [targetPos, POIS]);

  // Movement Logic
  useFrame((state, delta) => {
    if (speaking) {
      setIsMoving(false);
      setIsSitting(false);
      // Turn to face the boardroom center smoothly
      const center = new THREE.Vector3(0, 0, 0);
      const direction = center.clone().sub(currentPos).normalize();
      const targetRotation = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(direction.x, 0, direction.z).normalize()
      );
      if (group.current) group.current.quaternion.slerp(targetRotation, 0.1);
      return;
    }

    // Register position for collision repulsion
    agentPositions[label] = currentPos;

    const actualTarget = path.length > 0 ? path[0] : targetPos;

    if (currentPos.distanceTo(actualTarget) > 0.2) {
      setIsMoving(true);
      setIsSitting(false);

      const direction = actualTarget.clone().sub(currentPos).normalize();

      // Repel from other agents to avoid clipping
      const repulsion = new THREE.Vector3();
      Object.entries(agentPositions).forEach(([otherLabel, otherPos]) => {
        if (otherLabel !== label) {
          const dist = currentPos.distanceTo(otherPos);
          if (dist < 2.0 && dist > 0.01) {
            // Push away
            const push = currentPos.clone().sub(otherPos).normalize();
            repulsion.add(push.clone().multiplyScalar(0.1 / dist));

            // Add a perpendicular sidestep force to prevent head-on deadlocks
            const sidestep = new THREE.Vector3(-push.z, 0, push.x).multiplyScalar(0.15 / dist);
            repulsion.add(sidestep);
          }
        }
      });

      // Combine path direction with repulsion
      direction.add(repulsion).normalize();

      // Smoothly rotate toward walking direction
      const targetRotation = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(direction.x, 0, direction.z).normalize()
      );
      if (group.current) {
        group.current.quaternion.slerp(targetRotation, 0.15);
        const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(group.current.quaternion);
        currentPos.add(forward.multiplyScalar(delta * 1.8)); // Slightly faster walk
        group.current.position.copy(currentPos);
      }
    } else {
      if (path.length > 1) {
        // Reached current waypoint, pop it and move to next
        setPath(prev => prev.slice(1));
      } else {
        setIsMoving(false);
        // Snap to exact final position to avoid clipping
        currentPos.copy(targetPos);

        const atPoi = POIS.find(p => p.pos.distanceTo(targetPos) < 0.1);
        if (group.current) {
          group.current.position.copy(currentPos);

          // Turn to face the correct desk or table when arrived
          const lookTarget = atPoi?.lookAt || new THREE.Vector3(0, 0, 0);
          const lookDir = lookTarget.clone().sub(currentPos).normalize();
          if (lookDir.lengthSq() > 0.001) {
            const lookRotation = new THREE.Quaternion().setFromUnitVectors(
              new THREE.Vector3(0, 0, 1),
              new THREE.Vector3(lookDir.x, 0, lookDir.z).normalize()
            );
            group.current.quaternion.slerp(lookRotation, 0.1);
          }
        }

        if (atPoi && !isSitting) {
          if (atPoi.type === 'chair' || atPoi.type === 'desk') {
            setIsSitting(true);
          }
        }
      }
    }
  });

  // Intentional movement decisions (Context-aware)
  useEffect(() => {
    if (speaking) return;
    const interval = setInterval(() => {
      // 30% chance to move to a new POI every 5 seconds to keep the room dynamic
      if (Math.random() < 0.30) {
        const currentKey = targetPos.toArray().join(',');
        const availablePOIs = POIS.filter(p => !occupiedPOIs.has(p.pos.toArray().join(',')));

        if (availablePOIs.length > 0) {
          const nextPoi = availablePOIs[Math.floor(Math.random() * availablePOIs.length)];
          const nextKey = nextPoi.pos.toArray().join(',');

          occupiedPOIs.delete(currentKey);
          occupiedPOIs.add(nextKey);
          setTargetPos(nextPoi.pos);
        }
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [speaking, POIS, targetPos]);

  // Determine meaning-driven animations based on context
  const { activeSpeaker } = usePipelineStore();
  const isSomeoneElseSpeaking = activeSpeaker !== null && activeSpeaker !== label;

  useEffect(() => {
    // Evaluate context based on targetPos (which is reactive) instead of currentPos
    const atPoi = POIS.find(p => p.pos.distanceTo(targetPos) < 0.1);

    if (atPoi?.type === 'desk') {
      // The Sit_Work animation is actually a standing animation with a clipboard,
      // and 'Sit' is a bouncing animation. The only valid sitting animation is 'Sit_Idle'.
      setSitAnim('Sit_Idle');
    } else {
      // Cafe chairs are just idle
      setSitAnim('Sit_Idle');
    }

    // The 'Listen' animation in this model looks too much like 'Happy' / bouncing.
    // Default to 'Idle' for a more professional boardroom stance.
    setIdleAnim('Idle');
  }, [targetPos, POIS, isSomeoneElseSpeaking]);

  const lastMessage = useMemo(() => {
    return debateMessages.filter(m => m.sender === label).pop()?.text || "...";
  }, [debateMessages, label]);

  // High-Fidelity Animation Controller
  useEffect(() => {
    if (!actions) return;

    // Stop all currently playing actions first
    Object.values(actions).forEach(a => a?.fadeOut(0.3));

    const talkAction = actions['Talk'] || actions['Speak'] || actions['Idle'];
    const walkAction = actions['Walk'] || actions['Walking'] || actions['Idle'];
    const currentIdleAction = actions[idleAnim] || actions['Idle'];
    const currentSitAction = actions[sitAnim] || actions['Sit_Idle'] || actions['Idle'];

    if (speaking) {
      talkAction?.reset().fadeIn(0.2).play();
    } else if (isMoving) {
      if (walkAction) {
        walkAction.setEffectiveTimeScale(1.4);
        walkAction.reset().fadeIn(0.2).play();
      }
    } else if (isSitting) {
      currentSitAction?.reset().fadeIn(0.3).play();
    } else {
      currentIdleAction?.reset().fadeIn(0.4).play();
    }
  }, [speaking, isMoving, isSitting, actions, idleAnim, sitAnim]);

  return (
    <group position={startPos} ref={group}>
      <primitive object={clone} />

      {/* Brutalist Speech Bubble */}
      {speaking && (
        <Html position={[0, 2.8, 0]} center zIndexRange={[100, 0]}>
          <div className="bg-white border-4 border-black shadow-neo-black p-3 w-64 max-w-xs relative animate-bounce-slow font-['Fira_Code',monospace]">
            {/* Speech bubble pointer */}
            <div className="absolute -bottom-3 left-1/2 -translate-x-1/2 w-0 h-0 border-l-[10px] border-l-transparent border-r-[10px] border-r-transparent border-t-[12px] border-t-black" />
            <div className="absolute -bottom-[6px] left-1/2 -translate-x-1/2 w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[8px] border-t-white" />

            <div className="font-black text-[10px] uppercase tracking-widest text-[#FF4500] mb-1 border-b-2 border-black pb-1">
              {label}
            </div>
            <div className="font-bold text-xs text-black leading-tight line-clamp-3">
              {lastMessage}
            </div>
          </div>
        </Html>
      )}

      {/* Floating Pill Label (when NOT speaking) */}
      {!speaking && (
        <Html position={[0, 2.0, 0]} center zIndexRange={[50, 0]}>
          <div className="px-3 py-1 bg-black border-2 border-white shadow-neo-black text-[10px] font-black uppercase tracking-wider whitespace-nowrap text-white font-['Fira_Code',monospace]">
            {label}
          </div>
        </Html>
      )}

      {/* Active Speaker Ring */}
      {speaking && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.05, 0]}>
          <ringGeometry args={[0.7, 0.9, 32]} />
          <meshBasicMaterial color={color} transparent opacity={0.8} />
        </mesh>
      )}
    </group>
  );
}

function ImportedOffice() {
  const { scene } = useGLTF('/models/office.glb');

  const clone = useMemo(() => {
    const clonedScene = scene.clone();
    clonedScene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        child.receiveShadow = true;
        child.castShadow = true;
        const mat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial;
        if (mat) {
          if (mat.emissive) mat.emissiveIntensity = 0;
          mat.roughness = 0.6;
          mat.metalness = 0.1;

          const nodeName = child.name.toLowerCase();

          if (nodeName.includes('chair') || nodeName.includes('seat')) {
            mat.color.set('#34495e');
          } else if (nodeName.includes('floor')) {
            mat.color.set('#2c3e50');
          } else if (nodeName.includes('wall')) {
            mat.color.set('#f5f5f5');
          } else if (nodeName.includes('table') || nodeName.includes('desk')) {
            mat.color.set('#2c3e50');
          } else if (nodeName.includes('cupboard') || nodeName.includes('cabinet') || nodeName.includes('shelf') || nodeName.includes('wood')) {
            mat.color.set('#5d4037');
          }
        }
      }
    });
    return clonedScene;
  }, [scene]);

  return (
    <primitive object={clone} position={[0, 0, 0]} />
  );
}

function Boardroom3D() {
  const { personas, spawnedAgents, activeSpeaker } = usePipelineStore();

  const dynamicPersonas = useMemo(() => {
    const list = [
      { id: 'CEO', name: 'CEO', role: 'Chief Executive Officer' },
      { id: 'CTO', name: 'CTO', role: 'Chief Technology Officer' },
      { id: 'CISO', name: 'CISO', role: 'Chief Information Security Officer' },
      { id: 'CMO', name: 'CMO', role: 'Chief Marketing/Medical Officer' },
      { id: 'CFO', name: 'CFO', role: 'Chief Financial Officer' },
      { id: 'CPO', name: 'CPO', role: 'Chief Product Officer' },
      { id: 'Legal', name: 'Legal', role: 'General Counsel' },
      { id: 'Data', name: 'Data', role: 'Head of Data & ML' },
      { id: 'Sales', name: 'Sales', role: 'Head of Sales' },
      { id: 'CS', name: 'CS', role: 'Head of Customer Success' },
    ];

    const colors = [
      '#FF4500', // CEO
      '#3B82F6', // CTO
      '#8B5CF6', // CISO
      '#EC4899', // CMO
      '#10B981', // CFO
      '#F59E0B', // CPO
      '#F97316', // Legal
      '#14B8A6', // Data
      '#6366F1', // Sales
      '#A855F7', // CS
    ];

    const startPositions: [number, number, number][] = [
      [1.5, 0, 1.5],
      [-2, 0, 2.5],
      [-2.5, 0, -1],
      [0.5, 0, -2],
      [2.5, 0, -1.5],
      [-1.5, 0, 1.0],
      [1.0, 0, -1.0],
      [-0.5, 0, 2.0],
      [2.0, 0, 0.5],
      [-2.0, 0, -2.0],
    ];

    return list.map((p, index) => ({
      ...p,
      color: colors[index % colors.length],
      position: startPositions[index % startPositions.length],
    }));
  }, []);

  return (
    <div className="w-full h-full relative bg-black">
      <Canvas shadows dpr={[1, 2]}>
        {/* Isometric-style perspective */}
        <PerspectiveCamera makeDefault position={[8, 8, 8]} fov={35} />
        <OrbitControls enablePan={false} maxPolarAngle={Math.PI / 2.2} minPolarAngle={0.1} />

        {/* Bright, clean lighting to match "The Delegation" aesthetic on the 3D models */}
        <ambientLight intensity={0.7} />
        <directionalLight
          position={[10, 20, 10]}
          intensity={1.0}
          castShadow
          shadow-mapSize={[2048, 2048]}
          shadow-camera-far={50}
          shadow-camera-left={-10}
          shadow-camera-right={10}
          shadow-camera-top={10}
          shadow-camera-bottom={-10}
        />
        <pointLight position={[0, 5, 0]} intensity={0.5} color="#ffffff" />

        <Environment preset="city" />

        <group position={[0, 0, 0]}>
          <ImportedOffice />
          {dynamicPersonas.map((exec) => (
            <ImportedAgent
              key={exec.id}
              label={exec.name}
              position={exec.position as [number, number, number]}
              color={exec.color}
              speaking={activeSpeaker === exec.id}
              lookAt={[0, 0.5, 0]}
            />
          ))}
          {/* Holographic Centerpiece */}
          <mesh position={[0, 1.0, 0]}>
            <sphereGeometry args={[0.3, 32, 32]} />
            <meshPhysicalMaterial
              color="#FF4500"
              wireframe
              emissive="#FF4500"
              emissiveIntensity={2}
            />
          </mesh>
        </group>

        <EffectComposer disableNormalPass>
          <Bloom luminanceThreshold={0.9} luminanceSmoothing={0.9} height={300} intensity={0.15} />
        </EffectComposer>
      </Canvas>
    </div>
  );
}

function DebateChat() {
  const { debateMessages, consensusResult } = usePipelineStore();

  const verdictColor = consensusResult?.overall_verdict === 'APPROVE'
    ? { bg: 'bg-green-500', text: 'text-green-600', border: 'border-green-500' }
    : consensusResult?.overall_verdict === 'REJECT'
    ? { bg: 'bg-red-500', text: 'text-red-600', border: 'border-red-500' }
    : { bg: 'bg-yellow-400', text: 'text-yellow-600', border: 'border-yellow-400' };

  return (
    <div className="w-full h-full flex flex-col bg-white border-l-8 border-black font-mono">
      {/* Header */}
      <div className="bg-black text-white px-6 py-4 border-b-4 border-[#FF4500] flex items-center justify-between flex-none">
        <div className="flex items-center gap-3">
          <Cpu className="w-4 h-4 text-[#FF4500]" strokeWidth={3} />
          <h2 className="font-black text-sm uppercase tracking-widest">Adversarial Debate Log</h2>
        </div>
      </div>

      {/* Consensus Result panel */}
      {consensusResult && (
        <div className="flex-none border-b-4 border-black">
          {/* Verdict banner */}
          <div className={`flex items-center gap-3 px-5 py-3 ${verdictColor.bg} border-b-2 border-black`}>
            {consensusResult.overall_verdict === 'APPROVE'
              ? <CheckCircle2 className="w-5 h-5 text-white" strokeWidth={3} />
              : consensusResult.overall_verdict === 'REJECT'
              ? <XCircle className="w-5 h-5 text-white" strokeWidth={3} />
              : <MinusCircle className="w-5 h-5 text-white" strokeWidth={3} />}
            <span className="font-black text-sm uppercase text-white tracking-widest flex-1">
              Boardroom Consensus: {consensusResult.overall_verdict}
            </span>
            <span className="font-black text-xs text-white/80">
              {(consensusResult.approval_confidence * 100).toFixed(0)}% Confidence
            </span>
          </div>
          {/* Confidence bar */}
          <div className="h-2 bg-black/10">
            <div className="h-full transition-all duration-700" style={{ width: `${consensusResult.approval_confidence * 100}%`, backgroundColor: consensusResult.overall_verdict === 'APPROVE' ? '#22C55E' : consensusResult.overall_verdict === 'REJECT' ? '#EF4444' : '#F59E0B' }} />
          </div>
          {/* Stakeholder verdicts */}
          {Object.keys(consensusResult.stakeholder_verdicts).length > 0 && (
            <div className="grid divide-x-2 divide-black border-b-2 border-black" style={{ gridTemplateColumns: `repeat(${Object.keys(consensusResult.stakeholder_verdicts).length}, minmax(0, 1fr))` }}>
              {Object.entries(consensusResult.stakeholder_verdicts).map(([name, verdict]) => {
                const cleanName = cleanPersonaName(name);
                return (
                  <div key={name} className="p-2 text-center overflow-hidden">
                    <div className="font-black text-[9px] uppercase tracking-widest text-black/40 mb-1 truncate" title={name}>{cleanName}</div>
                    <span className={`font-black text-xs uppercase ${verdict === 'APPROVE' ? 'text-green-600' : verdict === 'REJECT' ? 'text-red-600' : 'text-yellow-600'}`}>
                      {verdict}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
          {/* Behavioral insights */}
          {consensusResult.behavioral_insights.length > 0 && (
            <div className="px-4 py-2 space-y-1">
              <div className="flex items-center gap-1.5 mb-1">
                <Brain className="w-3 h-3 text-brand" strokeWidth={3} />
                <span className="font-black text-xs uppercase tracking-widest text-brand">Behavioral Insights</span>
              </div>
              {consensusResult.behavioral_insights.slice(0, 2).map((insight: string, i: number) => (
                <p key={i} className="text-xs font-bold text-black/60 border-l-2 border-brand pl-2">{insight}</p>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {debateMessages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-4">
            <div className="w-12 h-12 bg-[#FF4500] border-4 border-black flex items-center justify-center animate-pulse">
              <Zap size={20} className="text-black" strokeWidth={3} />
            </div>
            <p className="font-black text-xs uppercase tracking-widest text-black/40">Awaiting debate data...</p>
          </div>
        ) : debateMessages.map((msg) => (
          <div
            key={msg.id}
            className={`p-3 border-4 ${msg.type === 'challenge' ? 'border-[#FF4500] bg-[#FF4500]/10' : 'border-black bg-white'}`}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-xs font-black uppercase tracking-widest ${msg.type === 'challenge' ? 'text-[#FF4500]' : 'text-black'}`}>
                {msg.sender}
              </span>
              {msg.type === 'challenge' && <ShieldAlert className="w-3 h-3 text-[#FF4500]" strokeWidth={3} />}
            </div>
            <p className="text-sm font-bold text-black/80">{msg.text}</p>
          </div>
        ))}
      </div>

      {/* Key Quotes ticker */}
      {consensusResult?.simulation_key_quotes?.length > 0 && (
        <div className="flex-none border-t-4 border-black bg-black/5 px-4 py-2">
          <div className="font-black text-xs uppercase tracking-widest text-brand mb-1">Key Quotes</div>
          <div className="space-y-1 max-h-20 overflow-y-auto">
            {consensusResult.simulation_key_quotes.slice(0, 3).map((q: string, i: number) => (
              <p key={i} className="text-xs font-bold text-black/60 italic">"{q.slice(0, 100)}{q.length > 100 ? '…' : ''}"</p>
            ))}
          </div>
        </div>
      )}

      {/* Status bar */}
      <div className="border-t-4 border-black p-4 flex items-center gap-3 bg-black/5 flex-none">
        <div className="w-2 h-2 bg-[#FF4500] animate-pulse" />
        <span className="text-xs font-black uppercase tracking-widest text-black/50">
          Logit-bias anti-sycophancy protocol active
        </span>
      </div>
    </div>
  );
}

export default function BoardroomDebate() {
  const { isConnected, simulationConfig } = usePipelineStore();
  return (
    <div className="w-full h-full flex" style={{ paddingTop: '72px' }}>
      {/* 3D Canvas */}
      <div className="flex-[2] relative overflow-hidden border-r-8 border-black">
        <BoardroomErrorBoundary>
          <Boardroom3D />
        </BoardroomErrorBoundary>
        <div className={`absolute top-4 left-4 border-4 px-4 py-2 text-xs font-mono font-black uppercase tracking-widest flex items-center gap-2 shadow-neo-black ${isConnected ? 'bg-black text-white border-[#FF4500]' : 'bg-white text-black border-black'}`}>
          <span className={`w-2 h-2 ${isConnected ? 'bg-[#FF4500] animate-pulse' : 'bg-black opacity-20'}`} />
          {isConnected ? 'Hindsight Link Active' : 'Bridge Offline'}
        </div>
        
        {/* Feature Discovery Banner */}
        {simulationConfig?.feature_title && (
          <div className="absolute bottom-4 left-4 max-w-lg border-4 border-black bg-white shadow-neo-black p-4 font-mono pointer-events-none">
            <div className="flex items-center gap-2 mb-2">
              <Cpu className="w-4 h-4 text-[#FF4500]" strokeWidth={3} />
              <div className="font-black text-[10px] uppercase tracking-widest text-[#FF4500]">
                Boardroom Agenda Item / Feature Discovery
              </div>
            </div>
            <div className="font-black text-lg text-black leading-tight border-b-2 border-black pb-2 mb-2">
              {simulationConfig.feature_title}
            </div>
            {simulationConfig.feature_description && (
              <div className="text-xs font-bold text-black/70 leading-relaxed max-h-32 overflow-y-auto pr-2 pointer-events-auto">
                {simulationConfig.feature_description}
              </div>
            )}
          </div>
        )}
      </div>
      {/* Chat */}
      <div className="flex-1">
        <DebateChat />
      </div>
    </div>
  );
}

// Preload the specific imported models
useGLTF.preload('/models/office.glb');
useGLTF.preload('/models/character.glb');
