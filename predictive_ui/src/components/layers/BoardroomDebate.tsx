import React, { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useGraph } from '@react-three/fiber';
import { PerspectiveCamera, Environment, OrbitControls, Html, useGLTF, useAnimations } from '@react-three/drei';
// SSAO removed — clean lighting is sufficient
import * as THREE from 'three';
import { SkeletonUtils } from 'three-stdlib';
import { Cpu, ShieldAlert, Zap, AlertCircle, CheckCircle2, XCircle, MinusCircle, Brain, TrendingUp, TrendingDown } from 'lucide-react';
import { usePipelineStore } from '../../store/usePipelineStore';
import { cleanPersonaName } from '../../utils/nameHelper';

// Error Boundary for the 3D Scene
// Global store to prevent agents from overlapping at the same POI
const occupiedPOIs = new Set<string>();
const agentPositions: { [key: string]: THREE.Vector3 } = {}; // For dynamic collision repulsion
const activeRoamers = new Set<string>(); // Global roamer slot system (max 3)

// Exact coordinates from office.glb with manual NavMesh escape routes
const BOARDROOM_POIS = [
  { pos: new THREE.Vector3(-3.16, 0, -4.43), type: 'chair', lookAt: new THREE.Vector3(-2.96, 0, -3.45), safePath: [new THREE.Vector3(-2.0, 0, -4.43), new THREE.Vector3(-1.8, 0, 0)] },
  { pos: new THREE.Vector3(-3.71, 0, -2.86), type: 'chair', lookAt: new THREE.Vector3(-2.96, 0, -3.45), safePath: [new THREE.Vector3(-2.0, 0, -2.86), new THREE.Vector3(-1.8, 0, 0)] },
  { pos: new THREE.Vector3(1.08, 0, -1.23), type: 'desk', lookAt: new THREE.Vector3(1.61, 0, -1.05), safePath: [new THREE.Vector3(1.08, 0, 0)] },
  { pos: new THREE.Vector3(1.57, 0, -3.68), type: 'desk', lookAt: new THREE.Vector3(1.39, 0, -3.15), safePath: [new THREE.Vector3(0.5, 0, -3.68), new THREE.Vector3(0.5, 0, 0)] },
  { pos: new THREE.Vector3(3.35, 0, -3.68), type: 'desk', lookAt: new THREE.Vector3(3.16, 0, -3.15), safePath: [new THREE.Vector3(4.2, 0, -3.68), new THREE.Vector3(4.2, 0, 0)] },
  { pos: new THREE.Vector3(3.10, 0, -1.27), type: 'desk', lookAt: new THREE.Vector3(2.57, 0, -1.45), safePath: [new THREE.Vector3(3.10, 0, 0)] },
  { pos: new THREE.Vector3(-4.13, 0, 4.47), type: 'desk', lookAt: new THREE.Vector3(-3.50, 0, 4.47), safePath: [new THREE.Vector3(-2.0, 0, 4.47), new THREE.Vector3(-1.8, 0, 0)] },
  { pos: new THREE.Vector3(-1.8, 0, 1.8), type: 'standing', lookAt: new THREE.Vector3(0, 0.5, 0), safePath: [] },
  { pos: new THREE.Vector3(-3.6, 0, 0.9), type: 'sofa', lookAt: new THREE.Vector3(0, 0, 0.9), safePath: [new THREE.Vector3(-2.0, 0, 0.9), new THREE.Vector3(-1.8, 0, 0)] }, // Sofa Seat 1 (Aligned perfectly on sofa cushions)
  { pos: new THREE.Vector3(-3.6, 0, 1.9), type: 'sofa', lookAt: new THREE.Vector3(0, 0, 1.9), safePath: [new THREE.Vector3(-2.0, 0, 1.9), new THREE.Vector3(-1.8, 0, 0)] }, // Sofa Seat 2 (Aligned perfectly on sofa cushions)

  // 4 New Standing POIs to unlock movement (Total 14 POIs for 10 agents = 4 vacant slots always)
  { pos: new THREE.Vector3(0.0, 0, 2.5), type: 'standing', lookAt: new THREE.Vector3(0, 0.5, 0), safePath: [] },
  { pos: new THREE.Vector3(-2.0, 0, -1.0), type: 'standing', lookAt: new THREE.Vector3(0, 0.5, 0), safePath: [] },
  { pos: new THREE.Vector3(2.0, 0, 2.0), type: 'standing', lookAt: new THREE.Vector3(-1.0, 0.5, 0), safePath: [] },
  { pos: new THREE.Vector3(2.5, 0, 0.0), type: 'standing', lookAt: new THREE.Vector3(0, 0.5, 0), safePath: [] },
];

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

function ImportedAgent({ startPos, targetPosProp, color, label, speaking }: { startPos: [number, number, number], targetPosProp: [number, number, number], color: string, label: string, speaking: boolean }) {
  const group = useRef<THREE.Group>(null);
  const { debateMessages } = usePipelineStore();
  const [currentPos, setCurrentPos] = React.useState(new THREE.Vector3(...startPos));
  const [targetPos, setTargetPos] = React.useState(new THREE.Vector3(...startPos));
  const [path, setPath] = React.useState<THREE.Vector3[]>([]);
  const [isMoving, setIsMoving] = React.useState(false);

  // Dynamic initial sitting state based on assigned starting POI type!
  const startSitting = useMemo(() => {
    const initialPoi = BOARDROOM_POIS.find(p => p.pos.distanceTo(new THREE.Vector3(...startPos)) < 0.1);
    return initialPoi ? (initialPoi.type === 'chair' || initialPoi.type === 'desk' || initialPoi.type === 'sofa') : false;
  }, [startPos]);

  const [isSitting, setIsSitting] = React.useState(startSitting);
  const [idleAnim, setIdleAnim] = React.useState('Idle');
  const [sitAnim, setSitAnim] = React.useState('Sit_Idle');

  // Sync positions dynamically when targetPosProp updates from the centralized scheduler
  useEffect(() => {
    const newTarget = new THREE.Vector3(...targetPosProp);
    setTargetPos(newTarget);
    setIsSitting(false); // Stand up to move
  }, [targetPosProp]);

  // Sync spawn position if startPos changes (e.g. after initial sofa POI resolution)
  useEffect(() => {
    const newPos = new THREE.Vector3(...startPos);
    setCurrentPos(newPos);
    if (group.current) group.current.position.copy(newPos);
  }, [startPos]);

  // Claim initial starting position on mount
  useEffect(() => {
    const key = new THREE.Vector3(...startPos).toArray().join(',');
    occupiedPOIs.add(key);
    return () => {
      occupiedPOIs.delete(key);
      activeRoamers.delete(label);
    };
  }, [startPos, label]);

  // Load the character model
  const { scene, animations } = useGLTF('/models/character.glb');

  const clone = useMemo(() => {
    const clonedScene = SkeletonUtils.clone(scene);

    // Hash for deterministic accessories based on label
    let hash = 0;
    for (let i = 0; i < label.length; i++) hash += label.charCodeAt(i);
    const accessoryType = hash % 3; // 0 = none, 1 = cap, 2 = headphones

    clonedScene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        const oldMat = mesh.material as THREE.MeshStandardMaterial;

        // Clean satin vinyl — no outlines, no glow, shape speaks for itself
        const newMat = new THREE.MeshStandardMaterial({
          color: oldMat?.color,
          map: oldMat?.map,
          roughness: 0.28,
          metalness: 0.18
        });
        mesh.material = newMat;

        const nameLower = mesh.name.toLowerCase();
        if (nameLower.includes('body') || nameLower.includes('skin') || nameLower.includes('torso') || nameLower.includes('arm') || nameLower.includes('leg')) {
          newMat.color.set('#ECEEF0'); // Soft warm white
          newMat.roughness = 0.55;  // Matte-satin, zero metalness = no glow
          newMat.metalness = 0.0;
        } else if (nameLower.includes('cap')) {
          mesh.visible = accessoryType === 1;
          newMat.color.set(color);
          newMat.roughness = 0.6;
          newMat.metalness = 0.0;
        } else if (nameLower.includes('headphones')) {
          mesh.visible = accessoryType === 2;
          newMat.color.set(color);
          newMat.roughness = 0.4;
          newMat.metalness = 0.3; // Slight sheen only on headphones
        }

        mesh.castShadow = true;
        mesh.receiveShadow = true;
      }
    });

    // Fix accessory rigging: attach them to the head bone so they move correctly
    const headBone = clonedScene.getObjectByName('head');
    const spineBone = clonedScene.getObjectByName('spine');

    if (headBone) {
      const capMesh = clonedScene.getObjectByName('cap');
      const hpMesh = clonedScene.getObjectByName('headphones');
      if (capMesh) headBone.attach(capMesh);
      if (hpMesh) headBone.attach(hpMesh);

      // Procedural Detailing: Stylish Glasses
      if (hash % 4 === 0) {
        const glasses = new THREE.Group();
        const lensMat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.1, metalness: 0.8 });
        const lensGeom = new THREE.BoxGeometry(0.12, 0.05, 0.02);
        const lLens = new THREE.Mesh(lensGeom, lensMat); lLens.position.set(-0.07, 0, 0);
        const rLens = new THREE.Mesh(lensGeom, lensMat); rLens.position.set(0.07, 0, 0);
        const bridge = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.01, 0.01), lensMat);
        glasses.add(lLens, rLens, bridge);
        glasses.position.set(0, 0.1, 0.12); // Approximate face offset
        headBone.add(glasses);
      }

      // Procedural Detailing: Neo-Brutalist Halo
      if (hash % 4 === 1) {
        const halo = new THREE.Mesh(
          new THREE.TorusGeometry(0.15, 0.02, 8, 8),
          new THREE.MeshBasicMaterial({ color: color, wireframe: true })
        );
        halo.position.set(0, 0.3, 0);
        halo.rotation.x = Math.PI / 2;
        headBone.add(halo);
      }
    }

    if (spineBone) {
      // Procedural Detailing: Bowtie
      if (hash % 4 === 2) {
        const bowtie = new THREE.Group();
        const tieMat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.8 });
        const tieGeom = new THREE.ConeGeometry(0.05, 0.08, 3);
        const lTie = new THREE.Mesh(tieGeom, tieMat); lTie.rotation.z = Math.PI / 2; lTie.position.set(-0.04, 0, 0);
        const rTie = new THREE.Mesh(tieGeom, tieMat); rTie.rotation.z = -Math.PI / 2; rTie.position.set(0.04, 0, 0);
        const knot = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.03, 0.04), tieMat);
        bowtie.add(lTie, rTie, knot);
        bowtie.position.set(0, 0.15, 0.12); // Approximate chest offset
        spineBone.add(bowtie);
      }

      // Procedural Detailing: Brutalist Chest Plate / Badge
      if (hash % 4 === 3) {
        const badge = new THREE.Mesh(
          new THREE.OctahedronGeometry(0.06),
          new THREE.MeshStandardMaterial({ color: color, flatShading: true })
        );
        badge.position.set(0, 0.1, 0.15); // Approximate chest offset
        spineBone.add(badge);
      }
    }

    return clonedScene;
  }, [scene, color, label]);

  const { actions } = useAnimations(animations, group);

  // Exact coordinates from office.glb with manual NavMesh escape routes
  const POIS = useMemo(() => {
    const rawPOIs = [
      { pos: new THREE.Vector3(-3.16, 0, -4.43), type: 'chair', lookAt: new THREE.Vector3(-2.96, 0, -3.45), safePath: [new THREE.Vector3(-2.0, 0, -4.43), new THREE.Vector3(-1.8, 0, 0)] },
      { pos: new THREE.Vector3(-3.71, 0, -2.86), type: 'chair', lookAt: new THREE.Vector3(-2.96, 0, -3.45), safePath: [new THREE.Vector3(-2.0, 0, -2.86), new THREE.Vector3(-1.8, 0, 0)] },
      { pos: new THREE.Vector3(1.08, 0, -1.23), type: 'desk', lookAt: new THREE.Vector3(1.61, 0, -1.05), safePath: [new THREE.Vector3(1.08, 0, 0)] },
      { pos: new THREE.Vector3(1.57, 0, -3.68), type: 'desk', lookAt: new THREE.Vector3(1.39, 0, -3.15), safePath: [new THREE.Vector3(0.5, 0, -3.68), new THREE.Vector3(0.5, 0, 0)] },
      { pos: new THREE.Vector3(3.35, 0, -3.68), type: 'desk', lookAt: new THREE.Vector3(3.16, 0, -3.15), safePath: [new THREE.Vector3(4.2, 0, -3.68), new THREE.Vector3(4.2, 0, 0)] },
      { pos: new THREE.Vector3(3.10, 0, -1.27), type: 'desk', lookAt: new THREE.Vector3(2.57, 0, -1.45), safePath: [new THREE.Vector3(3.10, 0, 0)] },
      { pos: new THREE.Vector3(-4.13, 0, 4.47), type: 'desk', lookAt: new THREE.Vector3(-3.50, 0, 4.47), safePath: [new THREE.Vector3(-2.0, 0, 4.47), new THREE.Vector3(-1.8, 0, 0)] },
      { pos: new THREE.Vector3(-1.8, 0, 1.8), type: 'standing', lookAt: new THREE.Vector3(0, 0.5, 0), safePath: [] }, // Moved from (0,0,0) to open lounge walkway
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
    const startPoi = BOARDROOM_POIS.find(p => p.pos.distanceTo(currentPos) < 0.1);
    const endPoi = BOARDROOM_POIS.find(p => p.pos.distanceTo(targetPos) < 0.1);

    if (startPoi && endPoi && startPoi !== endPoi && currentPos.distanceTo(targetPos) > 2.0) {
      const fullPath = [
        ...startPoi.safePath,
        new THREE.Vector3(-1.8, 0, 0), // Use wide open walkway side corridor instead of centerpiece
        ...[...endPoi.safePath].reverse(),
        endPoi.pos
      ];
      setPath(fullPath);
    } else {
      setPath([targetPos]);
    }
  }, [targetPos, currentPos]);

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

        // Strict boundary clamp to prevent agents from ever crossing the room borders / void
        currentPos.x = THREE.MathUtils.clamp(currentPos.x, -4.5, 4.3);
        currentPos.z = THREE.MathUtils.clamp(currentPos.z, -4.9, 4.9);

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

        const atPoi = BOARDROOM_POIS.find(p => p.pos.distanceTo(targetPos) < 0.1);
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
          if (atPoi.type === 'chair' || atPoi.type === 'desk' || atPoi.type === 'sofa') {
            setIsSitting(true);
            activeRoamers.delete(label); // Arrived! Release our roaming slot
          }
        }
      }
    }
  });

  // Centralized scheduler manages movement triggers now

  // Determine meaning-driven animations based on context
  const { activeSpeaker } = usePipelineStore();
  const isSomeoneElseSpeaking = activeSpeaker !== null && activeSpeaker !== label;

  useEffect(() => {
    // Evaluate context based on targetPos (which is reactive) instead of currentPos
    const atPoi = BOARDROOM_POIS.find(p => p.pos.distanceTo(targetPos) < 0.1);

    if (atPoi?.type === 'desk' || atPoi?.type === 'chair' || atPoi?.type === 'sofa') {
      setSitAnim('Sit_Idle');
    } else {
      setSitAnim('Sit_Idle');
    }

    setIdleAnim('Idle');
  }, [targetPos, isSomeoneElseSpeaking]);

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

      {/* Active Speaker Ring (Neo-brutalist octagon) */}
      {speaking && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.05, 0]}>
          <ringGeometry args={[0.7, 0.9, 8]} />
          <meshBasicMaterial color={color} />
        </mesh>
      )}
    </group>
  );
}

function ImportedOffice({ onLoad }: { onLoad: () => void }) {
  const { scene } = useGLTF('/models/office.glb');

  useEffect(() => {
    let sofaMesh: THREE.Mesh | null = null;
    scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh && child.name.toLowerCase().includes('sofa')) {
        sofaMesh = child as THREE.Mesh;
      }
    });

    if (sofaMesh) {
      const mesh = sofaMesh as THREE.Mesh;

      // 1. Get world position and rotation of the physical sofa mesh
      const worldPos = new THREE.Vector3();
      mesh.getWorldPosition(worldPos);
      const worldQuat = new THREE.Quaternion();
      mesh.getWorldQuaternion(worldQuat);

      // 2. Compute local bounding box to find the exact dimensions
      mesh.geometry.computeBoundingBox();
      const bbox = mesh.geometry.boundingBox;
      if (bbox) {
        const localSize = new THREE.Vector3();
        bbox.getSize(localSize);

        // Scale dimensions based on mesh scale
        const scale = mesh.scale;
        localSize.x *= scale.x;
        localSize.y *= scale.y;
        localSize.z *= scale.z;

        // 3. Find seating axis (typically Z axis for sofa in this model)
        const isZAward = localSize.z > localSize.x;
        const longAxisLen = isZAward ? localSize.z : localSize.x;

        // Distribute 2 seats on the left and right cushions along the long axis
        const offset = longAxisLen * 0.22;
        const seatOffset1 = isZAward ? new THREE.Vector3(0, 0.45, -offset) : new THREE.Vector3(-offset, 0.45, 0);
        const seatOffset2 = isZAward ? new THREE.Vector3(0, 0.45, offset) : new THREE.Vector3(offset, 0.45, 0);

        // 4. Transform local seat positions to world coordinates
        const worldSeat1 = seatOffset1.clone().applyQuaternion(worldQuat).add(worldPos);
        const worldSeat2 = seatOffset2.clone().applyQuaternion(worldQuat).add(worldPos);

        // 5. Determine the sofa's natural world forward direction (perpendicular to backrest)
        const localForward = isZAward ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1);
        const worldForward = localForward.clone().applyQuaternion(worldQuat).normalize();

        // Ensure worldForward points INTO the room (towards center corridor) rather than into the wall
        const toCenter = new THREE.Vector3(0, worldPos.y, worldPos.z).sub(worldPos).normalize();
        if (worldForward.dot(toCenter) < 0) {
          worldForward.negate();
        }

        // 6. Push the seats forward along the worldForward direction to sit on cushions instead of clipping the backrest
        const CUSHION_PULLFORWARD = 0.22; // Shift 0.22 units forward from backrest centerline
        worldSeat1.addScaledVector(worldForward, CUSHION_PULLFORWARD);
        worldSeat2.addScaledVector(worldForward, CUSHION_PULLFORWARD);

        // Ground level for characters is Y=0
        worldSeat1.y = 0;
        worldSeat2.y = 0;

        // 7. Calculate perfect lookAt targets directly along the worldForward seating direction
        const lookAt1 = worldSeat1.clone().addScaledVector(worldForward, 2.0);
        const lookAt2 = worldSeat2.clone().addScaledVector(worldForward, 2.0);

        // 8. Mutate global BOARDROOM_POIS entries dynamically
        const sofaPOIs = BOARDROOM_POIS.filter(p => p.type === 'sofa');
        if (sofaPOIs[0]) {
          sofaPOIs[0].pos.copy(worldSeat1);
          sofaPOIs[0].lookAt.copy(lookAt1);
          sofaPOIs[0].safePath = [
            worldSeat1.clone().addScaledVector(worldForward, 1.2),
            new THREE.Vector3(-1.8, 0, 0)
          ];
        }
        if (sofaPOIs[1]) {
          sofaPOIs[1].pos.copy(worldSeat2);
          sofaPOIs[1].lookAt.copy(lookAt2);
          sofaPOIs[1].safePath = [
            worldSeat2.clone().addScaledVector(worldForward, 1.2),
            new THREE.Vector3(-1.8, 0, 0)
          ];
        }

        console.log("DYNAMIC SOFA SEATING POIS INITIALIZED:", {
          sofaPos: worldPos,
          seat1: worldSeat1,
          lookAt1: lookAt1,
          seat2: worldSeat2,
          lookAt2: lookAt2
        });

        // Trigger reactive update of executive positions
        onLoad();
      }
    }
  }, [scene, onLoad]);

  const clone = useMemo(() => {
    const clonedScene = scene.clone();
    const meshesToOutline: THREE.Mesh[] = [];

    clonedScene.traverse((child) => {
      // Skip outline meshes we've already injected (prevents infinite recursion)
      if (child.userData.isOutline) return;
      if ((child as THREE.Mesh).isMesh) {
        child.receiveShadow = true;
        child.castShadow = true;
        const oldMat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial;
        if (oldMat) {
          const mat = oldMat.clone();
          (child as THREE.Mesh).material = mat;

          if (mat.emissive) mat.emissiveIntensity = 0;
          mat.roughness = 0.6;
          mat.metalness = 0.1;

          const nodeName = child.name.toLowerCase();

          // Refined Sleek Room Palette for High Contrast
          // High-End Retro-Tech & Mid-Century Designer Studio Palette
          const isPot = nodeName.includes('pot') || nodeName.includes('planter') || nodeName.includes('base') || nodeName.includes('vase');
          const isScreen = nodeName.includes('screen') || nodeName.includes('display') || nodeName.includes('glass');
          const isBox = nodeName.includes('box') || nodeName.includes('crate') || nodeName.includes('case') || nodeName.includes('pack') || nodeName.includes('container');

          if (nodeName.includes('plant')) {
            if (isPot) {
              mat.color.set('#D37257'); // Classic warm terracotta clay pot
              mat.roughness = 0.8;
              mat.metalness = 0.0;
            } else {
              mat.color.set('#428C64'); // Lighter, vibrant natural leaf green
              mat.roughness = 0.65;
              mat.metalness = 0.08;
            }
          } else if (nodeName.includes('floor') || nodeName.includes('border')) {
            mat.color.set('#3E4249'); // Mid-tone space-grey concrete floor
            mat.roughness = 0.85;
          } else if (nodeName.includes('desk') || nodeName.includes('counter')) {
            mat.color.set('#C69A75'); // Warm Scandinavian Beech/Oak wood top for working desks
            mat.roughness = 0.6;
            mat.metalness = 0.0;
          } else if (nodeName.includes('table')) {
            mat.color.set('#9CA0A8'); // Raw Brutalist Concrete (béton brut) for the round table
            mat.roughness = 0.9; // Coarse, raw matte stone texture
            mat.metalness = 0.0;
          } else if (nodeName.includes('sofa')) {
            mat.color.set('#E2A746'); // Iconic mid-century Mustard Yellow bouclé fabric
            mat.roughness = 0.85;
            mat.metalness = 0.0;
          } else if (nodeName.includes('chair')) {
            mat.color.set('#4A607A'); // Dusty Denim / Indigo Blue upholstery
            mat.roughness = 0.75;
            mat.metalness = 0.05;
          } else if (nodeName.includes('cabinet') || nodeName.includes('shelf') || nodeName.includes('rack')) {
            mat.color.set('#E5E2DA'); // Minimalist off-white/cream metal frame (high-end designer shelving)
            mat.roughness = 0.65;
            mat.metalness = 0.1;
          } else if (isBox) {
            mat.color.set('#C69A75'); // Warm honey-oak craft drawers/boxes inside the shelving unit
            mat.roughness = 0.55;
            mat.metalness = 0.0;
          } else if (nodeName.includes('flexo') || nodeName.includes('lamp')) {
            mat.color.set('#E63946'); // Classic Bauhaus Orange-Red designer accent lamp
            mat.roughness = 0.3;
            mat.metalness = 0.2;
          } else if (nodeName.includes('pc') || nodeName.includes('laptop')) {
            if (isScreen) {
              mat.color.set('#1A1D20'); // Dark off-state terminal screen
              mat.roughness = 0.2;
              mat.metalness = 0.8;
            } else {
              mat.color.set('#E1DDD5'); // Retro Macintosh-style Warm Beige/Cream plastic body
              mat.roughness = 0.45;
              mat.metalness = 0.05;
            }
          } else if (nodeName.includes('board')) {
            mat.color.set('#ECEEF0'); // Whiteboard surface
            mat.roughness = 0.15;
          } else {
            mat.color.set('#2C3038'); // Slate fallback
            mat.roughness = 0.9;
          }

          meshesToOutline.push(child as THREE.Mesh);
        }
      }
    });

    // Add outline wireframes after traverse completes to prevent infinite recursion crash!
    const wireMat = new THREE.MeshBasicMaterial({
      color: '#121418',
      wireframe: true,
      transparent: true,
      opacity: 0.18
    });
    meshesToOutline.forEach((mesh) => {
      const wireMesh = new THREE.Mesh(mesh.geometry, wireMat);
      wireMesh.userData.isOutline = true; // Tag so traversal always skips it
      mesh.add(wireMesh);
    });

    return clonedScene;
  }, [scene]);

  return (
    <primitive object={clone} position={[0, 0, 0]} />
  );
}

function BrutalistCenterpiece() {
  const innerOrbRef = useRef<THREE.Mesh>(null);
  const ringXRef = useRef<THREE.Mesh>(null);
  const ringYRef = useRef<THREE.Mesh>(null);
  const ringZRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    const elapsed = state.clock.getElapsedTime();

    // Smooth floating/hovering animation for the central core orb
    if (innerOrbRef.current) {
      innerOrbRef.current.position.y = 1.1 + Math.sin(elapsed * 2) * 0.05;
      innerOrbRef.current.rotation.y = elapsed * 0.5;
    }

    // Mesmerizing counter-rotations on different axes for the gimbal rings
    if (ringXRef.current) {
      ringXRef.current.rotation.x = elapsed * 0.4;
      ringXRef.current.rotation.y = elapsed * 0.1;
    }
    if (ringYRef.current) {
      ringYRef.current.rotation.y = -elapsed * 0.3;
      ringYRef.current.rotation.z = elapsed * 0.15;
    }
    if (ringZRef.current) {
      ringZRef.current.rotation.z = elapsed * 0.6;
      ringZRef.current.rotation.x = -elapsed * 0.2;
    }
  });

  return (
    <group>
      {/* 1. Heavy Industrial Pedestal (anchored to the boardroom desk center) */}
      <mesh position={[0, 0.4, 0]}>
        <cylinderGeometry args={[0.3, 0.35, 0.8, 16]} />
        <meshStandardMaterial color="#1A1C22" roughness={0.75} metalness={0.4} />
      </mesh>
      {/* Metallic highlight collar */}
      <mesh position={[0, 0.8, 0]}>
        <torusGeometry args={[0.28, 0.03, 8, 24]} />
        <meshStandardMaterial color="#8C97A5" roughness={0.2} metalness={0.8} />
      </mesh>

      {/* 2. Floating Gyroscopic Gimbal Assemblies */}
      <group position={[0, 1.1, 0]}>
        {/* Central Core: Hovering, Highly Polished Titanium/Chrome Orb representing the AI consensus mind */}
        <mesh ref={innerOrbRef}>
          <sphereGeometry args={[0.18, 32, 32]} />
          <meshStandardMaterial color="#A5B4FC" roughness={0.08} metalness={0.9} />
        </mesh>

        {/* Inner Ring (X-Axis) - Anodized Bronze/Copper */}
        <mesh ref={ringXRef}>
          <torusGeometry args={[0.35, 0.02, 8, 32]} />
          <meshStandardMaterial color="#C69A75" roughness={0.3} metalness={0.7} />
        </mesh>

        {/* Middle Ring (Y-Axis) - Polished Gunmetal/Titanium */}
        <mesh ref={ringYRef}>
          <torusGeometry args={[0.46, 0.02, 8, 32]} />
          <meshStandardMaterial color="#8C97A5" roughness={0.15} metalness={0.85} />
        </mesh>

        {/* Outer Ring (Z-Axis) - Matte Dark Charcoal Iron */}
        <mesh ref={ringZRef}>
          <torusGeometry args={[0.57, 0.015, 8, 32]} />
          <meshStandardMaterial color="#24272E" roughness={0.6} metalness={0.2} />
        </mesh>
      </group>
    </group>
  );
}

function Boardroom3D() {
  const { personas, spawnedAgents, activeSpeaker } = usePipelineStore();
  const [poisLoaded, setPoisLoaded] = React.useState(false);
  const [assignments, setAssignments] = React.useState<number[]>(() => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const [activeRoamerIdx, setActiveRoamerIdx] = React.useState<number>(0);

  // Centralized Roaming Coordinator (Token Rotation Protocol)
  // Ensures exactly one agent stands up and walks to a new spot every 20 seconds, cycling through all agents sequentially!
  useEffect(() => {
    const interval = setInterval(() => {
      // Pause roaming rotation if someone is speaking to keep focus on debate
      if (activeSpeaker !== null) return;

      const nextRoamerIdx = (activeRoamerIdx + 1) % 10;
      setActiveRoamerIdx(nextRoamerIdx);

      setAssignments((prev) => {
        const next = [...prev];
        const occupied = new Set(next);

        // Find all currently unoccupied POIs (including seats and standing spots)
        const vacantIndices: number[] = [];
        for (let i = 0; i < BOARDROOM_POIS.length; i++) {
          if (!occupied.has(i)) {
            vacantIndices.push(i);
          }
        }

        if (vacantIndices.length > 0) {
          // Relocate the active roamer to a random vacant POI, freeing up their old position
          const randomVacant = vacantIndices[Math.floor(Math.random() * vacantIndices.length)];
          next[nextRoamerIdx] = randomVacant;
        }
        return next;
      });
    }, 20000); // 20-second dynamic rotation interval

    return () => clearInterval(interval);
  }, [activeRoamerIdx, activeSpeaker]);

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
      '#D1493E', // CEO (Bauhaus Matte Red accent)
      '#3A6D8C', // CTO (Anodized Steel Blue)
      '#2A2E35', // CISO (Stealth Tactical Charcoal)
      '#D4A373', // CMO (Warm Sand / Terracotta)
      '#4E6B5A', // CFO (Functional Sage/Olive)
      '#E07A5F', // CPO (Caution/Safety Orange)
      '#D9A05B', // Legal (Structured Mustard Ochre)
      '#527A8C', // Data (Cobalt Grey-Blue)
      '#6F5E76', // Sales (Muted Slate Plum)
      '#8A9A86', // CS (Eucalyptus Sage Green)
    ];

    const startPositions = BOARDROOM_POIS.map(p => p.pos.toArray() as [number, number, number]);

    return list.map((p, index) => {
      const assignedPoiIdx = assignments[index] !== undefined ? assignments[index] : index;
      const targetPos = BOARDROOM_POIS[assignedPoiIdx].pos.toArray() as [number, number, number];

      return {
        ...p,
        color: colors[index % colors.length],
        startPosition: startPositions[index],
        targetPosition: targetPos,
      };
    });
  }, [poisLoaded, assignments]);

  return (
    <div className="w-full h-full relative bg-[#1F232B]">
      <Canvas shadows dpr={[1, 2]}>
        <color attach="background" args={['#1F232B']} />
        {/* Isometric-style perspective */}
        <PerspectiveCamera makeDefault position={[8, 8, 8]} fov={35} />
        <OrbitControls enablePan={false} maxPolarAngle={Math.PI / 2.2} minPolarAngle={0.1} />

        {/* Dramatic cinematic lighting setup */}
        <ambientLight intensity={0.6} />
        <directionalLight
          position={[10, 20, 10]}
          intensity={1.5}
          castShadow
          shadow-bias={-0.0005}
          shadow-mapSize={[2048, 2048]}
          shadow-camera-far={50}
          shadow-camera-left={-10}
          shadow-camera-right={10}
          shadow-camera-top={10}
          shadow-camera-bottom={-10}
        />
        {/* Rich warm accent fills and steel blue highlights */}
        <pointLight position={[-6, 4, -6]} intensity={0.6} color="#FF9E59" />
        <pointLight position={[6, 3, -4]} intensity={0.4} color="#5C9DFF" />

        {/* Atmospheric depth fog - fades smoothly into the medium-dark background */}
        <fog attach="fog" args={['#1F232B', 22, 45]} />


        <group position={[0, 0, 0]}>
          <ImportedOffice onLoad={() => setPoisLoaded(true)} />
          {/* Tactical Retro-Tech Floor Grid - extremely subtle to blend into the floor */}
          <gridHelper args={[20, 20, '#374151', '#111827']} position={[0, 0.01, 0]} />

          {dynamicPersonas.map((exec) => (
            <ImportedAgent
              key={exec.id}
              label={exec.name}
              startPos={exec.startPosition}
              targetPosProp={exec.targetPosition}
              color={exec.color}
              speaking={activeSpeaker === exec.id}
            />
          ))}

          {/* Animated Cybernetic Centerpiece */}
          <BrutalistCenterpiece />
        </group>
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

  const QABlock = ({ part, type }: { part: string, type: 'Q' | 'A' }) => {
    const [expanded, setExpanded] = React.useState(false);
    const cleanText = part.replace(/^[QA]:\s*/, '').trim();
    // Only allow answers to be collapsible, and only if they are long
    const isLong = cleanText.length > 150 && type === 'A';

    return (
      <div
        className={`p-3 shadow-[4px_4px_0_0_#000] relative mt-3 ${type === 'Q'
            ? 'bg-[#FAF9F6] border-[3px] border-black mt-4'
            : `bg-white border-[3px] border-black ml-6 ${isLong ? 'cursor-pointer hover:bg-neutral-50' : ''}`
          }`}
        onClick={(e) => {
          if (isLong) {
            e.stopPropagation();
            setExpanded(!expanded);
          }
        }}
      >
        <div className={`absolute -top-3 left-2 text-white text-[10px] font-black uppercase px-2 py-0.5 border-2 border-black ${type === 'Q' ? 'bg-[#FF4500]' : 'bg-[#10B981]'}`}>
          {type === 'Q' ? 'Interrogation' : 'Response'}
        </div>
        <span className="text-black font-bold block pt-1 text-[13px] whitespace-pre-wrap leading-relaxed">
          {(expanded || !isLong) ? cleanText : `${cleanText.slice(0, 150)}...`}
        </span>
        {isLong && (
          <div className="mt-2 text-right">
            <span className="text-[10px] font-black uppercase text-[#10B981] hover:underline">
              {expanded ? 'Collapse' : 'Expand'}
            </span>
          </div>
        )}
      </div>
    );
  };

  const InterrogationContent = ({ content }: { content: string }) => {
    if (!content.includes('Q:') && !content.includes('A:')) {
      return <span className="text-black/80 font-mono whitespace-pre-wrap">{content}</span>;
    }

    // Split only on Q: or A: at the START of a line to avoid splitting inside responses
    const parts = content.split(/(?=^Q:|^A:)/m);

    return (
      <div className="flex flex-col gap-4 my-2 font-mono">
        {parts.map((part, index) => {
          if (part.startsWith('Q:')) {
            return <QABlock key={index} part={part} type="Q" />;
          } else if (part.startsWith('A:')) {
            return <QABlock key={index} part={part} type="A" />;
          } else if (part.trim()) {
            return (
              <span key={index} className="text-black/80 font-mono whitespace-pre-wrap text-[13px] leading-relaxed">
                {part.trim()}
              </span>
            );
          }
          return null;
        })}
      </div>
    );
  };

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
            className={`p-4 border-4 transition-colors ${msg.type === 'challenge' ? 'border-[#FF4500] bg-[#FF4500]/5 hover:bg-[#FF4500]/10' : 'border-black bg-white hover:bg-neutral-50'}`}
          >
            <div className="flex items-center gap-2 mb-3 pb-2 border-b-2 border-black/10">
              <span className={`text-xs font-black uppercase tracking-widest ${msg.type === 'challenge' ? 'text-[#FF4500]' : 'text-black'}`}>
                {msg.sender}
              </span>
              {msg.type === 'challenge' && <ShieldAlert className="w-3.5 h-3.5 text-[#FF4500]" strokeWidth={3} />}
            </div>
            <div className="text-[13px] font-bold text-black/90">
              <InterrogationContent content={msg.text} />
            </div>
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
