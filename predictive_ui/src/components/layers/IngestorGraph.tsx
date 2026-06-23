import React, { useEffect, useCallback, useState } from 'react';
import * as THREE from 'three';
import ForceGraph3D from 'react-force-graph-3d';
import ReactFlow, { Background, Controls, addEdge, useNodesState, useEdgesState, MarkerType } from 'reactflow';
import type { Connection, Edge, Node } from 'reactflow';
import 'reactflow/dist/style.css';
import { Database, Filter, Layers, BrainCircuit, Loader2, Zap } from 'lucide-react';
import { usePipelineStore } from '../../store/usePipelineStore';

function buildFlowGraph(ingestionNodes: ReturnType<typeof usePipelineStore>['ingestionNodes']): { nodes: Node[]; edges: Edge[] } {
  if (ingestionNodes.length === 0) return { nodes: [], edges: [] };

  const inputNodes = ingestionNodes.filter(n => n.type === 'input');
  const processNodes = ingestionNodes.filter(n => n.type === 'process');
  const outputNodes = ingestionNodes.filter(n => n.type === 'output');

  const nodes: Node[] = [
    ...inputNodes.map((n, i) => ({
      id: n.id,
      position: { x: 50, y: 50 + i * 110 },
      data: { label: n.label },
      type: 'input' as const,
      style: {
        background: '#FFFFFF',
        color: '#000000',
        border: `4px solid ${n.status === 'pending' ? '#FF4500' : '#000000'}`,
        opacity: n.status === 'pending' ? 0.7 : 1,
        borderRadius: '0px',
        padding: '8px 12px',
        fontSize: '11px',
        fontWeight: 'bold',
        fontFamily: 'monospace',
        boxShadow: '4px 4px 0px 0px rgba(0,0,0,1)',
        width: 'auto',
        minWidth: '200px',
        maxWidth: '300px',
        wordWrap: 'break-word',
      },
    })),
    ...processNodes.map((n, i) => ({
      id: n.id,
      position: { x: 380, y: 80 + i * 120 },
      data: { label: n.label },
      type: 'default' as const,
      style: {
        background: '#FF4500',
        color: '#000000',
        border: '4px solid #000000',
        borderRadius: '0px',
        padding: '8px 12px',
        fontSize: '11px',
        fontWeight: 'bold',
        fontFamily: 'monospace',
        boxShadow: '4px 4px 0px 0px rgba(0,0,0,1)',
        width: 'auto',
        minWidth: '150px',
        maxWidth: '250px',
        wordWrap: 'break-word',
      },
    })),
    ...outputNodes.map((n, i) => ({
      id: n.id,
      position: { x: 700, y: 60 + i * 120 },
      data: { label: n.label },
      type: 'output' as const,
      style: {
        background: n.status === 'active' ? '#000000' : '#FFFFFF',
        color: n.status === 'active' ? '#FFFFFF' : '#666666',
        border: `4px solid ${n.status === 'active' ? '#000000' : '#AAAAAA'}`,
        fontWeight: 'bold',
        borderRadius: '0px',
        padding: '8px 12px',
        fontSize: '11px',
        fontFamily: 'monospace',
        boxShadow: n.status === 'active' ? '4px 4px 0px 0px rgba(255,69,0,1)' : '4px 4px 0px 0px rgba(0,0,0,0.2)',
        width: 'auto',
        minWidth: '150px',
        maxWidth: '250px',
        wordWrap: 'break-word',
      },
    })),
  ];

  const edges: Edge[] = [];
  inputNodes.forEach(input => {
    processNodes.forEach(proc => {
      edges.push({
        id: `e-${input.id}-${proc.id}`,
        source: input.id,
        target: proc.id,
        animated: true,
        style: { stroke: '#FF4500', strokeWidth: 3 },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#FF4500' },
      });
    });
  });
  processNodes.forEach(proc => {
    outputNodes.forEach(out => {
      edges.push({
        id: `e-${proc.id}-${out.id}`,
        source: proc.id,
        target: out.id,
        animated: true,
        style: { stroke: '#000000', strokeWidth: 3 },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#000000' },
      });
    });
  });

  return { nodes, edges };
}

export default function IngestorGraph() {
  const { ingestionNodes, isConnected, pipelineStages, kgNodes, kgEdges } = usePipelineStore();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [viewMode, setViewMode] = useState<'pipeline' | '3d'>('pipeline');

  const stageStatus = pipelineStages.layer1;
  const isWaiting = stageStatus === 'waiting';
  const isRunning = stageStatus === 'running';

  useEffect(() => {
    const { nodes: newNodes, edges: newEdges } = buildFlowGraph(ingestionNodes);
    setNodes(newNodes);
    setEdges(newEdges);
  }, [ingestionNodes, setNodes, setEdges]);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const inputCount = ingestionNodes.filter(n => n.type === 'input').length;
  const clusterCount = ingestionNodes.filter(n => n.type === 'output').length;
  const activeCount = ingestionNodes.filter(n => n.status === 'active').length;

  return (
    <div className="w-full h-full flex gap-0">
      
      {/* ── Side Panel ─────────────────────────────────── */}
      <div className="w-64 flex-none flex flex-col border-r-8 border-black bg-white">
        
        {/* Header */}
        <div className="p-5 border-b-8 border-black bg-black text-white">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-brand" strokeWidth={3} />
            <span className="font-black text-xs uppercase tracking-widest">Layer 1</span>
          </div>
          <h2 className="font-black text-lg uppercase leading-tight mt-1">Data Ingestion</h2>
        </div>

        {/* Stats */}
        <div className="flex flex-col divide-y-4 divide-black border-b-4 border-black">
          <div className="p-5">
            <div className="text-xs font-black uppercase tracking-widest text-black/50 mb-1">Sources Active</div>
            <div className="text-4xl font-black text-black">{inputCount}</div>
          </div>
          <div className="p-5">
            <div className="text-xs font-black uppercase tracking-widest text-black/50 mb-2">Nodes Online</div>
            <div className="flex items-center gap-2">
              <span className={`w-3 h-3 border-2 border-black ${isConnected ? 'bg-brand animate-pulse' : 'bg-red-500'}`} />
              <span className="text-lg font-black text-black">{activeCount} / {ingestionNodes.length}</span>
            </div>
          </div>
          <div className="p-5">
            <div className="text-xs font-black uppercase tracking-widest text-black/50 mb-1">Clusters Found</div>
            <div className="text-4xl font-black text-black">{clusterCount}</div>
          </div>
        </div>

        {/* Discovery */}
        <div className="p-5 border-b-4 border-black flex-1">
          <div className="flex items-center gap-2 mb-3">
            <BrainCircuit className="w-4 h-4 text-brand" strokeWidth={3} />
            <h3 className="font-black text-sm uppercase tracking-widest">Discovery</h3>
          </div>
          <p className="text-sm font-bold leading-snug text-black/70">
            {clusterCount > 0
              ? `${clusterCount} Tension Cluster${clusterCount > 1 ? 's' : ''} identified. High-density signal detected.`
              : 'Awaiting cluster analysis from semantic extractor...'}
          </p>
        </div>

        {/* Generate button */}
        <button className="m-4 py-3 bg-black text-white border-4 border-black shadow-neo-black font-black text-sm uppercase tracking-widest flex items-center justify-center gap-2 cursor-pointer transition-all hover:translate-x-1 hover:translate-y-1 hover:shadow-none">
          <Filter className="w-4 h-4" strokeWidth={3} /> Generate Proposal
        </button>
      </div>

      {/* ── Main Graph Panel ────────────────────────────── */}
      <div className="flex-1 relative overflow-hidden bg-white" style={{ minHeight: 0 }}>

        {/* Pending / Running Overlay */}
        {(isWaiting || isRunning) && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-white/90">
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
                  {isRunning ? 'Layer 1 - Ingesting' : 'Stage Pending'}
                </p>
                <p className="text-sm font-bold text-black/60">
                  {isRunning
                    ? 'Extracting signals from input documents...'
                    : 'Waiting for previous stages to complete.'}
                </p>
                {isRunning && (
                  <div className="mt-4 pt-4 border-t-4 border-black/20 flex flex-col gap-2">
                    <p className="text-xs font-black uppercase tracking-widest text-black/40">
                      ⏱ Est. ~5 min depending on LLM rate limits
                    </p>
                    <p className="text-xs font-bold text-black/50">
                      Open the <span className="font-black text-black uppercase">Terminal</span> in the top bar to track progress in real-time.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {viewMode === 'pipeline' ? (
          ingestionNodes.length === 0 && !isWaiting && !isRunning ? (
            <div className="w-full h-full flex flex-col items-center justify-center gap-4">
              <div className="w-16 h-16 bg-brand border-4 border-black flex items-center justify-center animate-pulse">
                <Zap size={28} className="text-black" strokeWidth={3} />
              </div>
              <p className="font-black text-sm uppercase tracking-widest">
                {isConnected ? 'Awaiting ingestion data...' : 'Connecting to backend...'}
              </p>
            </div>
          ) : (
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              fitView
              fitViewOptions={{ padding: 0.2 }}
              attributionPosition="bottom-left"
              proOptions={{ hideAttribution: true }}
            >
              <Background color="#EEEEEE" gap={20} />
              <Controls />
            </ReactFlow>
          )
        ) : (
          <div className="w-full h-full bg-white flex items-center justify-center">
            {kgNodes.length === 0 ? (
              <div className="w-full h-full flex flex-col items-center justify-center gap-4">
                <div className="w-16 h-16 bg-brand border-4 border-black flex items-center justify-center animate-pulse">
                  <Database size={28} className="text-black" strokeWidth={3} />
                </div>
                <p className="font-black text-sm uppercase tracking-widest">
                  No Graph Data Found
                </p>
              </div>
            ) : (
              <ForceGraph3D
                graphData={{
                  nodes: kgNodes.map(n => ({ ...n, name: n.label })),
                  links: kgEdges.map(e => ({ source: e.source, target: e.target, name: e.relationshipType }))
                }}
                nodeLabel={(node: any) => `
                  <div style="background: #FFFFFF; border: 4px solid #000000; padding: 12px; font-weight: 900; color: #000000; box-shadow: 6px 6px 0 0 #000000; font-family: monospace; text-transform: uppercase;">
                    <div style="font-size: 10px; color: ${node.entityType === 'Feature' ? '#FF4500' : '#666666'}; margin-bottom: 4px; letter-spacing: 0.1em;">${node.entityType || 'Node'}</div>
                    <div style="font-size: 14px; letter-spacing: 0.05em;">${node.name}</div>
                  </div>
                `}
                nodeThreeObject={(node: any) => {
                  const isFeature = node.entityType === 'Feature';
                  const color = isFeature ? '#FF4500' : '#000000';
                  const geometry = new THREE.SphereGeometry(isFeature ? 6 : 4, 32, 32);
                  const material = new THREE.MeshBasicMaterial({ color: color });
                  return new THREE.Mesh(geometry, material);
                }}
                linkColor={() => '#000000'}
                backgroundColor="#FFFFFF"
                linkDirectionalParticles={3}
                linkDirectionalParticleWidth={4}
                linkDirectionalParticleColor={() => '#FF4500'}
                linkDirectionalArrowLength={0}
                width={typeof window !== 'undefined' ? window.innerWidth - 256 : 800} // Subtract side panel width
              />
            )}
          </div>
        )}

        {/* View Toggle */}
        <div className="absolute top-4 right-4 flex gap-2 z-50">
          <button 
            onClick={() => setViewMode('pipeline')}
            className={`px-4 py-2 text-xs font-black uppercase tracking-widest border-4 transition-all ${
              viewMode === 'pipeline' 
                ? 'bg-black text-white border-black shadow-neo-black' 
                : 'bg-white text-black border-black hover:bg-gray-100'
            }`}
          >
            Pipeline
          </button>
          <button 
            onClick={() => setViewMode('3d')}
            className={`px-4 py-2 text-xs font-black uppercase tracking-widest border-4 transition-all flex items-center gap-2 ${
              viewMode === '3d' 
                ? 'bg-black text-white border-black shadow-neo-black' 
                : 'bg-white text-black border-black hover:bg-gray-100'
            }`}
          >
            <Layers className="w-3.5 h-3.5" strokeWidth={3} /> Live Knowledge Graph
          </button>
        </div>
      </div>
    </div>
  );
}
