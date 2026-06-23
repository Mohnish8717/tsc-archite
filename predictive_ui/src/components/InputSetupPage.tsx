import React, { useState, useRef } from 'react';
import { Activity, Play, Plus, X, FileText, Upload, File as FileIcon, Info, Terminal, Square, ArrowRightCircle, ShieldCheck } from 'lucide-react';

import { API_BASE_URL } from '../config';
import { AI_FITNESS_CONTEXT, AI_FITNESS_PROPOSAL, AI_FITNESS_TICKETS } from '../data/aiFitnessDefaults';

interface InputSetupPageProps {
  onStartSimulation: (filePaths: Record<string, string>, boardroomOnly: boolean) => Promise<void>;
  onSkip?: () => void;
}

interface DocumentInput {
  id: string;
  title: string;
  // Either a File (binary-safe) or raw text — never both active at once.
  file: File | null;
  text: string;
}

// Map a document title to the field name expected by /api/upload
function titleToFieldName(title: string): 'proposal' | 'context' | 'interviews' | 'support' | 'analytics' {
  const t = title.toLowerCase();
  if (t.includes('proposal') || t.includes('spec') || t.includes('feature')) return 'proposal';
  if (t.includes('interview') || t.includes('research') || t.includes('user')) return 'interviews';
  if (t.includes('support') || t.includes('ticket') || t.includes('complaint')) return 'support';
  if (t.includes('analytic') || t.includes('data')) return 'analytics';
  return 'context'; // Company Context + everything else
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const ACCEPTED_TYPES = '.txt,.md,.pdf,.docx,.json,.csv';

export default function InputSetupPage({ onStartSimulation, onSkip }: InputSetupPageProps) {
  const [documents, setDocuments] = useState<DocumentInput[]>([
    { id: '1', title: 'Feature Proposal', file: null, text: '' }
  ]);
  const [boardroomOnly, setBoardroomOnly] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileInputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  const addDocument = (title: string = 'New Document') => {
    const isRecommended = ['Company Context', 'Feature Proposal', 'Customer Interviews', 'Support Tickets', 'Analytics JSON'].includes(title);
    if (isRecommended && documents.some(d => d.title.toLowerCase() === title.toLowerCase())) {
      return;
    }
    setDocuments(prev => [...prev, { id: Math.random().toString(), title, file: null, text: '' }]);
  };

  const loadDefaultInputs = () => {
    setDocuments([
      { id: Math.random().toString(), title: 'Company Context', file: null, text: AI_FITNESS_CONTEXT },
      { id: Math.random().toString(), title: 'Feature Proposal', file: null, text: AI_FITNESS_PROPOSAL },
      { id: Math.random().toString(), title: 'Support Tickets', file: null, text: AI_FITNESS_TICKETS }
    ]);
  };

  const removeDocument = (id: string) => {
    setDocuments(prev => prev.filter(d => d.id !== id));
  };

  const updateTitle = (id: string, value: string) => {
    setDocuments(prev => prev.map(d => d.id === id ? { ...d, title: value } : d));
  };

  const updateText = (id: string, value: string) => {
    setDocuments(prev => prev.map(d => d.id === id ? { ...d, text: value, file: null } : d));
  };

  const handleFileChange = (id: string, e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    if (!file) return;
    // Store the File object directly — no FileReader, no corruption
    setDocuments(prev => prev.map(d =>
      d.id === id
        ? { ...d, file, text: '' }
        : d
    ));
  };

  const clearFile = (id: string) => {
    setDocuments(prev => prev.map(d => d.id === id ? { ...d, file: null, text: '' } : d));
    // Reset the hidden input so the same file can be re-selected
    if (fileInputRefs.current[id]) {
      fileInputRefs.current[id]!.value = '';
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const form = new FormData();
      // Track which fields have been populated to avoid double-overwriting
      const populated: Record<string, boolean> = {};

      for (const doc of documents) {
        const field = titleToFieldName(doc.title);

        if (doc.file) {
          // Binary-safe upload — preserve original filename so backend detects extension
          form.append(field, doc.file, doc.file.name);
          populated[field] = true;
        } else if (doc.text.trim()) {
          // Text-only — create a plain-text Blob, use .txt extension
          const safeName = `${field}_${Date.now()}.txt`;
          form.append(field, new Blob([doc.text], { type: 'text/plain' }), safeName);
          populated[field] = true;
        }
      }

      if (Object.keys(populated).length === 0) {
        setError('Add content to at least one document before starting.');
        setLoading(false);
        return;
      }

      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: form,
      });

      if (!response.ok) {
        const msg = await response.text();
        throw new Error(`Upload failed (${response.status}): ${msg}`);
      }

      const data = await response.json();
      // Pass the server-returned file paths to App.tsx
      await onStartSimulation(data.files, boardroomOnly);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-black font-sans selection:bg-black selection:text-white p-6 md:p-12">
      <div className="max-w-4xl mx-auto">
        <header className="mb-12 border-b-8 border-black pb-6 flex items-center justify-between">
          <div>
            <div className="inline-flex items-center gap-2 text-black font-black text-sm tracking-[4px] uppercase mb-4 bg-white px-4 py-2 border-4 border-black shadow-neo-black transform -rotate-1">
              Configuration
            </div>
            <h1 className="text-5xl md:text-7xl font-black uppercase tracking-tighter text-black leading-none">
              Define Scenario
            </h1>
          </div>
          <Activity className="text-black w-16 h-16 md:w-24 md:h-24 hidden md:block" strokeWidth={3} />
        </header>

        <div className="mb-8">
          <h2 className="text-xl font-black uppercase tracking-widest mb-4">Recommended Inputs:</h2>
          <div className="flex flex-wrap gap-4">
            {['Company Context', 'Feature Proposal', 'Customer Interviews', 'Support Tickets', 'Analytics JSON'].map(preset => {
              const exists = documents.some(d => d.title.toLowerCase() === preset.toLowerCase());
              return (
                <button
                  key={preset}
                  type="button"
                  onClick={() => addDocument(preset)}
                  disabled={exists}
                  className={`px-4 py-2 border-4 border-black font-black text-sm uppercase tracking-widest transition-all flex items-center gap-2 ${
                    exists
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed shadow-none opacity-60'
                      : 'bg-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:-translate-y-1 hover:translate-x-1 hover:shadow-none'
                  }`}
                >
                  <Plus size={16} strokeWidth={3} /> {preset}
                </button>
              );
            })}
            <button
              type="button"
              onClick={() => addDocument('Custom Document')}
              className="px-4 py-2 bg-black text-white border-4 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] font-black text-sm uppercase tracking-widest hover:-translate-y-1 hover:translate-x-1 hover:shadow-none transition-all flex items-center gap-2"
            >
              <Plus size={16} strokeWidth={3} /> Custom
            </button>
            <button
              type="button"
              onClick={loadDefaultInputs}
              className="px-4 py-2 bg-[#FF4500] text-black border-4 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] font-black text-sm uppercase tracking-widest hover:-translate-y-1 hover:translate-x-1 hover:shadow-none transition-all flex items-center gap-2 ml-auto"
            >
              <Play size={16} fill="currentColor" /> Use Default Inputs to Test Pipeline
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-6 px-5 py-4 bg-[#FF4500] border-4 border-black font-black text-sm uppercase tracking-widest">
            ⚠ {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex flex-col gap-8">
          {documents.map((doc) => (
            <div key={doc.id} className="bg-white border-4 border-black shadow-neo-black p-6 relative">
              <button
                type="button"
                onClick={() => removeDocument(doc.id)}
                className="absolute -top-4 -right-4 bg-[#FF4500] text-black border-4 border-black w-10 h-10 flex items-center justify-center hover:scale-110 transition-transform z-10"
              >
                <X strokeWidth={3} />
              </button>

              <div className="flex items-center gap-4 mb-4 border-b-4 border-black pb-4 justify-between">
                <div className="flex items-center gap-4 flex-1">
                  <FileText strokeWidth={3} size={32} />
                  <input
                    type="text"
                    value={doc.title}
                    onChange={(e) => updateTitle(doc.id, e.target.value)}
                    className="text-2xl font-black uppercase tracking-tighter w-full outline-none bg-transparent placeholder-gray-400"
                    placeholder="Document Title..."
                    required
                  />
                </div>
                <div>
                  <label
                    htmlFor={`file-upload-${doc.id}`}
                    className="cursor-pointer flex items-center gap-2 px-3 py-1 bg-white border-4 border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] font-black text-xs uppercase hover:-translate-y-0.5 hover:translate-x-0.5 hover:shadow-none transition-all"
                  >
                    <Upload size={14} strokeWidth={3} />
                    {doc.file ? 'Replace' : 'Upload File'}
                  </label>
                  <input
                    id={`file-upload-${doc.id}`}
                    ref={el => { fileInputRefs.current[doc.id] = el; }}
                    type="file"
                    accept={ACCEPTED_TYPES}
                    className="hidden"
                    onChange={(e) => handleFileChange(doc.id, e)}
                  />
                </div>
              </div>

              {/* File badge when a file is attached */}
              {doc.file ? (
                <div className="flex items-center gap-3 p-4 bg-black text-white border-4 border-black">
                  <FileIcon size={24} strokeWidth={2.5} className="shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="font-black text-sm uppercase tracking-widest truncate">{doc.file.name}</p>
                    <p className="text-xs font-mono opacity-70">{formatBytes(doc.file.size)}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => clearFile(doc.id)}
                    className="shrink-0 text-white hover:text-[#FF4500] transition-colors"
                    title="Remove file"
                  >
                    <X strokeWidth={3} size={18} />
                  </button>
                </div>
              ) : (
                <textarea
                  className="w-full h-40 outline-none text-sm font-mono font-bold resize-y text-black placeholder-gray-400 bg-transparent"
                  placeholder="Paste raw text, JSON, or CSV here — or upload a file above (PDF, DOCX, TXT, JSON, CSV)..."
                  value={doc.text}
                  onChange={e => updateText(doc.id, e.target.value)}
                />
              )}
            </div>
          ))}

          {documents.length === 0 && (
            <div className="text-center p-12 border-4 border-black border-dashed bg-white/50 font-black uppercase tracking-widest text-xl">
              Add at least one document to start the simulation.
            </div>
          )}

          <div className="flex flex-col gap-4 mt-8">
            <label className="flex items-center gap-2 font-black uppercase tracking-widest text-sm mb-2 text-gray-500 opacity-70">
              <input 
                type="checkbox" 
                checked={false}
                disabled={true}
                onChange={() => {}}
                className="w-5 h-5 accent-gray-500 cursor-not-allowed border-2 border-gray-400"
              />
              Skip Social Simulation (Boardroom Only) - Currently disabled, run full pipeline
            </label>
            <button
              type="submit"
              disabled={loading || documents.length === 0}
              className="w-full py-6 bg-black text-white border-4 border-black shadow-neo-white font-black uppercase tracking-[0.2em] text-2xl flex items-center justify-center gap-4 transition-all hover:translate-x-1 hover:-translate-y-1 disabled:opacity-50"
            >
              {loading ? (
                <span className="animate-pulse">Uploading & Starting...</span>
              ) : (
                <>
                  <Play fill="currentColor" size={28} /> Start Simulation
                </>
              )}
            </button>

            {onSkip && (
              <button
                type="button"
                onClick={onSkip}
                className="w-full py-4 bg-white text-black border-4 border-black shadow-neo-black font-black uppercase tracking-widest text-sm hover:translate-x-1 hover:translate-y-1 transition-all"
              >
                Skip to Dashboard (Developer Bypass)
              </button>
            )}
          </div>
        </form>

        {/* UI Usage Tips Section */}
        <div className="mt-12 mb-8 border-4 border-black p-6 bg-white shadow-neo-black">
          <div className="flex items-center gap-3 mb-6 pb-4 border-b-4 border-black">
            <Info size={28} className="text-brand" strokeWidth={3} />
            <h2 className="font-black text-xl uppercase tracking-widest">Suggestions & Usage Tips</h2>
          </div>
          
          <div className="flex flex-col gap-6">
            <div className="flex items-start gap-4">
              <div className="shrink-0 p-2 bg-black text-white mt-1">
                <Terminal size={20} strokeWidth={2.5} />
              </div>
              <div>
                <h3 className="font-black uppercase tracking-wider text-sm mb-1">Terminal</h3>
                <p className="text-sm font-bold text-black/70 leading-relaxed">
                  Located in the top navigation bar. Click this to toggle a drawer that shows the raw, real-time backend execution logs exactly as they appear on the server console. Great for debugging!
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="shrink-0 p-2 bg-red-600 text-white mt-1">
                <Square size={20} strokeWidth={2.5} fill="white" />
              </div>
              <div>
                <h3 className="font-black uppercase tracking-wider text-sm mb-1">Abort Simulation</h3>
                <p className="text-sm font-bold text-black/70 leading-relaxed">
                  Instantly aborts the entire pipeline at any stage. It functions as an emergency hard-stop that halts all background processes immediately without saving data.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="shrink-0 p-2 bg-red-500 text-black border-2 border-black mt-1">
                <Square size={20} strokeWidth={2.5} fill="black" />
              </div>
              <div>
                <h3 className="font-black uppercase tracking-wider text-sm mb-1">Stop & Proceed</h3>
                <p className="text-sm font-bold text-black/70 leading-relaxed">
                  Gracefully stops the current loop. If used during the OASIS Social Simulation (Layer 5), it will save all the social data generated up to that exact moment, conclude the phase, and smoothly transition to the next stage.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="shrink-0 p-2 bg-green-500 text-black border-2 border-black mt-1">
                <ShieldCheck size={20} strokeWidth={2.5} />
              </div>
              <div>
                <h3 className="font-black uppercase tracking-wider text-sm mb-1">Authorize Simulation</h3>
                <p className="text-sm font-bold text-black/70 leading-relaxed">
                  Found in the Seeds review layer (Layer 4). Click this once you are satisfied with the generated agent starting points. It locks in the inputs and officially launches the OASIS Social Simulation.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="shrink-0 p-2 bg-black text-white mt-1">
                <Play size={20} strokeWidth={2.5} fill="currentColor" />
              </div>
              <div>
                <h3 className="font-black uppercase tracking-wider text-sm mb-1">Start Simulation</h3>
                <p className="text-sm font-bold text-black/70 leading-relaxed">
                  Begins the full evaluation pipeline starting from Layer 1 data ingestion. Make sure you've uploaded your documents or clicked "Use Default Inputs" before starting.
                </p>
              </div>
            </div>

          </div>
        </div>

      </div>
    </div>
  );
}
