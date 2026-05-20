import React, { useState } from 'react';
import { Activity, Play, Plus, X, FileText, Upload } from 'lucide-react';

interface InputSetupPageProps {
  onStartSimulation: (inputs: any) => Promise<void>;
  onSkip?: () => void;
}

interface DocumentInput {
  id: string;
  title: string;
  content: string;
}

export default function InputSetupPage({ onStartSimulation, onSkip }: InputSetupPageProps) {
  const [documents, setDocuments] = useState<DocumentInput[]>([
    { id: '1', title: 'Feature Proposal', content: '' }
  ]);
  const [loading, setLoading] = useState(false);

  const addDocument = (title: string = 'New Document') => {
    const isRecommended = ['Company Context', 'Feature Proposal', 'Customer Interviews', 'Support Tickets', 'Analytics JSON'].includes(title);
    if (isRecommended && documents.some(d => d.title.toLowerCase() === title.toLowerCase())) {
      return; // Already exists
    }
    setDocuments([...documents, { id: Math.random().toString(), title, content: '' }]);
  };

  const removeDocument = (id: string) => {
    setDocuments(documents.filter(d => d.id !== id));
  };

  const updateDocument = (id: string, field: 'title' | 'content', value: string) => {
    setDocuments(documents.map(d => d.id === id ? { ...d, [field]: value } : d));
  };

  const handleFileUpload = (id: string, event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result;
      if (typeof text === 'string') {
        setDocuments(docs => docs.map(d => 
          d.id === id ? { ...d, content: text, title: file.name.replace(/\.[^/.]+$/, "") } : d
        ));
      }
    };
    reader.onerror = () => {
      console.error("Failed to read file");
      alert("Failed to read file");
    };
    reader.readAsText(file);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    // Map the dynamic documents into the backend's expected 5 categories
    const payload = {
      feature_proposal: '',
      company_context: '',
      support_tickets: '',
      customer_interviews: '',
      analytics: ''
    };

    documents.forEach(doc => {
      const t = doc.title.toLowerCase();
      if (t.includes('proposal') || t.includes('spec') || t.includes('feature')) {
        payload.feature_proposal += `\n\n--- ${doc.title} ---\n${doc.content}`;
      } else if (t.includes('support') || t.includes('ticket') || t.includes('complaint')) {
        payload.support_tickets += `\n\n--- ${doc.title} ---\n${doc.content}`;
      } else if (t.includes('interview') || t.includes('research') || t.includes('user')) {
        payload.customer_interviews += `\n\n--- ${doc.title} ---\n${doc.content}`;
      } else if (t.includes('analytic') || t.includes('data')) {
        payload.analytics += `\n\n--- ${doc.title} ---\n${doc.content}`;
      } else {
        payload.company_context += `\n\n--- ${doc.title} ---\n${doc.content}`;
      }
    });

    try {
      await onStartSimulation(payload);
    } catch (err) {
      console.error(err);
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
              onClick={() => addDocument('Custom Document')}
              className="px-4 py-2 bg-black text-white border-4 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] font-black text-sm uppercase tracking-widest hover:-translate-y-1 hover:translate-x-1 hover:shadow-none transition-all flex items-center gap-2"
            >
              <Plus size={16} strokeWidth={3} /> Custom
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-8">
          {documents.map((doc, index) => (
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
                    onChange={(e) => updateDocument(doc.id, 'title', e.target.value)}
                    className="text-2xl font-black uppercase tracking-tighter w-full outline-none bg-transparent placeholder-gray-400"
                    placeholder="Document Title..."
                    required
                  />
                </div>
                <div>
                  <label htmlFor={`file-upload-${doc.id}`} className="cursor-pointer flex items-center gap-2 px-3 py-1 bg-white border-4 border-black shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] font-black text-xs uppercase hover:-translate-y-0.5 hover:translate-x-0.5 hover:shadow-none transition-all">
                    <Upload size={14} strokeWidth={3} /> Upload File
                  </label>
                  <input 
                    id={`file-upload-${doc.id}`}
                    type="file" 
                    className="hidden" 
                    onChange={(e) => handleFileUpload(doc.id, e)} 
                  />
                </div>
              </div>
              <textarea 
                required
                className="w-full h-40 outline-none text-sm font-mono font-bold resize-y text-black placeholder-gray-400 bg-transparent"
                placeholder="Paste raw text, context, or JSON here..."
                value={doc.content}
                onChange={e => updateDocument(doc.id, 'content', e.target.value)}
              />
            </div>
          ))}

          {documents.length === 0 && (
            <div className="text-center p-12 border-4 border-black border-dashed bg-white/50 font-black uppercase tracking-widest text-xl">
              Add at least one document to start the simulation.
            </div>
          )}

          {/* Submit Button */}
          <div className="flex flex-col gap-4 mt-8">
            <button 
              type="submit" 
              disabled={loading || documents.length === 0}
              className="w-full py-6 bg-black text-white border-4 border-black shadow-neo-white font-black uppercase tracking-[0.2em] text-2xl flex items-center justify-center gap-4 transition-all hover:translate-x-1 hover:-translate-y-1 disabled:opacity-50"
            >
              {loading ? (
                <span className="animate-pulse">Starting Engine...</span>
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
      </div>
    </div>
  );
}
