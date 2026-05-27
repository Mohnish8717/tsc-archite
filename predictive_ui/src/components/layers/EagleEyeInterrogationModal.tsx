import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { sendEagleEyeCommand } from '../../api/commands';
import { usePipelineStore } from '../../store/usePipelineStore';

interface EagleEyeInterrogationModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialAgentId?: string;
}

export const EagleEyeInterrogationModal: React.FC<EagleEyeInterrogationModalProps> = ({
  isOpen,
  onClose,
  initialAgentId = '',
}) => {
  const { sessionId, spawnedAgents } = usePipelineStore();
  const [selectedAgentId, setSelectedAgentId] = useState<string>(initialAgentId);
  const [question, setQuestion] = useState<string>('');

  useEffect(() => {
    if (isOpen) {
      setSelectedAgentId(initialAgentId);
      setQuestion('');
      setError(null);
    }
  }, [isOpen, initialAgentId]);
  
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sessionId || !selectedAgentId || !question.trim()) return;

    setIsSubmitting(true);
    setError(null);

    try {
      await sendEagleEyeCommand(sessionId, selectedAgentId, question.trim());
      setQuestion('');
      onClose();
    } catch (err: any) {
      setError(err.message || 'An error occurred while sending the question.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const modalContent = (
    <div className="fixed inset-0 z-[2147483647] flex items-center justify-center bg-black/60 backdrop-blur-md font-mono p-4">
      <div className="bg-[#FAF9F6] border-[6px] border-black p-8 w-full max-w-2xl shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] text-black relative">
        
        {/* Close button top right */}
        <button 
          onClick={onClose}
          className="absolute top-4 right-4 bg-[#EF4444] text-white border-[3px] border-black shadow-[4px_4px_0_0_#000] px-3 py-1 font-black text-sm uppercase hover:-translate-y-1 hover:shadow-[6px_6px_0_0_#000] active:translate-y-0 active:shadow-[0_0_0_0_#000] transition-all"
        >
          X
        </button>

        <h2 className="text-3xl font-black mb-8 uppercase tracking-widest border-b-4 border-black pb-4">
          🦅 Interrogate Agent
        </h2>
        
        {error && (
          <div className="bg-[#EF4444] border-4 border-black text-white px-4 py-3 mb-6 font-bold uppercase shadow-[4px_4px_0_0_#000]">
            ERROR: {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-black uppercase tracking-widest">Select Target Agent</label>
            <select
              value={selectedAgentId}
              onChange={(e) => setSelectedAgentId(e.target.value)}
              required
              className="w-full bg-white border-4 border-black p-4 font-bold focus:outline-none focus:bg-[#F3F4F6] shadow-[4px_4px_0_0_#000] transition-colors appearance-none cursor-pointer"
              style={{ backgroundImage: 'linear-gradient(45deg, transparent 50%, #000 50%), linear-gradient(135deg, #000 50%, transparent 50%)', backgroundPosition: 'calc(100% - 20px) calc(1em + 6px), calc(100% - 15px) calc(1em + 6px)', backgroundSize: '5px 5px, 5px 5px', backgroundRepeat: 'no-repeat' }}
            >
              <option value="" disabled>-- Select an agent --</option>
              {Object.values(spawnedAgents).map(agent => (
                <option key={agent.agent_id} value={agent.agent_id}>
                  {agent.agent_name} ({agent.role || 'Agent'})
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-sm font-black uppercase tracking-widest">Your Question</label>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              required
              rows={4}
              placeholder="e.g. Why did you change your stance on the UI framework?"
              className="w-full bg-white border-4 border-black p-4 font-bold placeholder-gray-400 focus:outline-none focus:bg-[#F3F4F6] shadow-[4px_4px_0_0_#000] resize-none transition-colors"
            />
          </div>

          <div className="flex justify-end gap-6 mt-4">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="px-6 py-3 bg-white border-[4px] border-black text-black font-black uppercase tracking-widest shadow-[6px_6px_0_0_#000] hover:-translate-y-1 hover:shadow-[8px_8px_0_0_#000] active:translate-y-1 active:shadow-[2px_2px_0_0_#000] transition-all disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !selectedAgentId || !question.trim()}
              className="px-8 py-3 bg-[#10B981] border-[4px] border-black text-black font-black uppercase tracking-widest shadow-[6px_6px_0_0_#000] hover:-translate-y-1 hover:shadow-[8px_8px_0_0_#000] active:translate-y-1 active:shadow-[2px_2px_0_0_#000] transition-all disabled:opacity-50 flex items-center gap-2"
            >
              {isSubmitting ? 'Sending...' : 'Send Question'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );

  return createPortal(modalContent, document.body);
};
