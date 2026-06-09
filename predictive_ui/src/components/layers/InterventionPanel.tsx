import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { sendInterventionCommand } from '../../api/commands';

interface InterventionPanelProps {
  sessionId: string;
  isOpen: boolean;
  onClose: () => void;
}

export const InterventionPanel: React.FC<InterventionPanelProps> = ({
  sessionId,
  isOpen,
  onClose,
}) => {
  const [eventText, setEventText] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setEventText('');
      setError(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!eventText.trim()) {
      setError("Please enter an intervention event.");
      return;
    }
    
    setIsSubmitting(true);
    setError(null);
    
    try {
      await sendInterventionCommand(sessionId, eventText.trim());
      setEventText('');
      onClose();
    } catch (err: any) {
      setError(err.message || "Failed to send intervention.");
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

        <h2 className="text-3xl font-black mb-2 uppercase tracking-widest border-b-4 border-black pb-4 text-[#EF4444] flex items-center gap-3">
          <span>⚡</span> God's Eye Intervention
        </h2>
        <p className="mb-8 mt-2 text-sm font-bold opacity-80 uppercase">
          Inject a global event to stress-test the simulation. This will forcefully rewrite the agents' memory banks (Override Mechanism).
        </p>
        
        {error && (
          <div className="bg-[#EF4444] border-4 border-black text-white px-4 py-3 mb-6 font-bold uppercase shadow-[4px_4px_0_0_#000]">
            ERROR: {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-black uppercase tracking-widest">Intervention Scenario</label>
            <textarea
              value={eventText}
              onChange={(e) => setEventText(e.target.value)}
              required
              rows={4}
              placeholder="e.g. A major security breach has just been announced in the news..."
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
              disabled={isSubmitting || !eventText.trim()}
              className="px-8 py-3 bg-[#EF4444] border-[4px] border-black text-white font-black uppercase tracking-widest shadow-[6px_6px_0_0_#000] hover:-translate-y-1 hover:shadow-[8px_8px_0_0_#000] active:translate-y-1 active:shadow-[2px_2px_0_0_#000] transition-all disabled:opacity-50 flex items-center gap-2"
            >
              {isSubmitting ? 'Injecting...' : 'Inject Event'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );

  return createPortal(modalContent, document.body);
};
