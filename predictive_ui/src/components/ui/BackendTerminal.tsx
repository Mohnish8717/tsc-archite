import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePipelineStore } from '../../store/usePipelineStore';
import { X, Terminal } from 'lucide-react';

interface BackendTerminalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function BackendTerminal({ isOpen, onClose }: BackendTerminalProps) {
  const { systemLogs } = usePipelineStore();
  const endOfMessagesRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (autoScroll && endOfMessagesRef.current) {
      endOfMessagesRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [systemLogs, autoScroll, isOpen]);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const target = e.target as HTMLDivElement;
    const isAtBottom = target.scrollHeight - target.scrollTop <= target.clientHeight + 50;
    setAutoScroll(isAtBottom);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          drag
          dragConstraints={{ left: -500, right: 500, top: -500, bottom: 500 }}
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
          className="fixed bottom-4 right-4 w-[600px] h-[400px] bg-[#0c0c0c] border border-emerald-500/30 rounded-lg shadow-2xl flex flex-col overflow-hidden z-50 font-mono"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-3 py-2 bg-[#1a1a1a] border-b border-emerald-500/20 cursor-move">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-semibold text-emerald-400 tracking-wider">BACKEND_SYSTEM_LOGS</span>
            </div>
            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Logs Container */}
          <div 
            className="flex-1 overflow-y-auto p-3 text-xs leading-relaxed space-y-1"
            onScroll={handleScroll}
          >
            {systemLogs.length === 0 ? (
              <div className="text-gray-500 italic">Waiting for backend logs...</div>
            ) : (
              systemLogs.map((log, idx) => {
                let textColor = "text-gray-300";
                if (log.includes("ERROR") || log.includes("Exception")) textColor = "text-red-400";
                else if (log.includes("WARNING")) textColor = "text-yellow-400";
                else if (log.includes("INFO")) textColor = "text-emerald-300";

                return (
                  <div key={idx} className={`break-words ${textColor}`}>
                    {log}
                  </div>
                );
              })
            )}
            <div ref={endOfMessagesRef} />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
