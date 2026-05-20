import { usePipelineStore } from '../store/usePipelineStore';

export const runMockDebate = () => {
  const { addDebateMessage, setActiveSpeaker } = usePipelineStore.getState();

  const mockConvo = [
    { sender: 'CEO', text: 'We need to address the new dashboard metrics. The user retention is dropping rapidly.', type: 'normal' as const },
    { sender: 'CTO', text: 'The backend latency is causing timeouts. We need to refactor the database layer immediately.', type: 'normal' as const },
    { sender: 'CISO', text: 'Hold on. If we touch the database layer without a security audit, we risk exposing encrypted PI strings.', type: 'challenge' as const },
    { sender: 'Product', text: 'Users do not care about the architecture! They just want the dashboard to load fast. We must ship this week.', type: 'normal' as const },
    { sender: 'Legal', text: 'I agree with CISO. Data compliance is our first priority here. Do not deploy without the audit.', type: 'normal' as const },
  ];

  let i = 0;
  const interval = setInterval(() => {
    if (i >= mockConvo.length) {
      clearInterval(interval);
      setActiveSpeaker(null);
      return;
    }
    const msg = mockConvo[i];
    addDebateMessage({ id: Math.random().toString(), ...msg });
    setActiveSpeaker(msg.sender);
    i++;
  }, 4000);
};
