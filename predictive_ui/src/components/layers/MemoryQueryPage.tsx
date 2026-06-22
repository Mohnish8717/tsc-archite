import { useMemo, useState } from 'react';
import { Brain, Loader2, Search, Database, MessageSquare, AlertTriangle, ChevronRight } from 'lucide-react';
import { usePipelineStore } from '../../store/usePipelineStore';

// ── Types ─────────────────────────────────────────────────────────────────

type BankCategory = 'Simulation' | 'Boardroom';

interface Bank {
  bank_id: string;
  label: string;
  category: BankCategory;
}

// ── Bank ID derivation ────────────────────────────────────────────────────

function toBoardroomBankId(personaName: string): string {
  return `boardroom-${personaName.replace(/ /g, '_').replace(/\\./g, '')}`;
}

// ── Component ─────────────────────────────────────────────────────────────

export default function MemoryQueryPage() {
  const sessionId = usePipelineStore((s) => s.sessionId);
  const boardroomPersonas = usePipelineStore((s) => s.boardroomPersonas);
  const simulationConfig = usePipelineStore((s) => s.simulationConfig);

  const [selectedBank, setSelectedBank] = useState<Bank | null>(null);
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── Derive bank list from store (no API needed) ──────────────────────────
  const banks = useMemo<Bank[]>(() => {
    if (!sessionId) return [];
    const list: Bank[] = [];

    // 1. Unified OASIS Bank (Focus Group Interviews + Agent Traces)
    if (sessionId) {
      list.push({
        bank_id: `oasis-${sessionId}`,
        label: 'Unified Simulation Bank (Focus Groups & Agent Traces)',
        category: 'Simulation',
      });
    }

    // 2. Global World Bank (LightRAG)
    if (sessionId) {
      list.push({
        bank_id: `world-${sessionId}`,
        label: "Company Database",
        category: "Simulation"
      });
    }

    // 3. One bank per boardroom persona
    boardroomPersonas.forEach((p) => {
      list.push({
        bank_id: toBoardroomBankId(p.name),
        label: `${p.name}`,
        category: 'Boardroom',
      });
    });

    return list;
  }, [sessionId, simulationConfig, boardroomPersonas]);

  // ── Group banks by category ──────────────────────────────────────────────
  const grouped = useMemo(() => {
    const groups: Record<BankCategory, Bank[]> = { Simulation: [], Boardroom: [] };
    banks.forEach((b) => groups[b.category].push(b));
    return groups;
  }, [banks]);

  // ── Query submission ──────────────────────────────────────────────────────
  const handleSubmit = async () => {
    if (!selectedBank || !query.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('/api/hindsight/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bank_id: selectedBank.bank_id,
          query: query.trim(),
          max_tokens: 800,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data.answer);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
  };

  // ── Empty state — no active run ───────────────────────────────────────────
  if (!sessionId) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center gap-6 bg-white">
        <div className="w-20 h-20 bg-black border-4 border-black flex items-center justify-center shadow-neo-black">
          <Brain className="w-10 h-10 text-brand" strokeWidth={3} />
        </div>
        <div className="text-center">
          <h2 className="font-black text-2xl uppercase tracking-tighter mb-2">No Active Run</h2>
          <p className="font-bold text-sm uppercase tracking-widest text-black/40 max-w-xs mx-auto">
            Start a simulation pipeline first. Memory banks will appear here once the run begins.
          </p>
        </div>
      </div>
    );
  }

  // ── Main layout ──────────────────────────────────────────────────────────
  return (
    <div className="w-full h-full flex bg-white overflow-hidden">

      {/* ── Left: Bank selector ──────────────────────────────── */}
      <div className="w-72 flex-none flex flex-col border-r-8 border-black overflow-hidden">
        <div className="bg-black text-white px-5 py-4 border-b-4 border-brand flex items-center gap-3 flex-none">
          <Database className="w-4 h-4 text-brand" strokeWidth={3} />
          <h2 className="font-black text-sm uppercase tracking-widest">Memory Banks</h2>
        </div>

        {/* Run badge */}
        <div className="px-5 py-3 border-b-4 border-black bg-black/5 flex-none">
          <div className="text-xs font-black uppercase tracking-widest text-black/40 mb-1">Current Run</div>
          <div className="font-black text-xs text-black truncate">{sessionId}</div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {(['Simulation', 'Boardroom'] as BankCategory[]).map((cat) => {
            const catBanks = grouped[cat];
            if (catBanks.length === 0) return null;
            return (
              <div key={cat}>
                {/* Category header */}
                <div className="px-5 py-2 bg-black text-brand flex items-center gap-2 border-b-2 border-brand sticky top-0 z-10">
                  {cat === 'Simulation'
                    ? <Database className="w-3 h-3" strokeWidth={3} />
                    : <MessageSquare className="w-3 h-3" strokeWidth={3} />}
                  <span className="font-black text-xs uppercase tracking-widest">{cat}</span>
                </div>

                {catBanks.map((bank) => {
                  const isActive = selectedBank?.bank_id === bank.bank_id;
                  return (
                    <button
                      key={bank.bank_id}
                      id={`bank-${bank.bank_id}`}
                      onClick={() => {
                        setSelectedBank(bank);
                        setResult(null);
                        setError(null);
                      }}
                      className={`w-full flex items-center gap-3 px-5 py-3 text-left border-b-2 border-black/10 transition-colors cursor-pointer ${
                        isActive
                          ? 'bg-brand text-black border-l-4 border-l-black'
                          : 'bg-white text-black hover:bg-black/5 border-l-4 border-l-transparent'
                      }`}
                    >
                      <ChevronRight
                        className={`w-3 h-3 shrink-0 transition-opacity ${isActive ? 'opacity-100' : 'opacity-0'}`}
                        strokeWidth={3}
                      />
                      <span className="font-black text-xs uppercase tracking-wider leading-tight flex-1 min-w-0">
                        {bank.label}
                      </span>
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Right: Query + Result ────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="bg-black text-white px-6 py-4 border-b-4 border-brand flex items-center gap-3 flex-none">
          <Brain className="w-4 h-4 text-brand" strokeWidth={3} />
          <h2 className="font-black text-sm uppercase tracking-widest flex-1">
            {selectedBank ? `Query: ${selectedBank.label}` : 'Select a Memory Bank'}
          </h2>
          {selectedBank && (
            <span className="text-xs font-black text-black/50 bg-brand/20 px-2 py-0.5 border border-brand/40">
              {selectedBank.bank_id}
            </span>
          )}
        </div>

        <div className="flex-1 flex flex-col p-6 gap-5 overflow-y-auto">

          {/* No bank selected prompt */}
          {!selectedBank && (
            <div className="flex-1 flex flex-col items-center justify-center gap-4 opacity-40">
              <Search className="w-12 h-12" strokeWidth={2} />
              <p className="font-black text-sm uppercase tracking-widest text-center">
                Select a bank on the left to query
              </p>
            </div>
          )}

          {/* Query form */}
          {selectedBank && (
            <>
              <div className="border-4 border-black shadow-neo-black flex-none">
                <div className="bg-black text-white px-4 py-2 flex items-center gap-2">
                  <Search className="w-3 h-3 text-brand" strokeWidth={3} />
                  <span className="font-black text-xs uppercase tracking-widest">Your Question</span>
                  <span className="ml-auto font-bold text-xs text-white/30">⌘↩ to submit</span>
                </div>
                <textarea
                  id="hindsight-query-input"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={`Ask about ${selectedBank.label}...`}
                  rows={4}
                  className="w-full px-4 py-3 font-bold text-sm resize-none outline-none bg-white placeholder:text-black/20"
                />
              </div>

              <button
                id="hindsight-query-submit"
                onClick={handleSubmit}
                disabled={loading || !query.trim()}
                className={`flex items-center justify-center gap-3 py-4 border-4 border-black font-black text-sm uppercase tracking-widest transition-all duration-200
                  ${loading || !query.trim()
                    ? 'bg-black/10 text-black/30 cursor-not-allowed'
                    : 'bg-black text-white shadow-neo-brand cursor-pointer hover:translate-x-0.5 hover:translate-y-0.5 hover:shadow-none'
                  }`}
              >
                {loading
                  ? <><Loader2 className="w-4 h-4 animate-spin" strokeWidth={3} /> Querying Memory Bank…</>
                  : <><Brain className="w-4 h-4" strokeWidth={3} /> Query Bank</>
                }
              </button>

              {/* Error state */}
              {error && (
                <div className="border-4 border-red-500 shadow-[4px_4px_0px_0px_rgba(239,68,68,1)] p-5 bg-red-50">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className="w-4 h-4 text-red-600" strokeWidth={3} />
                    <span className="font-black text-xs uppercase tracking-widest text-red-600">Query Failed</span>
                  </div>
                  <p className="font-bold text-sm text-red-700">{error}</p>
                </div>
              )}

              {/* Result */}
              {result && !loading && (
                <div className="border-4 border-black shadow-neo-black flex-1 min-h-0 flex flex-col mt-4">
                  <div className="bg-black text-white px-4 py-2 flex items-center gap-2 border-b-2 border-brand flex-none">
                    <Brain className="w-3 h-3 text-brand" strokeWidth={3} />
                    <span className="font-black text-xs uppercase tracking-widest">Response</span>
                    <span className="ml-auto font-bold text-xs text-white/30">{selectedBank.label}</span>
                  </div>
                  <div className="flex-1 overflow-y-auto p-5">
                    <p className="font-bold text-sm leading-relaxed whitespace-pre-wrap text-black/80">
                      {result}
                    </p>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
