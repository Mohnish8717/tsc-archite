/**
 * OASIS Real-Time WebSocket Bridge (v3 — Persistent Storage + SQLite Query)
 * 
 * Data flow:
 *  Backend (simulation_engine.py)
 *    └→ /log/oasis_runs/run-{unix_timestamp}/
 *        ├── actions.jsonl        ← JSONL stream (real-time tailing)
 *        ├── {run_id}.sqlite      ← OASIS platform DB (users, posts, comments)
 *        ├── simulation_master.db ← TSC metadata DB
 *        └── prediction_report.json ← Final prediction output
 * 
 *  WebSocket Server (this file)
 *    └→ Watches for new run directories
 *    └→ Tails actions.jsonl for live streaming
 *    └→ Reads SQLite DB for rich historical data (comments, users)
 *    └→ Sends prediction_report.json when simulation completes
 *    └→ Broadcasts everything to all connected frontend clients
 */

import { WebSocketServer } from 'ws';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import Database from 'better-sqlite3';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 8080;
const wss = new WebSocketServer({ port: PORT });
console.log(`\n[WS] 🚀 Listening on ws://localhost:${PORT}`);

// ─── Search paths for simulation runs (persistent + legacy /tmp) ──────────
const OASIS_SEARCH_PATHS = [
  path.resolve(__dirname, '../../log/oasis_runs'),      // NEW: persistent workspace storage
  process.env.OASIS_RUNS_DIR,                            // ENV override
  '/tmp/oasis_runs',                                     // LEGACY: ephemeral /tmp
].filter(Boolean);

// ─── Connected clients registry ───────────────────────────────────────────
const clients = new Set();

function broadcast(obj) {
  const msg = JSON.stringify(obj);
  clients.forEach(ws => {
    if (ws.readyState === 1) ws.send(msg);
  });
}

// ─── Find all simulation run directories across all search paths ──────────
function getAllRunDirs() {
  const allRuns = [];
  for (const searchPath of OASIS_SEARCH_PATHS) {
    if (!searchPath || !fs.existsSync(searchPath)) continue;
    try {
      const entries = fs.readdirSync(searchPath);
      for (const name of entries) {
        const fullPath = path.join(searchPath, name);
        const actionsFile = path.join(fullPath, 'actions.jsonl');
        const pipelineFile = path.join(fullPath, 'pipeline.jsonl');
        const reportFile = path.join(fullPath, 'prediction_report.json');
        try {
          if (fs.statSync(fullPath).isDirectory() && (fs.existsSync(actionsFile) || fs.existsSync(pipelineFile))) {
            const mtime = fs.existsSync(actionsFile)
              ? fs.statSync(actionsFile).mtimeMs
              : fs.statSync(pipelineFile).mtimeMs;
            allRuns.push({ name, fullPath, actionsFile, pipelineFile, reportFile, mtime });
          }
        } catch {}
      }
    } catch {}
  }
  // Sort by modification time, newest first
  return allRuns.sort((a, b) => b.mtime - a.mtime);
}

function getLatestRun() {
  const runs = getAllRunDirs();
  return runs.length > 0 ? runs[0] : null;
}

// ─── Parse actions.jsonl ──────────────────────────────────────────────────
function parseActionsFile(filePath) {
  const agents = {};
  const actions = [];
  const events = [];
  try {
    const lines = fs.readFileSync(filePath, 'utf-8').split('\n').filter(l => l.trim());
    for (const line of lines) {
      try {
        const d = JSON.parse(line);
        if (d.type) { events.push(d); continue; }
        if (d.agent_id && d.action_type) {
          if (!agents[d.agent_id]) agents[d.agent_id] = { agent_id: d.agent_id, agent_name: d.agent_name };
          actions.push(d);
        }
      } catch {}
    }
  } catch {}
  return { agents, actions, events };
}

// ─── Read rich data from per-run SQLite database ──────────────────────────
function readSqliteData(run) {
  const dbFile = path.join(run.fullPath, `${run.name}.sqlite`);
  if (!fs.existsSync(dbFile) || fs.statSync(dbFile).size === 0) return null;

  try {
    const db = new Database(dbFile, { readonly: true });
    const result = {};

    // Users/agents registered on the platform
    try {
      result.users = db.prepare('SELECT user_id, user_name, name, bio, num_posts, num_comments FROM user').all();
    } catch { result.users = []; }

    // Social comments (the actual simulation interactions)
    try {
      result.comments = db.prepare(`
        SELECT c.comment_id, c.user_id, u.name as user_name, c.content, c.created_at, c.num_likes, c.num_dislikes
        FROM comment c LEFT JOIN user u ON c.user_id = u.user_id
        ORDER BY c.created_at DESC LIMIT 200
      `).all();
    } catch { result.comments = []; }

    // Posts (threads created by agents)
    try {
      result.posts = db.prepare(`
        SELECT p.post_id, p.user_id, u.name as user_name, p.content, p.created_at, p.num_comments
        FROM post p LEFT JOIN user u ON p.user_id = u.user_id
        ORDER BY p.created_at DESC LIMIT 50
      `).all();
    } catch { result.posts = []; }

    db.close();
    return result;
  } catch (err) {
    console.error(`[WS] SQLite read error for ${run.name}:`, err.message);
    return null;
  }
}

// ─── Bootstrap a client with a simulation run's complete data ─────────────
function bootstrapClient(ws, run) {
  const send = obj => { if (ws.readyState === 1) ws.send(JSON.stringify(obj)); };

  if (!run) {
    console.log('[WS] No simulation data found — sending waiting state to client');
    send({ type: 'pipeline_reset', stages: { layer1: 'waiting', layer3: 'waiting', layer5: 'waiting' } });
    return;
  }

  // 1. Replay pipeline.jsonl events first (ingestion_sync, persona_sync, pipeline_progress)
  let hasRealDebateMessages = false;
  if (run.pipelineFile && fs.existsSync(run.pipelineFile)) {
    try {
      const pipelineLines = fs.readFileSync(run.pipelineFile, 'utf-8').split('\n').filter(l => l.trim());
      for (const line of pipelineLines) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.type === 'debate_message') {
            hasRealDebateMessages = true;
          }
          send(parsed);
        } catch {}
      }
      console.log(`[WS] 📋 Replayed ${pipelineLines.length} pipeline events from ${run.pipelineFile} (hasRealDebateMessages: ${hasRealDebateMessages})`);
    } catch (err) {
      console.warn('[WS] Could not read pipeline.jsonl:', err.message);
      send({ type: 'pipeline_reset', stages: { layer1: 'waiting', layer3: 'waiting', layer5: 'waiting' } });
    }
  } else {
    send({ type: 'pipeline_reset', stages: { layer1: 'waiting', layer3: 'waiting', layer5: 'waiting' } });
  }

  if (!fs.existsSync(run.actionsFile)) {
    console.log('[WS] No actions.jsonl yet — waiting for simulation to begin');
    return;
  }

  console.log(`[WS] Bootstrapping client from: ${run.name} (${run.fullPath})`);

  // 2. Replay actions.jsonl IN EXACT CHRONOLOGICAL ORDER instantly
  let actionLinesCount = 0;
  const actions = [];
  try {
    const lines = fs.readFileSync(run.actionsFile, 'utf-8').split('\n').filter(l => l.trim());
    console.log(`[WS] 📋 Replaying ${lines.length} high-fidelity events from actions.jsonl`);
    for (const line of lines) {
      try {
        const d = JSON.parse(line);
        send(d);
        actionLinesCount++;
        if (d.agent_id && d.action_type) {
          actions.push(d);
        }
      } catch {}
    }
  } catch (err) {
    console.warn('[WS] Could not replay actions.jsonl:', err.message);
  }

  // 3. Read SQLite database for richer data (if available)
  const sqliteData = readSqliteData(run);
  if (sqliteData) {
    send({ type: 'sqlite_data', data: sqliteData, simulation_id: run.name });
    console.log(`[WS] 💾 Sent SQLite data: ${sqliteData.users?.length ?? 0} users, ${sqliteData.comments?.length ?? 0} comments, ${sqliteData.posts?.length ?? 0} posts`);
  }

  // 4. Seed boardroom debate from real agent comments (only if no real debate messages in pipeline)
  if (!hasRealDebateMessages) {
    const commentActions = actions
      .filter(a => a.action_type && a.action_type.toUpperCase().includes('COMMENT') && a.content)
      .slice(0, 10);

    commentActions.forEach((action, i) => {
      const thoughtEnd = action.content.indexOf('</thought>');
      const clean = thoughtEnd !== -1
        ? action.content.slice(thoughtEnd + 10).trim()
        : action.content.replace(/<[^>]+>/g, '').trim();
      const preview = clean.slice(0, 200);
      if (!preview) return;

      const isChallenge = /risk|gdpr|compliance|concern|privacy|legal/i.test(preview);

      // Map the agent's name/role to a boardroom persona (CEO, CTO, CISO, etc.)
      const roleLower = (action.role || '').toLowerCase();
      const nameLower = (action.agent_name || '').toLowerCase();
      let boardRole = 'CS'; // Default fallback
      if (roleLower.includes('security') || roleLower.includes('ciso')) boardRole = 'CISO';
      else if (roleLower.includes('tech') || roleLower.includes('cto') || roleLower.includes('engineer')) boardRole = 'CTO';
      else if (roleLower.includes('finance') || roleLower.includes('cfo')) boardRole = 'CFO';
      else if (roleLower.includes('product') || roleLower.includes('cpo')) boardRole = 'CPO';
      else if (roleLower.includes('legal') || roleLower.includes('law')) boardRole = 'Legal';
      else if (roleLower.includes('marketing') || roleLower.includes('cmo') || roleLower.includes('medical')) boardRole = 'CMO';
      else if (roleLower.includes('sales') || roleLower.includes('revenue')) boardRole = 'Sales';
      else if (roleLower.includes('data') || roleLower.includes('ml') || roleLower.includes('ai')) boardRole = 'Data';
      else if (roleLower.includes('ceo') || roleLower.includes('executive') || nameLower.includes('ceo')) boardRole = 'CEO';

      send({
        type: 'debate_message',
        message: {
          id: `boot_${i}`,
          sender: boardRole,
          text: `"${preview}${clean.length > 200 ? '…' : ''}"`,
          type: isChallenge ? 'challenge' : 'normal',
        }
      });
    });
  }

  // 5. Send prediction report if available (if not already sent via actions.jsonl replay)
  const hasReport = fs.existsSync(run.reportFile);
  if (hasReport) {
    try {
      const report = JSON.parse(fs.readFileSync(run.reportFile, 'utf-8'));
      send({
        type: 'simulation_report',
        simulation_id: run.name,
        feature_title: report.feature_title ?? run.name,
        nps: report.net_promoter_score ?? 0,
        churn_velocity: report.churn_velocity ?? 0,
        adoption_momentum: report.adoption_momentum ?? 0,
        population_size: report.population_size ?? 0,
        satisfaction_curve: report.satisfaction_curve ?? [],
        frustration_curve: report.frustration_curve ?? [],
        trust_curve: report.trust_curve ?? [],
        risk_distribution: report.risk_distribution ?? {},
        top_risk_factors: report.top_risk_factors ?? [],
        segments: report.segments ?? [],
        decision_events: report.decision_events ?? [],
        focus_group_insights: report.focus_group_insights ?? {},
        executive_summary: report.executive_summary ?? '',
      });
      console.log(`[WS] 📊 Sent prediction report for ${run.name}`);
    } catch (err) {
      console.error('[WS] Report error:', err.message);
    }
  }

  console.log(`[WS] 📦 Bootstrap complete: replayed ${actionLinesCount} events, sqlite=${!!sqliteData}`);
}

function getReportTitle(reportFile) {
  try { return JSON.parse(fs.readFileSync(reportFile, 'utf-8')).feature_title ?? 'Simulation'; }
  catch { return 'Simulation'; }
}

// ─── Live tailing of the active simulation ────────────────────────────────
let activeRun = null;

function stopActiveTail() {
  if (activeRun?.watcher) {
    try { activeRun.watcher.close(); } catch {}
    console.log(`[WS] ⏹ Stopped tailing: ${activeRun.name}`);
  }
  activeRun = null;
  // Also stop the pipeline tail when stopping the main simulation tail
  stopActivePipelineTail();
}

function startTailing(run) {
  stopActiveTail();
  if (!run) return;

  // ── Bug fix #1: actions.jsonl may not exist yet when the run dir is first detected.
  // Mirror the same directory-watch deferral pattern used by startTailingPipeline.
  if (!fs.existsSync(run.actionsFile)) {
    console.log(`[WS] 📡 Waiting for actions.jsonl to appear in: ${run.fullPath}`);
    const dirWatcher = fs.watch(run.fullPath, (eventType, filename) => {
      if (filename === 'actions.jsonl' && fs.existsSync(run.actionsFile)) {
        dirWatcher.close();
        startTailing(run); // retry now that the file exists
      }
    });
    activeRun = { ...run, watcher: dirWatcher, reportWatcher: null };
    return;
  }

  console.log(`[WS] 📡 LIVE tailing: ${run.actionsFile}`);
  let lastSize = fs.statSync(run.actionsFile).size;

  const watcher = fs.watch(run.actionsFile, (eventType) => {
    if (eventType !== 'change') return;
    try {
      const stats = fs.statSync(run.actionsFile);
      if (stats.size <= lastSize) return;

      const readStart = lastSize;
      const readEnd = stats.size;

      const stream = fs.createReadStream(run.actionsFile, {
        start: readStart,
        end: readEnd,
        encoding: 'utf-8',
      });

      stream.on('data', chunk => {
        chunk.split('\n').filter(l => l.trim()).forEach(line => {
          try { broadcast(JSON.parse(line)); } catch {}
        });
      });

      // ── Bug fix #2: advance lastSize only AFTER the stream has fully drained.
      // The original code advanced it synchronously, so a second fs.watch event
      // firing within the same tick saw size <= lastSize and silently skipped lines.
      // This was dropping the majority of agent action events during busy timesteps.
      stream.on('end', () => { lastSize = readEnd; });
    } catch {}
  });

  // Also watch for prediction_report.json and final_recommendation.json appearing
  const reportWatcher = fs.watch(path.dirname(run.actionsFile), (eventType, filename) => {
    if (filename === 'prediction_report.json') {
      const reportFile = path.join(path.dirname(run.actionsFile), filename);
      setTimeout(() => {
        try {
          if (!fs.existsSync(reportFile)) return;
          const report = JSON.parse(fs.readFileSync(reportFile, 'utf-8'));
          broadcast({
            type: 'simulation_report',
            simulation_id: run.name,
            feature_title: report.feature_title ?? run.name,
            nps: report.net_promoter_score ?? 0,
            churn_velocity: report.churn_velocity ?? 0,
            adoption_momentum: report.adoption_momentum ?? 0,
            population_size: report.population_size ?? 0,
            satisfaction_curve: report.satisfaction_curve ?? [],
            frustration_curve: report.frustration_curve ?? [],
            trust_curve: report.trust_curve ?? [],
            risk_distribution: report.risk_distribution ?? {},
            top_risk_factors: report.top_risk_factors ?? [],
            segments: report.segments ?? [],
            decision_events: report.decision_events ?? [],
            focus_group_insights: report.focus_group_insights ?? {},
            executive_summary: report.executive_summary ?? '',
          });
          console.log(`[WS] 📊 LIVE: Prediction report detected and sent!`);
        } catch {}
      }, 500);
    }
    if (filename === 'final_recommendation.json') {
      const recFile = path.join(path.dirname(run.actionsFile), filename);
      setTimeout(() => {
        try {
          if (!fs.existsSync(recFile)) return;
          const rec = JSON.parse(fs.readFileSync(recFile, 'utf-8'));
          broadcast({ type: 'final_recommendation', simulation_id: run.name, ...rec });
          console.log(`[WS] 🏁 LIVE: Final recommendation sent!`);
        } catch {}
      }, 500);
    }
  });

  activeRun = { ...run, watcher, reportWatcher, lastSize };
}

// ─── Polling fallback for fs.watch (macOS kqueue reliability) ─────────────────
// fs.watch can miss events on macOS for new subdirectories. Every 5 s we scan
// for run directories that appeared since the last check and start tailing them.
// Track which runs have been TAILED (not just seen) — a run dir can exist before its JSONL files
let _lastKnownRunNames = new Set(getAllRunDirs().map(r => r.name));
let _tailedRunNames = new Set(activeRun ? [activeRun.name] : []);

setInterval(() => {
  const currentRuns = getAllRunDirs(); // only returns dirs with actions.jsonl OR pipeline.jsonl
  for (const run of currentRuns) {
    const isNew = !_lastKnownRunNames.has(run.name);
    const isUntailed = !_tailedRunNames.has(run.name); // known dir but JSONL just appeared

    if (isNew || isUntailed) {
      _lastKnownRunNames.add(run.name);
      _tailedRunNames.add(run.name);
      console.log(`[WS] 🔄 Polling ${ isNew ? 'NEW' : 'JSONL-appeared'}: ${run.name}`);
      broadcast({
        type: 'pipeline_reset',
        session_id: run.name,
        stages: { layer1: 'waiting', layer3: 'waiting', layer5: 'waiting' },
      });
      startTailing(run);
      startTailingPipeline(run);
    }
  }
}, 5000);

// ─── Live tailing of pipeline.jsonl (ingestion, persona, progress events) ──
let activePipelineTail = null;

function stopActivePipelineTail() {
  if (activePipelineTail?.watcher) {
    try { activePipelineTail.watcher.close(); } catch {}
    console.log(`[WS] ⏹ Stopped tailing pipeline.jsonl: ${activePipelineTail.name}`);
  }
  activePipelineTail = null;
}

function startTailingPipeline(run) {
  stopActivePipelineTail();
  if (!run) return;

  const pipelineFile = run.pipelineFile || path.join(run.fullPath, 'pipeline.jsonl');

  // If the file doesn't exist yet, watch the directory and start tailing once it appears
  if (!fs.existsSync(pipelineFile)) {
    console.log(`[WS] 📡 Waiting for pipeline.jsonl to appear in: ${run.fullPath}`);
    const dirWatcher = fs.watch(run.fullPath, (eventType, filename) => {
      if (filename === 'pipeline.jsonl' && fs.existsSync(pipelineFile)) {
        dirWatcher.close();
        startTailingPipeline({ ...run, pipelineFile });
      }
    });
    activePipelineTail = { name: run.name, watcher: dirWatcher };
    return;
  }

  console.log(`[WS] 📡 LIVE tailing pipeline: ${pipelineFile}`);
  let lastSize = fs.statSync(pipelineFile).size;

  const watcher = fs.watch(pipelineFile, (eventType) => {
    if (eventType !== 'change') return;
    try {
      const stats = fs.statSync(pipelineFile);
      if (stats.size <= lastSize) return;

      const stream = fs.createReadStream(pipelineFile, {
        start: lastSize,
        end: stats.size,
        encoding: 'utf-8',
      });

      stream.on('data', chunk => {
        chunk.split('\n').filter(l => l.trim()).forEach(line => {
          try {
            const event = JSON.parse(line);
            broadcast(event);
            console.log(`[WS] 🔄 Pipeline event: ${event.type}`);
          } catch {}
        });
      });
      lastSize = stats.size;
    } catch {}
  });

  activePipelineTail = { name: run.name, watcher, lastSize };
}

// ─── Watch for NEW simulation directories across all search paths ──────────
function watchForNewSimulations() {
  const watchers = [];
  const knownDirs = new Set(getAllRunDirs().map(r => r.name));

  for (const searchPath of OASIS_SEARCH_PATHS) {
    if (!searchPath) continue;
    try { fs.mkdirSync(searchPath, { recursive: true }); } catch {}
    if (!fs.existsSync(searchPath)) continue;

    const watcher = fs.watch(searchPath, (eventType, filename) => {
      if (!filename || knownDirs.has(filename)) return;

      setTimeout(() => {
        const allRuns = getAllRunDirs();
        const newRun = allRuns.find(r => r.name === filename);
        if (!newRun || knownDirs.has(newRun.name)) return;
        knownDirs.add(newRun.name);

        console.log(`\n[WS] 🆕 NEW SIMULATION: ${newRun.name} at ${newRun.fullPath}`);
        broadcast({
          type: 'pipeline_reset',
          session_id: newRun.name,
          stages: { layer1: 'waiting', layer3: 'waiting', layer5: 'waiting' },
        });
        startTailing(newRun);
        startTailingPipeline(newRun);
      }, 500); // Short grace period — pipeline.jsonl is created nearly immediately
    });

    watchers.push(watcher);
    console.log(`[WS] 👁 Monitoring: ${searchPath}`);
  }

  return watchers;
}

// ─── Static data (always sent to every client) ─────────────────────────────
// NOTE: These are now REMOVED — data comes dynamically from pipeline.jsonl
// kept as a comment for reference only:
// INGESTION_NODES — now emitted by orchestrator.py after Layer 1 completes
// PERSONAS        — now emitted by orchestrator.py after Layer 2 persona generation

// ─── Startup ──────────────────────────────────────────────────────────────
const latestRun = getLatestRun();
if (latestRun) {
  console.log(`[WS] 📂 Latest run: ${latestRun.name} (${latestRun.fullPath})`);
  startTailing(latestRun);
  startTailingPipeline(latestRun);
} else {
  console.log('[WS] ⚠  No simulation runs found — waiting for new simulation to start');
}

const dirWatchers = watchForNewSimulations();

// ─── Connection handler ──────────────────────────────────────────────────
wss.on('connection', (ws) => {
  console.log('[WS] ✅ Client connected');
  clients.add(ws);
  ws.isAlive = true;
  ws.on('pong', () => { ws.isAlive = true; });

  // Re-discover latest run on each connection (in case new data arrived)
  const currentLatest = getLatestRun();

  // ── Switch tails if a newer run exists that we're not yet tailing ──
  // This covers the case where the WS server was running, a new simulation
  // started, but the polling hadn't picked it up yet (no JSONL at polling time).
  if (currentLatest && (!activeRun || currentLatest.name !== activeRun.name)) {
    console.log(`[WS] 🔀 Client connect: switching tail from ${activeRun?.name ?? 'none'} → ${currentLatest.name}`);
    _tailedRunNames.add(currentLatest.name);
    broadcast({
      type: 'pipeline_reset',
      session_id: currentLatest.name,
      stages: { layer1: 'waiting', layer3: 'waiting', layer5: 'waiting' },
    });
    startTailing(currentLatest);
    startTailingPipeline(currentLatest);
  }

  bootstrapClient(ws, currentLatest);

  ws.on('close', () => { clients.delete(ws); console.log('[WS] ❌ Disconnected'); });
  ws.on('error', (err) => { clients.delete(ws); console.error('[WS] Error:', err.message); });
});

// G7: Heartbeat — terminate dead connections every 30 s (ws protocol.md pattern)
const heartbeatInterval = setInterval(() => {
  wss.clients.forEach((ws) => {
    if (ws.isAlive === false) {
      console.log('[WS] 💀 Terminating dead connection');
      return ws.terminate();
    }
    ws.isAlive = false;
    ws.ping();
  });
}, 30000);

process.on('SIGINT', () => {
  clearInterval(heartbeatInterval);    // G7: stop heartbeat on shutdown
  stopActiveTail();
  dirWatchers.forEach(w => { try { w.close(); } catch {} });
  wss.close();
  process.exit(0);
});
