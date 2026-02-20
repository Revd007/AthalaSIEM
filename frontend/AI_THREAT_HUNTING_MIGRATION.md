# AI Analysis & Threat Hunting – Real Data Migration

## What Was Done

- **Auth**: On **login**, all queries with keys `ai`, `threat-hunting`, `dashboard-summary`, `dashboard` are invalidated so data refetches and metrics no longer stay at 0 after logout → login.
- **Logout**: Existing behavior kept (clear cache, remove tokens, redirect to `/login`).
- **API client**: `src/lib/ai-api.ts` – typed functions for all AI and threat-hunting endpoints. All calls use relative paths (`/api/ai-analysis/*`, `/api/threat-hunting/*`); .NET proxy forwards to Python (no direct Python URL in frontend).
- **Hooks**: `src/hooks/useAiData.ts` – `useAiOverview`, `useAiAnomalies`, `useAiBehavior`, `useAiPredictive`, `useAiAutomatedResponse`, `useAiOsint`, `useHuntDashboard`, `useHuntBehavior`, `useIocScan`, `useLiveHuntStart`, `useLiveHuntResults`.
- **Charts**: `src/components/charts/AnomalyTimeline.tsx`, `PredictionTimeline.tsx`, `MitreBarChart.tsx` – consume real API data.
- **Dashboard card**: `src/components/dashboard/AiOverviewCards.tsx` – uses `useAiOverview()`, shows real counts.
- **Pages**:  
  - `src/app/dashboard/ai-analysis/page.tsx` – full rewrite; every tab (Overview, Anomaly, Behavior, Predictive, Automated Response, OSINT) uses the hooks above and shows loading/error states and real data only.  
  - `src/app/dashboard/threat-hunting/page.tsx` – Dashboard and Behavior tabs use real API; IOC Scanner uses `POST /api/threat-hunting/ioc/scan`; Live Hunting uses `POST /api/threat-hunting/live/start` and `GET /api/threat-hunting/live/{id}/results`. YARA, Sigma, Threat Intel, Playbooks still use existing services (already wired to backend).
- **IOC Scanner**: `src/components/ThreatHunting/IOCScanner.tsx` – now calls `aiApi.scanIoc({ value })` per IOC; no mock or random data.

## Migration Steps

1. **No .env changes** for frontend; keep `NEXT_PUBLIC_API_URL` pointing at .NET (e.g. `http://localhost:9595`).
2. **.NET proxy**: Already forwards `/api/ai-analysis/*`, `/api/threat-hunting/*`, `/api/detection-rules/*`, `/api/threatintelligence/*`, `/api/playbooks` to Python (port 9797). No change required unless you moved Python to another port.
3. **Nothing to delete**: Old AI/ThreatHunting components (e.g. `AIThreatAnalyzer`, `AnomalyDetection`) remain in the codebase but are no longer used by the ai-analysis page; the new page uses inline tab content and the shared hooks/charts. You can remove them later if you want.
4. **Install deps**: If not already done, `npm install` in `frontend` (for TanStack Query persist packages used in layout).

## Query Keys (for invalidation / refetch)

| Key | Description |
|-----|-------------|
| `['ai', 'overview']` | AI overview stats |
| `['ai', 'anomalies', { period: '24h' }]` | Anomaly detection |
| `['ai', 'behavior']` | Behavioral analytics |
| `['ai', 'predictive']` | Predictive analysis |
| `['ai', 'automated-response']` | Playbook executions |
| `['ai', 'osint']` | OSINT |
| `['threat-hunting', 'dashboard']` | Hunt dashboard |
| `['threat-hunting', 'behavior']` | MITRE behavior |
| `['threat-hunting', 'live', sessionId]` | Live hunt results |

On **login**, all queries whose first key is `ai`, `threat-hunting`, `dashboard-summary`, or `dashboard` are invalidated so they refetch and show real numbers.

## Example API Responses (Python backend)

These shapes are what the UI expects; they produce non-zero/real data when the backend has data.

- **GET /api/ai-analysis/overview**  
  `{ activeThreats, avgConfidence, detectionRate24h, responseTime, mitreCoveragePercent, insightsTrend[], latestInsights[] }`

- **GET /api/ai-analysis/anomalies**  
  `{ anomalyScore, detectedToday, highSeverityAlerts, totalLogsAnalyzed, anomalyTimeline24h: [{ time, count }], detectedAnomalies[] }`

- **GET /api/ai-analysis/behavior**  
  `{ userActivityTimeline[], usersMonitored, anomaliesToday, avgRiskScore, highRiskUsers[] }`

- **GET /api/ai-analysis/predictive**  
  `{ activePredictionsCount, criticalAlerts, totalAlerts24h, highRiskPredictions, predictionTimeline[], activePredictions[] }`

- **GET /api/ai-analysis/automated-response**  
  `{ recentAutomatedActions[] }`

- **GET /api/threat-hunting/dashboard**  
  `{ huntActivityLast7Days[], activeHunts, totalFindings, avgHuntDuration, successRate, recentFindings[] }`

- **POST /api/threat-hunting/ioc/scan**  
  Body: `{ value: string }`. Response: `{ matchesFound, results[], historicalMatches[] }`

- **GET /api/threat-hunting/behavior**  
  `{ mitreTechniqueCounts: [{ technique, count }], processBehavior[], networkBehavior[], userBehavior[] }`

- **POST /api/threat-hunting/live/start**  
  Body: `{ query, timeRangeMinutes? }`. Response: `{ sessionId, findingsCount, status }`

- **GET /api/threat-hunting/live/{sessionId}/results**  
  `{ sessionId, status, findingsCount, findings[] }`

## Testing Checklist

1. **Login → Dashboard → Logout → Login**  
   AI Overview and other metrics should show real counts after second login (no reset to 0).
2. **Refresh while logged in**  
   With query persistence, data should not reset to zero.
3. **AI Analysis tabs**  
   Overview, Anomaly, Behavior, Predictive, Automated Response, OSINT all load from API; loading skeletons and error messages when the backend is down.
4. **Threat Hunting**  
   Dashboard and Behavior show real data; IOC Scanner and Live Hunting use the endpoints above.
5. **No mocks**  
   No `Math.random()`, hardcoded “No results”, or placeholder numbers in the new ai-analysis page or the updated threat-hunting/IOCScanner flow.
