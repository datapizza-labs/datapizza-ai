# 🔍 Web Research Agent - Ricerca Automatica Periodica

Sistema di ricerca automatica che monitora novità tecnologiche 3 volte a settimana (lunedì, mercoledì, venerdì alle 10:00).

## Cosa Fa

- **Ricerca automatica** su 9 topic tecnologici rilevanti (AI agents, multi-agent systems, LLM tools, interactive art, ecc.)
- **Analisi intelligente** con LLM per estrarre insights actionable
- **Report digest** salvati in `outputs/research/` con:
  - Executive summary (top 3 innovazioni + azioni raccomandate)
  - Detailed findings per ogni topic
  - Storico ricerche per analisi trend
- **Scheduling automatico** 3 volte a settimana

## Setup Scheduler Automatico

Per attivare l'esecuzione automatica su macOS:

### 1. Copia il LaunchAgent

```bash
cp com.mood.webresearch.plist ~/Library/LaunchAgents/
```

### 2. Carica il servizio

```bash
launchctl load ~/Library/LaunchAgents/com.mood.webresearch.plist
```

### 3. Verifica che sia attivo

```bash
launchctl list | grep mood.webresearch
```

Dovresti vedere l'output con il PID del servizio.

## Esecuzione Manuale

Puoi sempre eseguire una ricerca manualmente:

```bash
# Esegui solo se è un giorno schedulato
python tools/web_research_agent.py

# Forza esecuzione anche fuori schedule
python tools/web_research_agent.py --force

# Specifica directory output custom
python tools/web_research_agent.py --output /path/to/output
```

## Output

I report vengono salvati in `outputs/research/`:

- `research_digest_YYYY-MM-DD.md` - Report markdown leggibile
- `research_full_YYYY-MM-DD.json` - Dati completi in JSON
- `research_history.json` - Storico di tutte le ricerche
- `scheduler.log` - Log esecuzioni automatiche

## Esempio Report

```markdown
# 🔍 Research Digest - 2025-10-29

**Topics Researched:** 9

## Executive Summary

### Top 3 Most Promising Innovations
1. **LangGraph 0.2** - New orchestration patterns for multi-agent workflows
2. **AutoGen Studio** - Visual builder for agent teams
3. **EdgeAI HAT for RPi5** - 13 TOPS on-device inference

### Recommended Actions
1. Evaluate LangGraph for orchestrating Writer-Critic-Reviser pipeline
2. Test AutoGen Studio for rapid agent prototyping
3. Prototype edge deployment with RPi5 HAT

### Priority Integrations
Implement LangGraph state management for better agent coordination...
```

## Personalizzazione

### Cambia orario esecuzione

Modifica `com.mood.webresearch.plist` e cambia i valori di `Hour` e `Minute`:

```xml
<key>Hour</key>
<integer>14</integer>  <!-- 14:00 invece di 10:00 -->
```

Poi ricarica:

```bash
launchctl unload ~/Library/LaunchAgents/com.mood.webresearch.plist
launchctl load ~/Library/LaunchAgents/com.mood.webresearch.plist
```

### Aggiungi topic di ricerca

Modifica `tools/web_research_agent.py` e aggiungi topic in `RESEARCH_TOPICS`:

```python
RESEARCH_TOPICS = [
    "AI agents frameworks 2025",
    # ... topic esistenti ...
    "TUO NUOVO TOPIC QUI"
]
```

### Cambia frequenza

Per eseguire più o meno spesso, modifica `StartCalendarInterval` nel plist.

## Disattiva Scheduler

Se vuoi fermare le ricerche automatiche:

```bash
launchctl unload ~/Library/LaunchAgents/com.mood.webresearch.plist
```

Per riattivare:

```bash
launchctl load ~/Library/LaunchAgents/com.mood.webresearch.plist
```

## Troubleshooting

### Verificare log errori

```bash
tail -f outputs/research/launch_stderr.log
```

### Verificare esecuzioni

```bash
cat outputs/research/scheduler.log
```

### Testare manualmente

```bash
./tools/schedule_research.sh
```

Se funziona manualmente ma non automaticamente, verifica i permessi del LaunchAgent.

---

**Tip**: I digest vengono automaticamente usati dall'agente per proporre innovazioni aggiornate durante gli sprint settimanali!
