#!/bin/bash
# Scheduler per Web Research Agent
# Esegue ricerca automatica 3 volte a settimana (lun, mer, ven)

cd "$(dirname "$0")/.."

# Carica variabili ambiente
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Esegui ricerca
python tools/web_research_agent.py --output outputs/research

# Log esecuzione
echo "[$(date)] Research cycle completed" >> outputs/research/scheduler.log
