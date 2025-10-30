#!/bin/bash
# Script per avviare il dashboard Streamlit in modo automatico.

# Spostati nella directory dello script per garantire che i percorsi relativi funzionino
cd "$(dirname "$0")"

# 1. Carica la chiave API dal file .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "✅ Chiave API caricata da .env."
else
  echo "❌ Errore: file .env non trovato. Assicurati che esista e contenga la GOOGLE_API_KEY."
  exit 1
fi

if [ -z "$GOOGLE_API_KEY" ]; then
  echo "❌ Errore: GOOGLE_API_KEY non trovata nel file .env."
  exit 1
fi

# 2. Trova una porta libera tra 8501 e 8600
PORT=8501
while [ $(lsof -i:$PORT | wc -l) -ne 0 ] && [ $PORT -lt 8600 ]; do
  echo "Porta $PORT in uso, provo la successiva..."
  PORT=$((PORT + 1))
done

if [ $PORT -eq 8600 ]; then
  echo "❌ Errore: Nessuna porta libera trovata tra 8501 e 8599."
  exit 1
fi

URL="http://localhost:$PORT"

# 3. Avvia Streamlit sulla porta trovata
echo "🚀 Avvio del dashboard sulla porta $PORT..."
echo "URL: $URL"

streamlit run tools/outreach_dashboard.py --server.port $PORT --server.headless true &

# 4. Apri l'URL nel browser predefinito (solo su macOS)
sleep 5 # Attendi un momento per l'avvio del server
open $URL

echo "✅ Dashboard avviato. Puoi chiudere questo terminale quando hai finito."
