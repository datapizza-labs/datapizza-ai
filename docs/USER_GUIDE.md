# MOOD 2.0 — Manuale Utente e Tutorial

Benvenuto nella dashboard MOOD 2.0. Qui trovi: requisiti, setup, guida all’uso per ogni tab, automazioni, e una sezione di troubleshooting.

## Prerequisiti
- Python 3.10+ (ambiente virtuale consigliato; il repo crea automaticamente `.venv`)
- Streamlit installato (il progetto lo usa automaticamente)
- Connessione internet (per API Google, ricerche web, ecc.)

## Configurazione
1) .env (nella root del progetto)
- GOOGLE_API_KEY=... (obbligatoria per funzioni AI: Dev Agent, Email/Instagram/LinkedIn generation)
- GITHUB_TOKEN=... (opzionale; abilita creazione PR reali in Tab 10)

2) Email (locale e gitignored)
- File: `configs/email_config.yaml`
- Contenuto minimo:
  - smtp.host: smtp.gmail.com
  - smtp.port: 587
  - sender.email: la tua email
  - sender.name: il tuo nome
  - credentials.password: App Password (Gmail)

3) Directory di output (create se mancanti)
- `outputs/github`, `outputs/hardware`, `outputs/learning`, `outputs/research`

## Avvio Dashboard
- Avvio rapido:
  - `streamlit run tools/outreach_dashboard.py --server.port 8549`
- URL default: `http://127.0.0.1:8549`
- Se `localhost` non carica, usare 127.0.0.1 (issue IPv6/macOS)

## Navigazione
- Nel sidebar trovi il **dropdown "Seleziona Tab"** con tutte le 11 sezioni.
- La UI non usa più tabs orizzontali per evitare overflow.

## Panoramica Tab (11)
1. 🔍 Hunt Contacts — cerca contatti da target definiti in `configs/target_organizations.yaml`, salva su DB interno.
2. ✉️ Generate & Send Emails — genera email con team multi-agente (Writer→Critic→Reviser), approva e invia via SMTP.
3. 📸 Instagram Posts — genera post + (opz.) descrizioni immagine AI.
4. 📊 Dashboard — metriche di campagne, follow-up, risposte.
5. 🛠️ MOOD Dev Agent — azioni del Developer Agent (sprint, analisi tech, integrazioni software/hardware, demo, code review, doc).
6. 🌐 Research Insights — ricerche web 3x/settimana (Brave + DuckDuckGo), digest 30s, approvazioni.
7. ⚙️ Settings — impostazioni auto-send, provider immagini, info.
8. 💼 LinkedIn Personal — generazione post per profilo personale.
9. 🧠 Learning Agent — apprendimento da feedback, confidenza per azioni, report.
10. ⚡ GitHub Automation — pipeline Zero Click PR (branch→commit→push→PR). Legge/salva metadati in `outputs/github`.
11. 🎛️ Hardware Projects — generatori per Raspberry Pi 5, NVIDIA Jetson Orin, Audio Pro (config, requirements, struttura).

## Requisiti Chiave per le Funzioni
- AI (Tabs 2,3,5,6,8): richiedono GOOGLE_API_KEY valida (Gemini).
- GitHub PR (Tab 10): raccomandato GITHUB_TOKEN in `.env`.
- Email (Tab 2): `configs/email_config.yaml` correttamente impostato.

## Tutorial Rapido
- Email: Tab 1 trova contatti → Tab 2 genera email → approva/invia.
- Dev Agent: Tab 5 seleziona azione → Esegui → salva report, opz. genera progetto VS Code.
- Research: Tab 6 mostra digest e pending approvals → approva/rigenera.
- Learning: Tab 9 registra feedback → vedi confidenza e report.
- GitHub PR: Tab 10 inserisci nome progetto → genera pipeline; metadati in `outputs/github`.
- Hardware: Tab 11 seleziona piattaforma → imposta parametri → genera progetto (config/requirements).

## Troubleshooting
- Dashboard non si apre su `localhost`: usa `http://127.0.0.1:8549`.
- Errore Google API: aggiorna GOOGLE_API_KEY nel `.env`, poi riavvia Streamlit.
- `outputs/github` non trovata: crea la directory (`outputs/github`).
- SMTP errore: verifica App Password Gmail e `configs/email_config.yaml`.
- ImportError "No module named 'tools'": gli import sono stati corretti in `tools/outreach_dashboard.py`; riavvia se persiste.

## Self-Test
- Usa `tools/self_test.py` per testare componenti principali senza toccare servizi esterni (PR reali/SMTP opzionali).

