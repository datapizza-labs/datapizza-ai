# 🤖 MOOD Developer Agent

**Agente AI autonomo per sviluppo continuo e innovazione del sistema MOOD**

## 📋 Overview

Il MOOD Developer Agent è un agente AI specializzato che:

- ✅ **Monitora novità tecnologiche** (hardware, software, AI/ML)
- ✅ **Propone nuove feature** con design completo
- ✅ **Genera codice production-ready** per integrazioni
- ✅ **Fa code review** professionale
- ✅ **Crea demo e proof-of-concept**
- ✅ **Aggiorna documentazione** automaticamente
- ✅ **Monitora eventi live** (Ars Electronica, SXSW, Sonar+D, etc.)
- ✅ **Esegue weekly innovation sprints**

## 🚀 Quick Start

### Integrazione nel Dashboard (Streamlit)

Il MOOD Developer Agent è integrato nel dashboard web.

1. Avvia il dashboard:
  ```bash
  source .venv/bin/activate
  export GOOGLE_API_KEY=YOUR_KEY
  streamlit run tools/outreach_dashboard.py
  ```
2. Vai al tab "🛠️ MOOD Dev Agent"
3. Seleziona l'azione (es. Weekly Sprint, Analyze Technology, ecc.)
4. Esegui e salva il report: viene scritto in `docs/agent_reports/` e puoi scaricarlo dal browser

### Setup

```bash
# Assicurati di avere GOOGLE_API_KEY configurata
export GOOGLE_API_KEY=your_api_key_here

# Attiva virtual environment
cd /Users/mainenti/datapizza-ai-0.0.2
source .venv/bin/activate

# Rendi eseguibile
chmod +x tools/mood_dev_agent.py
```

### Uso Base

```bash
# Analizza nuova tecnologia
python tools/mood_dev_agent.py analyze-tech "Raspberry Pi 5" \
  --context "edge computing for MOOD" \
  --output docs/tech-analysis-rpi5.md

# Proponi nuova feature
python tools/mood_dev_agent.py propose-feature \
  "Multi-room orchestration with synchronized experiences" \
  --audience museums \
  --output docs/feature-multiroom.md

# Genera integrazione software
python tools/mood_dev_agent.py integrate-software "GrandMA3" \
  --protocol OSC \
  --output integrations/grandma_controller.py

# Monitora eventi rilevanti
python tools/mood_dev_agent.py monitor-events \
  --type all \
  --output reports/events-october-2025.md

# Weekly innovation sprint
python tools/mood_dev_agent.py weekly-sprint \
  --output reports/sprint-$(date +%Y%m%d).json

# Crea demo
python tools/mood_dev_agent.py create-demo \
  "Biometric sensor integration with real-time emotion detection" \
  --hardware "Raspberry Pi 5" \
  --output demos/biometric_demo.py

# Code review
python tools/mood_dev_agent.py code-review src/some_file.py \
  --focus performance \
  --output reviews/some_file_review.md
```

## 📚 Comandi Disponibili

### 1. `analyze-tech` - Analisi Tecnologia

Analizza una nuova tecnologia per valutare integrazione in MOOD.

**Esempio**:
```bash
python tools/mood_dev_agent.py analyze-tech "LiDAR Sensor" \
  --context "3D tracking audience position" \
  --output docs/lidar-analysis.md
```

**Output**: Report completo con:
- Overview tecnologia
- Technical analysis (specs, compatibility, performance)
- Integration plan (step-by-step, code examples)
- Pros & Cons
- Alternatives
- Recommendation (priorità, timeline)

---

### 2. `propose-feature` - Proposta Feature

Progetta una nuova feature per MOOD con architettura completa.

**Esempio**:
```bash
python tools/mood_dev_agent.py propose-feature \
  "AI-generated visuals based on audience emotions" \
  --audience "galleries" \
  --output docs/feature-ai-visuals.md
```

**Output**:
- Feature overview (user story, value prop)
- Architecture (components, data flow)
- Technical design (classes, OSC messages, DB schema)
- Implementation plan (MVP → Enhanced → Production)
- Code skeleton
- Testing strategy
- Documentation

---

### 3. `integrate-software` - Integrazione Software

Genera codice production-ready per integrare software esterni via OSC/MIDI/ArtNet.

**Esempio**:
```bash
# GrandMA (lighting control)
python tools/mood_dev_agent.py integrate-software "GrandMA3" \
  --protocol OSC \
  --output integrations/grandma_controller.py

# Resolume (video mapping)
python tools/mood_dev_agent.py integrate-software "Resolume Arena" \
  --protocol OSC \
  --output integrations/resolume_controller.py

# Ableton Live (audio)
python tools/mood_dev_agent.py integrate-software "Ableton Live" \
  --protocol MIDI \
  --output integrations/ableton_controller.py
```

**Output**: Codice Python completo con:
- Import & setup
- Main class con error handling
- Control methods (send/receive)
- Helper functions
- Example usage
- Testing mock objects

---

### 4. `integrate-hardware` - Integrazione Hardware

Implementazione completa hardware (analisi + demo + docs).

**Esempio**:
```bash
python tools/mood_dev_agent.py integrate-hardware "Raspberry Pi 5" \
  --output docs/rpi5-integration-complete.md
```

**Output**: 
- Technology analysis
- Working demo code
- Setup tutorial
- Integration guide

---

### 5. `monitor-events` - Monitor Eventi

Monitora eventi e conferenze rilevanti per MOOD.

**Esempio**:
```bash
python tools/mood_dev_agent.py monitor-events \
  --type interactive \
  --output reports/events-interactive-art.md
```

**Event Types**: `tech`, `art`, `ai`, `interactive`, `all`

**Output**: Per ogni evento:
- Overview (nome, date, focus)
- Highlights (tecnologie presentate, trends)
- Action items per MOOD (cosa implementare)
- Resources (link, repos, community)

**Eventi Monitorati**:
- Ars Electronica (Settembre) - Arte digitale, AI
- SXSW (Marzo) - Tech, innovation
- Sonar+D (Giugno) - Music, creativity, tech
- Biennale Venezia (Maggio-Nov) - Arte contemporanea
- MUTEK - Digital creativity
- CES (Gennaio) - Consumer electronics
- Google I/O (Maggio) - AI/ML
- WWDC (Giugno) - Apple tech
- Maker Faire - Hardware, DIY

---

### 6. `weekly-sprint` - Innovation Sprint

Esegue sprint settimanale automatico: monitora eventi, analizza tech, propone features.

**Esempio**:
```bash
python tools/mood_dev_agent.py weekly-sprint \
  --output reports/sprint-$(date +%Y%m%d).json
```

**Workflow**:
1. Monitor eventi e trends
2. Analyze nuova tecnologia rilevante
3. Propose nuova feature prioritaria
4. Generate report JSON completo

**Raccomandazione**: Schedulare con cron ogni domenica:
```bash
# Crontab entry
0 22 * * 0 cd /Users/mainenti/datapizza-ai-0.0.2 && source .venv/bin/activate && python tools/mood_dev_agent.py weekly-sprint --output reports/sprint-$(date +\%Y\%m\%d).json
```

---

### 7. `create-demo` - Crea Demo

Genera demo/proof-of-concept completo con codice + setup.

**Esempio**:
```bash
python tools/mood_dev_agent.py create-demo \
  "Real-time emotion detection with facial recognition" \
  --hardware "Jetson Nano" \
  --output demos/emotion_detection_demo.py
```

**Output**:
- Demo overview
- Hardware setup (componenti, wiring)
- Software requirements
- Complete working code
- Usage instructions
- README.md
- Next steps per produzione

---

### 8. `code-review` - Code Review

Review professionale di codice MOOD con suggerimenti.

**Esempio**:
```bash
python tools/mood_dev_agent.py code-review \
  src/osc_handler.py \
  --focus performance \
  --output reviews/osc_handler_review.md
```

**Focus Areas**: `performance`, `security`, `architecture`, `style`, `general`

**Output**:
- ✅ Good: Cosa è fatto bene
- ⚠️ Warning: Possibili problemi
- ❌ Issue: Problemi critici
- 💡 Suggestion: Miglioramenti con code examples

---

## 🎯 Use Cases Pratici

### Use Case 1: Integrare Raspberry Pi 5

```bash
# Step 1: Analizza tecnologia
python tools/mood_dev_agent.py analyze-tech "Raspberry Pi 5 with HAT AI" \
  --context "Replace Jetson Nano for cost reduction" \
  --output docs/rpi5-analysis.md

# Step 2: Crea demo
python tools/mood_dev_agent.py create-demo \
  "Basic computer vision with RPi5 HAT AI" \
  --hardware "Raspberry Pi 5" \
  --output demos/rpi5_vision_demo.py

# Step 3: Implementazione completa
python tools/mood_dev_agent.py integrate-hardware "Raspberry Pi 5" \
  --output docs/rpi5-integration-guide.md
```

### Use Case 2: Integrare GrandMA per Lighting

```bash
# Genera controller OSC
python tools/mood_dev_agent.py integrate-software "GrandMA3" \
  --protocol OSC \
  --output integrations/grandma_controller.py

# Code review
python tools/mood_dev_agent.py code-review \
  integrations/grandma_controller.py \
  --focus performance \
  --output reviews/grandma_review.md

# Crea demo con scene lighting
python tools/mood_dev_agent.py create-demo \
  "Dynamic lighting scenes controlled by audience mood" \
  --hardware "Server" \
  --output demos/grandma_mood_lighting.py
```

### Use Case 3: Weekly Innovation Workflow

**Automatico ogni domenica sera**:

```bash
# Setup cron job
crontab -e

# Add this line (assicurati che GOOGLE_API_KEY sia configurata nell'ambiente del cron in modo sicuro):
0 22 * * 0 cd /Users/mainenti/datapizza-ai-0.0.2 && source .venv/bin/activate && python tools/mood_dev_agent.py weekly-sprint --output /Users/mainenti/mood-reports/sprint-$(date +\%Y\%m\%d).json && python tools/mood_dev_agent.py monitor-events --output /Users/mainenti/mood-reports/events-$(date +\%Y\%m\%d).md
```

Suggerimento: evita di esportare chiavi direttamente nella riga di cron. Preferisci una delle seguenti opzioni:
- Imposta la variabile in `/etc/launchd.conf` o nel profilo dello user usato da cron (macOS: LaunchAgents con `EnvironmentVariables`).
- Usa un file `.env` caricato dallo script (non committato).
- Usa un Secret Manager.

**Manuale quando serve**:
```bash
# Run sprint
python tools/mood_dev_agent.py weekly-sprint

# Review risultati e scegli priority task
# Poi esegui implementazioni specifiche
```

---

## 🏗️ Architettura Agente

### MOODDeveloperAgent

Agent principale con 8 tool specializzati:

1. **`analyze_technology`** - Analisi tech completa
2. **`propose_feature`** - Design feature
3. **`implement_integration`** - Codice integrazione
4. **`monitor_events`** - Eventi & conferenze
5. **`code_review`** - Review professionale
6. **`create_demo`** - Demo/POC
7. **`update_documentation`** - Docs generator

### MOODDevelopmentTeam

Orchestrator che coordina workflow complessi:

- **`weekly_innovation_sprint()`** - Sprint automatico
- **`implement_hardware_integration()`** - Hardware completo
- **`implement_software_integration()`** - Software completo

---

## 📊 Output Examples

### Technology Analysis Output

```markdown
# RASPBERRY PI 5 - Analysis for MOOD Integration

## 1. OVERVIEW
Raspberry Pi 5 è il latest single-board computer della Raspberry Pi Foundation...

## 2. TECHNICAL ANALYSIS
- CPU: Broadcom BCM2712 (4x Cortex-A76 @ 2.4GHz)
- RAM: 4GB/8GB LPDDR4X
- GPIO: 40-pin header
- HAT AI: Hailo-8L AI accelerator (13 TOPS)
...

## 3. INTEGRATION PLAN
### Step 1: Setup Hardware
```bash
# Install OS
sudo apt update && sudo apt upgrade
pip install opencv-python
```

### Step 2: Install HAT AI drivers
```python
# Code example...
```

## 4. PROS & CONS
✅ Pros:
- Costo: €60 vs €200 Jetson
- HAT AI: 13 TOPS performance
- Community: Massive support

⚠️ Cons:
- Meno potente di Jetson Orin
- Setup più complesso

## 5. RECOMMENDATION
**Priorità: ALTA**
Timeline: 2-3 settimane
Budget: €100 (board + HAT AI)
```

---

## 🔄 Continuous Development Workflow

### Setup Iniziale (Una volta)

```bash
# 1. Clone MOOD repo
git clone https://github.com/ninuxi/TESTreale_OSC_mood-adaptive-art-system
cd TESTreale_OSC_mood-adaptive-art-system

# 2. Create reports directory
mkdir -p ~/mood-reports
mkdir -p ~/mood-integrations
mkdir -p ~/mood-demos

# 3. Setup cron for weekly sprint
crontab -e
# (add weekly sprint job)
```

### Workflow Settimanale

**Ogni domenica sera (automatico)**:
1. Weekly sprint eseguito automaticamente
2. Report salvato in `~/mood-reports/sprint-YYYYMMDD.json`
3. Eventi monitorati in `~/mood-reports/events-YYYYMMDD.md`

**Ogni lunedì mattina (manuale)**:
1. Leggi sprint report
2. Identifica 1-2 task prioritari
3. Esegui implementazioni:
   ```bash
   # Esempio: Integrate new sensor
   python tools/mood_dev_agent.py integrate-hardware "New Sensor XYZ"
   ```

**Quando serve (on-demand)**:
```bash
# Nuova tech interessante?
python tools/mood_dev_agent.py analyze-tech "Technology Name"

# Idea per feature?
python tools/mood_dev_agent.py propose-feature "Feature Description"

# Problema in codice?
python tools/mood_dev_agent.py code-review path/to/file.py
```

---

## 🎓 Best Practices

### 1. Usa Output Files
Sempre salva output per reference futuro:
```bash
python tools/mood_dev_agent.py analyze-tech "X" --output docs/X-analysis.md
```

### 2. Review Prima di Implementare
```bash
# 1. Analizza
python tools/mood_dev_agent.py analyze-tech "Tech"

# 2. Proponi
python tools/mood_dev_agent.py propose-feature "Feature"

# 3. Implementa
python tools/mood_dev_agent.py integrate-software "Software"

# 4. Review
python tools/mood_dev_agent.py code-review generated_code.py
```

### 3. Testa Demo Prima di Prod
```bash
# Crea demo
python tools/mood_dev_agent.py create-demo "Test Feature"

# Testa su hardware
python generated_demo.py

# Se OK, integra in MOOD
```

### 4. Documenta Tutto
```bash
# Dopo ogni integrazione
python tools/mood_dev_agent.py update-documentation "New Integration" \
  --type tutorial \
  --output docs/new-integration-guide.md
```

---

## 🚀 Roadmap Tasks Priority

### Alta Priorità (Q4 2025)
- [ ] Raspberry Pi 5 + HAT AI integration
- [ ] GrandMA OSC controller
- [ ] Resolume OSC controller
- [ ] Multi-room orchestration

### Media Priorità (Q1 2026)
- [ ] Biometric sensors integration
- [ ] Stable Diffusion real-time visuals
- [ ] Web dashboard for curators
- [ ] TouchDesigner integration

### Bassa Priorità (Q2+ 2026)
- [ ] MOOD Cloud Platform (SaaS)
- [ ] AR/VR integration
- [ ] Plugin system
- [ ] Open source community launch

---

## 📞 Support

- **Repository**: https://github.com/ninuxi/TESTreale_OSC_mood-adaptive-art-system
- **Email**: oggettosonoro@gmail.com
- **Created by**: Antonio Mainenti

---

**Generated**: 23 Ottobre 2025  
**Version**: 1.0  
**Status**: 🟢 Production Ready
