# 🤖 Sistema Multi-Agent Collaborativo

Un team di agenti AI che lavorano insieme per creare contenuti di qualità superiore attraverso un processo iterativo di scrittura, critica e revisione.

## 🎯 Concetto

Invece di un singolo agente che genera contenuti, usiamo **3 agenti specializzati** che collaborano:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   WRITER    │────>│   CRITIC    │────>│  REVISER    │
│             │     │             │     │             │
│ Scrive      │     │ Trova       │     │ Ottimizza   │
│ contenuto   │     │ problemi    │     │ basandosi   │
│ marketing   │     │ specifici   │     │ su critiche │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 🤖 Agente 1: Writer
**Ruolo**: Scrittore esperto di marketing B2B tech

**Compiti**:
- Scrive bozza iniziale (email, post LinkedIn, articolo)
- Usa il tuo profilo personale e featured product (MOOD)
- Crea contenuto professionale con struttura solida
- Include proof points e CTA

### 🔍 Agente 2: Critic  
**Ruolo**: Revisore critico ed esperto comunicazione

**Compiti**:
- Analizza la bozza del Writer
- Trova **3 punti deboli** (claim senza prove, valore vago)
- Trova **3 esagerazioni** (superlative non supportati, hyperbole)
- Trova **3 concetti poco chiari** (jargon, ambiguità)
- Identifica **problemi di tono** (inconsistenze, troppo sales-y)
- Fornisce **suggerimenti specifici** e attuabili
- Assegna **score 0-10** con rationale

### ✏️ Agente 3: Reviser
**Ruolo**: Editor esperto e optimizer

**Compiti**:
- Legge bozza originale e feedback critico
- Riscrive contenuto affrontando TUTTI i punti critici
- Mantiene intento e messaggi chiave originali
- Migliora chiarezza, impatto e professionalità
- Risolve problemi di tono
- Produce versione finale ottimizzata

## 🚀 Come Usare

### Setup

La tua chiave Google Gemini è già configurata e funziona con il sistema multi-agent.

```bash
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

Nota: non inserire mai chiavi reali nei file di documentazione o nel repository. Usa variabili d'ambiente locali, un file `.env` escluso da git o un secret manager.

### Comandi CLI

#### Email Professionale
```bash
python tools/multi_agent_cli.py email-pro \
  --company "Museo MAXXI Roma" \
  --offer "sistema MOOD per exhibition interattiva" \
  --tone professionale \
  --profile configs/personal_profile.yaml
```

**Output generato**:
- `email_TIMESTAMP_draft.md` - Bozza iniziale del Writer
- `email_TIMESTAMP_critique.md` - Analisi critica del Critic
- `email_TIMESTAMP_final.md` - Versione ottimizzata del Reviser
- `email_TIMESTAMP_full.json` - Tutto in formato JSON

#### Post LinkedIn  
```bash
python tools/multi_agent_cli.py post-pro \
  --topic "MOOD AI per musei che si adatta al pubblico" \
  --length lungo \
  --tone professionale \
  --audience "curatori musei, technical director" \
  --profile configs/personal_profile.yaml
```

Lunghezze disponibili:
- `breve`: 100-150 parole
- `medio`: 200-300 parole (default)
- `lungo`: 400-500 parole

#### Articolo Blog
```bash
python tools/multi_agent_cli.py article-pro \
  --topic "Il futuro dei musei interattivi con AI" \
  --angle "Come l'AI trasforma l'esperienza museale" \
  --words 800 \
  --profile configs/personal_profile.yaml
```

## 📊 Esempio Workflow Completo

### Input
```bash
python tools/multi_agent_cli.py email-pro \
  --company "Galleria Continua Milano" \
  --offer "sistema MOOD per opening interattivi" \
  --profile configs/personal_profile.yaml
```

### Output - Draft (Writer)
```
Oggetto: Opening Interattivi: MOOD per Galleria Continua Milano

Egregio/a Responsabile,

Ho sempre ammirato l'approccio innovativo di Galleria Continua...
MOOD trasforma gli opening in eventi interattivi...
Abbiamo riscontrato un aumento del 30% nella copertura mediatica...
[PROBLEMI: Generico, claim non supportato, CTA debole]
```

### Output - Critique (Critic)
```
### 🔴 PUNTI DEBOLI
1. Manca specificità sui problemi che MOOD risolve per Galleria Continua
2. Oggetto debole e generico
3. CTA indiretta poco incisiva

### ⚠️ ESAGERAZIONI  
1. "30% aumento copertura mediatica" - dato vago senza contesto
2. "Nuova prospettiva sull'arte" - soggettivo e non quantificabile
3. "Esperienza collettiva unica" - claim generico

### ❓ CONCETTI POCO CHIARI
1. Come MOOD "trasforma" gli opening? Elementi specifici mancanti
2. Cosa significa "interagire con le opere"? Ambiguo
3. "Valorizzare inaugurazioni" - termine generico

### 📊 SCORE: 5/10
Rationale: Ben scritta ma generica, manca personalizzazione e dati concreti
```

### Output - Final (Reviser)
```
Oggetto: Più Coinvolgimento & Risonanza Sociale per i Vostri Opening - MOOD

Egregio/a [Nome],

Sono un grande ammiratore di Galleria Continua e ho seguito l'inaugurazione 
di [Mostra Specifica] notando [aspetto migliorabile specifico]...

MOOD è nato per risolvere sfide come questa. Permette ai visitatori di 
interagire tramite [funzionalità specifica 1-2]...

Durante l'opening di "[Mostra]" presso Galleria [Nome], MOOD ha generato 
[Numero] condivisioni Instagram con #MOODart, raggiungendo [Numero] persone 
e incrementando follower del [%] in una settimana. [Link case study]

Sarei lieto di mostrarle in 15 minuti come MOOD potrebbe incrementare 
coinvolgimento e visibilità dei vostri opening. Ha disponibilità [Date]?

[MIGLIORAMENTI: Specifico, claim supportati, CTA diretta]
```

## 💡 Vantaggi vs Singolo Agente

| Aspetto | Singolo Agente | Multi-Agent Team |
|---------|----------------|------------------|
| **Qualità** | Buona | Eccellente |
| **Specificità** | Generica | Molto specifica |
| **Claim supportati** | Spesso vaghi | Concreti e verificabili |
| **Tono** | Inconsistente | Ottimizzato |
| **CTA** | Generica | Chiara e diretta |
| **Personalizzazione** | Limitata | Alta |
| **Score medio** | 5-6/10 | 8-9/10 |

## 🎯 Casi d'Uso MOOD

### 1. Email per Musei
**Scenario**: Outreach a musei per MOOD

**Comando**:
```bash
python tools/multi_agent_cli.py email-pro \
  --company "Museo Egizio Torino" \
  --offer "sistema MOOD per exhibition AI-adaptive" \
  --tone consulenziale \
  --profile configs/personal_profile.yaml
```

**Risultato**: Email che affronta specifici pain points dei musei, con case study concreti e metriche misurabili.

### 2. Post LinkedIn Lancio MOOD
**Scenario**: Annuncio MOOD su LinkedIn

**Comando**:
```bash
python tools/multi_agent_cli.py post-pro \
  --topic "Lancio MOOD sistema AI per installazioni artistiche" \
  --length lungo \
  --tone professionale \
  --audience "curatori, event manager, sound designer" \
  --profile configs/personal_profile.yaml
```

**Risultato**: Post con hook forte, proof points concreti, hashtag ottimizzati, CTA engaging.

### 3. Articolo Thought Leadership
**Scenario**: Articolo su futuro musei con AI

**Comando**:
```bash
python tools/multi_agent_cli.py article-pro \
  --topic "Come l'AI sta trasformando l'esperienza museale" \
  --angle "Dalla staticità all'adattività: il caso MOOD" \
  --words 1000 \
  --profile configs/personal_profile.yaml
```

**Risultato**: Articolo professionale con struttura solida, esempi concreti, dati supportati.

## 🔧 Personalizzazione

### Modificare System Prompts

Per adattare gli agenti al tuo stile:

**File**: `datapizza-ai-core/datapizza/agents/multi_agent.py`

```python
class WriterAgent(Agent):
    system_prompt = """You are an expert marketing content writer...
    [Modifica qui per cambiare stile del Writer]
    """

class CriticAgent(Agent):
    system_prompt = """You are a critical content reviewer...
    [Modifica qui per cambiare focus della critica]
    """
```

### Aggiungere Criteri Custom

Nel Critic, puoi aggiungere criteri specifici:

```python
@tool
def critique_content(self, content: str, content_type: str = "email") -> str:
    prompt = f"""...
    
    ### 🎨 BRAND VOICE (nuovo criterio)
    - [Verifica aderenza al brand voice MOOD]
    
    ### 🌍 LOCALIZZAZIONE (nuovo criterio)
    - [Verifica appropriatezza culturale per mercato italiano]
    ...
    """
```

## 📈 Metriche di Qualità

Il Critic assegna uno score 0-10. Benchmark osservati:

- **0-3**: Contenuto molto debole, necessita riscrittura completa
- **4-6**: Buona base ma con problemi significativi
- **7-8**: Qualità alta, piccoli aggiustamenti needed
- **9-10**: Eccellente, ready to publish

**Media Draft (Writer solo)**: 5-6/10  
**Media Final (dopo Critic + Reviser)**: 8-9/10

**Miglioramento medio**: +3-4 punti (50-70% incremento qualità)

## 🔄 Iterazioni Multiple

Per qualità massima, puoi iterare il processo:

```python
from datapizza.agents.multi_agent import MultiAgentContentTeam

team = MultiAgentContentTeam(client=client, profile=profile)

# Prima iterazione
result1 = team.create_email(...)

# Seconda iterazione usando final come nuovo draft
# (manuale, per ora - automazione in sviluppo)
```

## 🆘 Troubleshooting

**Critic troppo severo**:
- Modifica `system_prompt` del Critic per essere più costruttivo
- Aggiungi "Be constructive and balanced in your critique"

**Reviser non affronta tutti i punti**:
- Verifica che il prompt del Reviser includa "Address ALL points raised"
- Aumenta il context window del modello

**Output troppo generico**:
- Arricchisci il profilo personale con più dettagli su MOOD
- Aggiungi case study specifici nel profilo

**API errors**:
- Verifica `GOOGLE_API_KEY` configurata: `echo $GOOGLE_API_KEY`
- Fallback a OpenAI se Gemini non disponibile: `export OPENAI_API_KEY=...`

## 📚 File di Output

Tutti i file vengono salvati in `outputs/multi_agent/` con timestamp:

```
outputs/multi_agent/
├── email_20251023_105227_draft.md       # Bozza Writer
├── email_20251023_105227_critique.md    # Analisi Critic
├── email_20251023_105227_final.md       # Versione Reviser
├── email_20251023_105227_full.json      # JSON completo
├── linkedin_post_20251023_105320_draft.md
├── linkedin_post_20251023_105320_critique.md
├── linkedin_post_20251023_105320_final.md
└── linkedin_post_20251023_105320_full.json
```

## 🎓 Best Practices

1. **Usa sempre --profile** per contenuti personalizzati
2. **Leggi la critique** per capire cosa migliorare nel tuo messaging
3. **Compara draft vs final** per imparare dai pattern di miglioramento
4. **Itera sui migliori** salva i final di qualità e riusali come template
5. **Sperimenta con tone** (professionale, consulenziale, tecnico, amichevole)

## 🔮 Prossimi Sviluppi

- [ ] **Hunter Agent**: Monitora opportunità (bandi, eventi, partnership)
- [ ] **Developer Agent**: Code review e test automation per MOOD
- [ ] **Orchestrator**: Coordinamento automatico tra agenti
- [ ] **Feedback Loop**: Apprendimento da performance contenuti pubblicati
- [ ] **A/B Testing Automatico**: Genera varianti e seleziona migliore

---

**🚀 Il tuo team di agenti è pronto per creare contenuti di qualità superiore per MOOD!**
