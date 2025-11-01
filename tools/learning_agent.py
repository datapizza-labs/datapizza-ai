"""
Learning Agent: Agente che impara dai tuoi feedback e aumenta autonomia nel tempo.

Traccia:
- Decisioni dell'utente (Approva/Rifiuta)
- Calcola "fiducia" per tipi di azione
- Aumenta autonomia progressivamente
- Notifica via email azioni autonome eseguite
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class ActionType(Enum):
    """Tipi di azione tracciabili"""
    UPDATE_DEPENDENCY = "update_dependency"
    GENERATE_PROJECT = "generate_project"
    RUN_RESEARCH = "run_research"
    SEND_EMAIL = "send_email"
    APPROVE_PR = "approve_pr"
    FIX_LINT = "fix_lint"
    UPDATE_DOCS = "update_docs"
    OTHER = "other"


class FeedbackType(Enum):
    """Tipo di feedback utente"""
    APPROVED = "approved"
    REJECTED = "rejected"
    IGNORED = "ignored"
    EXECUTED_AUTONOMOUSLY = "executed_autonomously"


@dataclass
class ActionFeedback:
    """Tracciato di un feedback su un'azione"""
    action_id: str
    action_type: ActionType
    timestamp: str  # ISO format
    feedback: FeedbackType
    confidence_score_at_time: float
    notes: Optional[str] = None


@dataclass
class ActionTypeStats:
    """Statistiche per tipo di azione"""
    action_type: ActionType
    total_count: int
    approved_count: int
    rejected_count: int
    autonomous_count: int
    confidence_score: float  # 0.0 a 1.0
    last_updated: str  # ISO format
    trend: str  # "increasing", "decreasing", "stable"


class LearningAgent:
    """
    Agente che impara dalle decisioni dell'utente e aumenta l'autonomia.
    
    Algoritmo:
    1. Traccia ogni feedback dell'utente
    2. Calcola tasso di approvazione per ogni tipo di azione
    3. Assegna "fiducia" progressiva (0.0-1.0)
    4. Quando fiducia >= threshold, esegui autonomamente
    5. Notifica l'utente di azioni autonome eseguite
    """
    
    def __init__(self, output_dir: Path = None, config: Dict = None):
        """
        Inizializza Learning Agent.
        
        Args:
            output_dir: Directory per salvare feedback log
            config: Configurazione personalizzata
        """
        self.output_dir = output_dir or Path(__file__).parent.parent / "outputs" / "learning"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_log_path = self.output_dir / "feedback_log.json"
        self.stats_path = self.output_dir / "action_stats.json"
        self.learning_report_path = self.output_dir / "learning_report.md"
        
        # Configurazione
        self.config = config or {}
        self.autonomy_threshold = self.config.get("autonomy_threshold", 0.75)  # 75% confidenza
        self.min_samples = self.config.get("min_samples", 3)  # Almeno 3 azioni prima di diventare autonome
        self.decay_factor = self.config.get("decay_factor", 0.95)  # Decay settimanale della fiducia
        
        self.logger = logging.getLogger(__name__)
        
        # Carica stato
        self.feedback_history: List[ActionFeedback] = self._load_feedback_history()
        self.stats: Dict[str, ActionTypeStats] = self._load_stats()
        
    def record_feedback(
        self,
        action_id: str,
        action_type: ActionType,
        feedback: FeedbackType,
        notes: Optional[str] = None
    ) -> Tuple[bool, float]:
        """
        Registra feedback per un'azione.
        
        Args:
            action_id: ID univoco dell'azione
            action_type: Tipo di azione
            feedback: Tipo di feedback (APPROVED, REJECTED, ecc.)
            notes: Note opzionali
            
        Returns:
            (success, new_confidence_score)
        """
        # Calcola confidenza attuale
        current_confidence = self._calculate_confidence(action_type)
        
        # Crea entry feedback
        fb = ActionFeedback(
            action_id=action_id,
            action_type=action_type,
            timestamp=datetime.now().isoformat(),
            feedback=feedback,
            confidence_score_at_time=current_confidence,
            notes=notes
        )
        
        # Aggiungi alla storia
        self.feedback_history.append(fb)
        
        # Aggiorna statistiche
        self._update_stats(action_type, feedback)
        
        # Salva
        self._save_feedback_history()
        self._save_stats()
        
        # Calcola nuova confidenza
        new_confidence = self._calculate_confidence(action_type)
        
        self.logger.info(
            f"Feedback registrato: {action_type.value} → {feedback.value} "
            f"(confidenza: {current_confidence:.2f} → {new_confidence:.2f})"
        )
        
        return True, new_confidence
    
    def should_execute_autonomously(self, action_type: ActionType) -> Tuple[bool, float]:
        """
        Determina se un'azione dovrebbe essere eseguita autonomamente.
        
        Args:
            action_type: Tipo di azione
            
        Returns:
            (should_execute, confidence_score)
        """
        confidence = self._calculate_confidence(action_type)
        count = self._get_action_count(action_type)
        
        should_execute = (
            confidence >= self.autonomy_threshold and
            count >= self.min_samples
        )
        
        return should_execute, confidence
    
    def get_action_confidence(self, action_type: ActionType) -> float:
        """Ottiene livello di confidenza per un tipo di azione"""
        return self._calculate_confidence(action_type)
    
    def get_action_stats(self, action_type: ActionType) -> Optional[ActionTypeStats]:
        """Ottiene statistiche per tipo di azione"""
        key = action_type.value
        return self.stats.get(key)
    
    def get_learning_report(self) -> str:
        """Genera report di apprendimento"""
        report = "# 📊 Learning Agent Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Statistiche per tipo di azione
        report += "## Action Type Statistics\n\n"
        report += "| Action Type | Approved | Rejected | Autonomous | Confidence | Status |\n"
        report += "|-------------|----------|----------|-----------|------------|--------|\n"
        
        for key, stats in sorted(self.stats.items()):
            status = "🤖 AUTO" if stats.confidence_score >= self.autonomy_threshold else "👤 APPROVAL"
            report += (
                f"| {stats.action_type.value} | "
                f"{stats.approved_count} | "
                f"{stats.rejected_count} | "
                f"{stats.autonomous_count} | "
                f"{stats.confidence_score:.1%} | "
                f"{status} |\n"
            )
        
        # Trend di apprendimento
        report += "\n## Learning Trends\n\n"
        for key, stats in sorted(self.stats.items()):
            emoji = "📈" if stats.trend == "increasing" else "📉" if stats.trend == "decreasing" else "➡️"
            report += f"- {emoji} **{stats.action_type.value}**: {stats.trend}\n"
        
        # Prossimi step
        report += "\n## Next Steps\n\n"
        for key, stats in sorted(self.stats.items()):
            if stats.confidence_score < self.autonomy_threshold and stats.total_count > 0:
                needed = max(0, self.min_samples - stats.total_count)
                report += (
                    f"- **{stats.action_type.value}**: "
                    f"Confidence {stats.confidence_score:.1%} "
                    f"(+{needed} more samples for autonomy)\n"
                )
            elif stats.confidence_score >= self.autonomy_threshold:
                report += (
                    f"- ✅ **{stats.action_type.value}**: "
                    f"Ready for autonomous execution (confidence: {stats.confidence_score:.1%})\n"
                )
        
        return report
    
    # ========== PRIVATE METHODS ==========
    
    def _calculate_confidence(self, action_type: ActionType) -> float:
        """
        Calcola confidence score per tipo di azione.
        
        Formula:
        - Base: approval_rate = approved / (approved + rejected)
        - Bonus: min_samples multiplier
        - Decay: reduce se ultimi feedback negativi
        
        Returns:
            Score 0.0-1.0
        """
        recent_feedbacks = [
            fb for fb in self.feedback_history
            if fb.action_type == action_type
        ]
        
        if not recent_feedbacks:
            return 0.0
        
        # Calcola tasso di approvazione
        approved = sum(1 for fb in recent_feedbacks if fb.feedback == FeedbackType.APPROVED)
        rejected = sum(1 for fb in recent_feedbacks if fb.feedback == FeedbackType.REJECTED)
        total = approved + rejected
        
        if total == 0:
            return 0.0
        
        approval_rate = approved / total
        
        # Bonus per numero di campioni
        sample_bonus = min(1.0, len(recent_feedbacks) / self.min_samples)
        
        # Decay per feedback recenti negativi
        recent_decay = 1.0
        if recent_feedbacks[-1].feedback == FeedbackType.REJECTED:
            recent_decay = 0.7  # Penalità se ultimo feedback è rifiuto
        
        # Calcola confidenza finale
        confidence = approval_rate * sample_bonus * recent_decay
        
        return min(1.0, confidence)
    
    def _get_action_count(self, action_type: ActionType) -> int:
        """Conta azioni per tipo"""
        return sum(1 for fb in self.feedback_history if fb.action_type == action_type)
    
    def _update_stats(self, action_type: ActionType, feedback: FeedbackType):
        """Aggiorna statistiche per tipo di azione"""
        key = action_type.value
        
        if key not in self.stats:
            self.stats[key] = ActionTypeStats(
                action_type=action_type,
                total_count=0,
                approved_count=0,
                rejected_count=0,
                autonomous_count=0,
                confidence_score=0.0,
                last_updated=datetime.now().isoformat(),
                trend="stable"
            )
        
        stats = self.stats[key]
        stats.total_count += 1
        
        if feedback == FeedbackType.APPROVED:
            stats.approved_count += 1
        elif feedback == FeedbackType.REJECTED:
            stats.rejected_count += 1
        elif feedback == FeedbackType.EXECUTED_AUTONOMOUSLY:
            stats.autonomous_count += 1
        
        # Calcola nuova confidenza
        stats.confidence_score = self._calculate_confidence(action_type)
        stats.last_updated = datetime.now().isoformat()
        
        # Calcola trend
        if len(self.feedback_history) > 1:
            recent = [
                fb for fb in self.feedback_history[-5:]
                if fb.action_type == action_type
            ]
            if recent:
                recent_approved = sum(1 for fb in recent if fb.feedback == FeedbackType.APPROVED)
                older_approved = sum(1 for fb in self.feedback_history[-10:-5] if fb.feedback == FeedbackType.APPROVED)
                
                if recent_approved > older_approved:
                    stats.trend = "increasing"
                elif recent_approved < older_approved:
                    stats.trend = "decreasing"
                else:
                    stats.trend = "stable"
    
    def _load_feedback_history(self) -> List[ActionFeedback]:
        """Carica storia feedback da file"""
        if not self.feedback_log_path.exists():
            return []
        
        try:
            with open(self.feedback_log_path, 'r') as f:
                data = json.load(f)
                return [
                    ActionFeedback(
                        action_id=item['action_id'],
                        action_type=ActionType(item['action_type']),
                        timestamp=item['timestamp'],
                        feedback=FeedbackType(item['feedback']),
                        confidence_score_at_time=item['confidence_score_at_time'],
                        notes=item.get('notes')
                    )
                    for item in data
                ]
        except Exception as e:
            self.logger.error(f"Errore caricamento feedback history: {e}")
            return []
    
    def _save_feedback_history(self):
        """Salva storia feedback su file"""
        try:
            data = [
                {
                    'action_id': fb.action_id,
                    'action_type': fb.action_type.value,
                    'timestamp': fb.timestamp,
                    'feedback': fb.feedback.value,
                    'confidence_score_at_time': fb.confidence_score_at_time,
                    'notes': fb.notes
                }
                for fb in self.feedback_history
            ]
            with open(self.feedback_log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Errore salvataggio feedback history: {e}")
    
    def _load_stats(self) -> Dict[str, ActionTypeStats]:
        """Carica statistiche da file"""
        if not self.stats_path.exists():
            return {}
        
        try:
            with open(self.stats_path, 'r') as f:
                data = json.load(f)
                return {
                    key: ActionTypeStats(
                        action_type=ActionType(item['action_type']),
                        total_count=item['total_count'],
                        approved_count=item['approved_count'],
                        rejected_count=item['rejected_count'],
                        autonomous_count=item['autonomous_count'],
                        confidence_score=item['confidence_score'],
                        last_updated=item['last_updated'],
                        trend=item['trend']
                    )
                    for key, item in data.items()
                }
        except Exception as e:
            self.logger.error(f"Errore caricamento stats: {e}")
            return {}
    
    def _save_stats(self):
        """Salva statistiche su file"""
        try:
            data = {
                key: {
                    'action_type': stats.action_type.value,
                    'total_count': stats.total_count,
                    'approved_count': stats.approved_count,
                    'rejected_count': stats.rejected_count,
                    'autonomous_count': stats.autonomous_count,
                    'confidence_score': stats.confidence_score,
                    'last_updated': stats.last_updated,
                    'trend': stats.trend
                }
                for key, stats in self.stats.items()
            }
            with open(self.stats_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Errore salvataggio stats: {e}")


# ============ INTEGRATION FUNCTIONS ============

def notify_autonomous_execution(
    action_type: ActionType,
    action_description: str,
    result_details: Dict,
    email_config_path: Path = None
) -> bool:
    """
    Invia notifica email di esecuzione autonoma.
    
    Args:
        action_type: Tipo di azione
        action_description: Descrizione dell'azione
        result_details: Dettagli del risultato
        email_config_path: Path config email
        
    Returns:
        True se email inviata con successo
    """
    try:
        import yaml
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Carica config email
        email_config_path = email_config_path or Path(__file__).parent.parent / "configs" / "email_config.yaml"
        with open(email_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Prepara email
        sender = config['sender']['email']
        password = config['credentials']['password']
        
        msg = MIMEMultipart('alternative')
        msg['From'] = f"{config['sender']['name']} <{sender}>"
        msg['To'] = sender
        msg['Subject'] = f"🤖 MOOD Agent: Azione Autonoma Eseguita - {action_type.value}"
        
        # Corpo email
        body = f"""
Ciao Antonio,

Il tuo MOOD Agent ha appena eseguito autonomamente un'azione:

**Tipo:** {action_type.value}
**Descrizione:** {action_description}
**Timestamp:** {datetime.now().strftime('%d/%m/%Y alle %H:%M:%S')}

**Dettagli Risultato:**
"""
        for key, value in result_details.items():
            body += f"\n- **{key}**: {value}"
        
        body += f"\n\n{config.get('signature', '')}"
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Invia
        with smtplib.SMTP(config['smtp']['host'], config['smtp']['port']) as server:
            if config['smtp']['use_tls']:
                server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        
        return True
        
    except Exception as e:
        logging.error(f"Errore invio notifica autonoma: {e}")
        return False


if __name__ == "__main__":
    # Test
    agent = LearningAgent()
    
    # Simula feedback
    print("🧠 Testing Learning Agent\n")
    
    for i in range(5):
        result, conf = agent.record_feedback(
            action_id=f"test-action-{i}",
            action_type=ActionType.UPDATE_DEPENDENCY,
            feedback=FeedbackType.APPROVED if i < 4 else FeedbackType.REJECTED,
            notes=f"Test feedback {i}"
        )
        print(f"  Feedback {i}: confidence = {conf:.2f}")
    
    # Verifica autonomia
    should_auto, conf = agent.should_execute_autonomously(ActionType.UPDATE_DEPENDENCY)
    print(f"\n✅ Should execute autonomously: {should_auto} (confidence: {conf:.2f})")
    
    # Report
    print("\n" + agent.get_learning_report())
