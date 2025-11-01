"""
Test suite completa per MOOD 2.0 - con isolamento per ogni test
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

# Importa moduli da testare
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from learning_agent import (
    LearningAgent, ActionType, FeedbackType, ActionTypeStats
)
from hardware_integration import (
    HardwareIntegrationAgent, HardwarePlatform, AudioFramework, SensorType
)


class TestLearningAgent:
    """Test Learning Agent - apprendimento continuo"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup con temp directory per ogni test"""
        self.tmpdir = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def test_init(self):
        """Test inizializzazione"""
        agent = LearningAgent(output_dir=self.tmpdir)
        assert agent.autonomy_threshold == 0.75
        assert agent.min_samples == 3
        assert len(agent.stats) == 0
    
    def test_record_feedback(self):
        """Test registrazione feedback"""
        agent = LearningAgent(output_dir=self.tmpdir)
        
        success, conf = agent.record_feedback(
            action_id="test-1",
            action_type=ActionType.UPDATE_DEPENDENCY,
            feedback=FeedbackType.APPROVED,
            notes="Approved update"
        )
        
        assert success is True
        assert conf >= 0.0 and conf <= 1.0
        assert len(agent.feedback_history) == 1
    
    def test_confidence_calculation(self):
        """Test calcolo confidenza"""
        agent = LearningAgent(output_dir=self.tmpdir)
        
        # Aggiungi 5 feedback approvati
        for i in range(5):
            agent.record_feedback(
                action_id=f"test-{i}",
                action_type=ActionType.UPDATE_DEPENDENCY,
                feedback=FeedbackType.APPROVED
            )
        
        conf = agent.get_action_confidence(ActionType.UPDATE_DEPENDENCY)
        assert conf == 1.0  # 100% approvati
    
    def test_confidence_with_rejection(self):
        """Test confidenza con alcuni rifiuti"""
        agent = LearningAgent(output_dir=self.tmpdir)
        
        # 3 approvati, 1 rifiutato
        for i in range(3):
            agent.record_feedback(
                action_id=f"test-{i}",
                action_type=ActionType.UPDATE_DEPENDENCY,
                feedback=FeedbackType.APPROVED
            )
        
        agent.record_feedback(
            action_id="test-reject",
            action_type=ActionType.UPDATE_DEPENDENCY,
            feedback=FeedbackType.REJECTED
        )
        
        conf = agent.get_action_confidence(ActionType.UPDATE_DEPENDENCY)
        assert 0.5 <= conf <= 0.8
    
    def test_autonomous_execution_threshold(self):
        """Test soglia autonomia"""
        agent = LearningAgent(output_dir=self.tmpdir)
        
        for i in range(3):
            agent.record_feedback(
                action_id=f"test-{i}",
                action_type=ActionType.UPDATE_DEPENDENCY,
                feedback=FeedbackType.APPROVED
            )
        
        should_auto, conf = agent.should_execute_autonomously(
            ActionType.UPDATE_DEPENDENCY
        )
        
        assert should_auto is True
        assert conf >= agent.autonomy_threshold
    
    def test_insufficient_samples(self):
        """Test: non abbastanza campioni per autonomia"""
        agent = LearningAgent(
            output_dir=self.tmpdir,
            config={"min_samples": 10, "autonomy_threshold": 0.75}
        )
        
        for i in range(2):
            agent.record_feedback(
                action_id=f"test-{i}",
                action_type=ActionType.UPDATE_DEPENDENCY,
                feedback=FeedbackType.APPROVED
            )
        
        should_auto, conf = agent.should_execute_autonomously(
            ActionType.UPDATE_DEPENDENCY
        )
        
        assert should_auto is False
    
    def test_action_stats(self):
        """Test statistiche per azione"""
        agent = LearningAgent(output_dir=self.tmpdir)
        
        for i in range(5):
            agent.record_feedback(
                action_id=f"test-{i}",
                action_type=ActionType.GENERATE_PROJECT,
                feedback=FeedbackType.APPROVED if i < 4 else FeedbackType.REJECTED
            )
        
        stats = agent.get_action_stats(ActionType.GENERATE_PROJECT)
        assert stats is not None
        assert stats.approved_count == 4
        assert stats.rejected_count == 1
        assert stats.total_count == 5
    
    def test_learning_report(self):
        """Test generazione report"""
        agent = LearningAgent(output_dir=self.tmpdir)
        
        for action_type in [ActionType.UPDATE_DEPENDENCY, ActionType.SEND_EMAIL]:
            for i in range(3):
                agent.record_feedback(
                    action_id=f"test-{action_type.value}-{i}",
                    action_type=action_type,
                    feedback=FeedbackType.APPROVED
                )
        
        report = agent.get_learning_report()
        assert "Learning Agent Report" in report


class TestHardwareIntegration:
    """Test Hardware Integration Agent"""
    
    def test_init(self):
        """Test inizializzazione"""
        agent = HardwareIntegrationAgent()
        assert agent.hardware_db is not None
        assert HardwarePlatform.RASPBERRY_PI_5 in agent.hardware_db
        assert HardwarePlatform.NVIDIA_JETSON_ORIN in agent.hardware_db
    
    def test_raspberry_pi_project(self):
        """Test generazione progetto Raspberry Pi"""
        agent = HardwareIntegrationAgent()
        
        config = agent.generate_raspberry_pi_project(
            "Test Audio",
            "Test audio on Pi",
            audio_framework=AudioFramework.JACK,
            sensors=[SensorType.MICROPHONE, SensorType.CAMERA]
        )
        
        assert config["project"]["name"] == "Test Audio"
        assert config["project"]["platform"] == "raspberry_pi_5"
        assert config["project"]["audio_framework"] == "jack"
        assert "microphone" in config["project"]["sensors"]
        assert len(config["requirements"]) > 0
    
    def test_jetson_project(self):
        """Test generazione progetto Jetson"""
        agent = HardwareIntegrationAgent()
        
        config = agent.generate_jetson_project(
            "Test Inference",
            "Test GPU inference",
            use_gpu_inference=True
        )
        
        assert config["project"]["name"] == "Test Inference"
        assert config["project"]["platform"] == "nvidia_jetson_orin"
        assert config["project"]["gpu_inference"] is True
        assert config["cuda_config"]["cuda_enabled"] is True
    
    def test_audio_professional_project(self):
        """Test generazione progetto audio"""
        agent = HardwareIntegrationAgent()
        
        config = agent.generate_audio_professional_project(
            "Spatial Audio",
            "Spatial audio processing",
            framework=AudioFramework.JACK,
            sample_rate=192000,
            channels=8
        )
        
        assert config["project"]["name"] == "Spatial Audio"
        assert config["audio_config"]["sample_rate"] == 192000
        assert config["audio_config"]["channels"] == 8


class TestIntegration:
    """Test integrazioni tra moduli"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup"""
        self.tmpdir = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def test_learning_agent_persistence(self):
        """Test persistenza Learning Agent"""
        agent1 = LearningAgent(output_dir=self.tmpdir)
        agent1.record_feedback(
            action_id="test-persist",
            action_type=ActionType.UPDATE_DEPENDENCY,
            feedback=FeedbackType.APPROVED
        )
        
        agent2 = LearningAgent(output_dir=self.tmpdir)
        assert len(agent2.feedback_history) > 0
        assert agent2.feedback_history[0].action_id == "test-persist"
    
    def test_multiple_action_types(self):
        """Test tracking multipli tipi di azione"""
        agent = LearningAgent(output_dir=self.tmpdir)
        
        action_types = [
            ActionType.UPDATE_DEPENDENCY,
            ActionType.GENERATE_PROJECT,
            ActionType.RUN_RESEARCH,
            ActionType.SEND_EMAIL
        ]
        
        for action_type in action_types:
            for i in range(3):
                agent.record_feedback(
                    action_id=f"{action_type.value}-{i}",
                    action_type=action_type,
                    feedback=FeedbackType.APPROVED
                )
        
        report = agent.get_learning_report()
        assert "Learning Agent Report" in report


class TestEndToEnd:
    """Test end-to-end"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup"""
        self.tmpdir = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def test_learning_to_autonomous_workflow(self):
        """Test workflow completo apprendimento -> autonomia"""
        agent = LearningAgent(
            output_dir=self.tmpdir,
            config={"min_samples": 3, "autonomy_threshold": 0.75}
        )
        
        for i in range(5):
            success, conf = agent.record_feedback(
                action_id=f"workflow-{i}",
                action_type=ActionType.UPDATE_DEPENDENCY,
                feedback=FeedbackType.APPROVED
            )
            
            should_auto, _ = agent.should_execute_autonomously(
                ActionType.UPDATE_DEPENDENCY
            )
            
            if i >= 2:
                assert should_auto is True
                break
        
        assert success is True
        assert conf >= 0.75
    
    def test_hardware_project_completeness(self):
        """Test completezza configurazione progetto hardware"""
        hw_agent = HardwareIntegrationAgent()
        
        config = hw_agent.generate_jetson_project(
            "Complete Project",
            "Complete test",
            use_gpu_inference=True,
            requires_realtime=True
        )
        
        assert config["project"]["name"]
        assert config["project"]["description"]
        assert config["structure"]
        assert config["requirements"]
        assert config["cuda_config"]
        assert len(config["requirements"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
