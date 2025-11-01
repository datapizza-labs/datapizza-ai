"""
Hardware Integration Agent: Estende MOOD per generare progetti hardware-specific.

Supporta:
- Raspberry Pi (audio/video, GPIO, sensor)
- NVIDIA Jetson (inference, video processing)
- Progetti audio professionali (JACK, GStreamer)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class HardwarePlatform(Enum):
    """Piattaforme hardware supportate"""
    RASPBERRY_PI = "raspberry_pi"
    RASPBERRY_PI_5 = "raspberry_pi_5"
    NVIDIA_JETSON_NANO = "nvidia_jetson_nano"
    NVIDIA_JETSON_ORIN = "nvidia_jetson_orin"
    UBUNTU_PRO = "ubuntu_pro"


class AudioFramework(Enum):
    """Framework audio supportati"""
    JACK = "jack"
    ALSA = "alsa"
    GSTREAMER = "gstreamer"
    PULSEAUDIO = "pulseaudio"
    PIPEWIRE = "pipewire"


class SensorType(Enum):
    """Tipi di sensori supportati"""
    MICROPHONE = "microphone"
    CAMERA = "camera"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    ACCELEROMETER = "accelerometer"
    PROXIMITY = "proximity"


@dataclass
class HardwareRequirement:
    """Requisito hardware"""
    name: str
    platform: HardwarePlatform
    description: str
    installation_cmd: str
    python_package: Optional[str] = None


@dataclass
class HardwareProject:
    """Specifica di progetto hardware"""
    name: str
    description: str
    platform: HardwarePlatform
    audio_framework: Optional[AudioFramework] = None
    sensors: List[SensorType] = None
    use_gpu_inference: bool = False
    requires_realtime: bool = False
    power_consumption_class: str = "standard"  # low, standard, high


class HardwareIntegrationAgent:
    """
    Agente che estende VSCodeProjectGenerator per hardware.
    Fornisce template specifici per Raspberry Pi, Jetson, audio pro.
    """
    
    def __init__(self):
        """Inizializza Hardware Integration Agent"""
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(__file__).parent.parent / "outputs" / "hardware"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.hardware_db = self._init_hardware_database()
    
    def _init_hardware_database(self) -> Dict:
        """Inizializza database di hardware e configurazioni"""
        return {
            HardwarePlatform.RASPBERRY_PI_5: {
                "description": "Raspberry Pi 5 - high-end single board computer",
                "cpu": "ARM Cortex-A76 64-bit",
                "ram_variants": [4, 8],
                "storage": "microSD/NVMe",
                "interfaces": ["GPIO", "I2C", "SPI", "UART", "USB-C"],
                "audio_frameworks": [
                    AudioFramework.JACK,
                    AudioFramework.ALSA,
                    AudioFramework.GSTREAMER,
                    AudioFramework.PIPEWIRE
                ],
                "requirements": [
                    {
                        "name": "rpi.gpio",
                        "description": "GPIO control library",
                        "install": "pip install RPi.GPIO"
                    },
                    {
                        "name": "gpiozero",
                        "description": "Easy-to-use GPIO library",
                        "install": "pip install gpiozero"
                    },
                    {
                        "name": "pygame",
                        "description": "Multimedia library with audio support",
                        "install": "pip install pygame"
                    },
                    {
                        "name": "sounddevice",
                        "description": "Audio I/O library",
                        "install": "pip install sounddevice"
                    }
                ]
            },
            HardwarePlatform.NVIDIA_JETSON_ORIN: {
                "description": "NVIDIA Jetson Orin - AI inference at edge",
                "cpu": "ARM Cortex-A78AE 12-core",
                "gpu": "NVIDIA Ampere-based GPU (up to 275 TFLOPS)",
                "ram_variants": [8, 12, 64],
                "storage": "UFS or NVMe",
                "interfaces": ["PCIe", "USB", "Ethernet", "CSI", "GPIO"],
                "capabilities": ["TensorRT", "CUDA", "cuDNN", "DeepStream"],
                "audio_frameworks": [
                    AudioFramework.GSTREAMER,
                    AudioFramework.JACK
                ],
                "requirements": [
                    {
                        "name": "tensorrt",
                        "description": "NVIDIA inference optimizer",
                        "install": "pip install tensorrt"
                    },
                    {
                        "name": "cuda",
                        "description": "NVIDIA CUDA toolkit",
                        "install": "Manual installation from NVIDIA"
                    },
                    {
                        "name": "deepstream",
                        "description": "NVIDIA video analytics framework",
                        "install": "Jetpack package"
                    },
                    {
                        "name": "gstreamer",
                        "description": "Multimedia framework",
                        "install": "pip install gstreamer-python"
                    }
                ]
            }
        }
    
    def generate_raspberry_pi_project(
        self,
        project_name: str,
        description: str,
        audio_framework: AudioFramework = AudioFramework.JACK,
        sensors: List[SensorType] = None,
        output_dir: Path = None
    ) -> Dict:
        """
        Genera progetto specifico per Raspberry Pi 5.
        
        Args:
            project_name: Nome progetto
            description: Descrizione
            audio_framework: Framework audio da usare
            sensors: Sensori da integrare
            output_dir: Directory output
            
        Returns:
            Configurazione progetto
        """
        output_dir = output_dir or self.output_dir / "raspberry_pi" / project_name.lower().replace(" ", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            "project": {
                "name": project_name,
                "description": description,
                "platform": HardwarePlatform.RASPBERRY_PI_5.value,
                "audio_framework": audio_framework.value,
                "sensors": [s.value for s in (sensors or [])]
            },
            "structure": self._generate_pi_structure(project_name),
            "requirements": self._generate_pi_requirements(audio_framework, sensors),
            "setup_scripts": self._generate_pi_setup_scripts(audio_framework)
        }
        
        self.logger.info(f"✅ Progetto Raspberry Pi generato: {project_name}")
        return config
    
    def generate_jetson_project(
        self,
        project_name: str,
        description: str,
        use_gpu_inference: bool = True,
        requires_realtime: bool = False,
        output_dir: Path = None
    ) -> Dict:
        """
        Genera progetto specifico per NVIDIA Jetson Orin.
        
        Args:
            project_name: Nome progetto
            description: Descrizione
            use_gpu_inference: Abilita GPU inference
            requires_realtime: Requisiti realtime
            output_dir: Directory output
            
        Returns:
            Configurazione progetto
        """
        output_dir = output_dir or self.output_dir / "jetson" / project_name.lower().replace(" ", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            "project": {
                "name": project_name,
                "description": description,
                "platform": HardwarePlatform.NVIDIA_JETSON_ORIN.value,
                "gpu_inference": use_gpu_inference,
                "realtime": requires_realtime
            },
            "structure": self._generate_jetson_structure(project_name),
            "requirements": self._generate_jetson_requirements(use_gpu_inference),
            "cuda_config": self._generate_cuda_config(use_gpu_inference)
        }
        
        self.logger.info(f"✅ Progetto Jetson generato: {project_name}")
        return config
    
    def generate_audio_professional_project(
        self,
        project_name: str,
        description: str,
        framework: AudioFramework = AudioFramework.JACK,
        sample_rate: int = 48000,
        channels: int = 2,
        buffer_size: int = 256
    ) -> Dict:
        """
        Genera progetto audio professionale con JACK/GStreamer.
        
        Args:
            project_name: Nome progetto
            description: Descrizione
            framework: Framework audio
            sample_rate: Sample rate (Hz)
            channels: Numero canali
            buffer_size: Buffer size (samples)
            
        Returns:
            Configurazione progetto
        """
        config = {
            "project": {
                "name": project_name,
                "description": description,
                "type": "professional_audio",
                "framework": framework.value
            },
            "audio_config": {
                "sample_rate": sample_rate,
                "channels": channels,
                "buffer_size": buffer_size,
                "format": "float32"
            },
            "structure": self._generate_audio_structure(project_name, framework),
            "requirements": self._generate_audio_requirements(framework)
        }
        
        self.logger.info(f"✅ Progetto audio professionale generato: {project_name}")
        return config
    
    # ========== PRIVATE HELPER METHODS ==========
    
    def _generate_pi_structure(self, project_name: str) -> Dict:
        """Genera struttura directory per Raspberry Pi"""
        return {
            "root": {
                "README.md": "Documentazione progetto Raspberry Pi",
                "requirements.txt": "Dipendenze Python",
                "setup.sh": "Script setup per Raspberry Pi",
                "src": {
                    "__init__.py": "",
                    "main.py": "# TODO: [Copilot] Implementare logica principale per Raspberry Pi",
                    "hardware": {
                        "__init__.py": "",
                        "gpio_controller.py": "# TODO: [Copilot] Implementare GPIO control",
                        "sensors.py": "# TODO: [Copilot] Implementare sensori",
                        "audio_interface.py": "# TODO: [Copilot] Implementare interfaccia audio"
                    },
                    "utils": {
                        "__init__.py": "",
                        "config.py": "# TODO: [Copilot] Configurazioni hardware specifiche"
                    }
                },
                "tests": {
                    "__init__.py": "",
                    "test_hardware.py": "# TODO: [Copilot] Test hardware"
                }
            }
        }
    
    def _generate_pi_requirements(
        self,
        audio_framework: AudioFramework,
        sensors: List[SensorType]
    ) -> List[str]:
        """Genera requirements.txt per Raspberry Pi"""
        reqs = [
            "RPi.GPIO>=0.7.1",
            "gpiozero>=2.0",
            "sounddevice>=0.4.6",
            "numpy>=1.24.0",
        ]
        
        if audio_framework == AudioFramework.JACK:
            reqs.extend([
                "python-jack-client>=0.5.4",
                "jackd"
            ])
        elif audio_framework == AudioFramework.GSTREAMER:
            reqs.extend([
                "gstreamer-python>=0.10",
                "gstreamer"
            ])
        elif audio_framework == AudioFramework.PIPEWIRE:
            reqs.append("pipewire>=0.3")
        
        # Aggiungi sensori
        if sensors:
            if SensorType.CAMERA in sensors:
                reqs.extend([
                    "opencv-python>=4.8.0",
                    "picamera2"
                ])
            if SensorType.MICROPHONE in sensors:
                reqs.append("sounddevice>=0.4.6")
        
        return reqs
    
    def _generate_pi_setup_scripts(self, audio_framework: AudioFramework) -> Dict:
        """Genera script setup per Raspberry Pi"""
        return {
            "setup.sh": f"""#!/bin/bash
# Setup script per {audio_framework.value}

set -e

echo "🔧 Aggiornamento pacchetti..."
sudo apt-get update && sudo apt-get upgrade -y

echo "📦 Installazione dipendenze di sistema..."
sudo apt-get install -y python3-dev python3-pip libffi-dev libssl-dev

echo "🎵 Installazione framework audio: {audio_framework.value.upper()}..."
"""
        }
    
    def _generate_jetson_structure(self, project_name: str) -> Dict:
        """Genera struttura directory per Jetson"""
        return {
            "root": {
                "README.md": "Progetto NVIDIA Jetson Orin",
                "requirements.txt": "Dipendenze CUDA/TensorRT",
                "setup.sh": "Setup per Jetson",
                "src": {
                    "__init__.py": "",
                    "main.py": "# TODO: [Copilot] Implementare inference su GPU",
                    "inference": {
                        "__init__.py": "",
                        "tensorrt_engine.py": "# TODO: [Copilot] Optimizzazione TensorRT",
                        "video_processing.py": "# TODO: [Copilot] Video analytics con DeepStream",
                        "cuda_kernels.py": "# TODO: [Copilot] Custom CUDA kernels se necessario"
                    },
                    "models": {
                        "__init__.py": "",
                        "model_loader.py": "# TODO: [Copilot] Caricamento modelli ONNX/TensorFlow"
                    }
                },
                "benchmarks": {
                    "latency_test.py": "# TODO: [Copilot] Test latenza inference"
                }
            }
        }
    
    def _generate_jetson_requirements(self, use_gpu: bool) -> List[str]:
        """Genera requirements.txt per Jetson"""
        reqs = [
            "opencv-python>=4.8.0",
            "numpy>=1.24.0",
            "pyyaml>=6.0"
        ]
        
        if use_gpu:
            reqs.extend([
                "tensorrt>=8.6",
                "pycuda>=2022.2",
                "torch>=2.0",  # Per modelli AI
            ])
        
        return reqs
    
    def _generate_cuda_config(self, use_gpu: bool) -> Dict:
        """Genera configurazione CUDA"""
        return {
            "cuda_enabled": use_gpu,
            "device_id": 0,
            "memory_fraction": 0.8,
            "optimization_level": 3,
            "tensorrt_precision": "FP16" if use_gpu else "FP32"
        }
    
    def _generate_audio_structure(self, project_name: str, framework: AudioFramework) -> Dict:
        """Genera struttura per progetto audio"""
        return {
            "root": {
                "README.md": f"Progetto audio professionale con {framework.value}",
                "requirements.txt": "Dipendenze audio",
                "src": {
                    "__init__.py": "",
                    "audio_engine.py": f"# TODO: [Copilot] Implementare {framework.value} audio engine",
                    "dsp": {
                        "__init__.py": "",
                        "filters.py": "# TODO: [Copilot] Implementare filtri DSP",
                        "effects.py": "# TODO: [Copilot] Implementare effetti audio",
                        "spatial.py": "# TODO: [Copilot] Implementare audio spaziale 3D"
                    }
                },
                "config": {
                    "audio_config.yaml": f"sample_rate: 48000\nframework: {framework.value}"
                }
            }
        }
    
    def _generate_audio_requirements(self, framework: AudioFramework) -> List[str]:
        """Genera requirements per audio"""
        reqs = [
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "librosa>=0.10.0",
            "soundfile>=0.12.1",
        ]
        
        if framework == AudioFramework.JACK:
            reqs.extend([
                "python-jack-client>=0.5.4",
                "jackd"
            ])
        elif framework == AudioFramework.GSTREAMER:
            reqs.append("gstreamer-python>=0.10")
        elif framework == AudioFramework.PIPEWIRE:
            reqs.append("pipewire>=0.3")
        
        return reqs


def research_hardware_innovations() -> Dict:
    """
    WebResearchAgent integration: cerca innovazioni hardware per audio/AI.
    Ritorna topic specifici per ricerca settimanale.
    """
    return {
        "topics": [
            "Raspberry Pi audio processing 2024-2025",
            "NVIDIA Jetson real-time AI inference",
            "Professional audio on Raspberry Pi",
            "Spatial audio processing with TensorFlow",
            "Ultra-low latency audio DSP on ARM",
            "JACK vs PipeWire comparison"
        ],
        "search_keywords": [
            "raspberry pi 5 audio",
            "jetson orin inference",
            "ALSA configuration",
            "GStreamer audio pipeline",
            "JACK server setup"
        ]
    }


if __name__ == "__main__":
    # Test
    agent = HardwareIntegrationAgent()
    
    print("🤖 Hardware Integration Agent\n")
    
    # Genera progetto Raspberry Pi
    pi_config = agent.generate_raspberry_pi_project(
        "Audio Analyzer",
        "Real-time audio analysis on Raspberry Pi 5",
        audio_framework=AudioFramework.JACK,
        sensors=[SensorType.MICROPHONE, SensorType.CAMERA]
    )
    print(f"✅ Raspberry Pi project: {pi_config['project']['name']}")
    
    # Genera progetto Jetson
    jetson_config = agent.generate_jetson_project(
        "Video Analytics",
        "Real-time video analytics with GPU acceleration",
        use_gpu_inference=True
    )
    print(f"✅ Jetson project: {jetson_config['project']['name']}")
    
    # Genera progetto audio professionale
    audio_config = agent.generate_audio_professional_project(
        "Spatial Audio Studio",
        "Professional spatial audio processing",
        framework=AudioFramework.JACK,
        sample_rate=192000,
        channels=8
    )
    print(f"✅ Audio professional project: {audio_config['project']['name']}\n")
