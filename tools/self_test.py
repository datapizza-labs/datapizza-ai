"""
MOOD 2.0 - Self Test
Esegue test rapidi su componenti chiave (senza side-effects esterni).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

RESULTS: list[tuple[str, bool, str]] = []

def record(name: str, ok: bool, msg: str = ""):
    RESULTS.append((name, ok, msg))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{(' — ' + msg) if msg else ''}")

# Test 1: Learning Agent
try:
    from learning_agent import LearningAgent, ActionType, FeedbackType
    # Passa un Path, non una stringa, per evitare errori .mkdir su str
    la = LearningAgent(output_dir=ROOT / "outputs" / "learning")
    ok1, conf1 = la.record_feedback("t-1", ActionType.UPDATE_DEPENDENCY, FeedbackType.APPROVED)
    ok2, conf2 = la.record_feedback("t-2", ActionType.UPDATE_DEPENDENCY, FeedbackType.APPROVED)
    report = la.get_learning_report()
    assert "Learning Report" in report or "Report" in report
    record("LearningAgent", True, f"conf={conf2:.2f}")
except Exception as e:
    record("LearningAgent", False, str(e))

# Test 2: Hardware Integration
try:
    from hardware_integration import HardwareIntegrationAgent, AudioFramework, SensorType
    h = HardwareIntegrationAgent()
    cfg = h.generate_raspberry_pi_project(
        project_name="Audio Analyzer",
        description="Unit test",
        # Usa i nomi Enum corretti (maiuscoli)
        audio_framework=AudioFramework.JACK,
        sensors=[SensorType.MICROPHONE, SensorType.CAMERA],
    )
    assert "requirements" in cfg and cfg["requirements"]
    record("HardwareIntegration", True, f"reqs={len(cfg['requirements'])}")
except Exception as e:
    record("HardwareIntegration", False, str(e))

# Test 3: GitHub Automation (dry)
try:
    from github_automation import GitHubAutomation
    # In modalità dry-run non effettua chiamate git o rete
    gh = GitHubAutomation(dry_run=True)
    meta = gh.simulate_pr_creation(project_name="SelfTest Project", template="Python Project")
    assert meta.get("branch_name")
    record("GitHubAutomation(dry)", True, meta["branch_name"])
except Exception as e:
    record("GitHubAutomation(dry)", False, str(e))

# Test 4: Google API availability (optional)
try:
    api = os.getenv("GOOGLE_API_KEY")
    if not api:
        record("GoogleAPI", True, "SKIPPED (no key)")
    else:
        try:
            import google.generativeai as genai
        except ModuleNotFoundError:
            record("GoogleAPI", True, "SKIPPED (package not installed)")
        else:
            genai.configure(api_key=api)
            model = genai.GenerativeModel('gemini-2.0-flash')
            resp = model.generate_content("ping")
            ok = hasattr(resp, "text")
            record("GoogleAPI", ok, (resp.text[:30] + "...") if ok else "no text")
except Exception as e:
    record("GoogleAPI", False, str(e))

# Summary
print("\n=== SUMMARY ===")
passed = sum(1 for _, ok, _ in RESULTS if ok)
failed = len(RESULTS) - passed
for name, ok, msg in RESULTS:
    print(f"- {name}: {'PASS' if ok else 'FAIL'}")

if failed:
    raise SystemExit(1)
