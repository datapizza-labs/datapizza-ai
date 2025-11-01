#!/usr/bin/env python3
"""
🤖 MOOD Developer Agent CLI
============================
Interfaccia command-line per MOOD Developer Agent.

Usage:
    python tools/mood_dev_agent.py analyze-tech "Raspberry Pi 5"
    python tools/mood_dev_agent.py propose-feature "Multi-room sync"
    python tools/mood_dev_agent.py integrate-software "GrandMA" --protocol OSC
    python tools/mood_dev_agent.py integrate-hardware "Raspberry Pi 5"
    python tools/mood_dev_agent.py monitor-events
    python tools/mood_dev_agent.py weekly-sprint
    python tools/mood_dev_agent.py create-demo "Biometric sensor integration"
    python tools/mood_dev_agent.py code-review path/to/file.py
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "datapizza-ai-core"))

from datapizza.agents.mood_developer_agent import MOODDeveloperAgent, MOODDevelopmentTeam
from datapizza.clients.google import GoogleClient


def setup_agent():
    """Initialize MOOD Developer Agent"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("   Export it: export GOOGLE_API_KEY=your_key_here")
        sys.exit(1)
    
    client = GoogleClient(api_key=api_key, model="gemini-2.0-flash-exp")
    return MOODDeveloperAgent(client=client), MOODDevelopmentTeam(client=client)


def cmd_analyze_tech(args, agent):
    """Analizza nuova tecnologia"""
    print(f"\n🔬 Analyzing: {args.technology}")
    print("=" * 60)
    
    result = agent.analyze_technology(
        technology_name=args.technology,
        context=args.context or ""
    )
    
    print(result)
    
    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"\n💾 Saved to: {args.output}")


def cmd_propose_feature(args, agent):
    """Proponi nuova feature"""
    print(f"\n💡 Proposing feature: {args.feature}")
    print("=" * 60)
    
    result = agent.propose_feature(
        feature_description=args.feature,
        target_audience=args.audience or "museums"
    )
    
    print(result)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"\n💾 Saved to: {args.output}")


def cmd_integrate_software(args, agent):
    """Genera codice integrazione software"""
    print(f"\n🎛️ Integrating: {args.software}")
    print("=" * 60)
    
    result = agent.implement_integration(
        software=args.software,
        protocol=args.protocol or "OSC",
        mood_component=args.component or "server"
    )
    
    print(result)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"\n💾 Saved to: {args.output}")


def cmd_integrate_hardware(args, team):
    """Implementa integrazione hardware completa"""
    print(f"\n🔧 Hardware Integration: {args.hardware}")
    print("=" * 60)
    
    result = team.implement_hardware_integration(hardware=args.hardware)
    
    print(result)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"\n💾 Saved to: {args.output}")


def cmd_monitor_events(args, agent):
    """Monitora eventi e conferenze"""
    print("\n📅 Monitoring Events & Conferences")
    print("=" * 60)
    
    result = agent.monitor_events(event_type=args.type or "all")
    
    print(result)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"\n💾 Saved to: {args.output}")


def cmd_weekly_sprint(args, team):
    """Esegui weekly innovation sprint"""
    print("\n🚀 MOOD Weekly Innovation Sprint")
    print("=" * 60)
    
    result = team.weekly_innovation_sprint()
    
    print("\n📅 EVENTS REPORT:")
    print("-" * 60)
    print(result['events_report'])
    
    print("\n\n🔬 TECHNOLOGY ANALYSIS:")
    print("-" * 60)
    print(result['technology_analysis'])
    
    print("\n\n💡 FEATURE PROPOSAL:")
    print("-" * 60)
    print(result['feature_proposal'])
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Saved to: {args.output}")


def cmd_create_demo(args, agent):
    """Crea demo/proof-of-concept"""
    print(f"\n🎬 Creating Demo: {args.description}")
    print("=" * 60)
    
    result = agent.create_demo(
        demo_description=args.description,
        hardware=args.hardware or "Jetson Nano"
    )
    
    print(result)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"\n💾 Saved to: {args.output}")


def cmd_code_review(args, agent):
    """Fai code review"""
    print(f"\n🔍 Code Review: {args.file}")
    print("=" * 60)
    
    # Read file
    with open(args.file, 'r') as f:
        code = f.read()
    
    result = agent.code_review(
        code=code,
        focus=args.focus or "general"
    )
    
    print(result)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"\n💾 Saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="🤖 MOOD Developer Agent - Autonomous Development & Innovation"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # analyze-tech
    p_tech = subparsers.add_parser('analyze-tech', help='Analyze new technology')
    p_tech.add_argument('technology', help='Technology name (e.g., "Raspberry Pi 5")')
    p_tech.add_argument('--context', help='Context for analysis')
    p_tech.add_argument('--output', '-o', help='Output file path')
    
    # propose-feature
    p_feature = subparsers.add_parser('propose-feature', help='Propose new feature')
    p_feature.add_argument('feature', help='Feature description')
    p_feature.add_argument('--audience', help='Target audience (default: museums)')
    p_feature.add_argument('--output', '-o', help='Output file path')
    
    # integrate-software
    p_soft = subparsers.add_parser('integrate-software', help='Generate software integration code')
    p_soft.add_argument('software', help='Software name (e.g., "GrandMA", "Resolume")')
    p_soft.add_argument('--protocol', help='Protocol (default: OSC)')
    p_soft.add_argument('--component', help='MOOD component (default: server)')
    p_soft.add_argument('--output', '-o', help='Output file path')
    
    # integrate-hardware
    p_hard = subparsers.add_parser('integrate-hardware', help='Complete hardware integration')
    p_hard.add_argument('hardware', help='Hardware name (e.g., "Raspberry Pi 5")')
    p_hard.add_argument('--output', '-o', help='Output file path')
    
    # monitor-events
    p_events = subparsers.add_parser('monitor-events', help='Monitor events and conferences')
    p_events.add_argument('--type', help='Event type (tech, art, ai, interactive, all)')
    p_events.add_argument('--output', '-o', help='Output file path')
    
    # weekly-sprint
    p_sprint = subparsers.add_parser('weekly-sprint', help='Run weekly innovation sprint')
    p_sprint.add_argument('--output', '-o', help='Output file path (JSON)')
    
    # create-demo
    p_demo = subparsers.add_parser('create-demo', help='Create demo/proof-of-concept')
    p_demo.add_argument('description', help='Demo description')
    p_demo.add_argument('--hardware', help='Target hardware (default: Jetson Nano)')
    p_demo.add_argument('--output', '-o', help='Output file path')
    
    # code-review
    p_review = subparsers.add_parser('code-review', help='Review code file')
    p_review.add_argument('file', help='Python file to review')
    p_review.add_argument('--focus', help='Focus area (performance, security, architecture, style)')
    p_review.add_argument('--output', '-o', help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize agent
    print("🤖 Initializing MOOD Developer Agent...")
    agent, team = setup_agent()
    print("✅ Agent ready!\n")
    
    # Execute command
    if args.command == 'analyze-tech':
        cmd_analyze_tech(args, agent)
    elif args.command == 'propose-feature':
        cmd_propose_feature(args, agent)
    elif args.command == 'integrate-software':
        cmd_integrate_software(args, agent)
    elif args.command == 'integrate-hardware':
        cmd_integrate_hardware(args, team)
    elif args.command == 'monitor-events':
        cmd_monitor_events(args, agent)
    elif args.command == 'weekly-sprint':
        cmd_weekly_sprint(args, team)
    elif args.command == 'create-demo':
        cmd_create_demo(args, agent)
    elif args.command == 'code-review':
        cmd_code_review(args, agent)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
