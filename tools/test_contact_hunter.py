#!/usr/bin/env python3
"""
🧪 Test Contact Hunter - MAXXI Roma
====================================
Test del Contact Hunter Agent su un museo reale.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "datapizza-ai-core"))

from datapizza.agents.contact_hunter import ContactHunterAgent
from datapizza.database.contacts_db import ContactDatabase

def main():
    print("🎨 Testing Contact Hunter on MAXXI Roma...")
    print("=" * 60)
    
    # Initialize
    hunter = ContactHunterAgent(delay=2.0)
    db = ContactDatabase()
    
    # Target
    target_url = "https://www.maxxi.art"
    target_name = "MAXXI - Museo nazionale delle arti del XXI secolo"
    
    print(f"\n🔍 Hunting contacts from: {target_url}")
    print(f"📍 Organization: {target_name}\n")
    
    # Hunt!
    try:
        contacts = hunter.hunt_contacts(
            base_url=target_url,
            organization_name=target_name
        )
        
        print(f"\n✅ Found {len(contacts)} contacts!\n")
        
        # Display results
        print("📋 RESULTS:")
        print("-" * 60)
        
        for i, contact in enumerate(sorted(contacts, key=lambda c: c.confidence, reverse=True), 1):
            confidence_emoji = "🟢" if contact.confidence > 0.7 else "🟡" if contact.confidence > 0.4 else "🔴"
            
            print(f"\n{i}. {confidence_emoji} Confidence: {contact.confidence:.0%}")
            print(f"   📧 Email: {contact.email}")
            if contact.name:
                print(f"   👤 Name: {contact.name}")
            if contact.role:
                print(f"   💼 Role: {contact.role}")
            print(f"   🔗 Source: {contact.source_url}")
        
        # Save to database
        print(f"\n💾 Saving to database...")
        for contact in contacts:
            db.add_contact(contact)
        
        print(f"✅ Saved {len(contacts)} contacts to database!")
        
        # Show stats
        stats = db.get_stats()
        print(f"\n📊 DATABASE STATS:")
        print(f"   Total contacts: {stats['total_contacts']}")
        print(f"   By organization: {len(db.get_contacts(organization=target_name))}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
