"""
GitHub Automation: Crea PR completamente automatizzati da progetti generati.

Automatizza:
1. Creazione branch Git locale
2. Commit della struttura generata
3. Push su GitHub
4. Creazione Pull Request su datapizza-ai
5. Assegnazione PR e setup reviewer
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GitHubConfig:
    """Configurazione GitHub"""
    token: str
    repo_owner: str
    repo_name: str
    base_branch: str = "main"
    remote_name: str = "origin"


@dataclass
class PRMetadata:
    """Metadati Pull Request"""
    pr_number: int
    pr_url: str
    branch_name: str
    commit_hash: str
    created_at: str


class GitHubAutomation:
    """
    Automatizza creazione PR da progetti generati.
    """
    
    def __init__(self, github_config: Optional[GitHubConfig] = None, dry_run: bool = False):
        """
        Inizializza GitHub Automation.
        
        Args:
            github_config: Configurazione GitHub (token, repo, ecc.)
            dry_run: Se True, abilita funzioni di simulazione senza side-effects
        """
        self.config = github_config
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(__file__).parent.parent / "outputs" / "github"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run

    def simulate_pr_creation(self, project_name: str, template: str = "Python Project") -> Dict:
        """
        Simula la creazione di una PR senza usare git o la rete.
        Ritorna metadati fittizi utili per test/anteprima.
        """
        now = datetime.now()
        branch_name = f"feature/mood-{project_name.lower().replace(' ', '-')}-{now.strftime('%Y%m%d')}"
        fake_commit = now.strftime("%Y%m%d%H%M%S")
        pr_number = int(now.strftime("%H%M%S"))  # numero fittizio
        
        # Usa config reale se disponibile, altrimenti carica da .env o usa default
        if self.config:
            owner = self.config.repo_owner
            repo = self.config.repo_name
        else:
            try:
                import os
                from dotenv import load_dotenv
                load_dotenv()
                owner = os.getenv('GITHUB_REPO_OWNER', 'ninuxi')
                repo = os.getenv('GITHUB_REPO_NAME', 'datapizza-ai')
            except:
                owner = 'ninuxi'
                repo = 'datapizza-ai'
        
        pr_url = f"https://github.com/{owner}/{repo}/pull/{pr_number}"
        meta = {
            "pr_number": pr_number,
            "pr_url": pr_url,
            "branch_name": branch_name,
            "commit_hash": fake_commit,
            "created_at": now.isoformat(),
            "template": template,
        }
        # Salva anche un file di log della simulazione per tracciabilità
        try:
            sim_file = self.output_dir / f"pr_sim_{now.strftime('%Y%m%d_%H%M%S')}.json"
            with open(sim_file, 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Impossibile salvare simulazione PR: {e}")
        return meta
    
    def load_config_from_env(self) -> GitHubConfig:
        """Carica configurazione da .env"""
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            raise ValueError("GITHUB_TOKEN non trovato in .env")
        
        return GitHubConfig(
            token=token,
            repo_owner=os.getenv('GITHUB_REPO_OWNER', 'ninuxi'),
            repo_name=os.getenv('GITHUB_REPO_NAME', 'datapizza-ai'),
            base_branch=os.getenv('GITHUB_BASE_BRANCH', 'main')
        )
    
    def create_feature_branch(
        self,
        project_name: str,
        project_dir: Path
    ) -> Tuple[bool, str]:
        """
        Crea un feature branch locale e commita il progetto.
        
        Args:
            project_name: Nome del progetto
            project_dir: Directory del progetto generato
            
        Returns:
            (success, branch_name)
        """
        try:
            # Sanitizza nome branch
            branch_name = f"feature/mood-{project_name.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d')}"
            
            # Crea branch
            self.logger.info(f"Creazione branch: {branch_name}")
            subprocess.run(
                ['git', 'checkout', '-b', branch_name],
                cwd=project_dir,
                check=True,
                capture_output=True
            )
            
            # Aggiungi tutti i file
            self.logger.info("Aggiunta file al staging area")
            subprocess.run(
                ['git', 'add', '.'],
                cwd=project_dir,
                check=True,
                capture_output=True
            )
            
            # Commit
            commit_message = (
                f"feat: {project_name} project generated by MOOD Agent\n\n"
                f"Generated at: {datetime.now().isoformat()}\n"
                f"Project structure and boilerplate code with Copilot TODOs"
            )
            
            self.logger.info("Commit dei file")
            result = subprocess.run(
                ['git', 'commit', '-m', commit_message],
                cwd=project_dir,
                check=True,
                capture_output=True,
                text=True
            )
            
            self.logger.info(f"✅ Branch {branch_name} creato e committato")
            return True, branch_name
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Errore creazione branch: {e.stderr}")
            return False, ""
    
    def push_to_github(
        self,
        project_dir: Path,
        branch_name: str,
        remote_name: str = "origin"
    ) -> Tuple[bool, str]:
        """
        Fa push del branch su GitHub.
        
        Args:
            project_dir: Directory del progetto
            branch_name: Nome del branch
            remote_name: Nome del remote
            
        Returns:
            (success, commit_hash)
        """
        try:
            # Ottieni commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=project_dir,
                check=True,
                capture_output=True,
                text=True
            )
            commit_hash = result.stdout.strip()
            
            # Push
            self.logger.info(f"Push di {branch_name} su {remote_name}")
            subprocess.run(
                ['git', 'push', remote_name, branch_name],
                cwd=project_dir,
                check=True,
                capture_output=True
            )
            
            self.logger.info(f"✅ Push completato: {commit_hash}")
            return True, commit_hash
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Errore push: {e.stderr}")
            return False, ""
    
    def create_pull_request(
        self,
        project_name: str,
        branch_name: str,
        implementation_guide: str,
        templates_used: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[PRMetadata]]:
        """
        Crea Pull Request su GitHub usando l'API.
        
        Args:
            project_name: Nome del progetto
            branch_name: Nome del branch
            implementation_guide: Guida implementazione (da IMPLEMENTATION.md)
            templates_used: Lista di template utilizzati
            
        Returns:
            (success, PRMetadata)
        """
        try:
            import requests
            
            if not self.config:
                self.config = self.load_config_from_env()
            
            # Prepara body PR
            body = f"""# 🚀 New Project: {project_name}

**Generated by:** MOOD Agent System
**Generated at:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## 📋 Overview

This PR introduces a complete project structure and boilerplate code for **{project_name}**.

## 📁 Structure

- Complete project layout with organized directories
- Base configuration files
- Entry points and main modules
- Test structure
- Documentation

## 📖 Implementation Guide

{implementation_guide}

## 🎯 Templates Used

{chr(10).join([f"- {t}" for t in (templates_used or ["Standard Python Project"])])}

## ✅ Next Steps

1. Review the generated structure
2. Complete the TODO items (marked with `# TODO: [Copilot]`)
3. Run tests and validation
4. Merge and deploy

## 🤖 MOOD Agent

This project was generated using the MOOD Agent's intelligent project generation system. 
All code follows Copilot-ready patterns with detailed TODOs for rapid implementation.

---
*Auto-generated by MOOD AI Agent*
"""
            
            # API endpoint
            url = f"https://api.github.com/repos/{self.config.repo_owner}/{self.config.repo_name}/pulls"
            
            # Headers con autenticazione
            headers = {
                'Authorization': f'token {self.config.token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'MOOD-Agent'
            }
            
            # Payload
            payload = {
                'title': f"✨ feat: Add {project_name} project (MOOD-generated)",
                'head': branch_name,
                'base': self.config.base_branch,
                'body': body,
                'draft': False
            }
            
            # Crea PR
            self.logger.info(f"Creazione PR: {project_name}")
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 201:
                pr_data = response.json()
                
                metadata = PRMetadata(
                    pr_number=pr_data['number'],
                    pr_url=pr_data['html_url'],
                    branch_name=branch_name,
                    commit_hash=pr_data['head']['sha'],
                    created_at=pr_data['created_at']
                )
                
                self.logger.info(f"✅ PR creato: #{metadata.pr_number}")
                self._save_pr_metadata(metadata)
                
                return True, metadata
            else:
                self.logger.error(f"❌ Errore API GitHub: {response.status_code} - {response.text}")
                return False, None
                
        except Exception as e:
            self.logger.error(f"❌ Errore creazione PR: {e}")
            return False, None
    
    def create_project_pr(
        self,
        project_name: str,
        project_dir: Path,
        implementation_guide: str,
        templates_used: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[PRMetadata]]:
        """
        Pipeline completa: branch -> commit -> push -> PR.
        
        Args:
            project_name: Nome progetto
            project_dir: Directory progetto
            implementation_guide: Guida implementazione
            templates_used: Template utilizzati
            
        Returns:
            (success, PRMetadata)
        """
        print("\n" + "="*60)
        print(f"🚀 MOOD GitHub Automation - {project_name}")
        print("="*60 + "\n")
        
        # 1. Crea branch
        print("1️⃣  Creazione feature branch...")
        success, branch_name = self.create_feature_branch(project_name, project_dir)
        if not success:
            return False, None
        print(f"   ✅ Branch: {branch_name}\n")
        
        # 2. Push
        print("2️⃣  Push su GitHub...")
        success, commit_hash = self.push_to_github(project_dir, branch_name)
        if not success:
            return False, None
        print(f"   ✅ Commit: {commit_hash[:7]}\n")
        
        # 3. Crea PR
        print("3️⃣  Creazione Pull Request...")
        success, metadata = self.create_pull_request(
            project_name,
            branch_name,
            implementation_guide,
            templates_used
        )
        if not success or metadata is None:
            return False, None
        
        print(f"   ✅ PR: #{metadata.pr_number}")
        print(f"   🔗 URL: {metadata.pr_url}\n")
        
        print("="*60)
        print(f"✨ Progetto '{project_name}' pronto per review su GitHub!")
        print("="*60 + "\n")
        
        return True, metadata
    
    def _save_pr_metadata(self, metadata: PRMetadata):
        """Salva metadati PR per tracking"""
        try:
            pr_file = self.output_dir / f"pr_{metadata.pr_number}.json"
            with open(pr_file, 'w') as f:
                json.dump({
                    'pr_number': metadata.pr_number,
                    'pr_url': metadata.pr_url,
                    'branch_name': metadata.branch_name,
                    'commit_hash': metadata.commit_hash,
                    'created_at': metadata.created_at
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Errore salvataggio PR metadata: {e}")


def integrate_with_vscode_generator(
    vscode_project_dir: Path,
    project_name: str,
    github_config: Optional[GitHubConfig] = None
) -> Tuple[bool, Optional[PRMetadata]]:
    """
    Funzione di integrazione: prendi progetto da VSCodeProjectGenerator
    e crea PR completa automaticamente.
    
    Args:
        vscode_project_dir: Directory con progetto generato
        project_name: Nome progetto
        github_config: Configurazione GitHub
        
    Returns:
        (success, PRMetadata)
    """
    try:
        # Carica implementation guide
        impl_guide_path = vscode_project_dir / "IMPLEMENTATION.md"
        if impl_guide_path.exists():
            with open(impl_guide_path, 'r') as f:
                implementation_guide = f.read()
        else:
            implementation_guide = "See project structure and TODO items for implementation details."
        
        # Carica templates usati (da .copilot-instructions.md se possibile)
        templates = ["Python Project Template", "Copilot Integration"]
        
        # Crea automazione
        automator = GitHubAutomation(github_config or GitHubConfig(
            token="",
            repo_owner="ninuxi",
            repo_name="datapizza-ai"
        ))
        
        # Pipeline completa
        success, metadata = automator.create_project_pr(
            project_name,
            vscode_project_dir,
            implementation_guide,
            templates
        )
        
        return success, metadata
        
    except Exception as e:
        logging.error(f"Errore integrazione VSCode generator: {e}")
        return False, None


if __name__ == "__main__":
    # Test
    print("🧪 Testing GitHub Automation\n")
    
    # Simula configurazione
    config = GitHubConfig(
        token="test_token_placeholder",
        repo_owner="ninuxi",
        repo_name="datapizza-ai"
    )
    
    automator = GitHubAutomation(config)
    print("✅ GitHub Automation inizializzato")
