"""
🔧 VS Code Integration
======================
Genera progetti pronti per VS Code con scaffolding completo.
Quando MOOD Dev Agent propone una feature, questo sistema:
1. Genera struttura file completa
2. Crea boilerplate code
3. Aggiunge TODO con Copilot hints
4. Genera .vscode/tasks.json per build/run
5. Apre il progetto in VS Code

Autore: Antonio Mainenti
Data: 29 Ottobre 2025
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from google import genai


class VSCodeProjectGenerator:
    """
    Genera progetti completi da proposte MOOD Dev Agent.
    Output: struttura cartelle + file + VS Code config
    """
    
    def __init__(self, api_key: Optional[str] = None, workspace_root: str = "outputs/projects"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY non trovata")
        
        self.client = genai.Client(api_key=self.api_key.strip())
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
    
    def generate_project_from_proposal(self, proposal: Dict, project_name: Optional[str] = None) -> Dict:
        """
        Genera progetto completo da proposta Dev Agent.
        
        Args:
            proposal: Dict con feature_description, target_audience, technology, etc.
            project_name: Nome progetto (default: auto-generato)
            
        Returns:
            Dict con path progetto, file generati, istruzioni
        """
        if not project_name:
            project_name = self._sanitize_name(proposal.get('title', f"mood_project_{datetime.now().strftime('%Y%m%d_%H%M')}"))
        
        project_path = self.workspace_root / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Analizza proposta e genera struttura
        structure = self._analyze_and_design_structure(proposal)
        
        # Genera file
        generated_files = []
        
        # 1. README.md
        readme_content = self._generate_readme(proposal, structure)
        self._write_file(project_path / "README.md", readme_content)
        generated_files.append("README.md")
        
        # 2. Main code file(s)
        for code_file in structure.get('code_files', []):
            code_content = self._generate_code_file(proposal, code_file, structure)
            file_path = project_path / code_file['path']
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_file(file_path, code_content)
            generated_files.append(code_file['path'])
        
        # 3. requirements.txt / package.json
        if structure.get('language') == 'python':
            deps_content = self._generate_python_deps(structure)
            self._write_file(project_path / "requirements.txt", deps_content)
            generated_files.append("requirements.txt")
        elif structure.get('language') == 'javascript':
            deps_content = self._generate_package_json(structure, project_name)
            self._write_file(project_path / "package.json", deps_content)
            generated_files.append("package.json")
        
        # 4. VS Code configuration
        vscode_dir = project_path / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        # tasks.json
        tasks = self._generate_tasks_json(structure)
        self._write_file(vscode_dir / "tasks.json", json.dumps(tasks, indent=2))
        generated_files.append(".vscode/tasks.json")
        
        # launch.json
        launch = self._generate_launch_json(structure)
        self._write_file(vscode_dir / "launch.json", json.dumps(launch, indent=2))
        generated_files.append(".vscode/launch.json")
        
        # settings.json (with Copilot hints)
        settings = self._generate_settings_json(structure)
        self._write_file(vscode_dir / "settings.json", json.dumps(settings, indent=2))
        generated_files.append(".vscode/settings.json")
        
        # 5. .copilot-instructions.md (hints for GitHub Copilot)
        copilot_hints = self._generate_copilot_instructions(proposal, structure)
        self._write_file(project_path / ".copilot-instructions.md", copilot_hints)
        generated_files.append(".copilot-instructions.md")
        
        # 6. IMPLEMENTATION.md (step-by-step guide)
        impl_guide = self._generate_implementation_guide(proposal, structure)
        self._write_file(project_path / "IMPLEMENTATION.md", impl_guide)
        generated_files.append("IMPLEMENTATION.md")
        
        return {
            "project_name": project_name,
            "project_path": str(project_path.absolute()),
            "generated_files": generated_files,
            "structure": structure,
            "next_steps": [
                f"1. Apri in VS Code: code {project_path}",
                "2. Leggi IMPLEMENTATION.md per step-by-step",
                "3. Installa dipendenze (vedi README.md)",
                "4. Usa Copilot per completare i TODO",
                "5. Run con VS Code task (Cmd+Shift+B)"
            ]
        }
    
    def _analyze_and_design_structure(self, proposal: Dict) -> Dict:
        """Usa LLM per analizzare proposta e disegnare struttura progetto"""
        prompt = f"""You are a senior software architect. Analyze this project proposal and design a complete project structure.

**Proposal:**
{json.dumps(proposal, indent=2)}

Design:
1. **Language**: Best language for this project (Python, JavaScript, etc.)
2. **Architecture**: Pattern to use (MVC, microservices, script, etc.)
3. **File Structure**: List of files to create with purpose
4. **Dependencies**: Required libraries/packages
5. **Entry Point**: Main file to run
6. **Testing Strategy**: How to test

Output as JSON with this structure:
{{
  "language": "python",
  "architecture": "script",
  "entry_point": "main.py",
  "code_files": [
    {{"path": "main.py", "purpose": "Entry point", "type": "executable"}},
    {{"path": "utils.py", "purpose": "Helper functions", "type": "module"}}
  ],
  "dependencies": ["requests", "python-osc"],
  "test_strategy": "pytest",
  "vscode_extensions": ["ms-python.python", "GitHub.copilot"]
}}

Be concise and practical. Focus on MVP.
"""
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            # Parse JSON from response
            text = response.text
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                structure = json.loads(text[json_start:json_end])
                return structure
            else:
                # Fallback structure
                return self._default_structure(proposal)
                
        except Exception as e:
            print(f"Error analyzing structure: {e}")
            return self._default_structure(proposal)
    
    def _default_structure(self, proposal: Dict) -> Dict:
        """Struttura di default se LLM fallisce"""
        return {
            "language": "python",
            "architecture": "script",
            "entry_point": "main.py",
            "code_files": [
                {"path": "main.py", "purpose": "Entry point", "type": "executable"}
            ],
            "dependencies": ["requests"],
            "test_strategy": "manual",
            "vscode_extensions": ["ms-python.python", "GitHub.copilot"]
        }
    
    def _generate_readme(self, proposal: Dict, structure: Dict) -> str:
        """Genera README.md"""
        project_name = proposal.get('title', 'MOOD Project')
        description = proposal.get('feature_description', 'No description')
        
        return f"""# {project_name}

## 📝 Description
{description}

## 🎯 Target
{proposal.get('target_audience', 'General')}

## 🛠️ Technology Stack
- **Language**: {structure.get('language', 'Python')}
- **Architecture**: {structure.get('architecture', 'Script')}
- **Dependencies**: {', '.join(structure.get('dependencies', []))}

## 🚀 Quick Start

### 1. Install Dependencies
```bash
{'pip install -r requirements.txt' if structure.get('language') == 'python' else 'npm install'}
```

### 2. Run
```bash
{'python ' + structure.get('entry_point', 'main.py') if structure.get('language') == 'python' else 'npm start'}
```

### 3. Development with VS Code
- Open project: `code .`
- Use **GitHub Copilot** for auto-completion
- Read `.copilot-instructions.md` for context
- Check `IMPLEMENTATION.md` for step-by-step guide

## 📂 Project Structure
```
{self._format_structure_tree(structure)}
```

## 🔧 VS Code Tasks
- **Build**: Cmd+Shift+B
- **Run**: Cmd+Shift+P → "Run Task" → "Run Project"
- **Test**: Cmd+Shift+P → "Run Task" → "Test"

## 📖 Implementation Guide
See `IMPLEMENTATION.md` for detailed step-by-step instructions.

## 🤝 Contributing
Generated by **MOOD Dev Agent** on {datetime.now().strftime('%Y-%m-%d %H:%M')}.

## 📄 License
MIT
"""
    
    def _generate_code_file(self, proposal: Dict, file_spec: Dict, structure: Dict) -> str:
        """Genera contenuto file di codice con TODO per Copilot"""
        prompt = f"""Generate a code file for this project.

**Project Proposal:**
{proposal.get('feature_description', '')}

**File to Generate:**
- Path: {file_spec['path']}
- Purpose: {file_spec['purpose']}
- Type: {file_spec['type']}

**Language:** {structure.get('language', 'python')}
**Architecture:** {structure.get('architecture', 'script')}

Generate:
1. Complete boilerplate with imports
2. Main structure (functions/classes)
3. TODO comments with detailed hints for GitHub Copilot
4. Docstrings explaining what each part should do

**IMPORTANT**: 
- Add TODO comments like: "# TODO: [Copilot] Implement XYZ using ABC library"
- These hints help Copilot understand context
- Include example usage in docstrings

Keep code clean and well-structured. Focus on clarity.
"""
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            code = response.text
            
            # Rimuovi markdown code fences se presenti
            if code.startswith('```'):
                lines = code.split('\n')
                code = '\n'.join(lines[1:-1]) if len(lines) > 2 else code
            
            return code
            
        except Exception as e:
            return f"# Error generating code: {e}\n# TODO: Implement {file_spec['purpose']}\n"
    
    def _generate_python_deps(self, structure: Dict) -> str:
        """Genera requirements.txt"""
        deps = structure.get('dependencies', [])
        return '\n'.join(deps) + '\n'
    
    def _generate_package_json(self, structure: Dict, project_name: str) -> str:
        """Genera package.json"""
        return json.dumps({
            "name": project_name,
            "version": "0.1.0",
            "description": "Generated by MOOD Dev Agent",
            "main": structure.get('entry_point', 'index.js'),
            "scripts": {
                "start": f"node {structure.get('entry_point', 'index.js')}",
                "test": "echo \"Error: no test specified\" && exit 1"
            },
            "dependencies": {dep: "*" for dep in structure.get('dependencies', [])}
        }, indent=2)
    
    def _generate_tasks_json(self, structure: Dict) -> Dict:
        """Genera .vscode/tasks.json"""
        language = structure.get('language', 'python')
        entry = structure.get('entry_point', 'main.py')
        
        if language == 'python':
            return {
                "version": "2.0.0",
                "tasks": [
                    {
                        "label": "Run Project",
                        "type": "shell",
                        "command": "python",
                        "args": [entry],
                        "group": {
                            "kind": "build",
                            "isDefault": True
                        },
                        "presentation": {
                            "reveal": "always",
                            "panel": "new"
                        }
                    },
                    {
                        "label": "Test",
                        "type": "shell",
                        "command": "pytest",
                        "args": ["-v"],
                        "group": "test"
                    },
                    {
                        "label": "Install Dependencies",
                        "type": "shell",
                        "command": "pip",
                        "args": ["install", "-r", "requirements.txt"]
                    }
                ]
            }
        else:
            return {
                "version": "2.0.0",
                "tasks": [
                    {
                        "label": "Run Project",
                        "type": "npm",
                        "script": "start",
                        "group": {
                            "kind": "build",
                            "isDefault": True
                        }
                    }
                ]
            }
    
    def _generate_launch_json(self, structure: Dict) -> Dict:
        """Genera .vscode/launch.json per debugging"""
        language = structure.get('language', 'python')
        entry = structure.get('entry_point', 'main.py')
        
        if language == 'python':
            return {
                "version": "0.2.0",
                "configurations": [
                    {
                        "name": "Python: Current File",
                        "type": "debugpy",
                        "request": "launch",
                        "program": "${file}",
                        "console": "integratedTerminal"
                    },
                    {
                        "name": "Python: Main",
                        "type": "debugpy",
                        "request": "launch",
                        "program": "${workspaceFolder}/" + entry,
                        "console": "integratedTerminal"
                    }
                ]
            }
        else:
            return {
                "version": "0.2.0",
                "configurations": [
                    {
                        "name": "Node: Current File",
                        "type": "node",
                        "request": "launch",
                        "program": "${file}"
                    }
                ]
            }
    
    def _generate_settings_json(self, structure: Dict) -> Dict:
        """Genera .vscode/settings.json con hint per Copilot"""
        settings = {
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.organizeImports": "explicit"
            },
            "github.copilot.enable": {
                "*": True
            },
            "github.copilot.advanced": {
                "debug.overrideEngine": "gpt-4"
            }
        }
        
        if structure.get('language') == 'python':
            settings.update({
                "python.linting.enabled": True,
                "python.linting.pylintEnabled": True,
                "python.formatting.provider": "black"
            })
        
        return settings
    
    def _generate_copilot_instructions(self, proposal: Dict, structure: Dict) -> str:
        """Genera .copilot-instructions.md per dare contesto a Copilot"""
        return f"""# GitHub Copilot Instructions

## Project Context
{proposal.get('feature_description', 'No description')}

## Target Audience
{proposal.get('target_audience', 'General')}

## Architecture
{structure.get('architecture', 'Script')}

## Key Technologies
{', '.join(structure.get('dependencies', []))}

## Implementation Guidelines

### Style
- Follow {structure.get('language', 'Python')} best practices
- Clear variable names
- Comprehensive docstrings
- Error handling

### Patterns
- {structure.get('architecture', 'Simple script')} pattern
- Modular design
- Testable code

### Dependencies
Use these libraries:
{chr(10).join(f"- {dep}" for dep in structure.get('dependencies', []))}

## TODOs to Implement
Check all files for `TODO: [Copilot]` comments and implement them.

**Tips for Copilot:**
1. Read the TODO comment carefully
2. Consider the context (function/class purpose)
3. Use the suggested libraries
4. Add error handling
5. Write clear docstrings

Generated by MOOD Dev Agent on {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    
    def _generate_implementation_guide(self, proposal: Dict, structure: Dict) -> str:
        """Genera IMPLEMENTATION.md con step-by-step instructions"""
        return f"""# Implementation Guide

## 🎯 Goal
{proposal.get('feature_description', 'No description')}

## 📋 Step-by-Step Implementation

### Phase 1: Setup (5 minutes)
1. ✅ Open project in VS Code: `code .`
2. ✅ Install extensions:
   - GitHub Copilot
   - {', '.join(structure.get('vscode_extensions', []))}
3. ✅ Install dependencies:
   ```bash
   {'pip install -r requirements.txt' if structure.get('language') == 'python' else 'npm install'}
   ```

### Phase 2: Implementation (30-60 minutes)
1. **Read `.copilot-instructions.md`** for context
2. **Open `{structure.get('entry_point', 'main.py')}`**
3. **Find all `TODO: [Copilot]` comments**
4. **Use Copilot to implement each TODO**:
   - Position cursor after TODO
   - Press Tab to accept Copilot suggestion
   - Modify if needed
5. **Test each implementation** as you go

### Phase 3: Testing (15 minutes)
1. Run project: `Cmd+Shift+B` or `python {structure.get('entry_point', 'main.py')}`
2. Check output matches expected behavior
3. Fix any errors with Copilot help

### Phase 4: Refinement (Optional)
1. Add error handling
2. Improve docstrings
3. Add logging
4. Create tests

## 🔧 VS Code Shortcuts

| Shortcut | Action |
|----------|--------|
| `Cmd+Shift+B` | Build/Run project |
| `Cmd+I` | Open Copilot Chat |
| `Alt+]` | Next Copilot suggestion |
| `Alt+[` | Previous suggestion |
| `Tab` | Accept suggestion |

## 💡 Copilot Tips
- **Be specific** in TODO comments
- **Add examples** in docstrings
- **Describe edge cases** in comments
- **Break complex tasks** into smaller TODOs

## 🐛 Troubleshooting

### Copilot not working?
- Check you're logged into GitHub
- Enable Copilot in VS Code settings
- Reload window (Cmd+Shift+P → "Reload Window")

### Dependencies error?
- Check you installed all requirements
- Use correct Python/Node version

### Code not running?
- Check entry point file exists
- Verify all TODOs are implemented

## 🚀 Next Steps After Implementation
1. Commit code to git
2. Test with real data
3. Add documentation
4. Consider integration with MOOD system

---

**Generated by MOOD Dev Agent**  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    
    def _format_structure_tree(self, structure: Dict) -> str:
        """Formatta struttura file come tree"""
        lines = ["."]
        for file in structure.get('code_files', []):
            lines.append(f"├── {file['path']}")
        
        if structure.get('language') == 'python':
            lines.append("├── requirements.txt")
        else:
            lines.append("├── package.json")
        
        lines.extend([
            "├── .vscode/",
            "│   ├── tasks.json",
            "│   ├── launch.json",
            "│   └── settings.json",
            "├── .copilot-instructions.md",
            "├── IMPLEMENTATION.md",
            "└── README.md"
        ])
        
        return '\n'.join(lines)
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitizza nome progetto per filesystem"""
        return name.lower().replace(' ', '_').replace('-', '_')[:50]
    
    def _write_file(self, path: Path, content: str):
        """Scrive file con error handling"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Error writing {path}: {e}")
    
    def open_in_vscode(self, project_path: str) -> bool:
        """Apre progetto in VS Code"""
        try:
            subprocess.run(['code', project_path], check=True)
            return True
        except Exception as e:
            print(f"Error opening VS Code: {e}")
            return False


def main():
    """Test del generator"""
    generator = VSCodeProjectGenerator()
    
    # Esempio proposta da MOOD Dev Agent
    mock_proposal = {
        "title": "OSC MIDI Bridge",
        "feature_description": "Bridge between OSC protocol and MIDI for controlling lighting systems from audio software",
        "target_audience": "live performers",
        "technology": "Python with python-osc and mido libraries",
        "priority": "high"
    }
    
    result = generator.generate_project_from_proposal(mock_proposal)
    
    print("=" * 60)
    print("PROJECT GENERATED")
    print("=" * 60)
    print(f"Name: {result['project_name']}")
    print(f"Path: {result['project_path']}")
    print(f"\nFiles: {len(result['generated_files'])}")
    for f in result['generated_files']:
        print(f"  - {f}")
    
    print("\nNext Steps:")
    for step in result['next_steps']:
        print(f"  {step}")
    
    # Opzionale: apri in VS Code
    # generator.open_in_vscode(result['project_path'])


if __name__ == "__main__":
    main()
