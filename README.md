# Code Research Repository

A collection of exploratory code research projects using AI coding agents, inspired by [Simon Willison's async code research approach](https://simonwillison.net/2025/Nov/6/async-code-research/).

## Purpose

This repository serves as a workspace for conducting focused code research experiments. Each subdirectory contains a self-contained research project with its own scripts, data, and documentation.

## Structure

```
research/
├── projects/           # Individual research projects
│   └── example/       # Example project structure
│       ├── README.md  # Project documentation
│       ├── *.py       # Python scripts
│       ├── *.json     # Data and results
│       └── *.md       # Analysis and findings
├── .github/
│   └── workflows/     # Automation workflows
└── templates/         # Project templates
```

## Research Projects

<!-- PROJECT_LIST_START -->
*No projects yet. Each new research project will be automatically listed here.*
<!-- PROJECT_LIST_END -->

## Getting Started

### Creating a New Research Project

1. Create a new directory under `projects/`:
   ```bash
   mkdir -p projects/your-project-name
   ```

2. Copy the project template:
   ```bash
   cp -r templates/research-project/* projects/your-project-name/
   ```

3. Update the project README with your research question and goals

4. Develop your research scripts and document findings

### Tools and Dependencies

Common tools used in research projects:
- Python 3.8+
- pytest for testing
- Standard scientific libraries (numpy, pandas, etc.)
- LLM and AI coding tools

Install dependencies:
```bash
pip install -r requirements.txt
```

## Workflow

1. **Define Research Question**: Clearly articulate what you want to explore
2. **Develop Scripts**: Write focused code to investigate the question
3. **Document Findings**: Record results, insights, and conclusions
4. **Share Results**: Commit and push to make research accessible

## Contributing

This is a personal research repository, but the approach and templates can be adapted for your own research projects.

## License

MIT License - See LICENSE file for details
