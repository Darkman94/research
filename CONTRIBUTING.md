# Contributing to Research Projects

This guide explains how to add new research projects to this repository.

## Creating a New Research Project

### 1. Set Up Project Directory

Create a new directory under `projects/` with a descriptive name:

```bash
mkdir -p projects/my-research-topic
```

### 2. Use the Template

Copy the research project template:

```bash
cp -r templates/research-project/* projects/my-research-topic/
```

### 3. Update Project README

Edit `projects/my-research-topic/README.md`:

- Replace `[Project Name]` with your project title
- Define your research question clearly
- Add the start date
- Set initial status

### 4. Develop Your Research

- Write scripts to investigate your research question
- Document your methodology and approach
- Save results in JSON, CSV, or other appropriate formats
- Update the analysis.md with findings

### 5. Document Findings

As you complete your research:

- Update the README with findings and conclusions
- Include data visualizations or charts
- Link to relevant external resources
- Mark the status as "Completed" when done

## Best Practices

### Research Questions

Good research questions are:
- **Specific**: Clearly defined scope
- **Measurable**: Can be answered with data/code
- **Interesting**: Worth exploring and documenting
- **Actionable**: Results lead to insights or decisions

Examples:
- "How does Python's asyncio compare to threading for I/O-bound tasks?"
- "What are the performance characteristics of different JSON parsing libraries?"
- "Can we detect patterns in API error responses across different services?"

### Code Organization

- Keep scripts focused and well-documented
- Use meaningful variable and function names
- Include docstrings and comments
- Write tests for complex logic

### Documentation

- Document assumptions and limitations
- Explain why you chose specific approaches
- Include command-line examples
- Link to related research or documentation

### Data Management

- Store raw data separately from processed results
- Use version control for code, not large data files
- Document data sources and collection methods
- Consider using `.gitignore` for large datasets

## Workflow

1. **Start**: Create project directory and initial README
2. **Research**: Write code, run experiments, collect data
3. **Analyze**: Process results and identify patterns
4. **Document**: Write up findings in README and analysis.md
5. **Review**: Check that everything is clear and reproducible
6. **Commit**: Push your completed research

## Automation

The repository automatically updates the main README with project listings when you:
- Add a new project directory under `projects/`
- Update a project README
- Push to the main branch

The GitHub Actions workflow extracts metadata from your project README and updates the main README.

## Questions?

If you have questions or need help with your research project, open an issue or reach out to the maintainers.
