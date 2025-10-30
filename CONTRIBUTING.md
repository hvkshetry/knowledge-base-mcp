# Contributing to Semantic Search MCP Server

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Ways to Contribute

- **Bug Reports**: Report issues you encounter
- **Feature Requests**: Suggest new features or improvements
- **Documentation**: Improve or expand documentation
- **Code**: Fix bugs or implement new features
- **Examples**: Share example use cases or scripts
- **Testing**: Test on different platforms and configurations

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/knowledge-base-mcp.git
   cd knowledge-base-mcp
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up development environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Development Guidelines

### Code Style

- **Python**: Follow PEP 8 style guide
- **Line length**: 100 characters max
- **Imports**: Group by standard library, third-party, local
- **Type hints**: Use where helpful for clarity
- **Docstrings**: Use for all public functions and classes

### Testing

Before submitting, ensure:

1. **Services are running**:
   ```bash
   docker-compose ps
   ollama list
   ```

2. **Test ingestion** with sample documents:
   ```bash
   python ingest.py --root test_docs --collection test_kb
   ```

3. **Test search** in all modes:
   ```bash
   python validate_search.py --query "test" --collection test_kb --mode semantic
   python validate_search.py --query "test" --collection test_kb --mode rerank
   python validate_search.py --query "test" --collection test_kb --mode hybrid
   ```

4. **Test MCP server**:
   ```bash
   python server.py stdio
   # Send test MCP requests
   ```

### Commit Messages

Use clear, descriptive commit messages:

```
Add support for custom reranker models

- Add RERANKER_MODEL environment variable
- Update docker-compose.yml to use configurable model
- Document in README.md and .env.example
```

Format:
- **First line**: Brief summary (50 chars max)
- **Blank line**
- **Body**: Detailed explanation (wrap at 72 chars)
- **List changes** with bullet points

### Pull Request Process

1. **Update documentation** for any new features
2. **Add examples** if introducing new functionality
3. **Test thoroughly** on your local setup
4. **Update CHANGELOG.md** with your changes
5. **Create pull request** with clear description:
   - What problem does it solve?
   - How did you test it?
   - Any breaking changes?

### Pull Request Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- List specific changes
- Include file names

## Testing
How did you test these changes?

## Checklist
- [ ] Code follows project style
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] CHANGELOG.md updated
```

## Areas for Contribution

### High Priority

- **Testing framework**: Add unit and integration tests
- **Performance benchmarks**: Systematic performance testing
- **Alternative models**: Support for different embedding/reranking models
- **GPU support**: Add GPU acceleration options
- **Web UI**: Simple web interface for testing

### Medium Priority

- **Additional extractors**: Support more document formats
- **Query expansion**: Automatic query enhancement
- **Result caching**: Cache search results for speed
- **Monitoring/metrics**: Add observability tools
- **Multi-language support**: Non-English document support

### Documentation Improvements

- **Video tutorials**: Setup and usage walkthroughs
- **Architecture diagrams**: Improve visual documentation
- **Use case studies**: Real-world examples
- **Troubleshooting guides**: Expand FAQ
- **API documentation**: Detailed API reference

## Bug Reports

When reporting bugs, include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: Exact steps to trigger the bug
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**:
   - OS and version
   - Python version
   - Docker version
   - Ollama version
   - Hardware specs (RAM, CPU)
6. **Logs**: Relevant error messages or logs
7. **Configuration**: MCP config and environment variables (sanitize sensitive info)

### Bug Report Template

```markdown
**Description**
Brief description of the bug

**Steps to Reproduce**
1. Set up environment with...
2. Run command...
3. Observe error...

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: Ubuntu 22.04
- Python: 3.11.5
- Docker: 24.0.6
- Ollama: 0.1.24
- RAM: 16GB
- CPU: Intel i7-9700K

**Logs**
```
Paste relevant logs here
```

**Configuration**
```json
{
  "env": {
    "OLLAMA_MODEL": "snowflake-arctic-embed:xs",
    ...
  }
}
```
```

## Feature Requests

For feature requests, describe:

1. **Use case**: What problem does it solve?
2. **Proposed solution**: How would it work?
3. **Alternatives**: Other approaches considered?
4. **Impact**: Who would benefit from this?
5. **Implementation**: Ideas on how to implement?

## Code Review Process

All submissions require review. We look for:

- **Correctness**: Does it work as intended?
- **Performance**: Any performance implications?
- **Compatibility**: Works across platforms?
- **Documentation**: Is it documented?
- **Style**: Follows project conventions?
- **Tests**: Includes tests where appropriate?

## Community Guidelines

- **Be respectful**: Treat everyone with respect
- **Be constructive**: Provide helpful feedback
- **Be patient**: Maintainers volunteer their time
- **Be collaborative**: Work together toward solutions
- **Be inclusive**: Welcome all contributors

## Questions?

- **GitHub Issues**: For bugs and features
- **GitHub Discussions**: For questions and ideas
- **Documentation**: Check existing docs first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to making this project better!
