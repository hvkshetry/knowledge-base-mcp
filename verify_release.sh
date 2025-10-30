#!/bin/bash
#
# Pre-Release Verification Script
#
# This script checks that the repository is ready for public release
# by verifying that sensitive files are properly gitignored.

set -e

echo "=== Knowledge Base MCP - Public Release Verification ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

WARNINGS=0
ERRORS=0

# Check if in git repository
if [ ! -d .git ]; then
    echo -e "${YELLOW}⚠ Warning: Not a git repository${NC}"
    echo "  Run: git init"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✓ Git repository initialized${NC}"
fi

# Check if .gitignore exists
if [ ! -f .gitignore ]; then
    echo -e "${RED}✗ Error: .gitignore file missing${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓ .gitignore exists${NC}"
fi

# Check for sensitive directories
echo ""
echo "Checking for sensitive directories in git staging..."

SENSITIVE_DIRS=("*_kb" "data" ".claude" ".venv")
for dir_pattern in "${SENSITIVE_DIRS[@]}"; do
    if git ls-files --error-unmatch "$dir_pattern" 2>/dev/null | grep -q .; then
        echo -e "${RED}✗ Error: $dir_pattern is tracked by git${NC}"
        echo "  These should be gitignored!"
        ERRORS=$((ERRORS + 1))
    else
        echo -e "${GREEN}✓ $dir_pattern not tracked${NC}"
    fi
done

# Check for sensitive files
echo ""
echo "Checking for sensitive files in git staging..."

SENSITIVE_FILES=(".mcp.json" ".env" "*.db")
for file_pattern in "${SENSITIVE_FILES[@]}"; do
    if git ls-files --error-unmatch "$file_pattern" 2>/dev/null | grep -q .; then
        echo -e "${RED}✗ Error: $file_pattern is tracked by git${NC}"
        echo "  These should be gitignored!"
        ERRORS=$((ERRORS + 1))
    else
        echo -e "${GREEN}✓ $file_pattern not tracked${NC}"
    fi
done

# Check for hardcoded paths
echo ""
echo "Checking for hardcoded user paths in tracked files..."

if [ -d .git ]; then
    # Search for common path patterns
    PATTERNS=("/home/hvksh" "/mnt/c/Users/hvksh" "Circle H2O")

    for pattern in "${PATTERNS[@]}"; do
        matches=$(git grep -l "$pattern" -- ':!*.sh' ':!PUBLIC_RELEASE_SUMMARY.md' || true)
        if [ -n "$matches" ]; then
            echo -e "${YELLOW}⚠ Warning: Found '$pattern' in:${NC}"
            echo "$matches" | sed 's/^/  /'
            WARNINGS=$((WARNINGS + 1))
        fi
    done

    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✓ No hardcoded paths found${NC}"
    fi
fi

# Check for required template files
echo ""
echo "Checking for required template files..."

REQUIRED_TEMPLATES=(
    ".mcp.json.example"
    "claude_desktop_config.json.example"
    ".env.example"
    ".codex/config.toml.example"
)

for template in "${REQUIRED_TEMPLATES[@]}"; do
    if [ ! -f "$template" ]; then
        echo -e "${RED}✗ Error: $template missing${NC}"
        ERRORS=$((ERRORS + 1))
    else
        echo -e "${GREEN}✓ $template exists${NC}"
    fi
done

# Check for required documentation
echo ""
echo "Checking for required documentation..."

REQUIRED_DOCS=(
    "README.md"
    "INSTALLATION.md"
    "USAGE.md"
    "ARCHITECTURE.md"
    "FAQ.md"
    "CONTRIBUTING.md"
    "LICENSE"
    "CHANGELOG.md"
)

for doc in "${REQUIRED_DOCS[@]}"; do
    if [ ! -f "$doc" ]; then
        echo -e "${RED}✗ Error: $doc missing${NC}"
        ERRORS=$((ERRORS + 1))
    else
        echo -e "${GREEN}✓ $doc exists${NC}"
    fi
done

# Check for example scripts
echo ""
echo "Checking for example scripts..."

REQUIRED_EXAMPLES=(
    "examples/simple_ingest.sh"
    "examples/incremental_ingest.sh"
    "examples/multi_collection_setup.sh"
    "examples/README.md"
)

for example in "${REQUIRED_EXAMPLES[@]}"; do
    if [ ! -f "$example" ]; then
        echo -e "${RED}✗ Error: $example missing${NC}"
        ERRORS=$((ERRORS + 1))
    else
        echo -e "${GREEN}✓ $example exists${NC}"
    fi
done

# Summary
echo ""
echo "=== Verification Summary ==="
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Your repository is ready for public release."
    echo ""
    echo "Next steps:"
    echo "  1. Review staged files: git status"
    echo "  2. Create commit: git commit -m 'Initial public release'"
    echo "  3. Create GitHub repository"
    echo "  4. Push: git push -u origin main"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Verification completed with $WARNINGS warning(s)${NC}"
    echo ""
    echo "Review warnings above. The repository may be ready,"
    echo "but please double-check the flagged items."
    echo ""
    exit 0
else
    echo -e "${RED}✗ Verification failed with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before proceeding with release."
    echo ""
    exit 1
fi
