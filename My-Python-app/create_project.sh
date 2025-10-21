#!/bin/bash
# Script to create a Python project scaffold with CI, Docker, and Makefile automation

PROJECT_NAME="my-python-app"

echo "🚀 Creating project structure for $PROJECT_NAME..."

# Create directories
mkdir -p $PROJECT_NAME/{app,tests,.github/workflows}

# ========================
# Create main application
# ========================
cat > $PROJECT_NAME/app/main.py << 'EOF'
def add(a, b):
    """Simple add function"""
    return a + b

if __name__ == "__main__":
    print("Sum:", add(2, 3))
EOF

# ========================
# Create test file
# ========================
cat > $PROJECT_NAME/tests/test_main.py << 'EOF'
from app.main import add

def test_add():
    assert add(2, 3) == 5
EOF

# ========================
# Create requirements.txt
# ========================
cat > $PROJECT_NAME/requirements.txt << 'EOF'
pytest
flake8
black
EOF

# ========================
# Create Makefile
# ========================
cat > $PROJECT_NAME/Makefile << 'EOF'
.PHONY: install lint format test clean

install:
	pip install -r requirements.txt

lint:
	flake8 app tests

format:
	black app tests

test:
	pytest -v --disable-warnings

clean:
	rm -rf __pycache__ .pytest_cache
EOF

# ========================
# Create Dockerfile
# ========================
cat > $PROJECT_NAME/Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app/main.py"]
EOF

# ========================
# Create pyproject.toml
# ========================
cat > $PROJECT_NAME/pyproject.toml << 'EOF'
[tool.black]
line-length = 88
target-version = ['py310']

[tool.flake8]
max-line-length = 88
extend-ignore = E203
EOF

# ========================
# Create README.md
# ========================
cat > $PROJECT_NAME/README.md << 'EOF'
# My Python App

A simple Python project demonstrating:
- Makefile automation
- Linting and formatting
- Testing with pytest
- Docker containerization
- GitHub Actions CI/CD

## 🚀 Quick Start

```bash
make install
make lint
make format
make test
