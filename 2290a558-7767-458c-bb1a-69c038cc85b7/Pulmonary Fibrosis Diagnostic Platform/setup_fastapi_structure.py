import os

# Define project structure
directories = [
    'app',
    'templates',
    'static',
    'static/css',
    'static/js',
    'services',
    'data'
]

# Create all directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Created directory: {directory}/")

# Create app/main.py
main_py_content = '''"""
FastAPI Main Application Entry Point
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="FastAPI Project", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
'''

with open('app/main.py', 'w') as f:
    f.write(main_py_content)
print("✓ Created app/main.py")

# Create app/__init__.py
with open('app/__init__.py', 'w') as f:
    f.write('"""FastAPI Application Package"""\n')
print("✓ Created app/__init__.py")

# Create services/__init__.py
with open('services/__init__.py', 'w') as f:
    f.write('"""Services Package for Business Logic"""\n')
print("✓ Created services/__init__.py")

# Create requirements.txt
requirements_content = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
jinja2==3.1.2
python-multipart==0.0.6
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0
'''

with open('requirements.txt', 'w') as f:
    f.write(requirements_content)
print("✓ Created requirements.txt")

# Create .gitignore
gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment variables and credentials
.env
.env.local
*.env
credentials.json
secrets.json
config.ini
*.key
*.pem

# Data files
data/*
!data/.gitkeep
*.csv
*.xlsx
*.xls
*.db
*.sqlite
*.sqlite3

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/

# Logs
*.log
logs/
'''

with open('.gitignore', 'w') as f:
    f.write(gitignore_content)
print("✓ Created .gitignore")

# Create data/.gitkeep to preserve directory in git
with open('data/.gitkeep', 'w') as f:
    f.write('')
print("✓ Created data/.gitkeep")

# Create README.md
readme_content = '''# FastAPI Project

A scalable FastAPI project with a clean structure for building modern web applications.

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   └── main.py          # Main FastAPI application
├── templates/           # Jinja2 HTML templates
├── static/             # Static files (CSS, JS, images)
│   ├── css/
│   └── js/
├── services/           # Business logic and services
│   └── __init__.py
├── data/              # Data files (gitignored)
├── requirements.txt   # Python dependencies
├── .gitignore        # Git ignore rules
└── README.md         # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn app.main:app --reload
```

4. Open your browser to `http://localhost:8000`

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Features

- ✓ FastAPI framework with async support
- ✓ Static file serving
- ✓ Jinja2 templating
- ✓ Clean project structure
- ✓ Proper .gitignore for data and credentials
- ✓ Health check endpoint

## Development

Add your routes in `app/main.py` or create separate router modules.
Add business logic in the `services/` directory.
Store data files in `data/` (automatically gitignored).

## Security

- Never commit `.env` files or credentials
- Keep sensitive data in the `data/` directory
- Use environment variables for configuration
'''

with open('README.md', 'w') as f:
    f.write(readme_content)
print("✓ Created README.md")

# Create a sample HTML template
template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FastAPI Project</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <h1>Welcome to FastAPI!</h1>
    <p>Your project structure is ready.</p>
    <script src="/static/js/main.js"></script>
</body>
</html>
'''

with open('templates/index.html', 'w') as f:
    f.write(template_content)
print("✓ Created templates/index.html")

# Create sample CSS file
css_content = '''/* FastAPI Project Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
}

h1 {
    color: #009688;
}
'''

with open('static/css/style.css', 'w') as f:
    f.write(css_content)
print("✓ Created static/css/style.css")

# Create sample JS file
js_content = '''// FastAPI Project JavaScript
console.log("FastAPI project initialized!");
'''

with open('static/js/main.js', 'w') as f:
    f.write(js_content)
print("✓ Created static/js/main.js")

# Create .env.example
env_example_content = '''# Environment Configuration Example
# Copy this file to .env and fill in your actual values

DEBUG=True
API_KEY=your_api_key_here
DATABASE_URL=sqlite:///./data/app.db
'''

with open('.env.example', 'w') as f:
    f.write(env_example_content)
print("✓ Created .env.example")

print("\n" + "="*60)
print("🎉 FastAPI Project Structure Created Successfully!")
print("="*60)

# Verify all critical files exist
critical_files = [
    'app/main.py',
    'app/__init__.py',
    'services/__init__.py',
    'templates/index.html',
    'static/css/style.css',
    'static/js/main.js',
    'requirements.txt',
    'README.md',
    '.gitignore',
    'data/.gitkeep',
    '.env.example'
]

print("\n📋 Verification:")
all_exist = True
for file_path in critical_files:
    exists = os.path.exists(file_path)
    status = "✓" if exists else "✗"
    print(f"{status} {file_path}")
    if not exists:
        all_exist = False

# Check directories
print("\n📁 Directories:")
for directory in directories:
    exists = os.path.isdir(directory)
    status = "✓" if exists else "✗"
    print(f"{status} {directory}/")

if all_exist:
    print("\n✅ All files and directories created successfully!")
else:
    print("\n⚠️ Some files may be missing.")

project_structure = {
    'directories': directories,
    'files': critical_files,
    'status': 'complete' if all_exist else 'incomplete'
}