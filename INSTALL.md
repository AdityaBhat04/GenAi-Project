# Simple Text Adventure Game - Installation Guide

## Requirements
- Python 3.6 or higher
- A command prompt/terminal
- Git (optional, for cloning the repository)

## Installation Steps

### Step 1: Download the Project
Either download the ZIP file from GitHub or use git to clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/text_adventure.git
cd text_adventure
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install torch transformers pygame requests
```

If you have a slower computer or want a lighter installation:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers pygame requests
```

### Step 4: Run the Game
```bash
python start_game.py
```

Or with your own OpenRouter API key:
```bash
python start_game.py YOUR_API_KEY_HERE
```

## Troubleshooting

### Missing Background Images
If the game fails to download background images automatically, you can:
1. Create a `backgrounds` folder in the project directory
2. Add your own JPG images named: forest.jpg, cave.jpg, village.jpg, mountain.jpg, castle.jpg, dungeon.jpg, magical.jpg

### Windows-Specific Issues
If you encounter issues with PyTorch on Windows:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Memory Errors
If you encounter memory errors when loading models:
```bash
pip uninstall torch transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

## One-Line Installation (Windows)
Copy and paste this entire command into your Command Prompt:
```
git clone https://github.com/YOUR_USERNAME/text_adventure.git && cd text_adventure && python -m pip install torch transformers pygame requests && python start_game.py
```

## One-Line Installation (macOS/Linux)
Copy and paste this entire command into your Terminal:
```
git clone https://github.com/YOUR_USERNAME/text_adventure.git && cd text_adventure && python3 -m pip install torch transformers pygame requests && python3 start_game.py
``` 