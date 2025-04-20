# Text Adventure Game

A text-based adventure game powered by OpenRouter's GPT-4o integration!

## Quick Start

Simply run:

```
python start_game.py
```

This will start the game with the default OpenRouter API key.

## GitHub Usage

When pushing to GitHub, the virtual environment (venv folder) will be automatically excluded by the .gitignore file. This is standard practice as virtual environments are very large and specific to each user's system.

## Running the Game

There are two ways to run the game:

### Option 1: Using the starter script (Recommended)

```
python start_game.py [YOUR_API_KEY]
```

This starts the game from the root directory with an optional API key.

### Option 2: Running the standalone game directly

```
python src/standalone_game.py [YOUR_API_KEY]
```

This runs the game module directly if you prefer.

## Project Structure

The project has a simple structure:

```
text_adventure/
│
├── start_game.py        # Simple entry point
│
├── src/                 # Source code directory
│   └── standalone_game.py  # Self-contained game implementation
│
├── .gitignore           # Excludes virtual env and other large files
│
└── backgrounds/         # Background images for the game
```

## Game Controls

- Type commands in the input box at the bottom of the screen
- Press Enter to submit your command
- Type 'exit' to quit the game

## Features

- Dynamic story generation using OpenAI's GPT-4o through OpenRouter
- Fallback to local GPT-2 model if API connection fails
- Scene-aware descriptions for different locations
- Natural language input for game commands
- Background images that change based on your location
- Simplified language with explanations for complex words

## Installation

See the INSTALL.md file for detailed installation instructions.

## API Key

The game uses the OpenRouter API to access GPT-4o. While a default API key is included, you can use your own by:

1. Creating an account at [OpenRouter](https://openrouter.ai/)
2. Getting your own API key
3. Running the game with your key as a command line argument:
   ```
   python start_game.py YOUR_API_KEY
   ```
