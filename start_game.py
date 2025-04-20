#!/usr/bin/env python
# start_game.py - Simple entry point for the text adventure game

import os
import sys
import argparse

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the game directly from src
from src.standalone_game import main

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Text Adventure Game with OpenRouter API")
    parser.add_argument("api_key", nargs="?", 
                      help="OpenRouter API key (optional)",
                      default="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")
    args = parser.parse_args()
    
    # Run the game
    main(args.api_key) 