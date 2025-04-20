# src/ui/input_handler.py
import pygame
import re

class InputHandler:
    def __init__(self):
        """Initialize the input handler"""
        # Command patterns
        self.command_patterns = {
            "go": re.compile(r"^(?:go|move|walk|run|travel)(?:\s+(?:to|towards|into|in))?\s+(.+)$", re.IGNORECASE),
            "look": re.compile(r"^(?:look|examine|inspect|study|observe)(?:\s+(?:at|on|in))?\s+(.+)$", re.IGNORECASE),
            "take": re.compile(r"^(?:take|grab|pick|get|acquire)(?:\s+(?:up|the))?\s+(.+)$", re.IGNORECASE),
            "use": re.compile(r"^(?:use|utilize|employ|activate)(?:\s+(?:the))?\s+(.+)(?:\s+(?:on|with)\s+(.+))?$", re.IGNORECASE),
            "turn_on": re.compile(r"^(?:turn\s+on|activate|switch\s+on|light(?:\s+up)?)(?:\s+(?:the|a|my))?\s+(.+)$", re.IGNORECASE),
            "talk": re.compile(r"^(?:talk|speak|chat|converse)(?:\s+(?:to|with|at))?\s+(.+)$", re.IGNORECASE),
            "inventory": re.compile(r"^(?:inventory|items|backpack|possessions|i)$", re.IGNORECASE),
            "help": re.compile(r"^(?:help|commands|instructions|guide|hint|hints)$", re.IGNORECASE)
        }
    
    def parse_input(self, user_input):
        """Parse user input into game commands"""
        user_input = user_input.strip()
        
        # Check for special commands first
        if not user_input:
            return {"command": "invalid", "message": "Please enter a command."}
            
        if user_input.lower() in ["quit", "exit", "bye"]:
            return {"command": "quit"}
            
        if user_input.lower() in ["save", "save game"]:
            return {"command": "save"}
        
        if user_input.lower() in ["load", "load game"]:
            return {"command": "load"}
        
        # Check against command patterns
        for cmd_type, pattern in self.command_patterns.items():
            match = pattern.match(user_input)
            if match:
                if cmd_type == "inventory":
                    return {"command": "inventory"}
                elif cmd_type == "help":
                    return {"command": "help"}
                elif cmd_type == "turn_on":
                    # Special case for turn on commands - treat as "use" for game processing
                    return {
                        "command": "use",
                        "object": match.group(1),
                        "action": "turn_on"
                    }
                elif cmd_type == "use" and len(match.groups()) > 1 and match.group(2):
                    return {
                        "command": cmd_type,
                        "object": match.group(1),
                        "target": match.group(2)
                    }
                else:
                    return {
                        "command": cmd_type,
                        "target": match.group(1)
                    }
        
        # If no specific command pattern matched, treat as dialogue or custom action
        return {
            "command": "dialogue",
            "text": user_input
        }