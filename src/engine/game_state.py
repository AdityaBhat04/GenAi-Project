# src/engine/game_state.py
import json
import os
import time

class GameState:
    def __init__(self, player_name="Player"):
        """Initialize the game state"""
        self.player_name = player_name
        self.current_location = "start"
        self.inventory = []
        self.story_history = []  # Track the story segments so far
        self.npc_states = {}  # Track NPC states
        self.game_variables = {}  # Track game flags and variables
        self.start_time = time.time()
        self.play_time = 0
    
    def add_to_story(self, text):
        """Add a segment to the story history"""
        self.story_history.append({
            "text": text,
            "timestamp": time.time()
        })
    
    def get_full_story(self):
        """Get the full story so far"""
        return "\n\n".join([segment["text"] for segment in self.story_history])
    
    def get_recent_story(self, segments=3):
        """Get the most recent story segments"""
        recent_segments = self.story_history[-segments:] if self.story_history else []
        return "\n\n".join([segment["text"] for segment in recent_segments])
    
    def add_to_inventory(self, item):
        """Add an item to player inventory"""
        self.inventory.append(item)
    
    def remove_from_inventory(self, item):
        """Remove an item from player inventory"""
        if item in self.inventory:
            self.inventory.remove(item)
            return True
        return False
    
    def change_location(self, new_location):
        """Change the player's current location"""
        self.current_location = new_location
    
    def set_variable(self, key, value):
        """Set a game variable"""
        self.game_variables[key] = value
    
    def get_variable(self, key, default=None):
        """Get a game variable"""
        return self.game_variables.get(key, default)
    
    def update_play_time(self):
        """Update the total play time"""
        self.play_time = time.time() - self.start_time
    
    def register_npc(self, npc):
        """Register an NPC with the game state"""
        self.npc_states[npc.name] = npc
    
    def save_game(self, filename="savegame.json"):
        """Save the current game state"""
        self.update_play_time()
        
        # Convert NPCs to dict representation
        npc_data = {name: npc.to_dict() for name, npc in self.npc_states.items()}
        
        save_data = {
            "player_name": self.player_name,
            "current_location": self.current_location,
            "inventory": self.inventory,
            "story_history": self.story_history,
            "game_variables": self.game_variables,
            "play_time": self.play_time,
            "npcs": npc_data,
            "save_timestamp": time.time()
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(save_data, f, indent=2)
        
        return True
    
    @classmethod
    def load_game(cls, filename="savegame.json", npc_class=None):
        """Load a saved game state"""
        if not os.path.exists(filename):
            return None
            
        with open(filename, "r") as f:
            save_data = json.load(f)
        
        # Create a new game state
        game_state = cls(save_data["player_name"])
        game_state.current_location = save_data["current_location"]
        game_state.inventory = save_data["inventory"]
        game_state.story_history = save_data["story_history"]
        game_state.game_variables = save_data["game_variables"]
        game_state.play_time = save_data["play_time"]
        game_state.start_time = time.time() - game_state.play_time
        
        # Load NPCs if npc_class is provided
        if npc_class and "npcs" in save_data:
            for name, npc_data in save_data["npcs"].items():
                game_state.npc_states[name] = npc_class.from_dict(npc_data)
        
        return game_state