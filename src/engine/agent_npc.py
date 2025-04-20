# src/engine/agent_npc.py
import random
import json
import os

class NPCAgent:
    def __init__(self, name, personality, backstory, goals=None):
        """Initialize an NPC agent with personality traits and backstory"""
        self.name = name
        self.personality = personality  # Dict with traits like 'friendly', 'trustworthy', etc.
        self.backstory = backstory
        self.goals = goals or []
        self.memory = []  # Track interactions
        self.knowledge = {}  # What the NPC knows about the world and player
        
        # Track emotional state
        self.emotion = {
            "happiness": 50,  # 0-100 scale
            "trust": 50,
            "fear": 30,
            "anger": 20
        }
    
    def remember(self, event, importance=1):
        """Store an interaction or event in memory"""
        self.memory.append({
            "event": event,
            "importance": importance,
            "recency": 0  # Will increment as time passes
        })
        
    def update_knowledge(self, key, value):
        """Update NPC's knowledge about the world or player"""
        self.knowledge[key] = value
    
    def adjust_emotion(self, emotion_key, amount):
        """Adjust emotional state based on interactions"""
        if emotion_key in self.emotion:
            self.emotion[emotion_key] = max(0, min(100, self.emotion[emotion_key] + amount))
    
    def get_response(self, player_input, story_context, sentiment_analyzer=None):
        """Generate a contextual response based on NPC personality and game state"""
        # Analyze sentiment of player input if analyzer is available
        sentiment = None
        if sentiment_analyzer:
            sentiment = sentiment_analyzer.analyze_sentiment(player_input)
        
        # Update emotional state based on player input sentiment
        if sentiment:
            if sentiment["sentiment"] == "positive":
                self.adjust_emotion("happiness", 5)
                self.adjust_emotion("trust", 3)
                self.adjust_emotion("fear", -2)
            elif sentiment["sentiment"] == "negative":
                self.adjust_emotion("fear", 3)
                self.adjust_emotion("trust", -2)
                if sentiment["scores"]["negative"] > 0.7:  # Strong negative
                    self.adjust_emotion("anger", 5)
        
        # Generate response based on personality, emotions and context
        # In a full implementation, this might use a language model or template system
        
        # Simple rule-based response generation for demonstration
        responses = []
        
        # Personality-based responses
        if self.personality.get("friendly", 0) > 70:
            responses.append(f"*smiles warmly* {self.get_friendly_response(player_input)}")
        elif self.personality.get("mysterious", 0) > 70:
            responses.append(f"*looks thoughtful* {self.get_mysterious_response(player_input)}")
        
        # Emotion-based responses
        if self.emotion["happiness"] > 70:
            responses.append(f"*cheerfully* {self.get_happy_response(player_input)}")
        elif self.emotion["fear"] > 70:
            responses.append(f"*nervously* {self.get_fearful_response(player_input)}")
        elif self.emotion["anger"] > 70:
            responses.append(f"*angrily* {self.get_angry_response(player_input)}")
        
        # Add a memory or knowledge-based response if available
        if self.memory and random.random() < 0.3:
            # Reference a past interaction sometimes
            relevant_memory = random.choice(self.memory)
            responses.append(f"I remember {relevant_memory['event']}... ")
        
        # Choose and return a response
        if responses:
            return random.choice(responses)
        else:
            return self.get_neutral_response(player_input)
    
    # Response generators based on emotion or personality
    def get_friendly_response(self, player_input):
        friendly_responses = [
            f"I'm so glad we're talking about this, friend!",
            f"Your questions always brighten my day.",
            f"I'd be happy to help you with that."
        ]
        return random.choice(friendly_responses)
    
    def get_mysterious_response(self, player_input):
        mysterious_responses = [
            f"There's more to this than meets the eye...",
            f"Some secrets are better left untold.",
            f"I've seen things you wouldn't believe."
        ]
        return random.choice(mysterious_responses)
    
    def get_happy_response(self, player_input):
        happy_responses = [
            f"Wonderful! This makes me so happy!",
            f"What a delightful conversation!",
            f"Things are really looking up!"
        ]
        return random.choice(happy_responses)
    
    def get_fearful_response(self, player_input):
        fearful_responses = [
            f"I'm... I'm not sure this is safe to discuss.",
            f"We should be careful. They might be listening.",
            f"*glances around nervously* Keep your voice down..."
        ]
        return random.choice(fearful_responses)
    
    def get_angry_response(self, player_input):
        angry_responses = [
            f"I've had just about enough of this!",
            f"Do you think this is some kind of game?",
            f"You're testing my patience!"
        ]
        return random.choice(angry_responses)
    
    def get_neutral_response(self, player_input):
        neutral_responses = [
            f"I see.",
            f"That's interesting.",
            f"Tell me more about that."
        ]
        return random.choice(neutral_responses)
    
    def to_dict(self):
        """Convert NPC state to dictionary for saving"""
        return {
            "name": self.name,
            "personality": self.personality,
            "backstory": self.backstory,
            "goals": self.goals,
            "memory": self.memory,
            "knowledge": self.knowledge,
            "emotion": self.emotion
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create an NPC from saved data"""
        npc = cls(data["name"], data["personality"], data["backstory"], data["goals"])
        npc.memory = data["memory"]
        npc.knowledge = data["knowledge"]
        npc.emotion = data["emotion"]
        return npc
    
    @classmethod
    def load_npcs_from_file(cls, filename="data/npcs.json"):
        """Load all NPCs from a JSON file"""
        if not os.path.exists(filename):
            return []
            
        with open(filename, "r") as f:
            npc_data = json.load(f)
            
        return [cls.from_dict(data) for data in npc_data]