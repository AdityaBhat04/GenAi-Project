#!/usr/bin/env python
# src/standalone_game.py - Standalone text adventure game with OpenRouter

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pygame
import time
import os
import random
import requests
import json
import sys

class GameDisplay:
    def __init__(self, width=1024, height=768):
        """Initialize the Pygame display."""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Text Adventure")
        
        # Font setup
        self.font = pygame.font.Font(None, 32)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        
        # Text input setup
        self.input_text = ""
        self.input_rect = pygame.Rect(10, height - 40, width - 20, 30)
        
        # Story text setup
        self.story_surface = None
        self.story_rect = pygame.Rect(10, 10, width - 20, height - 60)
        
        # Create backgrounds directory if it doesn't exist
        if not os.path.exists('backgrounds'):
            os.makedirs('backgrounds')
        
        # Create or download default backgrounds
        self.download_default_backgrounds()
        
        # Load backgrounds
        self.backgrounds = self.load_backgrounds()
        self.current_background = 'forest'
        
    def download_default_backgrounds(self):
        """Download or create default background images."""
        default_backgrounds = {
            'forest': 'https://raw.githubusercontent.com/apekshik/genai_project/main/backgrounds/forest.jpg',
            'cave': 'https://raw.githubusercontent.com/apekshik/genai_project/main/backgrounds/cave.jpg',
            'village': 'https://raw.githubusercontent.com/apekshik/genai_project/main/backgrounds/village.jpg',
            'mountain': 'https://raw.githubusercontent.com/apekshik/genai_project/main/backgrounds/mountain.jpg',
            'castle': 'https://raw.githubusercontent.com/apekshik/genai_project/main/backgrounds/castle.jpg',
            'dungeon': 'https://raw.githubusercontent.com/apekshik/genai_project/main/backgrounds/dungeon.jpg',
            'magical': 'https://raw.githubusercontent.com/apekshik/genai_project/main/backgrounds/magical.jpg'
        }
        
        fallback_colors = {
            'forest': (34, 139, 34),      # Forest Green
            'cave': (64, 64, 64),         # Dark Gray
            'village': (210, 180, 140),   # Tan
            'mountain': (128, 128, 128),  # Gray
            'castle': (169, 169, 169),    # Light Gray
            'dungeon': (47, 79, 79),      # Dark Slate Gray
            'magical': (147, 112, 219)    # Purple
        }
        
        print("Setting up background images...")
        for name, url in default_backgrounds.items():
            filename = f'backgrounds/{name}.jpg'
            if not os.path.exists(filename):
                try:
                    print(f"Downloading {name} background...")
                    import urllib.request
                    urllib.request.urlretrieve(url, filename)
                except:
                    print(f"Creating fallback {name} background...")
                    surface = pygame.Surface((self.width, self.height))
                    surface.fill(fallback_colors[name])
                    pygame.image.save(surface, filename)
        print("Background setup complete!")
    
    def load_backgrounds(self):
        """Load all background images."""
        backgrounds = {}
        for filename in os.listdir('backgrounds'):
            if filename.endswith(('.jpg', '.png')):
                name = filename.split('.')[0]
                try:
                    image = pygame.image.load(os.path.join('backgrounds', filename))
                    image = pygame.transform.scale(image, (self.width, self.height))
                    backgrounds[name] = image
                except pygame.error:
                    print(f"Error loading {filename}, creating solid color fallback...")
                    surface = pygame.Surface((self.width, self.height))
                    surface.fill((100, 100, 100))  # Default gray
                    backgrounds[name] = surface
        return backgrounds
        
    def update_background(self, scene_type: str):
        """Update the current background."""
        if scene_type in self.backgrounds:
            self.current_background = scene_type
        
    def render_text(self, text: str):
        """Render the story text with improved wrapping and scrolling."""
        self.story_surface = pygame.Surface(
            (self.story_rect.width, self.story_rect.height), pygame.SRCALPHA)
        self.story_surface.fill((0, 0, 0, 180))
        
        # Split by words but preserve newlines
        text_lines = text.split('\n')
        formatted_lines = []
        
        for text_line in text_lines:
            # Handle empty lines
            if not text_line.strip():
                formatted_lines.append('')
                continue
                
            words = text_line.split()
            current_line = []
            current_width = 0
            max_width = self.story_rect.width - 40  # More padding for readability
            
            for word in words:
                word_surface = self.font.render(word, True, self.WHITE)
                word_width = word_surface.get_width()
                
                # Add space width except for first word
                if current_line:
                    space_width = self.font.render(' ', True, self.WHITE).get_width()
                    test_width = current_width + space_width + word_width
                else:
                    test_width = current_width + word_width
                
                if test_width <= max_width:
                    current_line.append(word)
                    current_width = test_width
                else:
                    formatted_lines.append(' '.join(current_line))
                    current_line = [word]
                    current_width = word_width
            
            if current_line:
                formatted_lines.append(' '.join(current_line))
        
        # Render text with proper line spacing
        y = 10
        line_height = self.font.get_linesize()
        
        for line in formatted_lines:
            if line:
                text_surface = self.font.render(line, True, self.WHITE)
                self.story_surface.blit(text_surface, (20, y))
            y += line_height + 2  # Reduced spacing for more text
            
            # Stop rendering if we've gone beyond the visible area
            if y > self.story_rect.height:
                # Add indicator that there's more text
                more_text = self.font.render("...", True, self.WHITE)
                self.story_surface.blit(more_text, 
                                      (self.story_rect.width - more_text.get_width() - 20, 
                                       self.story_rect.height - more_text.get_height() - 10))
                break

class StoryAdventure:
    def __init__(self, api_key=None):
        """Initialize the story adventure game."""
        # Set up OpenRouter API
        self.api_key = api_key
        self.use_api = self.api_key is not None
        
        if self.use_api:
            print("Using OpenRouter with GPT-4o for story generation...")
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "text_adventure_game",  # Site URL for rankings
                "X-Title": "Story Adventure Game"       # Site title for rankings
            }
        else:
            # Fallback to local GPT-2 if no API key provided
            print("No API key provided. Loading local GPT-2 model instead...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.model.eval()
        
        # Story memory management
        self.story_memory = []
        self.story_context = ""
        
        # Game state tracking
        self.game_state = {
            "location": "forest",
            "inventory": [],
            "characters_met": [],
            "current_quest": None,
            "quest_progress": 0
        }
        
        # System prompts for different story stages
        self.initial_story_system_prompt = (
            "You are a creative storyteller for an adventure game. Create an engaging starting scenario "
            "for a new adventure. Set the scene with vivid details, introduce intriguing elements, and "
            "give the player a sense of mystery and purpose. Keep it concise (3-4 sentences) yet immersive. "
            "IMPORTANT: Use simple, everyday language that's easy to understand. Avoid complex or archaic "
            "vocabulary when possible. If you must use an uncommon word, provide a brief explanation in [brackets] "
            "immediately after the word."
        )
        
        self.continuation_system_prompt = (
            "You are a creative story generator for an adventure game. Continue the story based on the "
            "player's actions with vivid, creative descriptions. Keep responses concise (2-3 sentences) "
            "and focused on advancing the narrative in a coherent way that builds upon previous events. "
            "IMPORTANT: Regularly introduce objects that the player can add to their inventory. "
            "Occasionally give the player clear tasks or quests to accomplish. "
            "For example: 'You notice a rusty key that might unlock the mysterious door' or "
            "'The village elder asks you to retrieve a special herb from the mountain'. "
            "Use simple, everyday language that most people can understand. Avoid complex or "
            "uncommon words when possible. If you must use a complex word, provide a brief explanation in [brackets] "
            "immediately after the word."
        )
        
        # Scene descriptions for better context
        self.scene_descriptions = {
            'forest': [
                "The ancient trees whisper secrets in the wind.",
                "Mysterious lights dance between the branches.",
                "The forest path winds deeper into darkness."
            ],
            'cave': [
                "Crystal formations glitter in the dim light.",
                "The cave walls echo with ancient mysteries.",
                "Underground streams flow through dark passages."
            ],
            'village': [
                "Bustling streets filled with magical merchants.",
                "Cozy cottages with smoking chimneys line the road.",
                "The village square hums with activity."
            ],
            'mountain': [
                "Snow-capped peaks pierce the clouds above.",
                "The mountain path winds treacherously upward.",
                "Ancient ruins cling to the mountainside."
            ],
            'castle': [
                "Towering spires reach toward the sky.",
                "Ancient banners flutter from the battlements.",
                "The castle halls echo with forgotten histories."
            ],
            'dungeon': [
                "Dark corridors stretch into the unknown.",
                "Ancient cells hold forgotten secrets.",
                "The air is thick with mystery and danger."
            ],
            'magical': [
                "Reality bends in impossible ways here.",
                "Magic flows visibly through the air.",
                "The laws of nature hold no power in this realm."
            ]
        }
        
        # Initialize display
        self.display = GameDisplay()
        
    def generate_initial_story(self, theme=None) -> str:
        """Generate the initial story setup based on an optional theme."""
        # Use the provided theme or ask the user for one
        if not theme:
            theme = self.get_user_theme()
            
        prompt = f"Create a starting scenario for an adventure with the theme: {theme}"
        
        if self.use_api:
            try:
                payload = {
                    "model": "openai/gpt-4o",
                    "messages": [
                        {"role": "system", "content": self.initial_story_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 150,
                    "temperature": 0.8,
                    "top_p": 0.9
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    initial_story = result["choices"][0]["message"]["content"].strip()
                    
                    # Update story memory with this initial segment
                    self.story_memory.append({"segment": "initial", "text": initial_story})
                    return initial_story
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    # Fall back to a default starting scenario
                    default_story = "You find yourself at the entrance of an ancient forest. The trees whisper ancient secrets, and a mysterious path leads deeper into the woods. A worn signpost points to multiple destinations, and an old map fragment lies at your feet."
                    self.story_memory.append({"segment": "initial", "text": default_story})
                    return default_story
                    
            except Exception as e:
                print(f"Error generating initial story: {e}")
                default_story = "You find yourself at the entrance of an ancient forest. Something tells you this journey will be unlike any other you've experienced before."
                self.story_memory.append({"segment": "initial", "text": default_story})
                return default_story
        else:
            # Use the local GPT-2 model as fallback for initial story
            initial_prompt = f"{prompt}\n\n"
            inputs = self.tokenizer.encode(initial_prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=150,
                    num_return_sequences=1,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            initial_story = full_text[len(initial_prompt):].strip()
            
            # Clean up the initial story if needed
            if len(initial_story) < 50:  # If generation is too short
                initial_story = "You find yourself at the entrance of an ancient forest. The trees whisper ancient secrets, and a mysterious path leads deeper into the woods."
            
            self.story_memory.append({"segment": "initial", "text": initial_story})
            return initial_story
        
    def get_user_theme(self) -> str:
        """Get a theme from the user input for story generation."""
        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()
        
        # Colors and font
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        BLUE = (0, 0, 200)
        
        # Font setup
        font = pygame.font.Font(None, 32)
        title_font = pygame.font.Font(None, 48)
        
        # Set up a temporary screen if we don't have one yet
        temp_screen = None
        if not hasattr(self, 'display') or not self.display.screen:
            temp_screen = pygame.display.set_mode((1024, 768))
            screen = temp_screen
        else:
            screen = self.display.screen
            
        # Explanation text
        title_text = title_font.render("Adventure Theme", True, WHITE)
        info_text = font.render("Enter a theme for your adventure (1-3 words):", True, WHITE)
        examples_text = font.render("Examples: 'haunted mansion', 'space exploration', 'underwater kingdom'", True, WHITE)
        
        # Input box
        input_box = pygame.Rect(100, 400, 824, 50)
        prompt_text = ""
        active = True
        
        # Background image (if available)
        background = None
        if hasattr(self, 'display') and self.display.backgrounds:
            # Use a default background like forest or magical
            if 'magical' in self.display.backgrounds:
                background = self.display.backgrounds['magical']
            elif 'forest' in self.display.backgrounds:
                background = self.display.backgrounds['forest']
        
        clock = pygame.time.Clock()
        done = False
        
        # Main input loop
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.KEYDOWN:
                    if active:
                        if event.key == pygame.K_RETURN:
                            # Return theme when Enter is pressed
                            if prompt_text.strip():
                                done = True
                        elif event.key == pygame.K_BACKSPACE:
                            prompt_text = prompt_text[:-1]
                        else:
                            # Only add printable characters
                            if event.unicode.isprintable():
                                prompt_text += event.unicode
            
            # Draw the screen
            screen.fill(BLACK)
            
            # Draw background if available
            if background:
                screen.blit(background, (0, 0))
                # Add semi-transparent overlay for better text visibility
                overlay = pygame.Surface((1024, 768), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180))  # Black with alpha
                screen.blit(overlay, (0, 0))
            
            # Draw the title and instruction text
            title_rect = title_text.get_rect(center=(512, 150))
            screen.blit(title_text, title_rect)
            
            info_rect = info_text.get_rect(center=(512, 250))
            screen.blit(info_text, info_rect)
            
            examples_rect = examples_text.get_rect(center=(512, 300))
            screen.blit(examples_text, examples_rect)
            
            # Draw the input box
            pygame.draw.rect(screen, WHITE, input_box, 2)
            
            # Render and display the prompt text
            text_surface = font.render(prompt_text, True, WHITE)
            # Blit to position with some padding
            screen.blit(text_surface, (input_box.x + 10, input_box.y + 10))
            
            # Draw a continue prompt if text is entered
            if prompt_text.strip():
                continue_text = font.render("Press Enter to continue", True, BLUE)
                continue_rect = continue_text.get_rect(center=(512, 500))
                screen.blit(continue_text, continue_rect)
            
            pygame.display.flip()
            clock.tick(30)
        
        # Return a default theme if somehow we got an empty string
        if not prompt_text.strip():
            return "fantasy adventure"
            
        return prompt_text.strip()
    
    def generate_continuation(self, action: str, max_length: int = 100) -> str:
        """Generate a story continuation based on the player's action and story history."""
        # Add scene description for better context
        scene_desc = random.choice(self.scene_descriptions[self.game_state['location']])
        
        # Compile the story context from memory
        story_so_far = "\n\n".join([item["text"] for item in self.story_memory])
        
        # Include inventory and quest information
        inventory_str = ", ".join(self.game_state["inventory"]) if self.game_state["inventory"] else "empty"
        quest_str = self.game_state["current_quest"] if self.game_state["current_quest"] else "none"
        
        # Create a context-aware prompt
        context_prompt = (
            f"Location: {self.game_state['location']} - {scene_desc}\n\n"
            f"Player inventory: {inventory_str}\n"
            f"Current quest: {quest_str}\n\n"
            f"Story so far:\n{story_so_far}\n\n"
            f"Player action: {action}\n\n"
            f"What happens next? (Include potential items for inventory or tasks for the player)"
        )
        
        if self.use_api:
            # Use the OpenRouter API
            try:
                payload = {
                    "model": "openai/gpt-4o",
                    "messages": [
                        {"role": "system", "content": self.continuation_system_prompt},
                        {"role": "user", "content": context_prompt}
                    ],
                    "max_tokens": max_length,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    continuation = result["choices"][0]["message"]["content"].strip()
                    
                    # Add to story memory
                    self.story_memory.append({"segment": "action", "text": f"You decide to {action}."})
                    self.story_memory.append({"segment": "continuation", "text": continuation})
                    
                    return continuation
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    fallback = "The adventure continues as you press forward..."
                    self.story_memory.append({"segment": "action", "text": f"You decide to {action}."})
                    self.story_memory.append({"segment": "continuation", "text": fallback})
                    return fallback
                    
            except Exception as e:
                print(f"Error calling OpenRouter API: {e}")
                fallback = "Your journey takes an unexpected turn..."
                self.story_memory.append({"segment": "action", "text": f"You decide to {action}."})
                self.story_memory.append({"segment": "continuation", "text": fallback})
                return fallback
        else:
            # Use the local GPT-2 model as fallback
            inputs = self.tokenizer.encode(context_prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the new content by finding the last occurrence of the question
            last_question_idx = full_text.rfind("What happens next?")
            if last_question_idx != -1:
                continuation = full_text[last_question_idx + len("What happens next?"):].strip()
            else:
                continuation = full_text[len(context_prompt):].strip()
            
            # Add to story memory
            self.story_memory.append({"segment": "action", "text": f"You decide to {action}."})
            self.story_memory.append({"segment": "continuation", "text": continuation})
            
            return continuation
    
    def update_game_state(self, action: str, continuation: str):
        """Update game state based on action and continuation."""
        combined_text = (action + " " + continuation).lower()
        
        # Check for location changes
        location_keywords = {
            'forest': ['forest', 'woods', 'trees'],
            'cave': ['cave', 'cavern', 'underground'],
            'village': ['village', 'town', 'settlement'],
            'mountain': ['mountain', 'peak', 'cliff'],
            'castle': ['castle', 'fortress', 'palace'],
            'dungeon': ['dungeon', 'prison', 'cell'],
            'magical': ['realm', 'dimension', 'magical world']
        }
        
        for location, keywords in location_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                self.game_state['location'] = location
                self.display.update_background(location)
                break
        
        # Check for inventory items
        inventory_action_words = ['pick up', 'take', 'grab', 'collect', 'acquire', 'obtain', 'pocket', 'get']
        potential_items = [
            'key', 'sword', 'map', 'potion', 'book', 'scroll', 'gem', 'amulet', 'torch', 'compass', 
            'dagger', 'shield', 'staff', 'wand', 'ring', 'coin', 'artifact', 'food', 'water', 'medicine',
            'rope', 'flint', 'lantern', 'cloak', 'boots', 'crystal', 'flower', 'herb', 'stone', 'feather'
        ]
        
        # Check for taking items
        for action_word in inventory_action_words:
            if action_word in combined_text:
                for item in potential_items:
                    # Check if the item is mentioned and not already in inventory
                    if item in combined_text and item not in self.game_state['inventory']:
                        self.game_state['inventory'].append(item)
                        print(f"Added {item} to inventory")
                        break
        
        # Check for quest-related content
        quest_keywords = ['quest', 'mission', 'task', 'objective', 'assignment', 'challenge', 'retrieve', 'find', 'rescue']
        if any(keyword in combined_text for keyword in quest_keywords) and not self.game_state['current_quest']:
            # Try to extract a quest from the text
            quest_phrases = [
                "find the", "retrieve the", "rescue the", "locate the", "bring back the", 
                "deliver the", "investigate the", "explore the", "defeat the", "solve the"
            ]
            
            for phrase in quest_phrases:
                if phrase in combined_text:
                    start_idx = combined_text.find(phrase)
                    # Get a reasonable chunk of text after the phrase
                    quest_snippet = combined_text[start_idx:start_idx + 50]
                    # Cut at the first period or comma
                    for char in ['.', ',']:
                        if char in quest_snippet:
                            quest_snippet = quest_snippet.split(char)[0]
                    
                    if len(quest_snippet) > 5:  # Ensure it's not just the phrase
                        self.game_state['current_quest'] = quest_snippet.strip()
                        self.game_state['quest_progress'] = 0
                        print(f"New quest: {self.game_state['current_quest']}")
                        break
    
    def get_formatted_story(self):
        """Get the formatted story from memory for display."""
        formatted_story = ""
        
        for item in self.story_memory:
            if item["segment"] == "initial":
                formatted_story += item["text"]
            elif item["segment"] == "action":
                formatted_story += f"\n\n{item['text']}"
            elif item["segment"] == "continuation":
                formatted_story += f" {item['text']}"
        
        return formatted_story
    
    def add_definitions_for_complex_words(self, text):
        """Add definitions for complex words if they don't already have explanations."""
        # Dictionary of complex words and their simple definitions
        complex_words = {
            # Architectural/Setting Terms
            "manor": "large country house",
            "estate": "large piece of land with a big house",
            "silhouette": "dark shape against a lighter background",
            "threshold": "entrance or doorway",
            "corridor": "hallway",
            "ornate": "decorated with lots of details",
            "facade": "front of a building",
            "alcove": "small space in a wall",
            "chamber": "room",
            "foyer": "entrance hall",
            "vestibule": "small entrance room",
            "catacombs": "underground tunnels with graves",
            
            # Descriptive Terms
            "looming": "appearing large and threatening",
            "sprawling": "spreading out over a large area",
            "eerie": "strange and frightening",
            "ominous": "suggesting something bad will happen",
            "resounding": "loud and echoing",
            "cryptic": "mysterious and hard to understand",
            "decrepit": "old and falling apart",
            "dilapidated": "in a state of disrepair",
            "ancient": "very old",
            "arcane": "mysterious and understood by few",
            "ethereal": "light and delicate, seeming not of this world",
            "spectral": "ghost-like",
            "harrowing": "extremely distressing",
            "forsaken": "abandoned",
            "malevolent": "having evil intentions",
            "ominous": "suggesting something bad will happen",
            
            # Emotional/Atmospheric Terms
            "dread": "great fear",
            "foreboding": "feeling that something bad will happen",
            "malice": "desire to harm others",
            "anguish": "extreme pain or suffering",
            "unease": "feeling of worry or nervousness",
            "dismay": "feeling of distress",
            "trepidation": "fear about something that's going to happen",
            
            # Action Words
            "beckoning": "signaling to come closer",
            "luring": "tempting or enticing someone",
            "recoiling": "moving back suddenly in fear",
            "cowering": "crouching down in fear",
            "lurking": "hiding in wait",
            "skulking": "moving in a secretive way",
            "cackling": "laughing in a harsh way"
        }
        
        # Only add definitions if they don't already exist in the text
        # Look for words that are already defined with [brackets]
        already_defined = []
        if "[" in text and "]" in text:
            import re
            defined_words = re.findall(r'(\w+)\s*\[(.*?)\]', text)
            for word, _ in defined_words:
                already_defined.append(word.lower())
        
        # Check for complex words and add definitions
        for word, definition in complex_words.items():
            # Only add definition if the word is in the text, 
            # isn't already defined, and isn't part of another word
            if f' {word} ' in f' {text} ' and word.lower() not in already_defined:
                # Replace the word with word + definition
                text = text.replace(f' {word} ', f' {word} [{definition}] ')
        
        return text
    
    def handle_inventory_commands(self, action: str) -> bool:
        """Handle inventory-specific commands and return True if processed."""
        action = action.lower().strip()
        
        # Check for inventory commands
        if action == 'inventory' or action == 'check inventory' or action == 'i':
            inventory_text = "Your inventory contains: "
            
            if not self.game_state['inventory']:
                inventory_text += "nothing."
            else:
                inventory_text += ", ".join(self.game_state['inventory']) + "."
            
            # Add this to the story memory without generating new content
            self.story_memory.append({"segment": "action", "text": "You check your inventory."})
            self.story_memory.append({"segment": "continuation", "text": inventory_text})
            
            # Update display
            self.story_context = self.get_formatted_story()
            self.display.render_text(self.story_context)
            return True
            
        # Check for using items from inventory
        use_patterns = ['use ', 'activate ', 'read ', 'open ', 'wear ', 'equip ', 'drink ', 'eat ']
        for pattern in use_patterns:
            if action.startswith(pattern):
                item_name = action[len(pattern):].strip()
                
                if item_name in self.game_state['inventory']:
                    # We'll let the normal story generation handle what happens when using an item
                    return False
                else:
                    # Item not in inventory
                    response = f"You don't have a {item_name} in your inventory."
                    self.story_memory.append({"segment": "action", "text": f"You try to {pattern}{item_name}."})
                    self.story_memory.append({"segment": "continuation", "text": response})
                    
                    # Update display
                    self.story_context = self.get_formatted_story()
                    self.display.render_text(self.story_context)
                    return True
        
        # Check for dropping items
        drop_patterns = ['drop ', 'discard ', 'throw away ', 'remove ']
        for pattern in drop_patterns:
            if action.startswith(pattern):
                item_name = action[len(pattern):].strip()
                
                if item_name in self.game_state['inventory']:
                    self.game_state['inventory'].remove(item_name)
                    response = f"You drop the {item_name}."
                    self.story_memory.append({"segment": "action", "text": f"You {pattern}{item_name}."})
                    self.story_memory.append({"segment": "continuation", "text": response})
                    
                    # Update display
                    self.story_context = self.get_formatted_story()
                    self.display.render_text(self.story_context)
                    return True
        
        # Not an inventory command
        return False
        
    def play(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True
        
        # Generate initial story setup based on user theme
        initial_story = self.generate_initial_story()
        # Add definitions for any complex words
        initial_story = self.add_definitions_for_complex_words(initial_story)
        self.story_memory[0]["text"] = initial_story
        self.story_context = initial_story
        self.display.render_text(self.story_context)
        
        input_text = ""
        
        # Create a font for UI elements
        ui_font = pygame.font.Font(None, 24)
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN and input_text:
                        action = input_text.strip().lower()
                        input_text = ""
                        
                        if action == 'exit':
                            running = False
                            break
                            
                        # Check if it's an inventory command first
                        if self.handle_inventory_commands(action):
                            continue
                        
                        # Generate continuation and update state
                        continuation = self.generate_continuation(action)
                        # Add definitions for any complex words
                        continuation = self.add_definitions_for_complex_words(continuation)
                        # Update the story memory with the processed text
                        self.story_memory[-1]["text"] = continuation
                        
                        self.update_game_state(action, continuation)
                        
                        # Update story context and display
                        self.story_context = self.get_formatted_story()
                        self.display.render_text(self.story_context)
                        
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.unicode.isprintable():
                        input_text += event.unicode
            
            # Draw everything
            self.display.screen.fill(self.display.BLACK)
            
            # Draw background
            if self.display.current_background in self.display.backgrounds:
                self.display.screen.blit(
                    self.display.backgrounds[self.display.current_background], (0, 0))
            
            # Draw story text
            if self.display.story_surface:
                self.display.screen.blit(self.display.story_surface, self.display.story_rect)
            
            # Draw inventory sidebar
            inventory_surface = pygame.Surface((200, self.display.height - 60), pygame.SRCALPHA)
            inventory_surface.fill((0, 0, 0, 180))  # Semi-transparent black
            
            # Title for inventory
            inventory_title = ui_font.render("INVENTORY", True, self.display.WHITE)
            inventory_surface.blit(inventory_title, (10, 10))
            
            # List inventory items
            y_offset = 40
            if not self.game_state["inventory"]:
                empty_text = ui_font.render("Empty", True, self.display.WHITE)
                inventory_surface.blit(empty_text, (20, y_offset))
            else:
                for item in self.game_state["inventory"]:
                    item_text = ui_font.render(f"• {item}", True, self.display.WHITE)
                    inventory_surface.blit(item_text, (20, y_offset))
                    y_offset += 25
            
            # Display current quest if any
            if self.game_state["current_quest"]:
                quest_y = y_offset + 20
                quest_title = ui_font.render("CURRENT QUEST:", True, (255, 215, 0))  # Gold color
                inventory_surface.blit(quest_title, (10, quest_y))
                
                # Wrap quest text
                quest_text = self.game_state["current_quest"]
                words = quest_text.split()
                quest_lines = []
                current_line = []
                line_width = 0
                
                for word in words:
                    word_surface = ui_font.render(word, True, self.display.WHITE)
                    word_width = word_surface.get_width()
                    
                    if line_width + word_width <= 180:  # Max width with some margin
                        current_line.append(word)
                        line_width += word_width + ui_font.size(' ')[0]  # Add space width
                    else:
                        quest_lines.append(' '.join(current_line))
                        current_line = [word]
                        line_width = word_width
                
                if current_line:
                    quest_lines.append(' '.join(current_line))
                
                for i, line in enumerate(quest_lines):
                    line_surf = ui_font.render(line, True, (255, 255, 150))  # Light yellow
                    inventory_surface.blit(line_surf, (20, quest_y + 25 + i * 20))
            
            # Draw the inventory sidebar on the right side
            self.display.screen.blit(inventory_surface, (self.display.width - 200, 10))
            
            # Draw input box
            pygame.draw.rect(self.display.screen, self.display.WHITE, self.display.input_rect, 2)
            text_surface = self.display.font.render(input_text, True, self.display.WHITE)
            self.display.screen.blit(text_surface, (self.display.input_rect.x + 5, self.display.input_rect.y + 5))
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

def main(api_key=None):
    """Main entry point for the game with optional API key parameter"""
    # If no API key provided, use the default
    if not api_key:
        api_key = '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'  # Default key
    
    # Or check environment variable as fallback
    elif os.environ.get("OPENROUTER_API_KEY"):
        api_key = os.environ.get("OPENROUTER_API_KEY")
    
    # Display a message about which API key we're using (masked for security)
    if api_key:
        print(f"Using API key: {api_key[:10]}...{api_key[-5:]}")
    
    # Create game instance and start
    game = StoryAdventure(api_key)
    game.play()

if __name__ == "__main__":
    # If run directly, check for command line argument
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()  