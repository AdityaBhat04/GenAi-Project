import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import pygame
import time
import os
from PIL import Image
import random
import urllib.request
import numpy as np
from textblob import TextBlob

class GameDisplay:
    def __init__(self, width=1024, height=768):
        """Initialize the Pygame display."""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Story Adventure")
        
        # Font setup
        self.title_font = pygame.font.Font(None, 74)
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GOLD = (255, 215, 0)
        
        # GUI elements
        self.sidebar_width = 200
        self.stats_height = 100
        self.input_height = 40
        
        # Text input setup
        self.input_text = ""
        self.input_rect = pygame.Rect(10, height - self.input_height - 10, width - self.sidebar_width - 20, self.input_height)
        
        # Story text setup
        self.story_surface = None
        self.story_rect = pygame.Rect(10, 10, width - self.sidebar_width - 20, height - self.input_height - 30)
        
        # Game state
        self.show_title_screen = True
        self.show_help = False
        self.health = 100
        self.stamina = 100
        self.gold = 0
        
        # Create backgrounds directory if it doesn't exist
        if not os.path.exists('backgrounds'):
            os.makedirs('backgrounds')
        
        # Background setup
        self.current_location = 'forest'
        self.current_time = 'day'
        self.create_backgrounds()
        self.current_background = self.get_background('forest', 'day')
        
        # Transition setup
        self.prev_background = None
        self.transition_alpha = 0
        self.is_transitioning = False
        
        # Try to load sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline('sentiment-analysis')
            print("Sentiment analysis loaded successfully")
        except:
            self.sentiment_analyzer = None
            print("Sentiment analysis not available")
    
    def create_backgrounds(self):
        """Create simple colored backgrounds for different locations."""
        self.background_colors = {
            'forest': {
                'day': (34, 139, 34),       # Forest Green
                'night': (0, 100, 0),       # Dark Green
                'dawn': (218, 165, 32)      # Golden
            },
            'cave': {
                'dark': (47, 79, 79),       # Dark Slate
                'lit': (105, 105, 105)      # Dim Gray
            },
            'village': {
                'day': (210, 180, 140),     # Tan
                'night': (139, 119, 101)    # Dark Tan
            },
            'mountain': {
                'day': (128, 128, 128),     # Gray
                'night': (105, 105, 105),   # Dark Gray
                'storm': (72, 72, 72)       # Stormy Gray
            },
            'castle': {
                'day': (169, 169, 169),     # Light Gray
                'night': (105, 105, 105),   # Dark Gray
                'interior': (139, 69, 19)   # Saddle Brown
            }
        }
        
        self.backgrounds = {}
        
        # Create backgrounds for each location and variation
        for location, variations in self.background_colors.items():
            self.backgrounds[location] = {}
            
            for variation, color in variations.items():
                # Create surface with solid color
                surface = pygame.Surface((self.width, self.height))
                surface.fill(color)
                
                # Add some texture/pattern to make it more interesting
                self.add_texture(surface, location, variation)
                
                # Save the surface
                self.backgrounds[location][variation] = surface
                
                # Save to file for future use
                filename = f'backgrounds/{location}_{variation}.png'
                if not os.path.exists(filename):
                    pygame.image.save(surface, filename)
                    print(f"Created background: {filename}")
    
    def add_texture(self, surface, location, variation):
        """Add texture to background surfaces to make them more interesting."""
        if location == 'forest':
            # Draw some tree-like shapes
            for _ in range(50):
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)
                h = random.randint(50, 150)
                w = random.randint(10, 20)
                color = (0, 100, 0) if variation == 'night' else (0, 80, 0)
                pygame.draw.rect(surface, color, (x, y, w, h))
                pygame.draw.circle(surface, color, (x + w//2, y), w)
        
        elif location == 'cave':
            # Draw some stalagmites
            for _ in range(30):
                x = random.randint(0, self.width)
                y = random.randint(self.height//2, self.height)
                h = random.randint(50, 150)
                w = random.randint(10, 30)
                color = (30, 30, 30) if variation == 'dark' else (80, 80, 80)
                pygame.draw.polygon(surface, color, [(x, y), (x + w, y), (x + w//2, y - h)])
        
        elif location == 'village':
            # Draw some simple houses
            for _ in range(20):
                x = random.randint(0, self.width)
                y = random.randint(self.height//2, self.height - 50)
                w = random.randint(40, 80)
                h = random.randint(30, 60)
                color = (139, 69, 19)  # Brown
                pygame.draw.rect(surface, color, (x, y, w, h))
                pygame.draw.polygon(surface, (150, 75, 0), [(x, y), (x + w, y), (x + w//2, y - 30)])
        
        elif location == 'mountain':
            # Draw mountain peaks
            for _ in range(10):
                x = random.randint(0, self.width)
                y = self.height
                w = random.randint(100, 200)
                h = random.randint(100, 300)
                color = (100, 100, 100)
                if variation == 'storm':
                    color = (50, 50, 70)
                pygame.draw.polygon(surface, color, [(x, y), (x + w, y), (x + w//2, y - h)])
        
        elif location == 'castle':
            # Draw a simple castle
            x = self.width // 2 - 100
            y = self.height // 2
            color = (100, 100, 100) if variation != 'interior' else (139, 69, 19)
            # Main structure
            pygame.draw.rect(surface, color, (x, y, 200, 150))
            # Towers
            pygame.draw.rect(surface, color, (x - 30, y - 30, 50, 180))
            pygame.draw.rect(surface, color, (x + 180, y - 30, 50, 180))
            # Battlements
            for i in range(5):
                pygame.draw.rect(surface, color, (x + i*40, y - 20, 20, 20))
    
    def get_background(self, location, variation=None):
        """Get the appropriate background surface."""
        if location not in self.backgrounds:
            location = 'forest'  # Default fallback
        
        if not variation:
            # Pick default variation for the location
            variations = list(self.backgrounds[location].keys())
            variation = variations[0]
        
        # Make sure the variation exists for this location
        if variation not in self.backgrounds[location]:
            # Get first available variation
            variation = list(self.backgrounds[location].keys())[0]
        
        return self.backgrounds[location][variation]
    
    def update_background(self, location, mood=None):
        """Start transition to a new background."""
        variation = self.determine_variation(location, mood)
        
        if location != self.current_location or variation != self.current_time:
            self.prev_background = self.current_background
            self.current_background = self.get_background(location, variation)
            self.current_location = location
            self.current_time = variation
            self.transition_alpha = 0
            self.is_transitioning = True
            print(f"Changing background to: {location} - {variation}")
    
    def determine_variation(self, location, mood):
        """Determine the appropriate variation based on location and mood."""
        # Default variations
        if location == 'cave':
            if mood == 'positive':
                return 'lit'
            return 'dark'
        
        if location == 'mountain' and mood == 'negative':
            return 'storm'
        
        if location == 'castle' and random.random() < 0.3:
            return 'interior'
        
        # Time of day for other locations
        hour = time.localtime().tm_hour
        if 6 <= hour < 8:
            return 'dawn' if 'dawn' in self.backgrounds[location] else 'day'
        elif 8 <= hour < 20:
            return 'day'
        else:
            return 'night'
    
    def update_transition(self):
        """Update background transition effect."""
        if self.is_transitioning:
            self.transition_alpha += 5
            if self.transition_alpha >= 255:
                self.is_transitioning = False
                self.transition_alpha = 255
    
    def draw_background(self):
        """Draw the current background with transition effect if needed."""
        if self.is_transitioning and self.prev_background:
            # Draw previous background
            self.screen.blit(self.prev_background, (0, 0))
            
            # Draw new background with alpha
            temp = self.current_background.copy()
            temp.set_alpha(self.transition_alpha)
            self.screen.blit(temp, (0, 0))
        else:
            # Just draw current background
            self.screen.blit(self.current_background, (0, 0))
    
    def draw_title_screen(self):
        """Draw the game's title screen."""
        self.screen.fill(self.BLACK)
        
        # Draw background
        if self.current_background:
            self.screen.blit(self.current_background, (0, 0))
        
        # Add semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.fill(self.BLACK)
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))
        
        # Draw title
        title = self.title_font.render("Story Adventure", True, self.GOLD)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 3))
        self.screen.blit(title, title_rect)
        
        # Draw menu options
        start_text = self.font.render("Press ENTER to Start", True, self.WHITE)
        help_text = self.font.render("Press H for Help", True, self.WHITE)
        quit_text = self.font.render("Press Q to Quit", True, self.WHITE)
        
        start_rect = start_text.get_rect(center=(self.width // 2, self.height // 2))
        help_rect = help_text.get_rect(center=(self.width // 2, self.height // 2 + 50))
        quit_rect = quit_text.get_rect(center=(self.width // 2, self.height // 2 + 100))
        
        self.screen.blit(start_text, start_rect)
        self.screen.blit(help_text, help_rect)
        self.screen.blit(quit_text, quit_rect)
    
    def draw_help_screen(self):
        """Draw the help screen."""
        self.screen.fill(self.BLACK)
        
        # Draw background
        if self.current_background:
            self.screen.blit(self.current_background, (0, 0))
        
        # Add semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.fill(self.BLACK)
        overlay.set_alpha(180)
        self.screen.blit(overlay, (0, 0))
        
        # Draw help content
        title = self.font.render("How to Play", True, self.GOLD)
        title_rect = title.get_rect(center=(self.width // 2, 50))
        self.screen.blit(title, title_rect)
        
        help_texts = [
            "- Type your actions to interact with the story",
            "- Use 'look' to examine your surroundings",
            "- Use 'inventory' to check your items",
            "- Use 'status' to check your health and stamina",
            "- Press ESC to pause the game",
            "- Type 'exit' to quit the game",
            "",
            "Press BACKSPACE to return"
        ]
        
        y = 120
        for text in help_texts:
            help_surface = self.font.render(text, True, self.WHITE)
            help_rect = help_surface.get_rect(center=(self.width // 2, y))
            self.screen.blit(help_surface, help_rect)
            y += 40
    
    def draw_sidebar(self, game_state):
        """Draw the game sidebar with player stats and inventory."""
        # Draw sidebar background
        sidebar_rect = pygame.Rect(self.width - self.sidebar_width, 0, self.sidebar_width, self.height)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), sidebar_rect)
        
        # Draw health bar
        self.draw_stat_bar("Health", self.health, self.RED, 20)
        
        # Draw stamina bar
        self.draw_stat_bar("Stamina", self.stamina, self.BLUE, 60)
        
        # Draw gold counter
        gold_text = self.font.render(f"Gold: {self.gold}", True, self.GOLD)
        self.screen.blit(gold_text, (self.width - self.sidebar_width + 10, 100))
        
        # Draw inventory
        inventory_title = self.font.render("Inventory", True, self.WHITE)
        self.screen.blit(inventory_title, (self.width - self.sidebar_width + 10, 140))
        
        y = 180
        for item in game_state["inventory"]:
            item_text = self.small_font.render(f"- {item}", True, self.WHITE)
            self.screen.blit(item_text, (self.width - self.sidebar_width + 20, y))
            y += 25
    
    def draw_stat_bar(self, name, value, color, y_offset):
        """Draw a stat bar (health/stamina)."""
        bar_width = self.sidebar_width - 20
        bar_height = 20
        x = self.width - self.sidebar_width + 10
        y = y_offset
        
        # Draw label
        label = self.small_font.render(name, True, self.WHITE)
        self.screen.blit(label, (x, y - 20))
        
        # Draw background
        pygame.draw.rect(self.screen, self.BLACK, (x, y, bar_width, bar_height))
        
        # Draw filled portion
        fill_width = int((value / 100) * bar_width)
        pygame.draw.rect(self.screen, color, (x, y, fill_width, bar_height))
        
        # Draw border
        pygame.draw.rect(self.screen, self.WHITE, (x, y, bar_width, bar_height), 1)
        
        # Draw value
        value_text = self.small_font.render(f"{value}%", True, self.WHITE)
        value_rect = value_text.get_rect(center=(x + bar_width // 2, y + bar_height // 2))
        self.screen.blit(value_text, value_rect)
    
    def render_text(self, text: str):
        """Render the story text."""
        self.story_surface = pygame.Surface(
            (self.story_rect.width, self.story_rect.height), pygame.SRCALPHA)
        self.story_surface.fill((0, 0, 0, 180))
        
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            text_surface = self.font.render(' '.join(current_line), True, self.WHITE)
            if text_surface.get_width() > self.story_rect.width - 20:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))
        
        y = 10
        for line in lines:
            text_surface = self.font.render(line, True, self.WHITE)
            self.story_surface.blit(text_surface, (10, y))
            y += text_surface.get_height() + 5

class StoryAdventure:
    def __init__(self):
        """Initialize the story adventure game."""
        print("Loading story generation model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        
        self.story_context = ""
        self.game_state = {
            "location": "forest",
            "inventory": [],
            "characters_met": [],
            "current_quest": None,
            "health": 100,
            "stamina": 100,
            "gold": 0
        }
        
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
        self.running = True
        
    def analyze_mood(self, text):
        """Analyze the mood of the text."""
        if self.display.sentiment_analyzer:
            try:
                result = self.display.sentiment_analyzer(text)
                return 'positive' if result[0]['label'] == 'POSITIVE' else 'negative'
            except Exception as e:
                print(f"Error in sentiment analysis: {str(e)}")
        
        # Fallback to keyword-based mood detection
        positive_words = ['happy', 'joy', 'success', 'beautiful', 'light', 'safe']
        negative_words = ['dark', 'danger', 'fear', 'death', 'evil', 'storm']
        
        text = text.lower()
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        return 'positive' if pos_count >= neg_count else 'negative'
    
    def generate_continuation(self, prompt: str, max_length: int = 100) -> str:
        """Generate a story continuation based on the given prompt."""
        # Add scene description for better context
        scene_desc = random.choice(self.scene_descriptions[self.game_state['location']])
        context_prompt = f"Location: {scene_desc}\n{prompt}"
        
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
        
        continuation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return continuation[len(context_prompt):].strip()
    
    def update_game_state(self, action: str, continuation: str):
        """Update game state based on action and continuation."""
        # Analyze mood
        mood = self.analyze_mood(continuation)
        print(f"Story mood: {mood}")
        
        # Check for location changes
        location_keywords = {
            'forest': ['forest', 'woods', 'trees'],
            'cave': ['cave', 'cavern', 'underground'],
            'village': ['village', 'town', 'settlement'],
            'mountain': ['mountain', 'peak', 'cliff'],
            'castle': ['castle', 'fortress', 'palace']
        }
        
        text = (action + " " + continuation).lower()
        for location, keywords in location_keywords.items():
            if any(keyword in text for keyword in keywords):
                self.game_state['location'] = location
                self.display.update_background(location, mood)
                break
    
    def play(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        
        while self.running:
            if self.display.show_title_screen:
                self.handle_title_screen()
            elif self.display.show_help:
                self.handle_help_screen()
            else:
                self.handle_game_screen()
            
            pygame.display.flip()
            clock.tick(60)
        
        # Ensure proper cleanup
        pygame.quit()
        import sys
        sys.exit()
    
    def handle_title_screen(self):
        """Handle title screen events and drawing."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.display.show_title_screen = False
                    self.start_game()
                elif event.key == pygame.K_h:
                    self.display.show_help = True
                    self.display.show_title_screen = False
                elif event.key == pygame.K_q:
                    self.running = False
                    return
        
        self.display.draw_title_screen()
    
    def handle_help_screen(self):
        """Handle help screen events and drawing."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.display.show_help = False
                    self.display.show_title_screen = True
        
        self.display.draw_help_screen()
    
    def handle_game_screen(self):
        """Handle main game screen events and drawing."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and self.display.input_text:
                    if not self.process_input():
                        self.running = False
                        return
                elif event.key == pygame.K_BACKSPACE:
                    self.display.input_text = self.display.input_text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    self.display.show_title_screen = True
                else:
                    self.display.input_text += event.unicode
        
        self.draw_game_screen()
    
    def start_game(self):
        """Initialize the game state and display initial story."""
        initial_prompt = """You find yourself at the entrance of an ancient forest. The trees whisper ancient secrets, 
and a mysterious path leads deeper into the woods. A worn signpost points to multiple destinations, 
and an old map fragment lies at your feet. What do you do?"""
        
        self.story_context = initial_prompt
        self.display.render_text(self.story_context)
    
    def process_input(self):
        """Process player input and update game state."""
        action = self.display.input_text.strip().lower()
        self.display.input_text = ""
        
        if action == 'exit':
            self.running = False
            return False
        
        # Generate continuation and update state
        continuation = self.generate_continuation(
            f"{self.story_context}\n\nYou decide to {action}. ")
        self.update_game_state(action, continuation)
        
        # Update story context and display
        self.story_context = f"{self.story_context}\n\nYou decide to {action}. {continuation}"
        self.display.render_text(self.story_context)
        
        # Update player stats based on action
        self.update_player_stats(action, continuation)
        return True
    
    def update_player_stats(self, action, continuation):
        """Update player stats based on action and story continuation."""
        # Decrease stamina for physical actions
        physical_actions = ['run', 'jump', 'climb', 'fight', 'swim', 'sprint']
        if any(word in action.lower() for word in physical_actions):
            self.display.stamina = max(0, self.display.stamina - random.randint(5, 15))
        else:
            self.display.stamina = min(100, self.display.stamina + 5)
        
        # Update health based on dangerous situations
        danger_words = ['hurt', 'wound', 'injury', 'damage', 'pain', 'hit']
        if any(word in continuation.lower() for word in danger_words):
            self.display.health = max(0, self.display.health - random.randint(5, 20))
        
        # Find gold in the story
        if 'gold' in continuation.lower() or 'coins' in continuation.lower():
            found_gold = random.randint(1, 10)
            self.display.gold += found_gold
    
    def draw_game_screen(self):
        """Draw the main game screen."""
        # Update background transition
        self.display.update_transition()
        
        # Draw background
        self.display.draw_background()
        
        # Draw story text
        if self.display.story_surface:
            self.display.screen.blit(self.display.story_surface, self.display.story_rect)
        
        # Draw sidebar
        self.display.draw_sidebar(self.game_state)
        
        # Draw input box
        pygame.draw.rect(self.display.screen, self.display.WHITE, self.display.input_rect, 2)
        text_surface = self.display.font.render(self.display.input_text, True, self.display.WHITE)
        self.display.screen.blit(text_surface, (self.display.input_rect.x + 5, self.display.input_rect.y + 5))

if __name__ == "__main__":
    game = StoryAdventure()
    game.play() 