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
import sys

class GameDisplay:
    def __init__(self, width=1024, height=768):
        """Initialize the Pygame display."""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Story Adventure (Dynamic Start)")
        
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
        self.input_height = 40
        
        # Text input setup
        self.input_text = ""
        self.input_rect = pygame.Rect(10, height - self.input_height - 10, width - self.sidebar_width - 20, self.input_height)
        
        # Story text setup
        self.story_display_rect = pygame.Rect(10, 10, width - self.sidebar_width - 20, height - self.input_height - 30)
        self.full_story_surface = None
        self.full_story_height = 0
        self.story_scroll_offset = 0
        self.line_height = self.font.get_linesize()
        self.line_spacing = 5
        
        # Player stats
        self.health = 100
        self.stamina = 100
        self.gold = 0

        # Game state flags
        self.show_title_screen = True
        self.show_help = False
        
        # Create backgrounds directory if it doesn't exist
        self.backgrounds_dir = 'backgrounds'
        if not os.path.exists(self.backgrounds_dir):
            os.makedirs(self.backgrounds_dir)
        
        # Background management
        self.current_location = 'forest'
        self.current_variation = 'day'
        self.backgrounds = self.load_or_create_backgrounds()
        self.current_background_surface = self.get_background_surface(self.current_location, self.current_variation)
        
        # Transition effect variables
        self.previous_background_surface = None
        self.transition_alpha = 0
        self.is_transitioning = False
        self.transition_speed = 5
        
        # Try to load sentiment analyzer
        self.sentiment_analyzer = self.load_sentiment_analyzer()

    def load_sentiment_analyzer(self):
        """Loads the sentiment analysis pipeline, handles errors."""
        try:
            analyzer = pipeline('sentiment-analysis')
            print("Sentiment analysis pipeline loaded successfully.")
            return analyzer
        except Exception as e:
            print(f"Warning: Failed to load sentiment analysis pipeline: {e}")
            print("Falling back to keyword-based mood detection.")
            return None

    def load_or_create_backgrounds(self):
        """Loads existing backgrounds or creates new ones if missing."""
        background_defs = {
            'forest': {'day': (34, 139, 34), 'night': (0, 100, 0), 'dawn': (218, 165, 32)},
            'cave': {'dark': (47, 79, 79), 'lit': (105, 105, 105)},
            'village': {'day': (210, 180, 140), 'night': (139, 119, 101)},
            'mountain': {'day': (128, 128, 128), 'night': (105, 105, 105), 'storm': (72, 72, 72)},
            'castle': {'day': (169, 169, 169), 'night': (105, 105, 105), 'interior': (139, 69, 19)}
        }
        loaded_backgrounds = {}
        for location, variations in background_defs.items():
            loaded_backgrounds[location] = {}
            for variation, color in variations.items():
                filename = os.path.join(self.backgrounds_dir, f'{location}_{variation}.png')
                if os.path.exists(filename):
                    try:
                        surface = pygame.image.load(filename).convert_alpha()
                        surface = pygame.transform.scale(surface, (self.width, self.height))
                    except Exception as e:
                        print(f"Error loading {filename}: {e}. Creating fallback.")
                        surface = self.create_fallback_background(color, location, variation)
                        pygame.image.save(surface, filename)
                else:
                    print(f"Creating background: {filename}")
                    surface = self.create_fallback_background(color, location, variation)
                    pygame.image.save(surface, filename)
                loaded_backgrounds[location][variation] = surface
        print("Background loading/creation complete.")
        return loaded_backgrounds

    def create_fallback_background(self, color, location=None, variation=None):
        """Creates a fallback background surface with color and optional texture."""
        surface = pygame.Surface((self.width, self.height))
        surface.fill(color)
        if location and variation:
            self.add_texture(surface, location, variation)
        return surface

    def add_texture(self, surface, location, variation):
        """Adds simple procedural texture to background surfaces."""
        num_elements = random.randint(20, 50)
        for _ in range(num_elements):
            try:
                if location == 'forest':
                    self.add_forest_texture(surface, variation)
                elif location == 'cave':
                    self.add_cave_texture(surface, variation)
                elif location == 'village':
                    self.add_village_texture(surface, variation)
                elif location == 'mountain':
                    self.add_mountain_texture(surface, variation)
                elif location == 'castle':
                    self.add_castle_texture(surface, variation)
            except Exception as e:
                print(f"Error adding texture for {location} {variation}: {e}")

    def add_forest_texture(self, surface, variation):
        x = random.randint(0, self.width)
        y = random.randint(int(self.height*0.4), self.height)
        h = random.randint(int(self.height*0.1), int(self.height*0.3))
        w = random.randint(10, 30)
        base_y = self.height - h
        tree_color = (0, random.randint(60, 120), 0) if variation != 'night' else (0, random.randint(30, 70), 0)
        pygame.draw.rect(surface, tree_color, (x, base_y, w, h))
        pygame.draw.circle(surface, tree_color, (x + w // 2, base_y), w * 2)

    def add_cave_texture(self, surface, variation):
        x = random.randint(0, self.width)
        is_stalactite = random.choice([True, False])
        h = random.randint(int(self.height*0.1), int(self.height*0.4))
        w = random.randint(15, 40)
        cave_color = (random.randint(20, 60), random.randint(20, 60), random.randint(20, 60)) if variation == 'dark' else (random.randint(70, 110), random.randint(70, 110), random.randint(70, 110))
        if is_stalactite:
            pygame.draw.polygon(surface, cave_color, [(x, 0), (x + w, 0), (x + w // 2, h)])
        else:
            base_y = self.height
            pygame.draw.polygon(surface, cave_color, [(x, base_y), (x + w, base_y), (x + w // 2, base_y - h)])

    def add_village_texture(self, surface, variation):
        x = random.randint(int(self.width*0.1), int(self.width*0.9) - 80)
        y = random.randint(int(self.height*0.6), self.height - 100)
        w = random.randint(50, 100)
        h = random.randint(40, 80)
        house_color = (random.randint(100, 150), random.randint(60, 100), random.randint(10, 50))
        roof_color = (random.randint(120, 180), random.randint(50, 90), 0)
        pygame.draw.rect(surface, house_color, (x, y, w, h))
        pygame.draw.polygon(surface, roof_color, [(x - 5, y), (x + w + 5, y), (x + w // 2, y - h // 2)])

    def add_mountain_texture(self, surface, variation):
        x = random.randint(-int(self.width*0.1), int(self.width*0.8))
        y = self.height
        w = random.randint(int(self.width*0.2), int(self.width*0.5))
        h = random.randint(int(self.height*0.3), int(self.height*0.7))
        peak_color = (random.randint(90, 140), random.randint(90, 140), random.randint(90, 140))
        if variation == 'storm':
            peak_color = (random.randint(40, 80), random.randint(40, 80), random.randint(50, 90))
        pygame.draw.polygon(surface, peak_color, [(x, y), (x + w, y), (x + w // 2, y - h)])

    def add_castle_texture(self, surface, variation):
        is_wall = random.random() > 0.3
        wall_color = (random.randint(80, 120), random.randint(80, 120), random.randint(80, 120)) if variation != 'interior' else (random.randint(100, 150), random.randint(60, 100), random.randint(20, 60))
        if is_wall:
            x = random.randint(0, self.width - 100)
            y = random.randint(int(self.height*0.3), int(self.height*0.7))
            w = random.randint(100, 300)
            h = random.randint(50, 150)
            pygame.draw.rect(surface, wall_color, (x, y, w, h))
        else:
            x = random.randint(int(self.width*0.2), int(self.width*0.8) - 50)
            y = random.randint(int(self.height*0.2), int(self.height*0.5))
            w = random.randint(40, 80)
            h = random.randint(100, 250)
            pygame.draw.rect(surface, wall_color, (x, y, w, h))

    def get_background_surface(self, location, variation):
        """Safely retrieves the background surface."""
        if location in self.backgrounds and variation in self.backgrounds[location]:
            return self.backgrounds[location][variation]
        else:
            print(f"Error: Background for {location}/{variation} not found. Returning default.")
            first_loc = list(self.backgrounds.keys())[0]
            first_var = list(self.backgrounds[first_loc].keys())[0]
            return self.backgrounds[first_loc][first_var]

    def trigger_background_change(self, new_location, new_variation):
        """Checks if a change is needed and initiates the transition."""
        if new_location == self.current_location and new_variation == self.current_variation:
            return

        print(f"Triggering background change: {self.current_location}_{self.current_variation} -> {new_location}_{new_variation}")
        self.previous_background_surface = self.current_background_surface
        self.current_background_surface = self.get_background_surface(new_location, new_variation)
        self.current_location = new_location
        self.current_variation = new_variation
        self.is_transitioning = True
        self.transition_alpha = 0

    def update_transition(self):
        """Updates the alpha for the fade transition."""
        if self.is_transitioning:
            self.transition_alpha += self.transition_speed
            if self.transition_alpha >= 255:
                self.transition_alpha = 255
                self.is_transitioning = False
                self.previous_background_surface = None

    def draw_background(self):
        """Draws the background, handling transitions."""
        if self.is_transitioning and self.previous_background_surface:
            self.screen.blit(self.previous_background_surface, (0, 0))
            new_bg_copy = self.current_background_surface.copy()
            new_bg_copy.set_alpha(self.transition_alpha)
            self.screen.blit(new_bg_copy, (0, 0))
        elif self.current_background_surface:
            self.screen.blit(self.current_background_surface, (0, 0))
        else:
            self.screen.fill(self.BLACK)

    def draw_title_screen(self):
        self.draw_background()
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        title = self.title_font.render("Story Adventure", True, self.GOLD)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 3))
        self.screen.blit(title, title_rect)
        
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
        self.draw_background()
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        title = self.font.render("How to Play", True, self.GOLD)
        title_rect = title.get_rect(center=(self.width // 2, 50))
        self.screen.blit(title, title_rect)
        
        help_texts = [
            "- Type your actions (e.g., 'look around', 'go north')",
            "- Use 'inventory' to check items",
            "- Use 'status' to see health/stamina",
            "- Press ESC to return to the title screen",
            "- Type 'exit' to quit the game",
            "",
            "Press BACKSPACE to return to Title Screen"
        ]
        
        y = 120
        for text in help_texts:
            help_surface = self.font.render(text, True, self.WHITE)
            help_rect = help_surface.get_rect(center=(self.width // 2, y))
            self.screen.blit(help_surface, help_rect)
            y += 40

    def draw_sidebar(self, game_state):
        """Draws the sidebar UI."""
        sidebar_surface = pygame.Surface((self.sidebar_width, self.height), pygame.SRCALPHA)
        sidebar_surface.fill((0, 0, 0, 180))
        self.screen.blit(sidebar_surface, (self.width - self.sidebar_width, 0))
        
        self.draw_stat_bar("Health", game_state['health'], self.RED, 20)
        self.draw_stat_bar("Stamina", game_state['stamina'], self.BLUE, 80)
        
        gold_text = self.font.render(f"Gold: {game_state['gold']}", True, self.GOLD)
        self.screen.blit(gold_text, (self.width - self.sidebar_width + 10, 140))
        
        inventory_title = self.font.render("Inventory", True, self.WHITE)
        self.screen.blit(inventory_title, (self.width - self.sidebar_width + 10, 180))
        
        y = 210
        if "inventory" in game_state:
            for item in game_state["inventory"]:
                item_text = self.small_font.render(f"- {item}", True, self.WHITE)
                self.screen.blit(item_text, (self.width - self.sidebar_width + 20, y))
                y += 25
                if y > self.height - 20:
                    break

    def draw_stat_bar(self, name, value, color, y_offset):
        """Draws a single stat bar."""
        bar_width = self.sidebar_width - 20
        bar_height = 20
        x = self.width - self.sidebar_width + 10
        y = y_offset
        
        label = self.small_font.render(name, True, self.WHITE)
        self.screen.blit(label, (x, y))
        
        bg_rect = pygame.Rect(x, y + 25, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.BLACK, bg_rect)
        
        fill_width = max(0, int((value / 100.0) * bar_width))
        fill_rect = pygame.Rect(x, y + 25, fill_width, bar_height)
        pygame.draw.rect(self.screen, color, fill_rect)
        
        pygame.draw.rect(self.screen, self.WHITE, bg_rect, 1)
        
        value_text = self.small_font.render(f"{int(value)}%", True, self.WHITE)
        value_rect = value_text.get_rect(center=bg_rect.center)
        self.screen.blit(value_text, value_rect)

    def render_text(self, text: str):
        """Renders the story text onto a potentially large internal surface."""
        words = text.split(' ')
        lines = []
        current_line = ''
        max_width = self.story_display_rect.width - 20

        for word in words:
            if '\n' in word:
                parts = word.split('\n')
                for i, part in enumerate(parts):
                    test_line = current_line + ' ' + part if current_line else part
                    if self.font.size(test_line)[0] <= max_width:
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = part
                    if i < len(parts) - 1:
                        lines.append(current_line)
                        current_line = ''
            else:
                test_line = current_line + ' ' + word if current_line else word
                if self.font.size(test_line)[0] <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
        lines.append(current_line)

        self.full_story_height = (len(lines) * (self.line_height + self.line_spacing)) + 20
        surface_height = max(self.full_story_height, self.story_display_rect.height)
        self.full_story_surface = pygame.Surface((self.story_display_rect.width, surface_height), pygame.SRCALPHA)
        self.full_story_surface.fill((0, 0, 0, 180))
        
        y = 10
        for line in lines:
            text_surface = self.font.render(line, True, self.WHITE)
            self.full_story_surface.blit(text_surface, (10, y))
            y += self.line_height + self.line_spacing
            
        max_scroll = max(0, self.full_story_height - self.story_display_rect.height)
        self.story_scroll_offset = max(0, min(self.story_scroll_offset, max_scroll))

    def scroll_text(self, dy):
        """Adjusts the story scroll offset."""
        if not self.full_story_surface:
            return
            
        max_scroll = max(0, self.full_story_height - self.story_display_rect.height)
        scroll_amount = dy * (self.line_height + self.line_spacing) * 3
        
        self.story_scroll_offset -= scroll_amount
        self.story_scroll_offset = max(0, min(self.story_scroll_offset, max_scroll))

    def draw_story_text(self):
        """Draws the visible portion of the story text onto the screen."""
        if self.full_story_surface:
            source_rect = pygame.Rect(0, self.story_scroll_offset, 
                                    self.story_display_rect.width, self.story_display_rect.height)
            self.screen.blit(self.full_story_surface, self.story_display_rect.topleft, source_rect)

class StoryAdventure:
    def __init__(self):
        """Initialize the game logic."""
        self.display = GameDisplay()
        self.running = True
        self.tokenizer, self.model = self.load_llm()
        
        self.story_context = ""
        self.game_state = {
            "location": "",  # Will be set based on initial story
            "inventory": [],
            "health": 100,
            "stamina": 100,
            "gold": 0
        }
        # Initialize display stats
        self.display.health = self.game_state['health']
        self.display.stamina = self.game_state['stamina']
        self.display.gold = self.game_state['gold']

    def load_llm(self):
        """Loads the LLM and tokenizer."""
        print("Loading story generation model (GPT-2)...")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            model.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
            print("GPT-2 model loaded successfully.")
            return tokenizer, model
        except Exception as e:
            print(f"Error loading GPT-2 model: {e}")
            print("Story generation will not function.")
            return None, None

    def generate_initial_story(self):
        """Generates a unique starting scenario for the game."""
        if not self.model or not self.tokenizer:
            return "You find yourself at the entrance of a mysterious forest.", "forest"

        starting_locations = [
            "forest", "mountain", "village", "castle", "cave"
        ]
        time_settings = [
            "dawn", "dusk", "morning", "night", "stormy day"
        ]
        story_elements = [
            "ancient prophecy", "lost treasure", "mysterious stranger",
            "magical artifact", "forgotten ruins", "local legend",
            "royal quest", "dark omen", "magical portal",
            "ancient map", "mystical creature"
        ]

        location = random.choice(starting_locations)
        time_setting = random.choice(time_settings)
        story_element = random.choice(story_elements)

        prompt = f"Write a short, engaging opening scene for a fantasy adventure. The scene takes place in/near a {location} during {time_setting}. The story should involve a {story_element}. The scene should be atmospheric and invite exploration. Keep it under 100 words."

        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            attention_mask = inputs.ne(self.tokenizer.pad_token_id).long()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            story = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
            story = story.strip()
            if story.endswith(('.', '!', '?')):
                return story, location
            else:
                return story + ".", location

        except Exception as e:
            print(f"Error generating initial story: {e}")
            fallback_stories = [
                (f"As {time_setting} settles over the {location}, you sense adventure in the air. A {story_element} beckons, promising mystery and excitement.", location),
                (f"The {location} looms before you in the {time_setting}. Whispers of a {story_element} draw you forward.", location),
                (f"You stand at the threshold of a {location}, the {time_setting} creating an otherworldly atmosphere. Tales of a {story_element} echo in your mind.", location)
            ]
            return random.choice(fallback_stories)

    def generate_continuation(self, prompt: str, max_length: int = 100) -> str:
        """Generates story continuation using the loaded LLM."""
        if not self.model or not self.tokenizer:
            return "[Story generation model not loaded]"
            
        max_context_len = 512
        context = self.story_context[-max_context_len:]
        full_prompt = f"Location: {self.game_state['location']}. Inventory: {', '.join(self.game_state['inventory'])}. Story so far: ...{context}\n\nContinue the story based on the player's action: {prompt}"

        try:
            inputs = self.tokenizer.encode(full_prompt, return_tensors='pt', max_length=1024, truncation=True)
            attention_mask = inputs.ne(self.tokenizer.pad_token_id).long()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            continuation = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
            continuation = continuation.strip()
            
            # Clean up the continuation
            if not continuation.endswith(('.', '!', '?')):
                continuation += '.'
                
            return continuation

        except Exception as e:
            print(f"Error during text generation: {e}")
            return "[Error generating story continuation]"

    def start_game(self):
        """Sets up the initial game state when starting a new game."""
        print("Generating initial story...")
        initial_story, initial_location = self.generate_initial_story()
        self.story_context = initial_story
        self.game_state["location"] = initial_location

        # Update display stats
        self.display.health = self.game_state['health']
        self.display.stamina = self.game_state['stamina']
        self.display.gold = self.game_state['gold']

        # Set initial background without transition
        self.display.current_location = initial_location
        self.display.current_variation = self.determine_variation(initial_location, 'neutral')
        self.display.current_background_surface = self.display.get_background_surface(
            self.display.current_location, 
            self.display.current_variation
        )
        self.display.is_transitioning = False
        self.display.render_text(self.story_context)
        print(f"Game started. Initial location: {initial_location}")

    def play(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        self.start_game()
        
        while self.running:
            if self.display.show_title_screen:
                if not self.handle_title_screen_events(): break
            elif self.display.show_help:
                if not self.handle_help_screen_events(): break
            else:
                if not self.handle_game_screen_events(): break
            
            self.display.update_transition()
            self.display.draw_background()
            
            if self.display.show_title_screen:
                self.display.draw_title_screen()
            elif self.display.show_help:
                self.display.draw_help_screen()
            else: 
                self.draw_game_screen_ui()
            
            pygame.display.flip()
            clock.tick(60)
        
        print("Exiting game.")
        pygame.quit()
        sys.exit()

    def handle_title_screen_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False; return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.display.show_title_screen = False
                elif event.key == pygame.K_h:
                    self.display.show_title_screen = False
                    self.display.show_help = True
                elif event.key == pygame.K_q:
                    self.running = False; return False
        return True

    def handle_help_screen_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False; return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.display.show_help = False
                    self.display.show_title_screen = True
        return True

    def handle_game_screen_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False; return False
            if event.type == pygame.MOUSEWHEEL:
                if self.display.story_display_rect.collidepoint(pygame.mouse.get_pos()):
                    self.display.scroll_text(event.y)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and self.display.input_text:
                    if not self.process_input(): 
                        return False
                elif event.key == pygame.K_BACKSPACE:
                    self.display.input_text = self.display.input_text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    self.display.show_title_screen = True
                elif event.key == pygame.K_PAGEUP:
                    self.display.scroll_text(-5)
                elif event.key == pygame.K_PAGEDOWN:
                    self.display.scroll_text(5)
                elif event.key == pygame.K_UP:
                    self.display.scroll_text(-1)
                elif event.key == pygame.K_DOWN:
                    self.display.scroll_text(1)
                elif event.unicode.isprintable():
                    self.display.input_text += event.unicode
        return True

    def process_input(self):
        """Handles player input, story generation, and state updates."""
        action = self.display.input_text.strip()
        self.display.input_text = ""
        if not action: return True
            
        print(f"> Player action: {action}")
        if action.lower() == 'exit':
            self.running = False
            return False

        if action.lower() == 'inventory':
            inv_text = "Inventory: " + (", ".join(self.game_state["inventory"]) if self.game_state["inventory"] else "Empty")
            self.story_context += f"\n\n> {action}\n{inv_text}"
            self.display.render_text(self.story_context)
            return True
        if action.lower() == 'status':
            status_text = f"Status: Health {self.game_state['health']}%, Stamina {self.game_state['stamina']}%, Gold {self.game_state['gold']}"
            self.story_context += f"\n\n> {action}\n{status_text}"
            self.display.render_text(self.story_context)
            return True

        continuation = self.generate_continuation(action)
        print(f"< Story Response: {continuation}")
        
        self.story_context += f"\n\n> {action}\n{continuation}"
        self.display.render_text(self.story_context)
        
        self.update_game_state(action, continuation)
        self.update_player_stats(action, continuation)
        
        self.display.health = self.game_state['health']
        self.display.stamina = self.game_state['stamina']
        self.display.gold = self.game_state['gold']
        
        return True

    def update_player_stats(self, action, continuation):
        """Updates health, stamina, gold based on text."""
        action_lower = action.lower()
        continuation_lower = continuation.lower()
        
        # Stamina update
        physical_actions = ['run', 'jump', 'climb', 'fight', 'swim', 'sprint', 'attack', 'dodge']
        stamina_change = 0
        if any(word in action_lower for word in physical_actions):
            stamina_change = -random.randint(5, 15)
        else:
            stamina_change = random.randint(1, 3)
        
        new_stamina = max(0, min(100, self.game_state["stamina"] + stamina_change))
        if new_stamina != self.game_state["stamina"]:
            self.game_state["stamina"] = new_stamina
            print(f"Stamina changed by {stamina_change} -> {new_stamina}%")

        # Health update
        danger_words = ['hurt', 'wound', 'injured', 'damage', 'pain', 'hit', 'attacked', 'fell', 'poisoned', 'burned']
        health_change = 0
        if any(word in continuation_lower for word in danger_words):
            health_change = -random.randint(10, 25)
        elif 'heal' in continuation_lower or 'recover' in continuation_lower or 'potion' in action_lower:
            health_change = random.randint(5, 20)
             
        new_health = max(0, min(100, self.game_state["health"] + health_change))
        if new_health != self.game_state["health"]:
            self.game_state["health"] = new_health
            print(f"Health changed by {health_change} -> {new_health}%")

        # Gold update
        gold_change = 0
        if 'gold' in continuation_lower or 'coins' in continuation_lower or 'treasure' in continuation_lower or 'reward' in continuation_lower:
            gold_change = random.randint(5, 20)
        elif 'paid' in action_lower or 'bought' in action_lower or 'spent' in action_lower:
            gold_change = -random.randint(1, 10)
        
        new_gold = max(0, self.game_state["gold"] + gold_change)
        if new_gold != self.game_state["gold"]:
            self.game_state["gold"] = new_gold
            print(f"Gold changed by {gold_change} -> {new_gold}")

    def draw_game_screen_ui(self):
        """Draws the main game UI elements."""
        self.display.draw_story_text()
        self.display.draw_sidebar(self.game_state)
        
        pygame.draw.rect(self.display.screen, self.display.WHITE, self.display.input_rect, 2)
        text_surface = self.display.font.render(self.display.input_text, True, self.display.WHITE)
        self.display.screen.blit(text_surface, (self.display.input_rect.x + 5, self.display.input_rect.y + 5))

    def determine_variation(self, location, mood):
        """Determines background variation based on location, mood, time."""
        # Specific mood overrides
        if location == 'cave':
            return 'lit' if mood == 'positive' else 'dark'
        if location == 'mountain' and mood == 'negative':
            return 'storm'
        if location == 'castle' and mood == 'negative':
            return 'night'
        if location == 'castle' and random.random() < 0.3:
            return 'interior'
        
        # Time of day determination
        hour = time.localtime().tm_hour
        time_variation = 'day'
        if 6 <= hour < 8:
            time_variation = 'dawn'
        elif 20 <= hour or hour < 6:
            time_variation = 'night'
        
        # Check if the time variation exists for the location
        if time_variation in self.display.backgrounds.get(location, {}):
            return time_variation
        elif 'day' in self.display.backgrounds.get(location, {}):
            return 'day'
        else:
            return list(self.display.backgrounds.get(location, {'day': None}).keys())[0]

    def update_game_state(self, action: str, continuation: str):
        """Updates location, background, inventory based on text."""
        mood = self.analyze_mood(continuation)
        print(f"Detected Mood: {mood}")
        
        current_location = self.game_state["location"]
        new_location = current_location
        
        location_keywords = {
            'forest': ['forest', 'woods', 'trees', 'path'],
            'cave': ['cave', 'cavern', 'underground', 'tunnel', 'dark place'],
            'village': ['village', 'town', 'settlement', 'inn', 'shop', 'market'],
            'mountain': ['mountain', 'peak', 'cliff', 'pass', 'summit', 'ridge'],
            'castle': ['castle', 'fortress', 'palace', 'keep', 'tower', 'hall']
        }
        
        # Check last sentence first for location change, then whole text
        sentences = continuation.split('.')
        relevant_text = sentences[-1] if sentences else ""
        text_lower = relevant_text.lower()
        found_loc_change = False
        for loc, keywords in location_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                if current_location != loc:
                    new_location = loc
                    found_loc_change = True
                    print(f"Location changed based on last sentence: {loc}")
                    break
        
        # If no change in last sentence, check whole continuation + action
        if not found_loc_change:
            text_lower = (action + " " + continuation).lower()
            for loc, keywords in location_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    if current_location != loc:
                        new_location = loc
                        print(f"Location changed based on full text: {loc}")
                        break
        
        # Determine the appropriate background variation
        new_variation = self.determine_variation(new_location, mood)
        
        # Trigger background change in GameDisplay
        self.display.trigger_background_change(new_location, new_variation)
        
        # Update the location in the game state
        self.game_state["location"] = new_location
        
        # Update Inventory
        text_lower = (action + " " + continuation).lower()
        found_items = []
        if "found" in text_lower or "picked up" in text_lower or "acquire" in text_lower or "take" in text_lower or "receive" in text_lower:
            potential_items = ["key", "sword", "shield", "potion", "map", "gem", "coin", "gold", "scroll", "book", "amulet", "ring"]
            for item in potential_items:
                for prefix in ["a ", "an ", "the ", "some ", " "]:
                    if prefix + item in text_lower and item not in self.game_state["inventory"]:
                        found_items.append(item)
                        break
        
        if found_items:
            for item in found_items:
                self.game_state["inventory"].append(item)
                print(f"Added '{item}' to inventory.")

    def analyze_mood(self, text):
        """Analyzes text mood using pipeline or keywords."""
        if self.display.sentiment_analyzer:
            try:
                max_len = 512
                truncated_text = text[-max_len:]
                result = self.display.sentiment_analyzer(truncated_text)
                score = result[0]['score']
                label = result[0]['label']
                if label == 'POSITIVE' and score > 0.7:
                    return 'positive'
                elif label == 'NEGATIVE' and score > 0.6:
                    return 'negative'
                else:
                    return 'neutral'
            except Exception as e:
                print(f"Sentiment analysis error: {e}")
        
        # Fallback keyword analysis
        positive_words = ['happy', 'joy', 'success', 'beautiful', 'light', 'safe', 'found', 'good', 'helpful', 'victory']
        negative_words = ['dark', 'danger', 'fear', 'death', 'evil', 'storm', 'lost', 'bad', 'hurt', 'injured', 'attack', 'scream']
        text_lower = text.lower()
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'

if __name__ == "__main__":
    game = StoryAdventure()
    game.play() 