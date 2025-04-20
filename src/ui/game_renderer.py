# src/ui/game_renderer.py
import pygame
import pygame.freetype

class GameRenderer:
    def __init__(self, width=800, height=600, title="AI Text Adventure"):
        """Initialize PyGame and set up the game window"""
        pygame.init()
        pygame.display.set_caption(title)
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        
        # Load fonts
        pygame.freetype.init()
        self.main_font = pygame.freetype.SysFont("Arial", 16)
        self.title_font = pygame.freetype.SysFont("Arial", 28, bold=True)
        self.input_font = pygame.freetype.SysFont("Courier New", 16)
        
        # Colors
        self.colors = {
            "bg": (10, 10, 30),
            "text": (220, 220, 220),
            "highlight": (180, 180, 255),
            "input_bg": (30, 30, 50),
            "input_text": (255, 255, 255),
            "prompt": (100, 200, 100),
            "npc_text": (200, 200, 100),
            "player_text": (100, 200, 255),
            "scrollbar": (80, 80, 120),
            "scrollbar_bg": (40, 40, 60)
        }
        
        # UI elements
        self.input_rect = pygame.Rect(20, height - 40, width - 40, 30)
        self.story_rect = pygame.Rect(20, 60, width - 40, height - 120)
        
        # Scrolling
        self.scroll_offset = 0
        self.max_scroll = 0
        self.scroll_speed = 20
        self.scrollbar_width = 12
        self.scrollbar_rect = pygame.Rect(
            self.story_rect.right - self.scrollbar_width,
            self.story_rect.top,
            self.scrollbar_width,
            self.story_rect.height
        )
        self.scrollbar_handle = pygame.Rect(0, 0, self.scrollbar_width, 40)
        self.dragging_scrollbar = False
        
        # State
        self.user_input = ""
        self.input_active = True
        
    def handle_events(self):
        """Handle pygame events like keypresses"""
        events = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return {"type": "QUIT"}
            
            elif event.type == pygame.KEYDOWN:
                if self.input_active:
                    if event.key == pygame.K_RETURN:
                        input_text = self.user_input
                        self.user_input = ""
                        return {"type": "INPUT", "text": input_text}
                    elif event.key == pygame.K_BACKSPACE:
                        self.user_input = self.user_input[:-1]
                    else:
                        self.user_input += event.unicode
                
                # Scroll with arrow keys
                if event.key == pygame.K_UP:
                    self.scroll_offset = max(0, self.scroll_offset - self.scroll_speed)
                elif event.key == pygame.K_DOWN:
                    self.scroll_offset = min(self.max_scroll, self.scroll_offset + self.scroll_speed)
                elif event.key == pygame.K_HOME:
                    self.scroll_offset = 0
                elif event.key == pygame.K_END:
                    self.scroll_offset = self.max_scroll
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.input_rect.collidepoint(event.pos):
                    self.input_active = True
                else:
                    self.input_active = False
                
                # Handle scrolling with mouse wheel
                if self.story_rect.collidepoint(event.pos):
                    if event.button == 4:  # Scroll up
                        self.scroll_offset = max(0, self.scroll_offset - self.scroll_speed)
                    elif event.button == 5:  # Scroll down
                        self.scroll_offset = min(self.max_scroll, self.scroll_offset + self.scroll_speed)
                
                # Handle scrollbar dragging
                if self.scrollbar_handle.collidepoint(event.pos):
                    self.dragging_scrollbar = True
                    self.drag_start_y = event.pos[1]
                    self.drag_start_scroll = self.scroll_offset
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self.dragging_scrollbar = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_scrollbar and self.max_scroll > 0:
                    # Move scrollbar and update scroll position
                    scroll_ratio = self.story_rect.height / self.max_scroll
                    drag_distance = event.pos[1] - self.drag_start_y
                    self.scroll_offset = min(self.max_scroll, max(0, 
                                        self.drag_start_scroll + drag_distance / scroll_ratio))
        
        return None
    
    def render(self, story_text, title="AI Text Adventure", npc_name=None):
        """Render the game screen"""
        # Fill background
        self.screen.fill(self.colors["bg"])
        
        # Draw title
        self.title_font.render_to(
            self.screen, 
            (20, 20), 
            title, 
            self.colors["highlight"]
        )
        
        # Draw NPC name if provided
        if npc_name:
            self.main_font.render_to(
                self.screen,
                (self.width - 200, 30),
                f"Speaking with: {npc_name}",
                self.colors["npc_text"]
            )
        
        # Draw story area
        pygame.draw.rect(self.screen, self.colors["input_bg"], self.story_rect, border_radius=5)
        
        # Calculate text dimensions to set up scrolling
        total_height = self.calculate_text_height(story_text, self.story_rect.width - 2 * 10 - self.scrollbar_width)
        viewport_height = self.story_rect.height - 2 * 10
        
        # Update max scroll
        self.max_scroll = max(0, total_height - viewport_height)
        
        # Setup clipping to story_rect
        clip_rect = self.story_rect.copy()
        clip_rect.inflate_ip(-2, -2)  # Slightly smaller to avoid clipping the border
        
        old_clip = self.screen.get_clip()
        self.screen.set_clip(clip_rect)
        
        # Render story text with word wrapping and scrolling
        self.render_text_with_wrapping(story_text, self.story_rect, scroll_offset=self.scroll_offset)
        
        # Reset clip
        self.screen.set_clip(old_clip)
        
        # Draw scrollbar if needed
        if self.max_scroll > 0:
            # Background
            pygame.draw.rect(
                self.screen, 
                self.colors["scrollbar_bg"], 
                self.scrollbar_rect,
                border_radius=3
            )
            
            # Handle
            scroll_ratio = min(1, self.story_rect.height / total_height)
            handle_height = max(30, scroll_ratio * self.story_rect.height)
            scroll_position = 0
            if self.max_scroll > 0:
                scroll_position = (self.scroll_offset / self.max_scroll) * (self.story_rect.height - handle_height)
            
            self.scrollbar_handle.x = self.scrollbar_rect.x
            self.scrollbar_handle.y = self.story_rect.y + scroll_position
            self.scrollbar_handle.height = handle_height
            
            pygame.draw.rect(
                self.screen, 
                self.colors["scrollbar"], 
                self.scrollbar_handle,
                border_radius=3
            )
        
        # Draw input area
        pygame.draw.rect(
            self.screen, 
            self.colors["input_bg" if not self.input_active else "input_text"], 
            self.input_rect, 
            width=2,
            border_radius=5
        )
        
        # Draw prompt
        self.main_font.render_to(
            self.screen,
            (25, self.height - 38),
            "> ",
            self.colors["prompt"]
        )
        
        # Draw user input
        self.input_font.render_to(
            self.screen,
            (45, self.height - 38),
            self.user_input,
            self.colors["input_text"]
        )
        
        # Draw scroll indicators if needed
        if self.scroll_offset > 0:
            # Up arrow indicator
            points = [(self.width - 25, 70), (self.width - 15, 80), (self.width - 35, 80)]
            pygame.draw.polygon(self.screen, self.colors["highlight"], points)
        
        if self.scroll_offset < self.max_scroll:
            # Down arrow indicator
            points = [(self.width - 25, self.height - 70), (self.width - 15, self.height - 80), (self.width - 35, self.height - 80)]
            pygame.draw.polygon(self.screen, self.colors["highlight"], points)
            
        # Update display
        pygame.display.flip()
        self.clock.tick(30)

    def calculate_text_height(self, text, max_width):
        """Calculate the total height of wrapped text for scrolling"""
        paragraphs = text.split("\n")
        total_height = 0
        line_height = self.main_font.get_sized_height()
        
        for paragraph in paragraphs:
            if paragraph.strip() == "":
                total_height += line_height
                continue
                
            words = paragraph.split(" ")
            line = ""
            
            for word in words:
                test_line = line + word + " "
                bounds = self.main_font.get_rect(test_line)
                
                if bounds.width > max_width:
                    # Line break
                    total_height += line_height
                    line = word + " "
                else:
                    line = test_line
            
            # Add the last line height
            if line:
                total_height += line_height
                
            # Extra space after paragraph
            total_height += line_height * 0.5
        
        return total_height

    def render_text_with_wrapping(self, text, rect, padding=10, scroll_offset=0):
        """Render text with word wrapping inside a rect, with scrolling support"""
        # Split text into paragraphs
        paragraphs = text.split("\n")
        
        x, y = rect.x + padding, rect.y + padding - scroll_offset
        max_width = rect.width - (padding * 2) - self.scrollbar_width
        line_height = self.main_font.get_sized_height()
        
        # Skip rendering if completely above viewport
        skip_height = 0
        
        for paragraph in paragraphs:
            if paragraph.strip() == "":
                y += line_height
                continue
                
            words = paragraph.split(" ")
            line = ""
            
            for word in words:
                test_line = line + word + " "
                bounds = self.main_font.get_rect(test_line)
                
                if bounds.width > max_width:
                    # Only render if within viewport
                    if y + line_height >= rect.y and y <= rect.y + rect.height:
                        self.main_font.render_to(self.screen, (x, y), line, self.colors["text"])
                    
                    y += line_height
                    line = word + " "
                else:
                    line = test_line
            
            # Render the last line of the paragraph
            if line:
                if y + line_height >= rect.y and y <= rect.y + rect.height:
                    self.main_font.render_to(self.screen, (x, y), line, self.colors["text"])
                y += line_height * 1.5  # Add extra space after paragraphs
    
    def quit(self):
        """Quit pygame cleanly"""
        pygame.quit()