import pygame
import pygame.freetype
import time

class TextEffects:
    def __init__(self, screen, font, color=(255, 255, 255)):
        """Initialize text effect renderer"""
        self.screen = screen
        self.font = font
        self.color = color
        self.animations = []
    
    def add_typewriter_text(self, text, position, speed=0.05, color=None):
        """Add a typewriter effect to display text gradually"""
        if color is None:
            color = self.color
            
        self.animations.append({
            "type": "typewriter",
            "text": text,
            "position": position,
            "progress": 0,
            "speed": speed,
            "color": color,
            "last_update": time.time()
        })
    
    def add_fade_text(self, text, position, duration=2.0, color=None):
        """Add a fade-in effect for text"""
        if color is None:
            color = self.color
            
        self.animations.append({
            "type": "fade",
            "text": text,
            "position": position,
            "progress": 0,
            "duration": duration,
            "color": color,
            "last_update": time.time()
        })
    
    def render(self):
        """Render all text effects and update their state"""
        completed = []
        
        for i, anim in enumerate(self.animations):
            if anim["type"] == "typewriter":
                self._render_typewriter(anim)
                if anim["progress"] >= len(anim["text"]):
                    completed.append(i)
                    
            elif anim["type"] == "fade":
                self._render_fade(anim)
                if anim["progress"] >= 1.0:
                    completed.append(i)
        
        # Remove completed animations (in reverse order to avoid index issues)
        for i in sorted(completed, reverse=True):
            self.animations.pop(i)
    
    def _render_typewriter(self, anim):
        """Render typewriter animation"""
        current_time = time.time()
        time_diff = current_time - anim["last_update"]
        
        # Update progress based on time and speed
        chars_to_add = int(time_diff / anim["speed"])
        if chars_to_add > 0:
            anim["progress"] += chars_to_add
            anim["last_update"] = current_time
        
        # Clamp progress to text length
        anim["progress"] = min(len(anim["text"]), anim["progress"])
        
        # Render visible portion of text
        visible_text = anim["text"][:anim["progress"]]
        self.font.render_to(self.screen, anim["position"], visible_text, anim["color"])
    
    def _render_fade(self, anim):
        """Render fade animation"""
        current_time = time.time()
        time_diff = current_time - anim["last_update"]
        
        # Update progress based on time
        anim["progress"] += time_diff / anim["duration"]
        anim["last_update"] = current_time
        
        # Clamp progress to [0, 1]
        anim["progress"] = min(1.0, anim["progress"])
        
        # Calculate alpha based on progress
        alpha = int(255 * anim["progress"])
        
        # Create a surface for the text
        text_surf, text_rect = self.font.render(anim["text"], anim["color"])
        text_surf.set_alpha(alpha)
        
        # Blit the text surface
        self.screen.blit(text_surf, anim["position"])