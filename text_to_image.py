import torch
from diffusers import StableDiffusionPipeline
import pygame
import time
import os
from PIL import Image
import sys

class TextToImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", output_dir="./generated_images"):
        """Initialize the text-to-image generator with Stable Diffusion."""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Check for CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model - use low memory option for CPU
        if self.device == "cpu":
            print("Loading model in CPU mode (will be slower)")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32
            )
        else:
            print("Loading model in GPU mode")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            )
            
        self.pipe.to(self.device)
        print("Model loaded successfully")
        
        # Set safety checker to None if you want to disable it
        # self.pipe.safety_checker = None
        
    def generate_image(self, prompt, negative_prompt=None, guidance_scale=7.5, steps=30, width=512, height=512):
        """Generate an image from a text prompt."""
        try:
            print(f"Generating image for prompt: {prompt}")
            start_time = time.time()
            
            # Generate image
            if negative_prompt:
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    width=width,
                    height=height
                ).images[0]
            else:
                image = self.pipe(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    width=width,
                    height=height
                ).images[0]
            
            # Save the image
            timestamp = int(time.time())
            filename = f"{self.output_dir}/image_{timestamp}.png"
            image.save(filename)
            
            generation_time = time.time() - start_time
            print(f"Image generated in {generation_time:.2f} seconds and saved to {filename}")
            
            return image, filename
            
        except Exception as e:
            print(f"Error generating image: {e}")
            return None, None

class ImageGeneratorApp:
    def __init__(self):
        """Initialize the image generator application."""
        # Initialize pygame
        pygame.init()
        self.width, self.height = 1024, 768
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Text to Image Generator")
        
        # Initialize fonts
        self.title_font = pygame.font.Font(None, 60)
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.LIGHT_BLUE = (173, 216, 230)
        self.GRAY = (128, 128, 128)
        
        # UI elements
        self.input_height = 40
        self.input_text = ""
        self.input_rect = pygame.Rect(50, 100, self.width - 100, self.input_height)
        self.generate_button_rect = pygame.Rect(self.width // 2 - 75, 160, 150, 40)
        
        # Negative prompt
        self.negative_prompt_text = ""
        self.negative_prompt_rect = pygame.Rect(50, 220, self.width - 100, self.input_height)
        self.is_editing_negative = False
        
        # Advanced settings
        self.steps = 30
        self.guidance_scale = 7.5
        self.steps_rect = pygame.Rect(50, 300, 100, 30)
        self.guidance_rect = pygame.Rect(50, 350, 100, 30)
        
        # Image display
        self.image_rect = pygame.Rect(50, 400, 512, 320)
        self.current_image = None
        
        # Status
        self.status_message = "Enter a prompt and click 'Generate'"
        self.loading = False
        self.loading_progress = 0
        
        # Initialize the generator
        self.generator = TextToImageGenerator()
        
    def draw_screen(self):
        """Draw the application UI."""
        self.screen.fill(self.BLACK)
        
        # Draw title
        title = self.title_font.render("Text to Image Generator", True, self.WHITE)
        title_rect = title.get_rect(center=(self.width // 2, 40))
        self.screen.blit(title, title_rect)
        
        # Draw prompt input
        pygame.draw.rect(self.screen, self.WHITE, self.input_rect, 2)
        prompt_label = self.font.render("Prompt:", True, self.WHITE)
        self.screen.blit(prompt_label, (self.input_rect.x, self.input_rect.y - 30))
        
        input_text_surface = self.font.render(self.input_text, True, self.WHITE)
        self.screen.blit(input_text_surface, (self.input_rect.x + 5, self.input_rect.y + 5))
        
        # Draw negative prompt input
        pygame.draw.rect(self.screen, self.WHITE, self.negative_prompt_rect, 2)
        neg_prompt_label = self.font.render("Negative Prompt:", True, self.WHITE)
        self.screen.blit(neg_prompt_label, (self.negative_prompt_rect.x, self.negative_prompt_rect.y - 30))
        
        neg_input_text_surface = self.font.render(self.negative_prompt_text, True, self.WHITE)
        self.screen.blit(neg_input_text_surface, (self.negative_prompt_rect.x + 5, self.negative_prompt_rect.y + 5))
        
        # Draw settings
        settings_label = self.font.render("Settings:", True, self.WHITE)
        self.screen.blit(settings_label, (50, 270))
        
        steps_label = self.small_font.render(f"Steps: {self.steps}", True, self.WHITE)
        self.screen.blit(steps_label, (self.steps_rect.x, self.steps_rect.y))
        pygame.draw.rect(self.screen, self.WHITE, self.steps_rect, 1)
        
        guidance_label = self.small_font.render(f"Guidance: {self.guidance_scale}", True, self.WHITE)
        self.screen.blit(guidance_label, (self.guidance_rect.x, self.guidance_rect.y))
        pygame.draw.rect(self.screen, self.WHITE, self.guidance_rect, 1)
        
        # Draw generate button
        button_color = self.LIGHT_BLUE if not self.loading else self.GRAY
        pygame.draw.rect(self.screen, button_color, self.generate_button_rect)
        pygame.draw.rect(self.screen, self.WHITE, self.generate_button_rect, 2)
        
        button_text = "Generate" if not self.loading else "Generating..."
        button_surface = self.font.render(button_text, True, self.BLACK)
        button_rect = button_surface.get_rect(center=self.generate_button_rect.center)
        self.screen.blit(button_surface, button_rect)
        
        # Draw image area
        pygame.draw.rect(self.screen, self.GRAY, self.image_rect)
        pygame.draw.rect(self.screen, self.WHITE, self.image_rect, 2)
        
        if self.current_image:
            self.screen.blit(self.current_image, self.image_rect)
        else:
            no_image_text = self.font.render("Image will appear here", True, self.WHITE)
            no_image_rect = no_image_text.get_rect(center=self.image_rect.center)
            self.screen.blit(no_image_text, no_image_rect)
        
        # Draw status
        status_text = self.small_font.render(self.status_message, True, self.WHITE)
        self.screen.blit(status_text, (50, self.height - 30))
        
        # Draw loading bar if needed
        if self.loading:
            bar_width = 300
            bar_height = 20
            bar_x = (self.width - bar_width) // 2
            bar_y = 185
            
            pygame.draw.rect(self.screen, self.WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
            fill_width = int((self.loading_progress / 100) * bar_width)
            pygame.draw.rect(self.screen, self.BLUE, (bar_x, bar_y, fill_width, bar_height))
        
    def handle_events(self):
        """Handle user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if generate button was clicked
                if self.generate_button_rect.collidepoint(event.pos) and not self.loading:
                    self.generate_image()
                
                # Check if prompt input was clicked
                elif self.input_rect.collidepoint(event.pos):
                    self.is_editing_negative = False
                
                # Check if negative prompt input was clicked
                elif self.negative_prompt_rect.collidepoint(event.pos):
                    self.is_editing_negative = True
                
                # Check if steps setting was clicked
                elif self.steps_rect.collidepoint(event.pos):
                    self.steps = (self.steps % 50) + 5  # Cycle from 5 to 50
                
                # Check if guidance setting was clicked
                elif self.guidance_rect.collidepoint(event.pos):
                    self.guidance_scale = round((self.guidance_scale + 0.5) % 15, 1)
                    if self.guidance_scale < 1:
                        self.guidance_scale = 1.0
            
            elif event.type == pygame.KEYDOWN:
                # Handle backspace
                if event.key == pygame.K_BACKSPACE:
                    if self.is_editing_negative:
                        self.negative_prompt_text = self.negative_prompt_text[:-1]
                    else:
                        self.input_text = self.input_text[:-1]
                
                # Handle return key for generation
                elif event.key == pygame.K_RETURN and not self.loading:
                    self.generate_image()
                
                # Handle tab key to switch between inputs
                elif event.key == pygame.K_TAB:
                    self.is_editing_negative = not self.is_editing_negative
                
                # Handle printable characters
                elif event.unicode.isprintable():
                    if self.is_editing_negative:
                        self.negative_prompt_text += event.unicode
                    else:
                        self.input_text += event.unicode
        
        return True
    
    def generate_image(self):
        """Generate an image from the current prompt."""
        if not self.input_text.strip():
            self.status_message = "Please enter a prompt first"
            return
        
        self.loading = True
        self.status_message = "Generating image... This may take a while"
        
        # Simulate progress (the actual model doesn't report progress)
        import threading
        
        def run_generation():
            # Actual image generation
            image, filename = self.generator.generate_image(
                prompt=self.input_text,
                negative_prompt=self.negative_prompt_text if self.negative_prompt_text else None,
                guidance_scale=self.guidance_scale,
                steps=self.steps
            )
            
            if image:
                # Convert PIL image to pygame surface
                pygame_image = pygame.image.fromstring(
                    image.resize((self.image_rect.width, self.image_rect.height)).tobytes(),
                    (self.image_rect.width, self.image_rect.height),
                    'RGB'
                )
                self.current_image = pygame_image
                self.status_message = f"Image generated and saved to {filename}"
            else:
                self.status_message = "Error generating image"
            
            self.loading = False
            self.loading_progress = 0
        
        def update_progress():
            progress_steps = self.steps
            for i in range(progress_steps + 1):
                if not self.loading:  # Check if generation completed early
                    break
                self.loading_progress = (i / progress_steps) * 100
                time.sleep(0.1)  # Simulate the step time
        
        # Start generation in a separate thread
        gen_thread = threading.Thread(target=run_generation)
        gen_thread.daemon = True
        gen_thread.start()
        
        # Start progress update in another thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()
    
    def run(self):
        """Main application loop."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            running = self.handle_events()
            self.draw_screen()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = ImageGeneratorApp()
    app.run() 