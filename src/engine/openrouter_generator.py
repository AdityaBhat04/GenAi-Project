# src/engine/openrouter_generator.py
import requests
import json
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

class OpenRouterStoryGenerator:
    def __init__(self, api_key=None):
        """Initialize the story generator with OpenRouter API"""
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
            print("No OpenRouter API key provided. Loading local GPT-2 model instead...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
            self.model.eval()
            
            # Required for generation
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Location descriptions for better context
        self.location_descriptions = {
            "desert island": [
                "The hot sun beats down on the sand. Palm trees sway in the ocean breeze.",
                "Waves crash rhythmically against the shore. Exotic birds call from the jungle interior.",
                "The salty air mingles with tropical scents. White sand stretches into crystal clear waters."
            ],
            "haunted mansion": [
                "Dusty furniture sits in the dim light. Portraits seem to follow your every move.",
                "The floorboards creak with each step. Whispers echo from empty rooms.",
                "Cobwebs hang from ornate chandeliers. The air is thick with the scent of decay."
            ],
            "space station": [
                "Blinking control panels line the metal walls. Stars twinkle through viewports.",
                "The hum of life support systems fills the air. Zero gravity creates an eerie weightlessness.",
                "Sealed doors lead to mysterious compartments. Emergency lights cast shadows in maintenance tunnels."
            ],
            "forest": [
                "Ancient trees whisper secrets in the wind. Mysterious lights dance between branches.",
                "The forest path winds deeper into darkness. Moss covers fallen logs and stones.",
                "Sunlight filters through the dense canopy. The sound of running water hints at a nearby stream."
            ],
            "cave": [
                "Crystal formations glitter in the dim light. The cave walls echo with ancient mysteries.",
                "Underground streams flow through dark passages. Stalactites hang like frozen daggers.",
                "The air is cool and damp against your skin. Strange rock formations create eerie shadows."
            ],
            "mountain": [
                "Snow-capped peaks pierce the clouds above. The wind howls between rocky outcroppings.",
                "The mountain path winds treacherously upward. A vast panorama unfolds beneath you.",
                "Ancient ruins cling to the mountainside. The air grows thinner with each step upward."
            ],
            "castle": [
                "Towering spires reach toward the sky. Ancient banners flutter from the battlements.",
                "The castle halls echo with forgotten histories. Suits of armor stand as silent sentinels.",
                "Stained glass windows cast colorful patterns on stone floors. Tapestries depict legendary battles."
            ],
            "dungeon": [
                "Dark corridors stretch into the unknown. Ancient cells hold forgotten secrets.",
                "The air is thick with the scent of damp stone. Rusty chains hang from the walls.",
                "Faint moans echo from deeper passages. Torches cast flickering shadows that play tricks on your mind."
            ],
            "magical realm": [
                "Reality bends in impossible ways here. Floating islands hover in the distance.",
                "Magic flows visibly through the air like gossamer threads. Plants glow with inner light.",
                "The laws of nature hold no power in this realm. Creatures of myth and legend roam freely."
            ]
        }
        
        # Track recent responses to avoid repetition
        self.recent_responses = []
        self.max_recent_responses = 5
        
        # System prompt for OpenRouter API
        self.system_prompt = """You are a creative story generator for an adventure game. Your task is to continue the story based on the player's actions with vivid, creative descriptions. Keep your responses focused on advancing the narrative with rich sensory details. 

Important rules:
1. Keep responses concise (2-3 sentences) and evocative
2. Match the tone and theme of the provided context
3. Describe the consequences of player actions realistically
4. Never break character or mention that you are an AI
5. Don't repeat recent descriptions or narrative patterns
6. Avoid using meta-commentary about the story or game"""
        
    def generate_story_continuation(self, prompt, max_length=100, num_return_sequences=1, 
                                   temperature=0.7, top_p=0.9, repetition_penalty=1.2):
        """Generate continuation of a story based on user input"""
        try:
            # Get location context from the prompt
            location_desc = self._extract_location_context(prompt)
            context_prompt = f"Location: {location_desc}\n{prompt}\n\nWhat happens next?"
            
            if self.use_api:
                # Use OpenRouter API with GPT-4o
                return self._generate_with_openrouter(
                    context_prompt, 
                    max_length,
                    temperature,
                    top_p
                )
            else:
                # Fallback to local GPT-2 model
                return self._generate_with_gpt2(
                    context_prompt,
                    max_length,
                    num_return_sequences,
                    temperature,
                    top_p,
                    repetition_penalty
                )
                
        except Exception as e:
            print(f"Error in story generation: {e}")
            return "As you continue your adventure, a new path reveals itself..."
    
    def _generate_with_openrouter(self, prompt, max_length, temperature, top_p):
        """Generate text using OpenRouter API"""
        try:
            payload = {
                "model": "openai/gpt-4o",
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_length,
                "temperature": temperature,
                "top_p": top_p
            }
            
            # Try up to 3 times in case of API errors
            for attempt in range(3):
                try:
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        generated_text = result["choices"][0]["message"]["content"].strip()
                        
                        # Check for repetition with recent responses
                        if self._is_too_similar_to_recent(generated_text):
                            if attempt == 2:  # Last attempt, return anyway
                                return generated_text
                            else:
                                # Try again with higher temperature
                                payload["temperature"] = min(payload["temperature"] + 0.1, 1.0)
                                continue
                                
                        # Store in recent responses to avoid repetition
                        self._update_recent_responses(generated_text)
                        return generated_text
                    else:
                        print(f"API Error (attempt {attempt+1}): {response.status_code} - {response.text}")
                        if attempt == 2:  # Last attempt
                            return "Your journey continues through this mysterious place..."
                except Exception as e:
                    print(f"API request failed (attempt {attempt+1}): {e}")
                    if attempt == 2:  # Last attempt
                        raise
            
            # If all attempts failed
            return "The adventure unfolds before you, full of possibilities..."
                
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            return "Your path takes an unexpected turn..."
    
    def _generate_with_gpt2(self, prompt, max_length, num_return_sequences, temperature, top_p, repetition_penalty):
        """Generate text using local GPT-2 model as fallback"""
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        with torch.no_grad():
            max_token_length = min(1024, max_length + len(input_ids[0]))
            output_sequences = self.model.generate(
                input_ids,
                max_length=max_token_length,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Extract only the newly generated content (after the prompt)
        prompt_decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        continuation = generated_text[len(prompt_decoded):].strip()
        
        # Store in recent responses to avoid repetition
        self._update_recent_responses(continuation)
        
        return continuation
    
    def _extract_location_context(self, prompt):
        """Extract relevant location description from the prompt"""
        # Try to identify the location from the prompt
        prompt_lower = prompt.lower()
        
        for location, descriptions in self.location_descriptions.items():
            if location in prompt_lower:
                return random.choice(descriptions)
        
        # Default description if no specific location is found
        return "A mysterious place full of possibilities and hidden secrets."
    
    def _update_recent_responses(self, text):
        """Add text to recent responses and maintain max length"""
        self.recent_responses.append(text)
        if len(self.recent_responses) > self.max_recent_responses:
            self.recent_responses.pop(0)
    
    def _is_too_similar_to_recent(self, text):
        """Check if text is too similar to recent responses"""
        for recent in self.recent_responses:
            similarity = self._calculate_similarity(text, recent)
            if similarity > 0.6:  # 60% similarity threshold
                return True
        return False
    
    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0
            
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union 