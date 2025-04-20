# src/engine/story_generator.py
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re

class StoryGenerator:
    def __init__(self, model_path="gpt2"):
        """Initialize the story generator with GPT-2 model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        print(f"Loading model: {model_path}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        
        # Check if we're using the standard or finetuned model
        self.is_finetuned = model_path != "gpt2"
        
        # Required for generation
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Track previously used responses to prevent repetition
        self.last_used_fallback = None
        self.last_used_index = -1
        self.recent_responses = []  # Store recent responses to avoid repetition
        self.max_recent_responses = 5
        
        # Cave exploration themed words and phrases for validation
        self.cave_themed_words = [
            "cave", "tunnel", "passage", "dark", "mist", "rock", "stone", "echo", "crystal", 
            "stalactite", "stalagmite", "underground", "damp", "wet", "cold", "torch", "light",
            "shadow", "chamber", "cavern", "corridor", "path", "ancient", "ruin", "mysterious",
            "whisper", "rumble", "water", "pool", "stream", "river", "formation", "mineral",
            "wall", "ceiling", "floor", "dust", "boulder", "pebble", "narrow", "wide", "descend",
            "ascend", "climb", "drop", "ledge", "pit", "hole", "entrance", "exit", "darkness"
        ]
        
        # Words that indicate the response is off-topic
        self.irrelevant_words = [
            "television", "TV", "phone", "computer", "internet", "website", "email", "modern",
            "car", "bus", "train", "airplane", "office", "school", "college", "university",
            "girlfriend", "boyfriend", "wife", "husband", "movie", "dating", "party", "club",
            "dance", "bar", "restaurant", "mall", "shop", "store", "supermarket", "grocery",
            "highway", "freeway", "road", "street", "city", "town", "building", "apartment",
            "house", "home", "kitchen", "bedroom", "bathroom", "living room", "dining room",
            "smartphone", "tablet", "laptop", "desktop", "social media", "Facebook", "Twitter"
        ]
        
        # Fallback responses for when generation fails
        self.fallback_responses = [
            "As you move deeper into the cave, the mist swirls around your feet. The air grows cooler and the strange echoes continue. You notice unusual rock formations on the walls, glimmering with an unearthly light. The path ahead splits in two directions.",
            
            "You proceed carefully through the cave. Water drips from the ceiling, creating eerie, melodic sounds. The walls are covered in strange, glowing symbols that seem to pulse with ancient energy. Something scurries across the ground ahead of you.",
            
            "The cave narrows as you continue, forcing you to duck your head. Crystals embedded in the rock ceiling cast a soft blue glow over everything. You hear what sounds like distant whispers coming from somewhere up ahead.",
            
            "Moving forward, you discover a large chamber with a small underground lake in the center. The water is perfectly still, reflecting the stalactites above like a mirror. There's a narrow pathway around the edge of the water.",
            
            "The ground slopes downward as you progress. The air becomes warmer and you can smell something sulfurous. Strange rock formations rise from the floor, shaped almost like statues. You get the feeling you're being watched from the shadows."
        ]
        
    def generate_story_continuation(self, prompt, max_length=100, num_return_sequences=1, 
                              temperature=0.7, top_p=0.9, repetition_penalty=1.2):
        """Generate continuation of a story based on user input"""
        try:
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Standard GPT-2 can handle longer sequences
            if self.is_finetuned:
                # Finetuned model might have been trained with specific parameters
                max_token_length = max_length + len(input_ids[0])
                generation_params = {
                    "input_ids": input_ids,
                    "max_length": max_token_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": True,
                    "num_return_sequences": num_return_sequences,
                    "pad_token_id": self.tokenizer.eos_token_id
                }
            else:
                # Standard model gets enhanced parameters
                max_token_length = min(1024, max_length + len(input_ids[0]))
                generation_params = {
                    "input_ids": input_ids,
                    "max_length": max_token_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": 50,  # Add top_k sampling for better quality
                    "repetition_penalty": repetition_penalty,
                    "do_sample": True,
                    "num_return_sequences": num_return_sequences,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "no_repeat_ngram_size": 3  # Prevent repeating 3-grams
                }
            
            # Generate text with appropriate parameters
            output_sequences = self.model.generate(**generation_params)
            
            # Decode and return generated text
            generated_texts = []
            for sequence in output_sequences:
                text = self.tokenizer.decode(sequence, skip_special_tokens=True)
                # Extract only the newly generated content (after the prompt)
                generated_content = text[len(self.tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
                generated_texts.append(generated_content)
            
            # Return raw generated text without any validation or filtering
            return generated_texts[0] if num_return_sequences == 1 else generated_texts
                
        except Exception as e:
            print(f"Error in story generation: {e}")
            return "Error generating story continuation."
    
    def _get_non_repeating_fallback(self):
        """Get a fallback response that wasn't recently used"""
        import random
        
        # Ensure we don't repeat the last fallback
        if len(self.fallback_responses) > 1:
            available_indices = [i for i in range(len(self.fallback_responses)) if i != self.last_used_index]
            chosen_index = random.choice(available_indices)
            self.last_used_index = chosen_index
            self.last_used_fallback = self.fallback_responses[chosen_index]
            return self.last_used_fallback
        else:
            # If we only have one fallback, use it
            return self.fallback_responses[0]
    
    def _is_valid_response(self, text):
        """Check if the generated text is valid and on-topic"""
        # Clean the text
        text = text.lower()
        
        # If the text is very short, it's probably not good
        if len(text.split()) < 10:
            return False
            
        # Check for irrelevant content
        for word in self.irrelevant_words:
            if word.lower() in text:
                return False
        
        # Check for at least some cave-themed content
        cave_word_count = 0
        for word in self.cave_themed_words:
            if word.lower() in text:
                cave_word_count += 1
        
        # Check for repetition with recent responses
        for recent in self.recent_responses:
            # If there's significant overlap (more than 60% similarity)
            similarity = self._calculate_text_similarity(text, recent)
            if similarity > 0.6:
                return False
                
        # Require at least a few cave words to be present
        return cave_word_count >= 2
    
    def _calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two text snippets"""
        # Simple similarity based on shared words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Jaccard similarity
        if not words1 or not words2:
            return 0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union
    
    def generate_story_branch(self, context, options=3, length_per_option=50):
        """Generate multiple potential story branches from current context"""
        branches = self.generate_story_continuation(
            context, 
            max_length=length_per_option, 
            num_return_sequences=options,
            temperature=0.9  # Higher temperature for more diverse options
        )
        
        return branches if isinstance(branches, list) else [branches]