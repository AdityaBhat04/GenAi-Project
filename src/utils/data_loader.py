# src/utils/data_loader.py
import os
import pandas as pd
from datasets import load_dataset

class DataLoader:
    @staticmethod
    def load_rocstories():
        """Load ROCStories dataset"""
        try:
            # First try to load from Hugging Face
            dataset = load_dataset("roemmele/rocstories")
            return dataset
        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            
            # Fallback to local file if available
            local_path = os.path.join("data", "rocstories", "rocstories.csv")
            if os.path.exists(local_path):
                return pd.read_csv(local_path)
            else:
                raise FileNotFoundError("ROCStories dataset not found")
    
    @staticmethod
    def preprocess_for_gpt2(stories, max_length=512):
        """Process stories for GPT-2 fine-tuning"""
        # Combine the 5 sentences into full stories
        processed_stories = []
        
        for story in stories:
            if isinstance(story, dict):
                # If using Hugging Face dataset
                full_story = " ".join([
                    story.get("sentence1", ""),
                    story.get("sentence2", ""),
                    story.get("sentence3", ""), 
                    story.get("sentence4", ""),
                    story.get("sentence5", "")
                ])
            else:
                # If using pandas dataframe
                full_story = " ".join([
                    story["sentence1"],
                    story["sentence2"],
                    story["sentence3"], 
                    story["sentence4"],
                    story["sentence5"]
                ])
                
            processed_stories.append(full_story)
        
        return processed_stories
    
    @staticmethod
    def prepare_sentiment_data(stories):
        """Extract sentences and assign sentiment labels for training"""
        # This would normally require manual labeling or using another sentiment model
        # For demonstration, we'll use a simple heuristic based on positive/negative words
        
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        
        # Download VADER lexicon if needed
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        
        sia = SentimentIntensityAnalyzer()
        
        sentences = []
        labels = []
        
        for story in stories:
            if isinstance(story, dict):
                story_sentences = [
                    story.get("sentence1", ""),
                    story.get("sentence2", ""),
                    story.get("sentence3", ""), 
                    story.get("sentence4", ""),
                    story.get("sentence5", "")
                ]
            else:
                story_sentences = [
                    story["sentence1"],
                    story["sentence2"],
                    story["sentence3"], 
                    story["sentence4"],
                    story["sentence5"]
                ]
            
            for sentence in story_sentences:
                if sentence.strip():
                    sentiment_score = sia.polarity_scores(sentence)
                    
                    # Convert to simple label: 0=negative, 1=neutral, 2=positive
                    if sentiment_score['compound'] >= 0.05:
                        label = 2  # Positive
                    elif sentiment_score['compound'] <= -0.05:
                        label = 0  # Negative
                    else:
                        label = 1  # Neutral
                    
                    sentences.append(sentence)
                    labels.append(label)
        
        return sentences, labels