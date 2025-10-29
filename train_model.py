"""
Simple Neural Network Language Model trained on CV data
This implementation doesn't require downloading pre-trained models
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import pickle
import os


class SimpleLanguageModel(nn.Module):
    """Simple LSTM-based language model"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(SimpleLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        return output, hidden


class Tokenizer:
    """Simple word-level tokenizer"""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def fit(self, texts):
        """Build vocabulary from texts"""
        words = []
        for text in texts:
            words.extend(text.lower().split())
        
        # Count words and keep most common
        word_counts = Counter(words)
        vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + [word for word, _ in word_counts.most_common(1000)]
        
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(vocab)
        
    def encode(self, text, max_length=50):
        """Convert text to sequence of indices"""
        words = text.lower().split()
        indices = [self.word2idx.get('<START>', 2)]
        for word in words[:max_length-2]:
            indices.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        indices.append(self.word2idx.get('<END>', 3))
        
        # Pad sequence
        while len(indices) < max_length:
            indices.append(self.word2idx['<PAD>'])
            
        return indices[:max_length]
    
    def decode(self, indices):
        """Convert sequence of indices back to text"""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if word in ['<PAD>', '<END>']:
                break
            if word != '<START>':
                words.append(word)
        return ' '.join(words)


def create_training_data():
    """Create training data from CV"""
    from cv_knowledge_base import CV_DATA, QA_PAIRS
    
    training_pairs = []
    
    # Add Q&A pairs
    for question, answer in QA_PAIRS.items():
        training_pairs.append((f"question {question}", answer))
    
    # Add CV information
    pi = CV_DATA['personal_info']
    training_pairs.append(
        ("tell me about yourself", 
         f"I am {pi['name']}, a {pi['title']} based in {pi['location']}. {CV_DATA['summary'][:200]}")
    )
    
    # Add experience
    for exp in CV_DATA['experience']:
        training_pairs.append(
            (f"tell me about {exp['company']}", 
             f"I worked as {exp['title']} at {exp['company']} from {exp['duration']}. {exp['responsibilities'][0]}")
        )
    
    # Add education
    edu = CV_DATA['education']
    training_pairs.append(
        ("what is your education",
         f"I have a {edu['degree']} from {edu['institution']} from {edu['duration']}.")
    )
    
    # Add skills
    for skill_cat, skills in CV_DATA['skills'].items():
        cat_name = skill_cat.replace('_', ' ')
        training_pairs.append(
            (f"what are your {cat_name}",
             f"My {cat_name} include {', '.join(skills[:5])}.")
        )
    
    return training_pairs


def train_model(output_dir="./cv_model_simple", num_epochs=50, batch_size=4):
    """Train a simple neural language model on CV data"""
    
    print("=" * 80)
    print("Training Simple Neural Language Model on CV Data")
    print("=" * 80)
    
    # Create training data
    print("\n1. Creating training data...")
    training_pairs = create_training_data()
    print(f"   Created {len(training_pairs)} training pairs")
    
    # Build tokenizer
    print("\n2. Building vocabulary...")
    tokenizer = Tokenizer()
    all_texts = []
    for q, a in training_pairs:
        all_texts.append(q)
        all_texts.append(a)
    tokenizer.fit(all_texts)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Encode data
    print("\n3. Encoding training data...")
    train_data = []
    for question, answer in training_pairs:
        q_enc = tokenizer.encode(question, max_length=30)
        a_enc = tokenizer.encode(answer, max_length=50)
        train_data.append((q_enc, a_enc))
    
    # Initialize model
    print("\n4. Initializing model...")
    model = SimpleLanguageModel(tokenizer.vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\n5. Training...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Prepare batch
            questions = torch.tensor([q for q, _ in batch])
            answers = torch.tensor([a for _, a in batch])
            
            # Forward pass - use answer as both input and target (shifted)
            optimizer.zero_grad()
            output, _ = model(answers[:, :-1])
            
            # Calculate loss
            loss = criterion(output.reshape(-1, tokenizer.vocab_size), answers[:, 1:].reshape(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / (len(train_data) / batch_size)
            print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    print(f"\n6. Saving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(model.state_dict(), f"{output_dir}/model.pt")
    
    # Save tokenizer as JSON instead of pickle
    tokenizer_data = {
        'word2idx': tokenizer.word2idx,
        'idx2word': {str(k): v for k, v in tokenizer.idx2word.items()},  # Convert int keys to str for JSON
        'vocab_size': tokenizer.vocab_size
    }
    with open(f"{output_dir}/tokenizer.json", 'w') as f:
        json.dump(tokenizer_data, f, indent=2)
    
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump({
            'vocab_size': tokenizer.vocab_size,
            'num_epochs': num_epochs,
            'num_training_pairs': len(training_pairs)
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nModel saved to: {output_dir}")
    print(f"Training pairs: {len(training_pairs)}")
    print(f"Epochs: {num_epochs}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    return output_dir


if __name__ == "__main__":
    train_model()
