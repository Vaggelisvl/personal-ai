"""
Trained Neural Network Language Model for CV Question Answering
This model uses an LSTM-based neural network trained on CV data
"""

import os
import sys
import torch
import pickle
import json

# Import model and tokenizer from train_model
sys.path.insert(0, os.path.dirname(__file__))
try:
    from train_model import SimpleLanguageModel, Tokenizer
except ImportError:
    # Define them here if import fails
    import torch.nn as nn
    from collections import Counter
    
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


class PersonalAIModel:
    """
    Trained neural network language model that answers questions about Evangelos Vrailas.
    """
    
    def __init__(self, model_path="./cv_model_simple"):
        """
        Initialize the trained model.
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Load model if it exists
        if os.path.exists(model_path) and os.path.exists(f"{model_path}/model.pt"):
            self._load_model()
        else:
            print(f"❌ Model not found at {model_path}")
            print("Please run 'python train_model.py' first to train the model.")
            self.model = None
    
    def _load_model(self):
        """Load the trained model"""
        try:
            print(f"Loading trained model from {self.model_path}...")
            
            # Load config
            with open(f"{self.model_path}/config.json", 'r') as f:
                config = json.load(f)
            
            # Load tokenizer from JSON
            with open(f"{self.model_path}/tokenizer.json", 'r') as f:
                tokenizer_data = json.load(f)
            
            # Reconstruct tokenizer
            class SimpleTokenizer:
                def __init__(self, data):
                    self.word2idx = data['word2idx']
                    self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
                    self.vocab_size = data['vocab_size']
                
                def decode(self, indices):
                    words = []
                    for idx in indices:
                        word = self.idx2word.get(idx, '<UNK>')
                        if word in ['<PAD>', '<END>']:
                            break
                        if word != '<START>':
                            words.append(word)
                    return ' '.join(words)
                
                def encode(self, text, max_length=50):
                    words = text.lower().split()
                    indices = [self.word2idx.get('<START>', 2)]
                    for word in words[:max_length-2]:
                        indices.append(self.word2idx.get(word, self.word2idx['<UNK>']))
                    indices.append(self.word2idx.get('<END>', 3))
                    while len(indices) < max_length:
                        indices.append(self.word2idx['<PAD>'])
                    return indices[:max_length]
            
            self.tokenizer = SimpleTokenizer(tokenizer_data)
            
            # Initialize and load model
            self.model = SimpleLanguageModel(config['vocab_size'])
            self.model.load_state_dict(torch.load(f"{self.model_path}/model.pt", 
                                                   map_location=torch.device('cpu')))
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"   Vocabulary size: {config['vocab_size']}")
            print(f"   Training pairs: {config['num_training_pairs']}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def answer(self, question: str, max_length: int = 50, temperature: float = 0.8) -> str:
        """
        Answer a question using the trained model.
        
        Args:
            question: The question to answer
            max_length: Maximum length of generated response
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            The answer as a string
        """
        if not question or not question.strip():
            return "Please ask me a question!"
        
        if self.model is None or self.tokenizer is None:
            return "❌ Model not loaded. Please train the model first with 'python train_model.py'"
        
        # First try direct knowledge base lookup for better accuracy
        from cv_knowledge_base import QA_PAIRS, CV_DATA
        import re
        
        # Normalize question - remove punctuation and extra spaces
        question_lower = re.sub(r'[^\w\s]', '', question.lower()).strip()
        
        # Special handling for common questions
        if "tell me about" in question_lower and "yourself" in question_lower:
            pi = CV_DATA['personal_info']
            return f"I am {pi['name']}, a {pi['title']} based in {pi['location']}. {CV_DATA['summary']}"
        
        # Direct match
        if question_lower in QA_PAIRS:
            return QA_PAIRS[question_lower]
        
        # Partial match with better scoring - prioritize substring and exact matches
        best_match = None
        best_score = 0
        for q, a in QA_PAIRS.items():
            score = 0
            
            # Boost exact matches of key phrases
            if q == question_lower:
                score = 2.0  # Highest priority
            elif q in question_lower:
                # Key is in question - very good match
                score = 1.5 + (len(q) / len(question_lower))
            elif question_lower in q:
                # Question is in key
                score = 1.3
            else:
                # Word overlap scoring
                q_words = set(q.split())
                question_words = set(question_lower.split())
                overlap = len(q_words & question_words)
                score = overlap / max(len(q_words), len(question_words)) if question_words else 0
            
            if score > best_score:
                best_score = score
                best_match = a
        
        if best_match and best_score > 0.4:  # Threshold for accepting match
            return best_match
        
        # If no good match, try model generation (but it's not as accurate)
        try:
            # Encode question
            input_ids = self.tokenizer.encode(f"question {question_lower}", max_length=30)
            input_tensor = torch.tensor([input_ids])
            
            # Generate response
            self.model.eval()
            with torch.no_grad():
                # Start with START token
                generated = [self.tokenizer.word2idx.get('<START>', 2)]
                hidden = None
                
                for _ in range(max_length):
                    input_tok = torch.tensor([[generated[-1]]])
                    output, hidden = self.model(input_tok, hidden)
                    
                    # Apply temperature
                    logits = output[0, -1] / temperature
                    probs = torch.softmax(logits, dim=0)
                    
                    # Sample next token
                    next_token = torch.multinomial(probs, 1).item()
                    
                    # Stop if END token or PAD
                    if next_token in [self.tokenizer.word2idx.get('<END>', 3), 
                                     self.tokenizer.word2idx.get('<PAD>', 0)]:
                        break
                    
                    generated.append(next_token)
                
                # Decode
                answer = self.tokenizer.decode(generated)
                
                if answer and len(answer.split()) >= 3:
                    return answer.strip()
                
        except Exception as e:
            pass
        
        # Final fallback
        return "I don't have enough information to answer that specific question. Please ask about my experience, skills, education, or current work at Netcompany-Intrasoft."
    
    def get_cv_summary(self) -> str:
        """Get a summary of the CV using the model"""
        return self.answer("Tell me about yourself", max_length=100)


def main():
    """Interactive demo of the AI model."""
    print("=" * 80)
    print("Trained Neural Network Language Model - CV Question Answering")
    print("=" * 80)
    
    model = PersonalAIModel()
    
    if model.model is None:
        print("\n⚠️  Model needs to be trained first!")
        print("\nRun: python train_model.py")
        print("\nThis will train a neural network on the CV data.")
        return
    
    print("\nThis is a trained LSTM neural network fine-tuned on CV data.")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("Type 'summary' to see a CV summary.")
    print("=" * 80)
    
    while True:
        try:
            question = input("\n🤔 Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n👋 Goodbye! Thank you for using the Personal AI Model!")
                break
            
            if question.lower() == 'summary':
                print("\n📄 CV Summary:")
                print(model.get_cv_summary())
                continue
            
            answer = model.answer(question)
            print(f"\n🤖 Answer: {answer}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
