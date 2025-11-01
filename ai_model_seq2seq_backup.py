"""
Real AI Model - Seq2Seq with Attention trained on CV data
This is a genuine neural network that learns to answer questions through training
"""

import torch
import json
import os
import sys

# Import the model architecture
sys.path.insert(0, os.path.dirname(__file__))
from train_seq2seq import Seq2SeqModel, Tokenizer as TrainTokenizer


class Tokenizer:
    """Tokenizer for inference"""
    def __init__(self, word2idx, idx2word, vocab_size):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab_size = vocab_size
    
    def encode(self, text, max_length=50):
        words = text.lower().split()
        indices = [self.word2idx.get('<SOS>', 1)]
        for word in words[:max_length-2]:
            indices.append(self.word2idx.get(word, self.word2idx.get('<UNK>', 3)))
        indices.append(self.word2idx.get('<EOS>', 2))
        return indices
    
    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == self.word2idx.get('<EOS>', 2) or idx == self.word2idx.get('<PAD>', 0):
                break
            if idx != self.word2idx.get('<SOS>', 1):
                word = self.idx2word.get(str(idx), '<UNK>')
                if word not in ['<UNK>', '<PAD>']:
                    words.append(word)
        return ' '.join(words)


class PersonalAIModel:
    """
    Real AI model using Seq2Seq with Attention - trained neural network
    """
    
    def __init__(self, model_path="./cv_seq2seq_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        if os.path.exists(model_path) and os.path.exists(f"{model_path}/model.pt"):
            self._load_model()
        else:
            print(f"❌ Model not found at {model_path}")
            print("Please run 'python train_seq2seq.py' first to train the model.")
    
    def _load_model(self):
        """Load the trained seq2seq model"""
        try:
            print(f"Loading Seq2Seq model from {self.model_path}...")
            
            # Load config
            with open(f"{self.model_path}/config.json", 'r') as f:
                config = json.load(f)
            
            # Load tokenizer
            with open(f"{self.model_path}/tokenizer.json", 'r') as f:
                tok_data = json.load(f)
            
            self.tokenizer = Tokenizer(
                tok_data['word2idx'],
                tok_data['idx2word'],
                tok_data['vocab_size']
            )
            
            # Load model
            self.model = Seq2SeqModel(config['vocab_size'])
            self.model.load_state_dict(torch.load(f"{self.model_path}/model.pt", 
                                                   map_location=torch.device('cpu')))
            self.model.eval()
            
            print(f"✅ Seq2Seq model loaded!")
            print(f"   Architecture: {config.get('architecture', 'Seq2Seq with Attention')}")
            print(f"   Vocabulary: {config['vocab_size']} tokens")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def answer(self, question: str, max_length: int = 60) -> str:
        """
        Answer using the trained neural network
        
        Args:
            question: The question to answer
            max_length: Maximum length of answer
            
        Returns:
            Generated answer from the neural network
        """
        if not question or not question.strip():
            return "Please ask me a question!"
        
        if self.model is None or self.tokenizer is None:
            return "❌ Model not loaded. Please train the model with 'python train_seq2seq.py'"
        
        try:
            # Encode question
            question_enc = self.tokenizer.encode(question, max_length=30)
            src = torch.tensor([question_enc])
            
            # Generate answer
            self.model.eval()
            with torch.no_grad():
                # Encode
                encoder_outputs, hidden, cell = self.model.encoder(src)
                
                # Start decoding
                outputs = [self.tokenizer.word2idx.get('<SOS>', 1)]
                
                for _ in range(max_length):
                    input_token = torch.tensor([[outputs[-1]]])
                    
                    prediction, hidden, cell, _ = self.model.decoder(
                        input_token, hidden, cell, encoder_outputs
                    )
                    
                    # Get best token
                    top_token = prediction.argmax(1).item()
                    outputs.append(top_token)
                    
                    # Stop if EOS
                    if top_token == self.tokenizer.word2idx.get('<EOS>', 2):
                        break
                
                # Decode
                answer = self.tokenizer.decode(outputs)
                
                if answer and len(answer.strip()) > 0:
                    # Capitalize first letter
                    answer = answer.strip()
                    if answer:
                        answer = answer[0].upper() + answer[1:] if len(answer) > 1 else answer.upper()
                    return answer
                else:
                    return "I don't have enough training data to answer that question accurately."
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_cv_summary(self) -> str:
        """Get CV summary"""
        return self.answer("tell me about yourself")


def main():
    """Interactive demo"""
    print("=" * 80)
    print("Real AI Model - Seq2Seq with Attention")
    print("=" * 80)
    
    model = PersonalAIModel()
    
    if model.model is None:
        print("\n⚠️  Model not found!")
        print("\nTrain the model first:")
        print("  python train_seq2seq.py")
        return
    
    print("\nThis is a real trained neural network (Seq2Seq with Attention).")
    print("\nType 'quit' to exit, 'summary' for CV summary.")
    print("=" * 80)
    
    while True:
        try:
            question = input("\n🤔 Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!")
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
