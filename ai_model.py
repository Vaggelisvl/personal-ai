"""
Real AI Model - Fine-tuned GPT-2 on CV data
This is a genuine transformer-based neural network fine-tuned on CV Q&A data
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import json


class PersonalAIModel:
    """
    Real AI model using fine-tuned GPT-2 - transformer-based neural network
    """
    
    def __init__(self, model_path="./cv_gpt2_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        if os.path.exists(model_path) and os.path.exists(f"{model_path}/pytorch_model.bin"):
            self._load_model()
        else:
            print(f"❌ Model not found at {model_path}")
            print("Please run 'python train_gpt2.py' first to train the model.")
            print("Training will take 10-20 minutes.")
    
    def _load_model(self):
        """Load the fine-tuned GPT-2 model"""
        try:
            print(f"Loading fine-tuned GPT-2 model from {self.model_path}...")
            
            # Load tokenizer and model
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            self.model.eval()
            
            # Load training info if available
            info_path = f"{self.model_path}/training_info.json"
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                print(f"✅ GPT-2 model loaded!")
                print(f"   Architecture: Fine-tuned {info.get('model_type', 'GPT-2')}")
                print(f"   Training pairs: {info.get('num_training_pairs', 'N/A')}")
                print(f"   Vocabulary: {info.get('vocab_size', len(self.tokenizer))} tokens")
            else:
                print(f"✅ GPT-2 model loaded!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def answer(self, question: str, max_new_tokens: int = 100) -> str:
        """
        Answer using the fine-tuned GPT-2 model
        
        Args:
            question: The question to answer
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated answer from the neural network
        """
        if not question or not question.strip():
            return "Please ask me a question!"
        
        if self.model is None or self.tokenizer is None:
            return "❌ Model not loaded. Please train the model with 'python train_gpt2.py'"
        
        try:
            # Format input in the same way as training
            prompt = f"Question: {question.strip()}\nAnswer:"
            
            # Encode input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Generate response
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    temperature=0.7,  # Slightly creative but focused
                    top_p=0.9,  # Nucleus sampling
                    top_k=50,  # Top-k sampling
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Avoid repetition
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part
            if "Answer:" in full_response:
                answer = full_response.split("Answer:")[-1].strip()
                
                # Clean up if there's another "Question:" in the response
                if "Question:" in answer:
                    answer = answer.split("Question:")[0].strip()
                
                # Remove any remaining special tokens or artifacts
                answer = answer.replace("<|endoftext|>", "").strip()
                
                if answer and len(answer) > 0:
                    # Ensure answer ends with proper punctuation
                    if answer and answer[-1] not in ['.', '!', '?']:
                        answer += '.'
                    return answer
                else:
                    return "I don't have enough information to answer that question accurately."
            else:
                return "I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating response: {str(e)}"
    
    def get_cv_summary(self) -> str:
        """Get comprehensive CV summary"""
        return self.answer("Tell me about yourself and your background")


def main():
    """Interactive demo"""
    print("=" * 80)
    print("Real AI Model - Fine-tuned GPT-2")
    print("=" * 80)
    
    model = PersonalAIModel()
    
    if model.model is None:
        print("\n⚠️  Model not found!")
        print("\nTrain the model first:")
        print("  python train_gpt2.py")
        print("\nThis will fine-tune a GPT-2 model on your CV data (takes 10-20 minutes).")
        return
    
    print("\nThis is a real fine-tuned transformer model (GPT-2).")
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
