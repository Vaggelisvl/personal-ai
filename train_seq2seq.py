"""
Seq2Seq Neural Network with Attention for CV Question Answering
A real trained AI model (not retrieval-based) that learns to answer questions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import json
import os
import pickle
from collections import Counter
import random


class Encoder(nn.Module):
    """Encoder with LSTM"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    """Attention mechanism"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hidden_dim]
        # encoder_outputs: [batch, seq_len, hidden_dim]
        
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, hidden_dim]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch, seq_len]
        
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    """Decoder with attention"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden, cell, encoder_outputs):
        # x: [batch, 1]
        embedded = self.dropout(self.embedding(x))  # [batch, 1, emb_dim]
        
        # Calculate attention
        attn_weights = self.attention(hidden[-1], encoder_outputs)  # [batch, seq_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch, 1, seq_len]
        
        # Apply attention to encoder outputs
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch, 1, hidden_dim]
        
        # Concatenate embedding and context
        lstm_input = torch.cat((embedded, context), dim=2)  # [batch, 1, emb_dim + hidden_dim]
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))  # [batch, vocab_size]
        
        return prediction, hidden, cell, attn_weights.squeeze(1)


class Seq2SeqModel(nn.Module):
    """Seq2Seq model with attention"""
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size)
        
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First input to decoder is SOS token
        input_token = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs


class Tokenizer:
    """Simple word tokenizer"""
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.vocab_size = 4
        
    def fit(self, texts):
        """Build vocabulary"""
        words = []
        for text in texts:
            words.extend(text.lower().split())
        
        word_counts = Counter(words)
        # Add most common words
        for word, _ in word_counts.most_common(2000):
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text, max_length=50, add_sos_eos=True):
        """Encode text to indices"""
        words = text.lower().split()
        indices = []
        
        if add_sos_eos:
            indices.append(self.word2idx['<SOS>'])
        
        for word in words[:max_length-2]:
            indices.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        
        if add_sos_eos:
            indices.append(self.word2idx['<EOS>'])
        
        return indices
    
    def decode(self, indices):
        """Decode indices to text"""
        words = []
        for idx in indices:
            if idx == self.word2idx['<EOS>'] or idx == self.word2idx['<PAD>']:
                break
            if idx != self.word2idx['<SOS>']:
                word = self.idx2word.get(idx, '<UNK>')
                if word != '<UNK>':
                    words.append(word)
        return ' '.join(words)


def create_training_data():
    """Create Q&A pairs from CV"""
    from cv_knowledge_base import CV_DATA, QA_PAIRS
    
    pairs = []
    
    # Use Q&A pairs
    for q, a in QA_PAIRS.items():
        pairs.append((q, a))
    
    # Add more variations
    pi = CV_DATA['personal_info']
    
    # Personal info variations
    pairs.extend([
        ("who is evangelos vrailas", f"{pi['name']} is a {pi['title']} based in {pi['location']}"),
        ("tell me about evangelos", f"{pi['name']} is a {pi['title']} specializing in Java and Spring Boot"),
        ("evangelos vrailas background", CV_DATA['summary'][:150]),
    ])
    
    # Experience variations
    for exp in CV_DATA['experience']:
        pairs.append((
            f"tell me about {exp['company'].lower()}",
            f"At {exp['company']}, I worked as {exp['title']} from {exp['duration']}. {exp['responsibilities'][0]}"
        ))
        pairs.append((
            f"what did you do at {exp['company'].lower()}",
            exp['responsibilities'][0]
        ))
    
    # Skills variations
    langs = ', '.join(CV_DATA['skills']['languages_frameworks'][:4])
    pairs.append(("what technologies do you use", f"I work with {langs} and more"))
    pairs.append(("technical stack", f"My stack includes {langs}"))
    
    return pairs


def train_model(output_dir="./cv_seq2seq_model", num_epochs=50, batch_size=8):
    """Train the seq2seq model"""
    
    print("=" * 80)
    print("Training Seq2Seq Neural Network with Attention")
    print("=" * 80)
    
    # Create data
    print("\n1. Creating training data...")
    pairs = create_training_data()
    
    # Augment by adding each pair multiple times
    augmented = []
    for q, a in pairs:
        for _ in range(3):  # 3x augmentation (reduced from 5x)
            augmented.append((q, a))
    pairs = augmented
    
    print(f"   Training pairs: {len(pairs)}")
    
    # Build tokenizer
    print("\n2. Building vocabulary...")
    tokenizer = Tokenizer()
    all_texts = []
    for q, a in pairs:
        all_texts.extend([q, a])
    tokenizer.fit(all_texts)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Prepare data
    print("\n3. Preparing training data...")
    train_data = []
    for q, a in pairs:
        q_enc = tokenizer.encode(q, max_length=30)
        a_enc = tokenizer.encode(a, max_length=60)
        train_data.append((q_enc, a_enc))
    
    # Initialize model
    print("\n4. Initializing Seq2Seq model with attention...")
    model = Seq2SeqModel(tokenizer.vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\n5. Training for {num_epochs} epochs...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        random.shuffle(train_data)
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Pad sequences
            src_batch = [torch.tensor(q) for q, _ in batch]
            trg_batch = [torch.tensor(a) for _, a in batch]
            
            src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
            trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=0)
            
            optimizer.zero_grad()
            output = model(src_padded, trg_padded, teacher_forcing_ratio=0.5)
            
            # Calculate loss
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg_padded[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / (len(train_data) / batch_size)
            print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save
    print(f"\n6. Saving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(model.state_dict(), f"{output_dir}/model.pt")
    
    with open(f"{output_dir}/tokenizer.json", 'w') as f:
        json.dump({
            'word2idx': tokenizer.word2idx,
            'idx2word': {str(k): v for k, v in tokenizer.idx2word.items()},
            'vocab_size': tokenizer.vocab_size
        }, f)
    
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump({
            'vocab_size': tokenizer.vocab_size,
            'num_epochs': num_epochs,
            'architecture': 'Seq2Seq with Attention'
        }, f)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nModel: Seq2Seq with Attention")
    print(f"Vocabulary: {tokenizer.vocab_size} tokens")
    print(f"Training pairs: {len(train_data)}")
    
    return output_dir


if __name__ == "__main__":
    train_model()
