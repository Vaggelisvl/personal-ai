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
    """Create comprehensive Q&A pairs from CV for recruiter and technical questions"""
    from cv_knowledge_base import CV_DATA, QA_PAIRS
    
    pairs = []
    
    # Use Q&A pairs from knowledge base
    for q, a in QA_PAIRS.items():
        pairs.append((q, a))
    
    pi = CV_DATA['personal_info']
    
    # === GREETINGS AND CONVERSATIONAL (for recruiters) ===
    greetings = [
        ("hi", "Hello! I'm happy to discuss my background and experience with you."),
        ("hello", "Hello! I'm Evangelos Vrailas, a Software Developer. How can I help you?"),
        ("hey", "Hey! I'm Evangelos Vrailas. What would you like to know about my experience?"),
        ("good morning", "Good morning! I'm Evangelos Vrailas, happy to discuss my qualifications."),
        ("good afternoon", "Good afternoon! I'm Evangelos Vrailas. What can I tell you about my experience?"),
        ("how are you", "I'm doing great! I'm excited to discuss my background in software development."),
        ("how are you doing", "I'm doing well! I'm a Software Developer specializing in Java and Spring Boot."),
        ("nice to meet you", "Nice to meet you too! I'm Evangelos Vrailas, a Full Stack Developer."),
        ("tell me about yourself", f"I'm {pi['name']}, a {pi['title']} based in {pi['location']}. I specialize in building robust, production-ready systems with Java and Spring Boot. I have experience across full-stack development and am currently working at Netcompany-Intrasoft."),
        ("introduce yourself", f"I'm {pi['name']}, a passionate Software Developer with expertise in Java, Spring Boot, and full-stack development. I'm based in {pi['location']} and currently work at Netcompany-Intrasoft."),
    ]
    pairs.extend(greetings)
    
    # === RECRUITER QUESTIONS ===
    recruiter_questions = [
        ("why should we hire you", "I bring strong technical skills in Java and Spring Boot, proven experience building enterprise systems, and a track record of delivering production-ready code. I follow clean code practices, TDD, and Agile methodologies."),
        ("why should i hire you", "I have deep expertise in Java development, experience with modern cloud technologies like Kubernetes and Docker, and a strong foundation in building scalable systems. I'm passionate about writing high-quality, maintainable code."),
        ("what makes you a good fit", "My combination of Java expertise, full-stack development skills, and experience with enterprise systems makes me well-suited for challenging projects. I adapt quickly to new technologies and focus on delivering business value."),
        ("what are your strengths", "My strengths include deep Java and Spring Boot expertise, strong problem-solving skills, ability to learn new technologies quickly, and commitment to clean code and best practices like TDD and Agile."),
        ("what is your greatest strength", "My greatest strength is my ability to build robust, production-ready systems using Java and Spring Boot while maintaining high code quality through TDD and clean code practices."),
        ("tell me about your experience", "I have professional software development experience since 2021. I've worked as a Full Stack Developer at Netcompany-Intrasoft, Junior Software Developer at Public Group, and as a Computer Science Professor Assistant. I specialize in Java, Spring Boot, and enterprise systems."),
        ("what is your experience", "I have experience building enterprise-grade systems with Java and .NET platforms, developing microservices with Kubernetes, and working with modern frameworks like Spring Boot, ReactJS, and Angular."),
        ("describe your work history", "I started as a Computer Science Professor Assistant in 2021, then became a Junior Software Developer at Public Group in 2022, and I'm currently a Full Stack Developer at Netcompany-Intrasoft since October 2024."),
        ("what motivates you", "I'm motivated by building high-quality software that solves real problems. I enjoy working on technically challenging projects and using the right tools for each job."),
        ("why did you choose software development", "I'm passionate about building robust, production-ready systems and solving complex problems through code. My background in C/C++ gives me a strong foundation that helps me adapt to any technology."),
        ("where do you see yourself in 5 years", "I see myself as a senior engineer working on technically challenging projects, mentoring junior developers, and continuing to deepen my expertise in distributed systems and cloud technologies."),
        ("what are your career goals", "My goal is to continue growing as a software engineer, taking on more complex technical challenges, and eventually leading architecture decisions for enterprise systems."),
        ("are you a team player", "Yes, I work well in teams. I've collaborated with cross-functional teams at Netcompany-Intrasoft and Public Group, and I value knowledge sharing and pair programming."),
        ("can you work under pressure", "Yes, I handle pressure well. I've delivered projects with tight deadlines and maintain code quality even under time constraints by following established best practices."),
        ("when can you start", "I can discuss start dates once we align on the role and responsibilities. I'm committed to professional transitions."),
    ]
    pairs.extend(recruiter_questions)
    
    # === TECHNICAL QUESTIONS (for technical interviewers) ===
    technical_questions = [
        ("do you know java", "Yes, I have extensive experience with Java, particularly versions 11, 17, and 21. I use Java daily for building enterprise applications with Spring Boot."),
        ("are you familiar with java", "Yes, Java is my primary programming language. I work with Java 11, 17, and 21, and specialize in Spring Boot for building production-ready applications."),
        ("what java versions do you know", "I have experience with Java 11, 17, and 21. I work with modern Java features and best practices."),
        ("do you know spring boot", "Yes, Spring Boot is my main framework. I use it to build robust, production-ready systems with microservices architecture."),
        ("tell me about your spring boot experience", "I have extensive Spring Boot experience building enterprise applications, RESTful APIs, and microservices. I use Spring Boot for backend development in my current role."),
        ("do you know kubernetes", "Yes, I have hands-on experience with Kubernetes. I deploy and monitor microservices in K8s clusters, implement metrics collection, and manage system health."),
        ("what is your kubernetes experience", "I deploy microservices to Kubernetes clusters, set up monitoring and alerting, and manage containerized applications. I work with K8s regularly."),
        ("do you know docker", "Yes, I use Docker for containerization. I create Docker images, manage containers, and work with Docker in CI/CD pipelines."),
        ("what databases do you know", "I have experience with Oracle, MongoDB, Microsoft SQL Server, and NoSQL databases. I can work with both relational and non-relational databases."),
        ("do you know reactjs", "Yes, I use ReactJS for frontend development. I build modern web applications with React and integrate them with backend services."),
        ("do you know angular", "Yes, I have experience with Angular for frontend development. I use it to build enterprise web applications."),
        ("what is your frontend experience", "I work with ReactJS and Angular for frontend development, building modern web applications that integrate with backend services."),
        ("do you know python", "Yes, I'm proficient in Python. I use it for scripting, automation, and backend development when appropriate."),
        ("what about c plus plus", "Yes, I have a deep background in C/C++. This foundation helps me understand low-level concepts and adapt quickly to new technologies."),
        ("do you know c plus plus", "Yes, C++ is one of my core languages. My background in C/C++ gives me strong fundamentals in memory management and system programming."),
        ("tell me about your technical skills", "I'm proficient in Java, Spring Boot, C++, SQL, and Python. I work with ReactJS and Angular for frontend, use Docker and Kubernetes for deployment, and follow TDD and Agile practices."),
        ("what technologies do you use", "I work with Java, Spring Boot, ReactJS, Angular, Docker, Kubernetes, Oracle, MongoDB, and various DevOps tools like Git and CI/CD pipelines."),
        ("what is your tech stack", "My stack includes Java 11/17/21, Spring Boot, ReactJS, Angular, Docker, Kubernetes, Oracle, MongoDB, and modern DevOps tools."),
        ("do you follow best practices", "Yes, I follow clean code principles, Test-Driven Development (TDD), and Agile methodologies. I prioritize code quality and maintainability."),
        ("do you write tests", "Yes, I practice Test-Driven Development (TDD). I write unit tests, integration tests, and believe in comprehensive test coverage."),
        ("what is your approach to coding", "I follow clean code principles, write maintainable code, practice TDD, and believe in using the right tool for the job. Code quality is important to me."),
        ("do you know agile", "Yes, I work in Agile environments using Scrum methodology. I'm comfortable with sprints, stand-ups, and iterative development."),
        ("have you worked with microservices", "Yes, I have experience building and deploying microservices. I use Spring Boot for service development and Kubernetes for orchestration."),
        ("what about devops", "I use Docker for containerization, Kubernetes for orchestration, and work with CI/CD tools like Bitbucket Pipelines and Azure DevOps for automated deployments."),
        ("do you know git", "Yes, I use Git for version control daily. I'm familiar with branching strategies, pull requests, and collaborative development workflows."),
    ]
    pairs.extend(technical_questions)
    
    # === SPECIFIC CV INFORMATION ===
    # Personal info variations
    pairs.extend([
        ("who is evangelos vrailas", f"{pi['name']} is a {pi['title']} based in {pi['location']} specializing in Java and Spring Boot"),
        ("who is vaggelis", f"{pi['name']} is a Software Developer based in Athens, Greece"),
        ("who are you", f"I'm {pi['name']}, a {pi['title']} based in {pi['location']}"),
        ("what is your background", f"I'm a Software Developer with expertise in Java, Spring Boot, and full-stack development. I have a Bachelor's in Informatics from the National and Kapodistrian University of Athens."),
    ])
    
    # Experience variations
    for exp in CV_DATA['experience']:
        pairs.append((
            f"tell me about {exp['company'].lower()}",
            f"At {exp['company']}, I worked as {exp['title']} from {exp['duration']}. {exp['responsibilities'][0]}"
        ))
        pairs.append((
            f"what did you do at {exp['company'].lower()}",
            f"As {exp['title']}, {exp['responsibilities'][0]}"
        ))
        pairs.append((
            f"your experience at {exp['company'].lower()}",
            f"I was {exp['title']} at {exp['company']} from {exp['duration']}, where {exp['responsibilities'][0]}"
        ))
    
    # Skills variations
    langs = ', '.join(CV_DATA['skills']['languages_frameworks'][:5])
    pairs.extend([
        ("what technologies do you use", f"I work with {langs} and more"),
        ("technical stack", f"My stack includes {langs}"),
        ("what tools do you use", "I use Java, Spring Boot, ReactJS, Angular, Docker, Kubernetes, Git, and various CI/CD tools"),
    ])
    
    # Education
    edu = CV_DATA['education']
    pairs.extend([
        ("what is your education", f"I have a {edu['degree']} from {edu['institution']} ({edu['duration']})"),
        ("where did you study", f"I studied at {edu['institution']} where I earned my {edu['degree']}"),
        ("what degree do you have", f"I have a {edu['degree']} from {edu['institution']}"),
        ("your educational background", f"{edu['degree']} from {edu['institution']}, graduated {edu['duration']}"),
    ])
    
    return pairs
    return pairs


def train_model(output_dir="./cv_seq2seq_model", num_epochs=40, batch_size=16):
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
        for _ in range(2):  # 2x augmentation for faster training
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
