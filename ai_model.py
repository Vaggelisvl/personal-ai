"""
Personal AI Model - Trained on CV data with general question answering capabilities
This model combines retrieval-based QA with generative capabilities for natural responses.
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from cv_knowledge_base import CV_DATA, QA_PAIRS


class PersonalAIModel:
    """
    AI Model that answers questions about Evangelos Vrailas and general queries.
    Uses a hybrid approach: retrieval for CV-specific questions and pattern matching
    for general questions.
    """
    
    def __init__(self):
        self.cv_data = CV_DATA
        self.qa_pairs = QA_PAIRS
        self.context = self._build_context()
        
    def _build_context(self) -> str:
        """Build a comprehensive context from CV data."""
        context_parts = []
        
        # Personal info
        pi = self.cv_data['personal_info']
        context_parts.append(f"Name: {pi['name']}")
        context_parts.append(f"Title: {pi['title']}")
        context_parts.append(f"Location: {pi['location']}")
        context_parts.append(f"Email: {pi['email']}")
        
        # Summary
        context_parts.append(f"\nSummary: {self.cv_data['summary']}")
        
        # Experience
        context_parts.append("\nExperience:")
        for exp in self.cv_data['experience']:
            context_parts.append(f"- {exp['title']} at {exp['company']} ({exp['duration']})")
            for resp in exp['responsibilities']:
                context_parts.append(f"  * {resp}")
        
        # Education
        edu = self.cv_data['education']
        context_parts.append(f"\nEducation: {edu['degree']} from {edu['institution']} ({edu['duration']})")
        
        # Skills
        context_parts.append("\nSkills:")
        for skill_cat, skills in self.cv_data['skills'].items():
            context_parts.append(f"- {skill_cat.replace('_', ' ').title()}: {', '.join(skills)}")
        
        return "\n".join(context_parts)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        return re.sub(r'[^\w\s]', '', text.lower()).strip()
    
    def _calculate_similarity(self, query: str, text: str) -> float:
        """Calculate simple word overlap similarity."""
        query_words = set(self._normalize_text(query).split())
        text_words = set(self._normalize_text(text).split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words & text_words
        return len(intersection) / len(query_words)
    
    def _find_best_qa_match(self, query: str) -> Optional[Tuple[str, float]]:
        """Find the best matching Q&A pair."""
        best_match = None
        best_score = 0.0
        best_question = None
        
        normalized_query = self._normalize_text(query)
        
        for question, answer in self.qa_pairs.items():
            # Longer, more specific questions should score higher
            # Check for substring match first (more accurate for short queries)
            if question in normalized_query:
                # Give higher score to longer matches (more specific)
                score = 0.9 + (len(question) / 100.0)  # Bonus for length
            elif normalized_query in question:
                score = 0.8
            else:
                score = self._calculate_similarity(normalized_query, question)
            
            if score > best_score:
                best_score = score
                best_match = answer
                best_question = question
        
        # Return match if score is above threshold
        if best_score > 0.25:
            return best_match, best_score
        
        return None
    
    def _search_cv_data(self, query: str) -> List[str]:
        """Search CV data for relevant information."""
        results = []
        normalized_query = self._normalize_text(query)
        query_words = set(normalized_query.split())
        
        # Search in experience
        for exp in self.cv_data['experience']:
            exp_text = f"{exp['title']} {exp['company']} {' '.join(exp['responsibilities'])}"
            if self._calculate_similarity(query, exp_text) > 0.2:
                result = f"{exp['title']} at {exp['company']} ({exp['duration']})"
                if exp['responsibilities']:
                    result += ": " + exp['responsibilities'][0]
                results.append(result)
        
        # Search in skills
        for skill_cat, skills in self.cv_data['skills'].items():
            for skill in skills:
                if self._calculate_similarity(query, skill) > 0.3:
                    results.append(f"Skill: {skill}")
        
        # Search in education
        edu = self.cv_data['education']
        if any(word in normalized_query for word in ['education', 'university', 'degree', 'study', 'studied']):
            results.append(f"Education: {edu['degree']} from {edu['institution']} ({edu['duration']})")
        
        return results
    
    def _answer_general_question(self, query: str) -> str:
        """Answer general questions not specific to the CV."""
        normalized = self._normalize_text(query)
        
        # Skip if it's clearly a CV-specific question (but allow tech-specific experience questions)
        cv_keywords = ['your email', 'your contact', 'where do you', 'your name', 'who are you',
                       'your job', 'where did you study', 'your education', 'tell me about you']
        if any(keyword in normalized for keyword in cv_keywords):
            return None
        
        # Greetings
        greetings = ['hello', 'hi', 'hey', 'greetings']
        if any(greet in normalized for greet in greetings) and len(normalized.split()) <= 3:
            return "Hello! I'm an AI assistant trained on Evangelos Vrailas's CV. I can answer questions about his background, skills, experience, and general questions too. How can I help you?"
        
        # Time/Date (but not "do you know")
        time_date_words = ['what time', 'what date', 'what is today', 'what is now']
        if any(word in normalized for word in time_date_words):
            from datetime import datetime
            return f"I don't have access to real-time information, but I can answer questions about Evangelos Vrailas's background and general knowledge."
        
        # Math
        if any(word in normalized for word in ['add', 'sum', 'plus', 'minus', 'multiply', 'divide']) or any(op in query for op in ['+', '-', '*', '/']):
            # Simple math parser
            try:
                import re
                numbers = re.findall(r'\d+', query)
                if len(numbers) >= 2 and '+' in query:
                    result = sum(int(n) for n in numbers)
                    return f"The sum is {result}."
                elif len(numbers) >= 2 and '-' in query:
                    result = int(numbers[0]) - int(numbers[1])
                    return f"The difference is {result}."
                elif len(numbers) >= 2 and ('*' in query or 'x' in normalized or 'multiply' in normalized):
                    result = int(numbers[0]) * int(numbers[1])
                    return f"The product is {result}."
                elif len(numbers) >= 2 and ('/' in query or 'divide' in normalized):
                    result = int(numbers[0]) / int(numbers[1])
                    return f"The result is {result}."
            except:
                pass
        
        # Capital cities (basic knowledge)
        capitals = {
            'greece': 'Athens',
            'france': 'Paris',
            'germany': 'Berlin',
            'italy': 'Rome',
            'spain': 'Madrid',
            'uk': 'London',
            'usa': 'Washington D.C.',
            'japan': 'Tokyo',
            'china': 'Beijing',
            'india': 'New Delhi'
        }
        if 'capital' in normalized:
            for country, capital in capitals.items():
                if country in normalized:
                    return f"The capital of {country.title()} is {capital}."
        
        # What is questions (only for general tech/knowledge, not CV-related)
        if normalized.startswith('what is') or normalized.startswith('what are'):
            if 'programming' in normalized and 'your' not in normalized:
                return "Programming is the process of creating instructions for computers to follow. It involves writing code in programming languages like Java, Python, C++, etc."
            elif 'java' in normalized and 'your' not in normalized:
                return "Java is a high-level, class-based, object-oriented programming language. It's widely used for enterprise applications, Android development, and web services. Evangelos has extensive experience with Java (versions 11, 17, and 21) and Spring Boot."
            elif 'kubernetes' in normalized or 'k8s' in normalized:
                return "Kubernetes (K8s) is an open-source container orchestration platform for automating deployment, scaling, and management of containerized applications. Evangelos has experience deploying and monitoring microservices in Kubernetes clusters."
            elif 'docker' in normalized:
                return "Docker is a platform for developing, shipping, and running applications in containers. Containers package software with all dependencies, making applications portable and consistent across environments."
            elif 'spring boot' in normalized:
                return "Spring Boot is a Java-based framework for building production-ready applications quickly. It simplifies Spring application development with auto-configuration and embedded servers. Evangelos specializes in Spring Boot development."
        
        # How questions
        if normalized.startswith('how'):
            if 'are you' in normalized:
                return "I'm doing great! I'm an AI assistant trained on Evangelos Vrailas's CV data. How can I help you today?"
        
        return None
    
    def answer(self, question: str) -> str:
        """
        Answer a question using the AI model.
        
        Args:
            question: The question to answer
            
        Returns:
            The answer as a string
        """
        if not question or not question.strip():
            return "Please ask me a question!"
        
        # Try to answer general question first (for non-CV questions)
        general_answer = self._answer_general_question(question)
        if general_answer:
            return general_answer
        
        # Then, try to find a direct Q&A match for CV questions
        qa_match = self._find_best_qa_match(question)
        if qa_match:
            return qa_match[0]
        
        # Search CV data
        cv_results = self._search_cv_data(question)
        if cv_results:
            if len(cv_results) == 1:
                return cv_results[0]
            else:
                return "Here's what I found:\n" + "\n".join(f"- {r}" for r in cv_results[:5])
        
        # Fallback response
        return ("I don't have specific information about that. I can answer questions about "
                "Evangelos Vrailas's background, experience, skills, education, and some general questions. "
                "Try asking about his work experience, technical skills, or education!")
    
    def get_cv_summary(self) -> str:
        """Get a summary of the CV."""
        return f"""
{self.cv_data['personal_info']['name']} - {self.cv_data['personal_info']['title']}
Location: {self.cv_data['personal_info']['location']}

{self.cv_data['summary']}

Current Role: {self.cv_data['experience'][0]['title']} at {self.cv_data['experience'][0]['company']}

Key Skills: Java, Spring Boot, C++, Python, ReactJS, Angular, Kubernetes, Docker, RabbitMQ, and more.

Education: {self.cv_data['education']['degree']} from {self.cv_data['education']['institution']}

Contact: {self.cv_data['personal_info']['email']}
"""


def main():
    """Interactive demo of the AI model."""
    model = PersonalAIModel()
    
    print("=" * 80)
    print("Personal AI Model - Trained on Evangelos Vrailas's CV")
    print("=" * 80)
    print("\nThis AI can answer questions about:")
    print("- Evangelos's background, experience, and skills")
    print("- General knowledge questions (basic math, capitals, tech definitions, etc.)")
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
