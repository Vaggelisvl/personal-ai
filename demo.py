"""
Demo script showcasing the Personal AI Model capabilities
"""

from ai_model import PersonalAIModel


def run_demo():
    """Run a demonstration of the AI model's capabilities"""
    model = PersonalAIModel()
    
    print("\n" + "=" * 80)
    print(" " * 20 + "PERSONAL AI MODEL DEMONSTRATION")
    print("=" * 80)
    print("\nThis AI model is trained on Evangelos Vrailas's CV and can answer:")
    print("  • Questions about his background, experience, and skills")
    print("  • General knowledge questions (math, capitals, tech definitions)")
    print("=" * 80)
    
    # Demo questions organized by category
    demos = {
        "Personal Information": [
            "Who are you?",
            "What is your email?",
            "Where are you located?",
        ],
        "Current Work & Experience": [
            "What is your current job?",
            "Tell me about your experience at Public Group",
            "What was your role as a professor assistant?",
        ],
        "Technical Skills": [
            "What are your programming skills?",
            "Do you know Spring Boot?",
            "What experience do you have with Kubernetes?",
            "What databases do you work with?",
        ],
        "Education": [
            "Where did you study?",
            "What is your education?",
        ],
        "General Knowledge": [
            "What is Spring Boot?",
            "What is Kubernetes?",
            "What is the capital of Greece?",
            "What is 15 + 27?",
        ],
    }
    
    for category, questions in demos.items():
        print("\n" + "=" * 80)
        print(f"📋 {category}")
        print("=" * 80)
        
        for question in questions:
            print(f"\n❓ {question}")
            answer = model.answer(question)
            print(f"💡 {answer}")
            print("-" * 80)
    
    # Show CV Summary
    print("\n" + "=" * 80)
    print("📄 CV SUMMARY")
    print("=" * 80)
    print(model.get_cv_summary())
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE - Try the interactive mode with: python ai_model.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_demo()
