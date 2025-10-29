"""
Test suite for the Personal AI Model
Tests all sections of the CV and general question answering capabilities
"""

from ai_model import PersonalAIModel


def test_personal_information():
    """Test questions about personal information"""
    model = PersonalAIModel()
    
    tests = [
        ("Who are you?", "Evangelos Vrailas"),
        ("What is your name?", "Evangelos Vrailas"),
        ("What is your email?", "e.vrailas.dev@gmail.com"),
        ("Where are you located?", "Athens, Greece"),
    ]
    
    print("\n" + "=" * 80)
    print("Testing: Personal Information")
    print("=" * 80)
    
    passed = 0
    for question, expected_substring in tests:
        answer = model.answer(question)
        if expected_substring.lower() in answer.lower():
            print(f"✓ PASS: {question}")
            passed += 1
        else:
            print(f"✗ FAIL: {question}")
            print(f"  Expected substring: {expected_substring}")
            print(f"  Got: {answer}")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_current_work():
    """Test questions about current work"""
    model = PersonalAIModel()
    
    tests = [
        ("What is your current job?", "Full Stack Developer"),
        ("Where do you work?", "Netcompany-Intrasoft"),
        ("What is your current position?", "Full Stack Developer"),
    ]
    
    print("\n" + "=" * 80)
    print("Testing: Current Work")
    print("=" * 80)
    
    passed = 0
    for question, expected_substring in tests:
        answer = model.answer(question)
        if expected_substring.lower() in answer.lower():
            print(f"✓ PASS: {question}")
            passed += 1
        else:
            print(f"✗ FAIL: {question}")
            print(f"  Expected substring: {expected_substring}")
            print(f"  Got: {answer}")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_skills():
    """Test questions about skills"""
    model = PersonalAIModel()
    
    tests = [
        ("What are your programming skills?", "Java"),
        ("What programming languages do you know?", "Java"),
        ("Do you know Spring Boot?", "Spring Boot"),
        ("What databases do you work with?", "MongoDB"),
        ("What experience do you have with Kubernetes?", "Kubernetes"),
    ]
    
    print("\n" + "=" * 80)
    print("Testing: Skills")
    print("=" * 80)
    
    passed = 0
    for question, expected_substring in tests:
        answer = model.answer(question)
        if expected_substring.lower() in answer.lower():
            print(f"✓ PASS: {question}")
            passed += 1
        else:
            print(f"✗ FAIL: {question}")
            print(f"  Expected substring: {expected_substring}")
            print(f"  Got: {answer}")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_experience():
    """Test questions about work experience"""
    model = PersonalAIModel()
    
    tests = [
        ("Tell me about your experience at Public Group", "Public Group"),
        ("What was your previous job?", "Public Group"),
        ("What was your role as a professor assistant?", "professor"),
    ]
    
    print("\n" + "=" * 80)
    print("Testing: Work Experience")
    print("=" * 80)
    
    passed = 0
    for question, expected_substring in tests:
        answer = model.answer(question)
        if expected_substring.lower() in answer.lower():
            print(f"✓ PASS: {question}")
            passed += 1
        else:
            print(f"✗ FAIL: {question}")
            print(f"  Expected substring: {expected_substring}")
            print(f"  Got: {answer}")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_education():
    """Test questions about education"""
    model = PersonalAIModel()
    
    tests = [
        ("Where did you study?", "National and Kapodistrian University of Athens"),
        ("What is your education?", "Bachelor"),
    ]
    
    print("\n" + "=" * 80)
    print("Testing: Education")
    print("=" * 80)
    
    passed = 0
    for question, expected_substring in tests:
        answer = model.answer(question)
        if expected_substring.lower() in answer.lower():
            print(f"✓ PASS: {question}")
            passed += 1
        else:
            print(f"✗ FAIL: {question}")
            print(f"  Expected substring: {expected_substring}")
            print(f"  Got: {answer}")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_general_questions():
    """Test general knowledge questions"""
    model = PersonalAIModel()
    
    tests = [
        ("What is Spring Boot?", "framework"),
        ("What is the capital of Greece?", "Athens"),
        ("What is the capital of France?", "Paris"),
        ("What is 5 + 3?", "8"),
        ("What is 10 - 4?", "6"),
        ("Hello!", "Hello"),
    ]
    
    print("\n" + "=" * 80)
    print("Testing: General Questions")
    print("=" * 80)
    
    passed = 0
    for question, expected_substring in tests:
        answer = model.answer(question)
        if expected_substring.lower() in answer.lower():
            print(f"✓ PASS: {question}")
            passed += 1
        else:
            print(f"✗ FAIL: {question}")
            print(f"  Expected substring: {expected_substring}")
            print(f"  Got: {answer}")
    
    print(f"\nPassed: {passed}/{len(tests)}")
    return passed == len(tests)


def test_cv_summary():
    """Test CV summary generation"""
    model = PersonalAIModel()
    
    print("\n" + "=" * 80)
    print("Testing: CV Summary")
    print("=" * 80)
    
    summary = model.get_cv_summary()
    
    required_elements = [
        "Evangelos Vrailas",
        "Software Developer",
        "Athens, Greece",
        "Java",
        "Spring Boot",
        "e.vrailas.dev@gmail.com"
    ]
    
    passed = 0
    for element in required_elements:
        if element in summary:
            print(f"✓ PASS: Summary contains '{element}'")
            passed += 1
        else:
            print(f"✗ FAIL: Summary missing '{element}'")
    
    print(f"\nPassed: {passed}/{len(required_elements)}")
    return passed == len(required_elements)


def run_all_tests():
    """Run all test suites"""
    print("\n" + "=" * 80)
    print("PERSONAL AI MODEL - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    results = {
        "Personal Information": test_personal_information(),
        "Current Work": test_current_work(),
        "Skills": test_skills(),
        "Work Experience": test_experience(),
        "Education": test_education(),
        "General Questions": test_general_questions(),
        "CV Summary": test_cv_summary(),
    }
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print(f"🎉 ALL TESTS PASSED ({total_passed}/{total_tests})")
    else:
        print(f"⚠️  SOME TESTS FAILED ({total_passed}/{total_tests} passed)")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
