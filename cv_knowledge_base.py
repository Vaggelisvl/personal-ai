"""
CV Knowledge Base - Structured information about Evangelos Vrailas
This module contains structured CV data for accurate information retrieval.
"""

CV_DATA = {
    "personal_info": {
        "name": "Evangelos Vrailas",
        "title": "Software Developer",
        "email": "e.vrailas.dev@gmail.com",
        "linkedin": "https://www.linkedin.com/in/evangelos-vrailas-66b31122a/",
        "portfolio": "https://vaggelisvl.github.io/portfolio/",
        "location": "Athens, Greece"
    },
    
    "summary": """I am a Software Engineer specializing in robust, production-ready systems built 
with Java and Spring Boot. I am driven by the principle of using the right tool for 
the job, and my deep background in C/C++ allows me to quickly adapt to new 
technologies like .NET and various frontend frameworks. I seek out projects that 
demand high technical proficiency and have a meaningful purpose.""",
    
    "experience": [
        {
            "title": "Full Stack Developer",
            "company": "Netcompany-Intrasoft",
            "duration": "October 2024 – Present",
            "location": "Athens, Greece",
            "responsibilities": [
                "Developed and maintained backend services using Java 17, Spring Boot, and advanced programming techniques such as reflection and custom annotations alongside with React for Frontend",
                "Built and enhanced .NET-based Case Management Systems, ensuring modularity, maintainability, and testability, utilizing Angular for frontend development",
                "Deployed and monitored microservices in Kubernetes (K8s) clusters, implementing metrics collection, alerting, and system health dashboards",
                "Designed and implemented asynchronous messaging solutions using RabbitMQ, ensuring robust, scalable inter-service communication",
                "Proactively extended involvement in key project phases to verify performance stability and ensure that code changes do not introduce regressions or bottlenecks",
                "Participated in cross-team collaboration and code reviews, promoting clean code practices and system optimization",
                "Regularly communicated in English with international teams and clients, contributing to requirement analysis and daily stand-ups"
            ]
        },
        {
            "title": "Junior Software Developer",
            "company": "Public Group",
            "duration": "2022 - 2024",
            "location": "Athens, Greece",
            "responsibilities": [
                "Developed and maintained enterprise-level applications using Java and Spring Boot",
                "Implemented message brokering solutions with ActiveMQ and RabbitMQ",
                "Collaborated with cross-functional teams to design, develop, and optimize software solutions",
                "Provided comprehensive documentation and technical support, resolving bugs and issues",
                "Leveraged MongoDB, Oracle, and Microsoft SQL Server databases to support application data storage and retrieval"
            ]
        },
        {
            "title": "Computer Science Professor Assistant (subject: Data structures Programming techniques)",
            "company": "National and Kapodistrian University of Athens",
            "duration": "2021 - 2022",
            "location": "Athens, Greece",
            "responsibilities": [
                "Tutored 30–40 students weekly, improving understanding of algorithm design and C development",
                "Reviewed student code and projects, providing individualized technical feedback"
            ]
        }
    ],
    
    "education": {
        "degree": "Bachelor's in Informatics and Telecommunications",
        "institution": "National and Kapodistrian University of Athens",
        "duration": "2019 - 2024",
        "location": "Athens, Greece"
    },
    
    "skills": {
        "languages_frameworks": ["Java (11, 17, 21)", "Spring Boot", "C++", "SQL", "Python"],
        "frontend": ["ReactJS", "Angular"],
        "messaging_integration": ["RabbitMQ", "ActiveMQ"],
        "databases": ["Oracle", "MongoDB", "Microsoft SQL Server", "NoSQL"],
        "devops_tools": ["Docker", "Kubernetes (K8s)", "Git", "CI/CD (Bitbucket Pipelines, Azure DevOps)", "Jira"],
        "development_practices": ["Clean code", "TDD", "Agile (Scrum)", "CI/CD"]
    },
    
    "certifications": [
        {
            "name": "Public Next Graduate Program",
            "description": "Completed a 500-hour in-depth Learning & Development Program with Instructor-Led training, On-The-Job training and e-learning courses provided by Code.Hub and Public Group"
        }
    ]
}

# Question-Answer pairs for common queries
QA_PAIRS = {
    "who are you": "I am Evangelos Vrailas, a Software Developer from Athens, Greece, specializing in robust, production-ready systems built with Java and Spring Boot.",
    "what is your name": "My name is Evangelos Vrailas.",
    "name": "My name is Evangelos Vrailas.",
    "what do you do": "I am a Software Developer specializing in robust, production-ready systems built with Java and Spring Boot. I work with various technologies including .NET, frontend frameworks, and have a deep background in C/C++.",
    "where do you work": "I currently work as a Full Stack Developer at Netcompany-Intrasoft in Athens, Greece, since October 2024.",
    "current job": "I am currently a Full Stack Developer at Netcompany-Intrasoft, where I work on enterprise-grade systems across Java and .NET platforms since October 2024.",
    "current position": "I am currently a Full Stack Developer at Netcompany-Intrasoft, where I work on enterprise-grade systems across Java and .NET platforms since October 2024.",
    "what is your current": "I am currently a Full Stack Developer at Netcompany-Intrasoft, where I work on enterprise-grade systems across Java and .NET platforms since October 2024.",
    "previous job": "Before Netcompany-Intrasoft, I worked as a Junior Software Developer at Public Group from 2022 to 2024.",
    "education": "I have a Bachelor's degree in Informatics and Telecommunications from the National and Kapodistrian University of Athens (2019-2024).",
    "where did you study": "I studied at the National and Kapodistrian University of Athens, where I earned my Bachelor's degree in Informatics and Telecommunications.",
    "skills": "I have expertise in Java (11, 17, 21), Spring Boot, C++, SQL, Python, ReactJS, Angular, RabbitMQ, ActiveMQ, Oracle, MongoDB, Docker, Kubernetes, and more. I follow clean code practices, TDD, and Agile methodologies.",
    "what are your skills": "I have expertise in Java (11, 17, 21), Spring Boot, C++, SQL, Python, ReactJS, Angular, RabbitMQ, ActiveMQ, Oracle, MongoDB, Docker, Kubernetes, and more. I follow clean code practices, TDD, and Agile methodologies.",
    "programming skills": "I am proficient in Java (versions 11, 17, 21), C++, SQL, and Python. I also work with frameworks like Spring Boot, ReactJS, and Angular.",
    "programming languages": "I am proficient in Java (versions 11, 17, 21), C++, SQL, and Python.",
    "languages": "I am proficient in programming languages: Java (versions 11, 17, 21), C++, SQL, and Python.",
    "experience": "I have professional experience since 2021. I've worked as a Full Stack Developer at Netcompany-Intrasoft (2024-present), Junior Software Developer at Public Group (2022-2024), and Computer Science Professor Assistant at National and Kapodistrian University of Athens (2021-2022).",
    "work experience": "I have professional experience since 2021. I've worked as a Full Stack Developer at Netcompany-Intrasoft (2024-present), Junior Software Developer at Public Group (2022-2024), and Computer Science Professor Assistant at National and Kapodistrian University of Athens (2021-2022).",
    "contact": "You can reach me at e.vrailas.dev@gmail.com or connect with me on LinkedIn at https://www.linkedin.com/in/evangelos-vrailas-66b31122a/. You can also check my portfolio at https://vaggelisvl.github.io/portfolio/.",
    "email": "My email is e.vrailas.dev@gmail.com",
    "your email": "My email is e.vrailas.dev@gmail.com",
    "location": "I am based in Athens, Greece.",
    "where are you": "I am based in Athens, Greece.",
    "frameworks": "I work with Spring Boot for backend, and ReactJS and Angular for frontend development.",
    "databases": "I have experience with Oracle, MongoDB, Microsoft SQL Server, and NoSQL databases.",
    "devops": "I work with Docker, Kubernetes (K8s), Git, CI/CD tools like Bitbucket Pipelines and Azure DevOps, and Jira.",
    "kubernetes": "Yes, I have extensive experience with Kubernetes. I deploy and monitor microservices in K8s clusters, implement metrics collection, alerting, and system health dashboards.",
    "kubernetes experience": "Yes, I have extensive experience with Kubernetes. I deploy and monitor microservices in K8s clusters, implement metrics collection, alerting, and system health dashboards.",
    "experience with kubernetes": "Yes, I have extensive experience with Kubernetes. I deploy and monitor microservices in K8s clusters, implement metrics collection, alerting, and system health dashboards.",
    "do you have experience with kubernetes": "Yes, I have extensive experience with Kubernetes. I deploy and monitor microservices in K8s clusters, implement metrics collection, alerting, and system health dashboards.",
    "what experience do you have with kubernetes": "Yes, I have extensive experience with Kubernetes. I deploy and monitor microservices in K8s clusters, implement metrics collection, alerting, and system health dashboards.",
    "k8s": "Yes, I have extensive experience with Kubernetes (K8s). I deploy and monitor microservices in K8s clusters, implement metrics collection, alerting, and system health dashboards.",
    "rabbitmq": "Yes, I have designed and implemented asynchronous messaging solutions using RabbitMQ for robust, scalable inter-service communication.",
    "spring boot": "Yes, I specialize in Spring Boot and have extensive experience developing and maintaining backend services using Spring Boot with Java 11, 17, and 21.",
    "java": "I have extensive experience with Java, particularly versions 11, 17, and 21. I specialize in Spring Boot and have developed enterprise-level applications using Java.",
    "frontend": "I work with ReactJS and Angular for frontend development.",
    "backend": "I specialize in backend development using Java and Spring Boot. I also have experience with .NET for backend systems.",
    "certifications": "I completed the Public Next Graduate Program, a 500-hour in-depth Learning & Development Program with Instructor-Led training, On-The-Job training and e-learning courses provided by Code.Hub and Public Group.",
}
