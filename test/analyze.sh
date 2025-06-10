#!/bin/bash

echo "Testing analyze endpoint..."
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "job_title": "Data Engineer",
    "company": "TechCorp",
    "location": "Brussels",
    "description": "Design and maintain data pipelines for analytics.",
    "requirements": "Python, SQL, 3+ years experience",
    "posted_date": "2025-06-10"
  }'

echo "Testing batch analyze endpoint..."
curl -X POST http://localhost:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{
    "jobs": [
      {
        "job_title": "Data Engineer",
        "company": "TechCorp",
        "location": "Brussels",
        "description": "Design and maintain data pipelines for analytics.",
        "requirements": "Python, SQL, 3+ years experience",
        "posted_date": "2025-06-10"
      },
      {
        "job_title": "Data Scientist",
        "company": "DataCorp",
        "location": "Ghent",
        "description": "Analyze data to extract insights and drive business decisions.",
        "requirements": "Python, R, 5+ years experience",
        "posted_date": "2025-06-10"
      },
      {
        "job_title": "Java Ontwikkelaar",
        "company": "KBC",
        "location": "Brussel",
        "description": "Ontwikkel Java applicaties voor de bank",
        "requirements": "5+ jaren Java ontwikkelaar ervaring, Java 17, Spring Boot, Hibernate, MySQL, Docker, Kubernetes",
        "posted_date": "2025-06-10"
      }
    ]
  }'

