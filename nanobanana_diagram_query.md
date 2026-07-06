# Nanobanana Diagram Queries

## Opción 1: CLI de nanobanana

```bash
nanobanana architecture "AWS Cloud architecture for Iris Social Media Downloader (ChismesitoGPT). The system has: an Application Load Balancer distributing traffic to Amazon ECS Fargate tasks running a Python Shiny web app with Chrome/Selenium for browser automation. ECS connects to Amazon RDS PostgreSQL for relational data, Amazon OpenSearch Serverless as a vector database for semantic search, and Amazon S3 for static assets and file exports. Secrets are managed via AWS Secrets Manager. CloudWatch handles monitoring and logging. The Shiny app calls multiple external APIs: Google Gemini LLM, OpenAI, YouTube Data API, Twitter API v2, Reddit API, Google Maps API, and Google Play Store. Route 53 provides DNS, CloudFront + WAF provide CDN and security. Include ECR for container registry. Show data flows between all components."
```

## Opción 2: Diagrama técnico detallado con /diagram

```
/diagram AWS architecture for social media scraping analytics app: ALB -> ECS Fargate (Shiny Python + Chrome/Selenium) -> RDS PostgreSQL, OpenSearch Vector DB, S3. Secrets Manager, CloudWatch, Route 53, CloudFront + WAF. External APIs: Gemini, OpenAI, Twitter, YouTube, Reddit, Google Maps. Show data flows. --type=architecture --complexity=comprehensive --style=technical
```

## Opción 3: Prompt para MCP Nano Banana

```
Generate an AWS cloud architecture diagram for the Iris Social Media Downloader application. 

The application is a Python Shiny web app that scrapes social media comments from Twitter/X, YouTube, Facebook, Instagram, Reddit, and Google Maps, analyzes sentiment/emotions/topics using LLMs (Gemini, OpenAI), stores data in PostgreSQL and vector databases, and generates visualizations.

Architecture components:
1. Route 53 DNS → CloudFront CDN + WAF
2. Application Load Balancer (ALB) for HTTPS/WebSocket routing
3. Amazon ECS Fargate running Shiny Python app containers with Chrome/Selenium for browser automation
4. Amazon ECR for Docker image storage
5. Amazon RDS PostgreSQL for relational data (users, comments, analyses)
6. Amazon OpenSearch Serverless (vector engine) for semantic search on embeddings
7. Amazon S3 for static assets, exports (CSV/Excel/PDF), and demo data
8. AWS Secrets Manager for API keys (Gemini, OpenAI, Twitter, YouTube, etc.)
9. Amazon CloudWatch for centralized logging and monitoring
10. External API connections: Google Gemini, OpenAI, YouTube Data API, Twitter API v2, Reddit (PRAW), Google Maps API, Google Play Store, RapidAPI

Style: Clean corporate diagram with blue/green color scheme, show all components with labeled arrows indicating data flow direction. Group components into layers: Networking, Compute, Storage, Security, External. Use AWS icon styling.
```
