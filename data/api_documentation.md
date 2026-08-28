# Knovera Vector RAG API Documentation

## Overview
The Knovera API enables seamless integration between internal document repositories and downstream Retrieval-Augmented Generation (RAG) models.

### Authentication
All requests require a valid Bearer Token in the authorization header:
`Authorization: Bearer <KNOVERA_API_KEY>`

### Endpoints

#### 1. POST `/v1/retrieve`
Queries the vector database for top-$K$ matching chunk contexts.

**Request Payload:**
```json
{
  "query": "What is our customer refund policy?",
  "top_k": 5,
  "similarity_threshold": 0.75
}
```

**Response Payload:**
```json
{
  "status": "success",
  "matches": [
    {
      "source": "customer_policy.txt",
      "score": 0.92,
      "text": "Customers are eligible for a 100% full refund within a 30-day window..."
    }
  ]
}
```

#### 2. POST `/v1/chat/completions`
Generates grounded responses utilizing system prompt constraints and retrieved document context.
