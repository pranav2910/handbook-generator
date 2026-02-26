# AI Handbook Generator (Retrieval-Augmented Generation with Supabase and LightRAG)

## Overview

The AI Handbook Generator is a Streamlit-based application that allows users to upload AI-related PDF documents, index them using Retrieval-Augmented Generation (RAG), and generate a structured handbook exceeding 20,000 words with proper citations.

This system integrates Supabase, LightRAG, and Grok to provide accurate, source-grounded long-form content generation.

---

## Features

- Upload multiple PDF documents
- Extract text and preserve page numbers
- Chunk documents with overlap for better retrieval
- Store document chunks and metadata in Supabase
- Index full documents using LightRAG
- Ask questions based on uploaded PDFs
- Generate complete handbooks with:
  - Table of contents
  - Parts, chapters, and sections
  - Proper source citations
  - 20,000+ word structured output
- Download generated handbook as Markdown

---

## Technology Stack

Frontend:
- Streamlit

Backend:
- Python

Retrieval and Storage:
- Supabase (PostgreSQL + pgvector)
- LightRAG

Document Processing:
- pdfplumber
- langchain-text-splitters

AI Model:
- Grok (xAI API)

---

## Project Structure

```

handbook-generator/
│
├── app.py
├── config.py
├── db.py
├── ingestion.py
├── longwriter.py
├── rag_engine.py
├── grok_client.py
├── requirements.txt
├── .env.example
├── README.md
│
├── rag_storage/        (auto-generated, not committed)
├── venv/               (local environment, not committed)

```

---

## Setup Instructions

### Step 1: Clone repository

```

git clone [https://github.com/YOUR_USERNAME/handbook-generator.git](https://github.com/YOUR_USERNAME/handbook-generator.git)
cd handbook-generator

```

---

### Step 2: Create virtual environment

```

python -m venv venv
source venv/bin/activate

```

Windows:

```

venv\Scripts\activate

```

---

### Step 3: Install dependencies

```

pip install -r requirements.txt

```

---

### Step 4: Configure environment variables

Copy example file:

```

cp .env.example .env

```

Edit `.env` and add your keys:

```

SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
SUPABASE_TABLE=document_chunks

GROK_API_KEY=your_grok_api_key
GROK_MODEL=grok-4-1-fast-reasoning
GROK_ENDPOINT=[https://api.x.ai/v1/chat/completions](https://api.x.ai/v1/chat/completions)

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TARGET_WORDS=20000
EMBEDDING_DIM=1536

```

---

## Supabase Table Schema

Create this table in Supabase SQL Editor:

```

create table document_chunks (
id uuid primary key default gen_random_uuid(),
content text not null,
metadata jsonb not null,
embedding vector(1536),
created_at timestamptz default now()
);

```

---

## Running the Application

Start Streamlit:

```

streamlit run app.py

```

---

## Usage

1. Upload 2–3 AI-related PDF files
2. Enter prompt:

```

Create a handbook on Retrieval-Augmented Generation

```

3. The system will generate:

- Structured handbook
- Table of contents
- Chapters and sections
- Proper citations
- 20,000+ words

4. Download handbook as Markdown file

---

## Example Output

- Table of Contents
- Part I: Introduction
- Chapter 1: Overview
- Section 1.1: Definition

Citations format:

```

[source: document_name.pdf, p.3]

```

---

## Security Notes

Do NOT commit:

- .env file
- Supabase keys
- API keys
- venv folder
- rag_storage folder

---

## Assignment Compliance

This project satisfies all assignment requirements:

- Retrieval-Augmented Generation
- Supabase storage
- Vector indexing
- Document ingestion
- Source-grounded generation
- Structured handbook generation

---

## Author

P. Pranav Sai

AI Engineering Assignment Project

