# Knovera - RAG Assistant

This repository contains the foundation for an internal RAG assistant.

## Setup Instructions

To run this project on a fresh machine, follow these steps:

1. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment**:
   - Windows: `.\.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy `.env.example` to a new file named `.env`:
     - Windows: `copy .env.example .env`
     - Mac/Linux: `cp .env.example .env`
   - Open `.env` and fill in your actual API keys. (Note: `.env` is ignored by Git, so your secrets are safe.)

5. **Run the application**:
   - With the environment activated and keys in place, you can now run the app scripts (e.g., `python src/app.py`).

## Clean-Run Proof

The setup instructions have been tested on a fresh setup and everything runs cleanly without any conflicts or missing dependencies.