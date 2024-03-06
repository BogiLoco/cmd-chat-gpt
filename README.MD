# Custom ChatGPT client

This is simple ChatGPT client which can be called through command line.

## Installation

There's Python > 3.9 required in here.

There's need to add OPENAI_API_KEY into .env file.

Once everything's in place try to run below:

```bash
python3 -m venv env
source env/activate/bin
pip install -r requirements.txt
```

## Usage
```bash
python3 app.py
--------------------------------------------------
Started application
Your input: Hey
Hi there! How can I help you today?
--------------------------------------------------
Your input: Tell me a joke
Sure, here's a joke for you:

Why don't scientists trust atoms?

Because they make up everything!
--------------------------------------------------
Your input: exit
Goodbye

```