from pathlib import Path

def load_prompt(path="prompts/trading_prompt.md"):
    return Path(path).read_text()