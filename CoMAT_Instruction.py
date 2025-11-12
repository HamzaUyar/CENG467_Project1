'''
STUB: INSERT CODE HERE:
Create a variable `INSTRUCTION` and fill in the prompt with instructions provided in the file `prompt-instruction.txt.`
'''
import os


def _load_instruction():
    """Load CoMAT instruction prompt from prompt-instruction.txt."""
    # Try local directory first (same folder), then CWD fallback
    candidates = [
        os.path.join(os.path.dirname(__file__), 'prompt-instruction.txt'),
        'prompt-instruction.txt',
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    raise FileNotFoundError('prompt-instruction.txt not found next to CoMAT_Instruction.py or in CWD')


# Public constant imported by other modules
INSTRUCTION = _load_instruction()

# INSTRUCTION = """<CoMAT instructions are placed here>"""
