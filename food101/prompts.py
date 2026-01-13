# prompts.py
# Simple prompt templates. Keep it stable; no prompt-engineering needed.

TEMPLATES = [
    "a photo of {}",
    "a photo of a plate of {}",
    "a close-up photo of {}",
]

def normalize_classname(name: str) -> str:
    # Food101 uses names like "chicken_curry"
    return name.replace("_", " ").strip()

def build_zeroshot_prompts(classnames):
    prompts = []
    for c in classnames:
        c2 = normalize_classname(c)
        for t in TEMPLATES:
            prompts.append(t.format(c2))
    return prompts

