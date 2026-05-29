import os
import subprocess

def get_git_content(filepath):
    result = subprocess.run(['git', 'show', f'HEAD:{filepath}'], capture_output=True, text=True)
    return result.stdout

def get_local_content(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def estimate_tokens(text):
    return len(text) // 4

filepath = "tsc/oasis/oasis_persona_gen.py"
old_content = get_git_content(filepath)
new_content = get_local_content(filepath)

def extract_prompt(content, variable_name):
    try:
        start_idx = content.index(f'{variable_name} = """\\') + len(f'{variable_name} = """\\')
        end_idx = content.index('"""', start_idx)
        return content[start_idx:end_idx]
    except ValueError:
        try:
            start_idx = content.index(f'{variable_name} = """') + len(f'{variable_name} = """')
            end_idx = content.index('"""', start_idx)
            return content[start_idx:end_idx]
        except ValueError:
            return ""

old_segment = extract_prompt(old_content, "SEGMENT_INFERENCE_SYSTEM")
old_persona = extract_prompt(old_content, "PERSONA_GEN_SYSTEM")

new_segment = extract_prompt(new_content, "SEGMENT_INFERENCE_SYSTEM")
new_persona = extract_prompt(new_content, "PERSONA_GEN_SYSTEM")

print(f"Old Segment Inference: {len(old_segment)} chars (~{estimate_tokens(old_segment)} tokens)")
print(f"New Segment Inference: {len(new_segment)} chars (~{estimate_tokens(new_segment)} tokens)")
print(f"Old Persona Gen: {len(old_persona)} chars (~{estimate_tokens(old_persona)} tokens)")
print(f"New Persona Gen: {len(new_persona)} chars (~{estimate_tokens(new_persona)} tokens)")
