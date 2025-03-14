# scripts/llm_annotation.py
import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from cad_processing.utils.utils import ensure_dir

def generate_multi_level_instructions(shape_description, llm_tokenizer, llm_model, device):
    """Generate multi-level text instructions using an LLM."""
    # Prepare prompt
    prompt = f"""
    You are a senior CAD engineer and you are tasked to provide natural language instructions to a junior CAD designer for generating a parametric CAD model.
    
    Shape Description: {shape_description}
    
    Please provide four levels of instructions with increasing detail:
    
    1. Abstract level (L0): A simple description of the final shape without technical details.
    2. Beginner level (L1): Basic instructions focusing on the overall approach, suitable for someone with minimal CAD experience.
    3. Intermediate level (L2): More detailed instructions with approximate measurements and specific steps.
    4. Expert level (L3): Precise instructions with exact coordinates, dimensions, and operations.
    
    Format your response as:
    
    Abstract: [abstract level instructions]
    
    Beginner: [beginner level instructions]
    
    Intermediate: [intermediate level instructions]
    
    Expert: [expert level instructions]
    """
    
    # Generate response using the LLM
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(
        inputs.input_ids, 
        max_length=2048, 
        num_return_sequences=1, 
        temperature=0.7
    )
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse the response to extract the different levels
    levels = {}
    
    # Extract sections
    for level in ["Abstract", "Beginner", "Intermediate", "Expert"]:
        pattern = f"{level}: (.*?)(?=(Abstract|Beginner|Intermediate|Expert):|$)"
        import re
        match = re.search(pattern, response, re.DOTALL)
        if match:
            levels[level.lower()] = match.group(1).strip()
        else:
            levels[level.lower()] = f"{level} instructions not found"
    
    return levels

def main(args):
    """Generate multi-level text instructions for CAD models using LLM."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load LLM model
    print("Loading LLM model...")
    llm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    llm_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(device)
    
    # Load shape descriptions
    shape_file = os.path.join(args.input_dir, "shape_descriptions.json")
    
    if not os.path.exists(shape_file):
        raise FileNotFoundError(f"Shape descriptions file not found: {shape_file}")
    
    with open(shape_file, 'r') as f:
        shape_descriptions = json.load(f)
    
    print(f"Loaded shape descriptions for {len(shape_descriptions)} CAD models")
    
    # Limit to max_samples
    if args.max_samples:
        uids = list(shape_descriptions.keys())[:args.max_samples]
        shape_descriptions = {uid: shape_descriptions[uid] for uid in uids}
    
    # Generate multi-level instructions
    all_instructions = {}
    
    for uid, description in tqdm(shape_descriptions.items(), desc="Generating instructions"):
        instructions = generate_multi_level_instructions(description, llm_tokenizer, llm_model, device)
        all_instructions[uid] = instructions
    
    # Save results
    ensure_dir(args.output_dir)
    output_file = os.path.join(args.output_dir, "text_annotations.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_instructions, f, indent=2)
    
    # Also save as CSV for easier inspection
    import pandas as pd
    
    rows = []
    for uid, instructions in all_instructions.items():
        row = {"uid": uid}
        row.update(instructions)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_file = os.path.join(args.output_dir, "text_annotations.csv")
    df.to_csv(csv_file, index=False)
    
    print(f"Generated multi-level instructions for {len(all_instructions)} CAD models")
    print(f"Saved to {output_file} and {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-level text instructions for CAD models using LLM")
    parser.add_argument("--input_dir", required=True, help="Input directory containing shape descriptions")
    parser.add_argument("--output_dir", required=True, help="Output directory for text annotations")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU instead of GPU")
    
    args = parser.parse_args()
    main(args)


