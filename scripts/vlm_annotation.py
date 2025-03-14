import os
import json
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import pipeline

def ensure_dir(directory):
    """Ensure a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def render_cad_model(cad_data, output_dir, n_views=6):
    """
    Render multiple views of a CAD model.
    
    This is a simplified placeholder that creates simple visualizations
    of the CAD model from different viewpoints.
    """
    try:
        # Create output directory
        ensure_dir(output_dir)
        
        # Create a basic visualization of the model
        # In a real implementation, you would use a proper CAD renderer
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.grid(True)
        
        # Extract entities from CAD data
        entities = cad_data.get("entities", {})
        sketches = []
        
        # Find sketch entities
        for entity_id, entity in entities.items():
            if entity.get("type") == "Sketch":
                sketches.append(entity)
        
        # Simplified rendering as 2D projections from different angles
        image_paths = []
        for i in range(n_views):
            angle = i * (360 / n_views)
            
            # Clear the plot
            ax.clear()
            ax.set_aspect('equal')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.grid(True)
            
            # Draw a simple representation of the sketches
            # In a real implementation, you would render actual 3D views
            ax.text(0, 0, f"CAD Model - View {i+1}\nAngle: {angle}°",
                    ha='center', va='center', fontsize=12)
            
            # Save the figure
            img_path = os.path.join(output_dir, f"view_{i:02d}.png")
            plt.savefig(img_path)
            image_paths.append(img_path)
        
        plt.close(fig)
        return image_paths
    
    except Exception as e:
        print(f"Error rendering CAD model: {e}")
        return None

def generate_shape_description(image_paths, device):
    """Generate a shape description using a vision-language model via pipeline."""
    if not image_paths:
        return "Unknown shape"
    
    descriptions = []
    
    # Cargar el pipeline de transformers
    pipe = pipeline("image-text-to-text", model="llava-hf/llama3-llava-next-8b-hf", device=0 if device == 'cuda' else -1)
    
    # Definir el mensaje que se pasará al pipeline
    prompt = """
    This is an image of a Computer Aided Design (CAD) model. 
    You are a senior CAD engineer who knows the object name, where and how the CAD model is used. 
    Give an accurate natural language description about the CAD model to a junior CAD designer 
    who can design it from your simple description.
    
    Focus on shape, structure, and geometric features. 
    Do not mention colors, materials, or rendering aspects.
    """
    
    for img_path in image_paths:
        # Cargar imagen
        image = Image.open(img_path).convert("RGB")
        
        # Crear el mensaje de entrada
        messages = [{"role": "user", "content": prompt}]
        
        # Usar el pipeline para obtener la descripción
        output = pipe({"text": prompt, "image": image})  # Pasa tanto el texto como la imagen
        description = output[0]['generated_text']  # Obtener la descripción generada
        
        descriptions.append(description)
    
    # Extraer partes relevantes de las descripciones
    filtered_descriptions = []
    for desc in descriptions:
        # Si el formato de la descripción tiene un prefijo o marcador innecesario, lo eliminamos
        if ":" in desc:
            desc = desc.split(":", 1)[1].strip()
        filtered_descriptions.append(desc)
    
    # Combinar todas las descripciones
    combined_description = " ".join(filtered_descriptions)
    
    return combined_description

def process_cad_file(json_path, output_dir, vlm_processor, vlm_model, device, args):
    """Process a single CAD file to generate shape description."""
    try:
        # Extract UID from the file path
        if args.dataset == "deepcad":
            uid = "/".join(json_path.strip(".json").split("/")[-2:])  # 0003/00003121
        else:
            uid = "/".join(json_path.split("/")[-4:-2])
        
        # Load the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Create render directory
        render_dir = os.path.join(output_dir, "renders", uid)
        ensure_dir(render_dir)
        
        # Render the CAD model
        image_paths = render_cad_model(data, render_dir, n_views=args.n_views)
        
        if not image_paths:
            return None
        
        # Generate shape description
        shape_description = generate_shape_description(image_paths, device)
        
        return uid, shape_description
    
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None

def main(args):
    """Generate shape descriptions for CAD models using VLM."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load VLM model
    print("Loading VLM model...")
    
    # Load split file
    with open(args.split_json, 'r') as f:
        data = json.load(f)
    
    # Get UIDs for the specified split
    all_uids = data.get(args.split, [])
    
    # Limit to max_samples
    if args.max_samples:
        all_uids = all_uids[:args.max_samples]
    
    print(f"Processing {len(all_uids)} CAD models from '{args.split}' split")
    
    # Process files
    results = {}
    
    for uid in tqdm(all_uids, desc="Generating shape descriptions"):
        json_path = os.path.join(args.input_dir, f"{uid}.json")
        
        if not os.path.exists(json_path):
            print(f"File not found: {json_path}")
            continue
        
        result = process_cad_file(json_path, args.output_dir, None, None, device, args)
        
        if result:
            uid, description = result
            results[uid] = description
    
    # Save results
    output_file = os.path.join(args.output_dir, "shape_descriptions.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Generated shape descriptions for {len(results)} CAD models")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate shape descriptions for CAD models using VLM")
    parser.add_argument("--input_dir", required=True, help="Input directory containing CAD JSON files")
    parser.add_argument("--output_dir", required=True, help="Output directory for rendered images and descriptions")
    parser.add_argument("--split_json", required=True, help="Path to train/val/test split JSON file")
    parser.add_argument("--split", default="train", help="Dataset split to process (train, test, validation)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--n_views", type=int, default=6, help="Number of views to render")
    parser.add_argument("--dataset", choices=["deepcad", "fusion360"], default="deepcad", help="Dataset type")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU instead of GPU")
    
    args = parser.parse_args()
    main(args)
