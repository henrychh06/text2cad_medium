import os
import json
import csv
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    """Carga el modelo Qwen2.5-72B-Instruct y el tokenizador."""
    print(f"Cargando modelo {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, system_message="You are Qwen, a helpful CAD design expert."):
    """Genera una respuesta utilizando el modelo para un prompt dado."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,  # Aumentamos para generar respuestas más largas
        temperature=0.7,      # Añadimos algo de creatividad
        do_sample=True        # Habilitamos muestreo para diversidad
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def load_minimal_json(json_path):
    """Carga un archivo JSON minimal."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_abstract_prompt(json_data):
    """
    Crea un prompt para generar una descripción abstracta basada en los metadatos del JSON.
    Esta función simula lo que haría un VLM con imágenes.
    """
    prompt = """
    You are a senior CAD engineer. Based on the following CAD metadata, provide a brief abstract description (Level 0) of the shape.
    Focus on describing what the overall shape looks like in simple terms that a non-technical person would understand.
    Use 1-2 sentences only.
    
    CAD metadata:
    """
    
    # Extraemos información útil del JSON minimal
    if "final_shape" in json_data:
        prompt += f"\nFinal shape description: {json_data['final_shape']}"
    
    if "parts" in json_data:
        for part_id, part_info in json_data["parts"].items():
            prompt += f"\n\nPart {part_id}:"
            
            if "coordinate_system" in part_info:
                coords = part_info["coordinate_system"]
                prompt += f"\n- Coordinate system: Euler angles {coords.get('Euler Angles', 'N/A')}, Translation {coords.get('Translation Vector', 'N/A')}"
            
            if "sketch" in part_info:
                sketch = part_info["sketch"]
                prompt += "\n- Sketch contains: "
                for face_id, face in sketch.items():
                    for loop_id, loop in face.items():
                        for primitive_id, primitive in loop.items():
                            if "Circle" in primitive_id:
                                prompt += f"Circle (Center: {primitive.get('Center', 'N/A')}, Radius: {primitive.get('Radius', 'N/A')}), "
                            elif "Line" in primitive_id:
                                prompt += f"Line (Start: {primitive.get('Start', 'N/A')}, End: {primitive.get('End', 'N/A')}), "
                            elif "Arc" in primitive_id:
                                prompt += f"Arc, "
            
            if "extrusion" in part_info:
                extrusion = part_info["extrusion"]
                prompt += f"\n- Extrusion: Depth towards normal = {extrusion.get('extrude_depth_towards_normal', 'N/A')}, "
                prompt += f"Depth opposite normal = {extrusion.get('extrude_depth_opposite_normal', 'N/A')}, "
                prompt += f"Scale = {extrusion.get('sketch_scale', 'N/A')}, "
                prompt += f"Operation = {extrusion.get('operation', 'N/A')}"
            
            if "description" in part_info:
                desc = part_info["description"]
                prompt += f"\n- Description: {desc.get('shape', '')}"
                if "length" in desc and "width" in desc and "height" in desc:
                    prompt += f"\n- Dimensions: length={desc.get('length', 'N/A')}, width={desc.get('width', 'N/A')}, height={desc.get('height', 'N/A')}"
    
    prompt += """
    
    Respond ONLY with a simple, concise description of what this CAD model looks like visually.
    Do not include technical terms, parameters, or dimensions in your response.
    """
    
    return prompt

def create_beginner_prompt(abstract_description, json_data):
    """Crea un prompt para generar instrucciones de nivel principiante."""
    prompt = f"""
    You are a senior CAD engineer. Based on the following information, provide simplified step-by-step instructions (Level 1) for creating this CAD model.
    
    Abstract shape description: {abstract_description}
    
    CAD metadata overview:
    """
    
    # Añadimos información resumida del JSON
    if "parts" in json_data:
        n_parts = len(json_data["parts"])
        prompt += f"\nThe CAD model consists of {n_parts} part(s)."
        
        for part_id, part_info in json_data["parts"].items():
            primitives = []
            if "sketch" in part_info:
                for face in part_info["sketch"].values():
                    for loop in face.values():
                        primitives.extend(list(loop.keys()))
            
            primitive_counts = {}
            for p in primitives:
                p_type = p.split("_")[0]
                primitive_counts[p_type] = primitive_counts.get(p_type, 0) + 1
            
            primitive_summary = ", ".join([f"{count} {p_type}(s)" for p_type, count in primitive_counts.items()])
            
            extrusion_info = ""
            if "extrusion" in part_info:
                extrusion = part_info["extrusion"]
                extrusion_info = f"extruded with depth {extrusion.get('extrude_depth_towards_normal', 0)} units"
    
            prompt += f"\nPart {part_id} contains {primitive_summary} and is {extrusion_info}."
    
    prompt += """
    
    Using this information, write beginner-friendly instructions for creating this CAD model.
    Your instructions should:
    1. Use simple language without technical jargon
    2. Provide a general overview of the steps (no exact coordinates or parameters)
    3. Focus on the basic shape and structure
    4. Keep the explanation under 5-6 sentences
    5. Explain the process in a way that a CAD beginner could understand
    """
    
    return prompt

def create_intermediate_prompt(abstract_description, beginner_description, json_data):
    """Crea un prompt para generar instrucciones de nivel intermedio."""
    prompt = f"""
    You are a senior CAD engineer. Based on the following information, provide intermediate-level instructions (Level 2) for creating this CAD model.
    
    Abstract shape description: {abstract_description}
    
    Beginner instructions: {beginner_description}
    
    CAD metadata:
    """
    
    # Añadimos información más detallada pero sin llegar al nivel experto
    if "parts" in json_data:
        for part_id, part_info in json_data["parts"].items():
            prompt += f"\n\nPart {part_id}:"
            
            if "sketch" in part_info:
                sketch = part_info["sketch"]
                prompt += "\n- Sketch contains: "
                primitives = []
                for face in sketch.values():
                    for loop in face.values():
                        for primitive_id, primitive in loop.items():
                            p_type = primitive_id.split("_")[0]
                            primitives.append(p_type)
                
                primitive_counts = {}
                for p in primitives:
                    primitive_counts[p] = primitive_counts.get(p, 0) + 1
                
                prompt += ", ".join([f"{count} {p}(s)" for p, count in primitive_counts.items()])
            
            if "extrusion" in part_info:
                extrusion = part_info["extrusion"]
                prompt += f"\n- Extrusion depth: {extrusion.get('extrude_depth_towards_normal', 0)} units"
                prompt += f"\n- Sketch scaling: {extrusion.get('sketch_scale', 1)}"
                prompt += f"\n- Operation: {extrusion.get('operation', 'N/A')}"
            
            if "description" in part_info and "shape" in part_info["description"]:
                prompt += f"\n- Description: {part_info['description']['shape']}"
    
    prompt += """
    
    Using this information, write intermediate-level CAD instructions that:
    1. Provide a general overview of the geometric properties
    2. Include approximate dimensions and relations between parts
    3. Describe the main construction steps in moderate detail
    4. Avoid exact coordinates but include relative positions
    5. Target an audience with basic CAD knowledge
    6. Keep the explanation under 10 sentences
    """
    
    return prompt

def create_expert_prompt(abstract_description, beginner_description, intermediate_description, json_data):
    """Crea un prompt para generar instrucciones de nivel experto."""
    prompt = f"""
    You are a senior CAD engineer. Based on the following information, provide expert-level instructions (Level 3) for creating this CAD model with precise parameters.
    
    Abstract description: {abstract_description}
    
    Beginner instructions: {beginner_description}
    
    Intermediate instructions: {intermediate_description}
    
    Detailed CAD metadata:
    """
    
    # Añadimos toda la información técnica disponible
    if "parts" in json_data:
        for part_id, part_info in json_data["parts"].items():
            prompt += f"\n\nPart {part_id}:"
            
            if "coordinate_system" in part_info:
                coords = part_info["coordinate_system"]
                prompt += f"\n- Coordinate system: Euler angles {coords.get('Euler Angles', 'N/A')}, Translation {coords.get('Translation Vector', 'N/A')}"
            
            if "sketch" in part_info:
                sketch = part_info["sketch"]
                for face_id, face in sketch.items():
                    prompt += f"\n- {face_id}:"
                    for loop_id, loop in face.items():
                        prompt += f"\n  - {loop_id}:"
                        for primitive_id, primitive in loop.items():
                            if "Circle" in primitive_id:
                                prompt += f"\n    - {primitive_id}: Center={primitive.get('Center', 'N/A')}, Radius={primitive.get('Radius', 'N/A')}"
                            elif "Line" in primitive_id:
                                prompt += f"\n    - {primitive_id}: Start={primitive.get('Start', 'N/A')}, End={primitive.get('End', 'N/A')}"
                            elif "Arc" in primitive_id:
                                prompt += f"\n    - {primitive_id}: Start={primitive.get('Start', 'N/A')}, Mid={primitive.get('Mid', 'N/A')}, End={primitive.get('End', 'N/A')}"
                            else:
                                prompt += f"\n    - {primitive_id}: {primitive}"
            
            if "extrusion" in part_info:
                extrusion = part_info["extrusion"]
                prompt += f"\n- Extrusion:"
                prompt += f"\n  - Depth towards normal: {extrusion.get('extrude_depth_towards_normal', 'N/A')}"
                prompt += f"\n  - Depth opposite normal: {extrusion.get('extrude_depth_opposite_normal', 'N/A')}"
                prompt += f"\n  - Sketch scale: {extrusion.get('sketch_scale', 'N/A')}"
                prompt += f"\n  - Operation: {extrusion.get('operation', 'N/A')}"
    
    prompt += """
    
    Based on this detailed information, write expert-level CAD modeling instructions that:
    1. Include precise coordinates, parameters, and measurements
    2. Follow a logical step-by-step process for creating the model
    3. Use proper CAD terminology
    4. Provide complete details for sketch creation, scaling, and extrusion
    5. Include all necessary transformations and operations
    6. Target an audience with advanced CAD knowledge
    
    Your instructions should be detailed enough that a CAD expert could reproduce the model exactly.
    """
    
    return prompt

def create_keywords_prompt(abstract_description, json_data):
    """Crea un prompt para generar palabras clave descriptivas."""
    prompt = f"""
    You are a senior CAD engineer. Based on the following CAD model description, provide 5-10 relevant keywords or tags that describe this model.
    
    Abstract description: {abstract_description}
    
    CAD metadata:
    """
    
    # Añadimos información resumida del JSON
    if "parts" in json_data:
        for part_id, part_info in json_data["parts"].items():
            if "description" in part_info and "shape" in part_info["description"]:
                prompt += f"\n- Part {part_id} description: {part_info['description']['shape']}"
    
    prompt += """
    
    Provide a comma-separated list of 5-10 relevant keywords or tags that describe this CAD model.
    Focus on shape characteristics, geometric features, potential applications, and general categories.
    
    Format your response as: keyword1, keyword2, keyword3, ...
    """
    
    return prompt

def create_nli_prompt(abstract_description, json_data):
    """Crea un prompt para generar instrucciones en lenguaje natural (NLI)."""
    prompt = f"""
    You are a senior CAD engineer tasked with providing natural language instructions for a junior CAD designer.
    Based on the following CAD model information, create detailed natural language instructions that describe how to model this CAD object step by step.
    
    Abstract description: {abstract_description}
    
    CAD metadata:
    """
    
    # Añadimos toda la información disponible
    if "parts" in json_data:
        for part_id, part_info in json_data["parts"].items():
            prompt += f"\n\nPart {part_id}:"
            
            if "coordinate_system" in part_info:
                coords = part_info["coordinate_system"]
                prompt += f"\n- Coordinate system: Euler angles {coords.get('Euler Angles', 'N/A')}, Translation {coords.get('Translation Vector', 'N/A')}"
            
            if "sketch" in part_info:
                sketch = part_info["sketch"]
                for face_id, face in sketch.items():
                    prompt += f"\n- {face_id}:"
                    for loop_id, loop in face.items():
                        prompt += f"\n  - {loop_id}:"
                        for primitive_id, primitive in loop.items():
                            if "Circle" in primitive_id:
                                prompt += f"\n    - {primitive_id}: Center={primitive.get('Center', 'N/A')}, Radius={primitive.get('Radius', 'N/A')}"
                            elif "Line" in primitive_id:
                                prompt += f"\n    - {primitive_id}: Start={primitive.get('Start', 'N/A')}, End={primitive.get('End', 'N/A')}"
                            elif "Arc" in primitive_id:
                                prompt += f"\n    - {primitive_id}: Start={primitive.get('Start', 'N/A')}, Mid={primitive.get('Mid', 'N/A')}, End={primitive.get('End', 'N/A')}"
                            else:
                                prompt += f"\n    - {primitive_id}: {primitive}"
            
            if "extrusion" in part_info:
                extrusion = part_info["extrusion"]
                prompt += f"\n- Extrusion:"
                prompt += f"\n  - Depth towards normal: {extrusion.get('extrude_depth_towards_normal', 'N/A')}"
                prompt += f"\n  - Depth opposite normal: {extrusion.get('extrude_depth_opposite_normal', 'N/A')}"
                prompt += f"\n  - Sketch scale: {extrusion.get('sketch_scale', 'N/A')}"
                prompt += f"\n  - Operation: {extrusion.get('operation', 'N/A')}"
            
            if "description" in part_info:
                desc = part_info["description"]
                if "shape" in desc:
                    prompt += f"\n- Shape description: {desc['shape']}"
                if "length" in desc and "width" in desc and "height" in desc:
                    prompt += f"\n- Dimensions: length={desc.get('length', 'N/A')}, width={desc.get('width', 'N/A')}, height={desc.get('height', 'N/A')}"
    
    prompt += """
    
    Your task is to write comprehensive, natural language instructions that explain how to create this CAD model step by step.
    Your instructions should:
    1. Be written in clear, detailed paragraphs
    2. Include all necessary steps from start to finish
    3. Specify the exact parameters, coordinates, and operations needed
    4. Provide guidance on the modeling approach and technique
    5. Be understandable to someone with moderate CAD experience
    
    Write these instructions as if you were explaining to a colleague how to recreate this model from scratch.
    """
    
    return prompt

def process_single_cad(uid, json_path, model, tokenizer):
    """
    Procesa un solo archivo CAD y genera todas las anotaciones necesarias.
    """
    print(f"Procesando UID: {uid}")
    
    # Cargar datos JSON
    json_data = load_minimal_json(json_path)
    
    # Generar descripción abstracta (simulando lo que haría un VLM)
    abstract_prompt = create_abstract_prompt(json_data)
    abstract_description = generate_response(model, tokenizer, abstract_prompt)
    print(f"Abstract Description generada: {abstract_description[:100]}...")
    
    # Generar instrucciones de nivel principiante
    beginner_prompt = create_beginner_prompt(abstract_description, json_data)
    beginner_description = generate_response(model, tokenizer, beginner_prompt)
    print(f"Beginner Instructions generadas: {beginner_description[:100]}...")
    
    # Generar instrucciones de nivel intermedio
    intermediate_prompt = create_intermediate_prompt(abstract_description, beginner_description, json_data)
    intermediate_description = generate_response(model, tokenizer, intermediate_prompt)
    print(f"Intermediate Instructions generadas: {intermediate_description[:100]}...")
    
    # Generar instrucciones de nivel experto
    expert_prompt = create_expert_prompt(abstract_description, beginner_description, intermediate_description, json_data)
    expert_description = generate_response(model, tokenizer, expert_prompt)
    print(f"Expert Instructions generadas: {expert_description[:100]}...")
    
    # Generar palabras clave
    keywords_prompt = create_keywords_prompt(abstract_description, json_data)
    keywords = generate_response(model, tokenizer, keywords_prompt)
    print(f"Keywords generadas: {keywords[:100]}...")
    
    # Generar NLI (instrucciones en lenguaje natural)
    nli_prompt = create_nli_prompt(abstract_description, json_data)
    nli_data = generate_response(model, tokenizer, nli_prompt)
    print(f"NLI generado: {nli_data[:100]}...")
    
    # Crear un objeto para capturar todos los niveles de datos
    all_level_data = {
        "abstract": abstract_description,
        "beginner": beginner_description,
        "intermediate": intermediate_description,
        "expert": expert_description,
        "keywords": keywords
    }
    
    # Devolver todos los datos generados
    return {
        "uid": uid,
        "abstract": abstract_description,
        "beginner": beginner_description,
        "intermediate": intermediate_description,
        "expert": expert_description,
        "description": json_data.get("final_shape", ""),  # Usar final_shape si está disponible
        "keywords": keywords,
        "all_level_data": json.dumps(all_level_data),
        "nli_data": nli_data
    }

def main():
    parser = argparse.ArgumentParser(description='Generar anotaciones de CAD usando Qwen2.5-72B-Instruct')
    parser.add_argument('--input_dir', required=True, help='Directorio raíz con los archivos minimal_json')
    parser.add_argument('--split_json', required=True, help='Archivo JSON con los splits train/test/validation')
    parser.add_argument('--output_file', required=True, help='Archivo CSV de salida')
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-72B-Instruct', help='Nombre del modelo Qwen a utilizar')
    parser.add_argument('--split', default='train', choices=['train', 'test', 'validation', 'all'], 
                        help='Split a procesar (train, test, validation, all)')
    parser.add_argument('--max_samples', type=int, default=None, help='Número máximo de muestras a procesar')
    args = parser.parse_args()
    
    # Cargar modelo y tokenizador
    model, tokenizer = load_model(args.model_name)
    
    # Cargar split JSON
    with open(args.split_json, 'r') as f:
        split_data = json.load(f)
    
    # Determinar qué UIDs procesar
    if args.split == 'all':
        uids = split_data.get('train', []) + split_data.get('test', []) + split_data.get('validation', [])
    else:
        uids = split_data.get(args.split, [])
    
    if args.max_samples:
        uids = uids[:args.max_samples]
    
    print(f"Procesando {len(uids)} UIDs del split '{args.split}'")
    
    # Crear el archivo CSV de salida
    fieldnames = [
        'uid', 'abstract', 'beginner', 'intermediate', 'expert', 
        'description', 'keywords', 'all_level_data', 'nli_data'
    ]
    
    with open(args.output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Procesar cada UID
        for uid in tqdm(uids):
            root_id, sample_id = uid.split('/')
            json_path = os.path.join(args.input_dir, uid, "minimal_json", f"{sample_id}.json")
            
            # Verificar si el archivo existe
            if not os.path.exists(json_path):
                print(f"Advertencia: Archivo no encontrado: {json_path}")
                # Intentar una ruta alternativa
                alt_json_path = os.path.join(args.input_dir, root_id, sample_id, "minimal_json", f"{sample_id}.json")
                if os.path.exists(alt_json_path):
                    json_path = alt_json_path
                    print(f"Usando ruta alternativa: {json_path}")
                else:
                    print(f"Error: No se pudo encontrar el archivo JSON para {uid}")
                    continue
            
            # Procesar el archivo
            try:
                result = process_single_cad(uid, json_path, model, tokenizer)
                writer.writerow(result)
                # Guardar después de cada muestra para no perder datos en caso de error
                csvfile.flush()
            except Exception as e:
                print(f"Error procesando {uid}: {e}")
                continue

if __name__ == "__main__":
    main()