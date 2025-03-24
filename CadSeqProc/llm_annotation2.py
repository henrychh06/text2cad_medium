import os
import json
import csv
import re
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from glob import glob

def load_model(model_name):
    """Carga el modelo Qwen2.5-72B-Instruct y el tokenizador."""
    print(f"Cargando modelo {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
        torch_dtype=torch.float16,  # Reduce el consumo de memoria
        low_cpu_mem_usage=True,
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

def extract_keywords_from_annotation(annotation_dir, sample_id):
    """
    Extrae palabras clave del archivo de anotación final_*.json.
    Busca la etiqueta <KEYWORDS></KEYWORDS> en el contenido.
    """
    try:
        # Buscar el archivo final_*.json
        annotation_files = glob(os.path.join(annotation_dir, f"final_{sample_id}.json"))
        
        if not annotation_files:
            return ""  # No se encontró el archivo
            
        # Leer el contenido del archivo
        with open(annotation_files[0], 'r') as f:
            content = f.read()
            
        # Usar regex para extraer las palabras clave entre las etiquetas <KEYWORDS></KEYWORDS>
        keyword_match = re.search(r'<KEYWORDS>(.*?)</KEYWORDS>', content, re.DOTALL)
        if keyword_match:
            return keyword_match.group(1).strip()
        else:
            return ""  # No se encontraron las etiquetas
    except Exception as e:
        print(f"Error al extraer palabras clave: {e}")
        return ""

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
    This level should be clearly different from both the abstract description and beginner level instructions.
    
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
    7. Make sure this level is distinctly different from both the abstract and beginner levels
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

def create_nli_prompt(json_data):
    """
    Crea un prompt NLI según el formato mostrado en el paper.
    """
    prompt = """[INST]
You are a senior CAD engineer and you are tasked to provide natural language instructions to a junior CAD designer for generating a parametric CAD model.

Overview information about the CAD assembly JSON:
1. The CAD assembly json lists the process of constructing a CAD model.
2. Every CAD model consists of one or multiple intermediate CAD parts.
3. These intermediate CAD parts are listed in the "parts" key of the CAD assembly JSON.
4. The first intermediate CAD part is the base part and the subsequent parts build upon the previously constructed parts using the operation defined for that part.
5. All intermediate parts combine to a final cad model.

Every intermediate CAD part is generated using the following steps:
Step 1: Draw a 2D sketch.
Step 2: Scale the 2D sketch using the sketch_scale scaling parameter.
Step 3: Transform the scaled 2D sketch into 3D Sketch using the euler angles and translation.
Step 4: Extrude the 2D sketch to generate the 3D model.

Detailed CAD assembly JSON:
"""
    
    # Añadimos los datos JSON de manera más limpia y formateada
    if "parts" in json_data:
        prompt += f"\nThis model has {len(json_data['parts'])} part(s).\n"
        
        for part_id, part_info in json_data["parts"].items():
            prompt += f"\nPart {part_id}:"
            
            if "coordinate_system" in part_info:
                coords = part_info["coordinate_system"]
                prompt += f"\n- Coordinate system: Euler angles {coords.get('Euler Angles', 'N/A')}, Translation Vector {coords.get('Translation Vector', 'N/A')}"
            
            if "sketch" in part_info:
                sketch = part_info["sketch"]
                prompt += "\n- Sketch details:"
                for face_id, face in sketch.items():
                    prompt += f"\n  - {face_id}:"
                    for loop_id, loop in face.items():
                        prompt += f"\n    - {loop_id}:"
                        for primitive_id, primitive in loop.items():
                            if "Circle" in primitive_id:
                                prompt += f"\n      - {primitive_id}: Center={primitive.get('Center', 'N/A')}, Radius={primitive.get('Radius', 'N/A')}"
                            elif "Line" in primitive_id:
                                prompt += f"\n      - {primitive_id}: Start={primitive.get('Start', 'N/A')}, End={primitive.get('End', 'N/A')}"
                            elif "Arc" in primitive_id:
                                prompt += f"\n      - {primitive_id}: Start={primitive.get('Start', 'N/A')}, Mid={primitive.get('Mid', 'N/A')}, End={primitive.get('End', 'N/A')}"
            
            if "extrusion" in part_info:
                extrusion = part_info["extrusion"]
                prompt += f"\n- Extrusion details:"
                prompt += f"\n  - Extrude depth towards normal: {extrusion.get('extrude_depth_towards_normal', 'N/A')}"
                prompt += f"\n  - Extrude depth opposite normal: {extrusion.get('extrude_depth_opposite_normal', 'N/A')}"
                prompt += f"\n  - Sketch scale: {extrusion.get('sketch_scale', 'N/A')}"
                prompt += f"\n  - Operation: {extrusion.get('operation', 'N/A')}"
            
            # Añadir información de descripción si está disponible
            if "description" in part_info:
                desc = part_info["description"]
                if "shape" in desc:
                    prompt += f"\n- Shape description: {desc['shape']}"
                if "length" in desc and "width" in desc and "height" in desc:
                    prompt += f"\n- Dimensions: length={desc.get('length', 'N/A')}, width={desc.get('width', 'N/A')}, height={desc.get('height', 'N/A')}"
    
    # Añadir la información de forma final si está disponible
    if "final_shape" in json_data:
        prompt += f"\n\nFinal shape description: {json_data['final_shape']}"
    
    prompt += """

Based on the CAD assembly JSON information provided above, write detailed natural language instructions for creating this CAD model. Your instructions should cover the complete process from initial setup to the final model, including all steps for creating and transforming each part.
[/INST]"""
    
    return prompt

def process_single_cad(uid, json_path, model, tokenizer, annotation_dir=None):
    """
    Procesa un solo archivo CAD y genera todas las anotaciones necesarias.
    """
    print(f"Procesando UID: {uid}")
    
    # Cargar datos JSON
    json_data = load_minimal_json(json_path)
    
    # Extraemos el abstract directamente del campo final_shape
    abstract_description = json_data.get("final_shape", "")
    if not abstract_description and "final_name" in json_data:
        abstract_description = json_data.get("final_name", "")
    
    # Si no hay descripción, generamos una
    if not abstract_description:
        print("No se encontró final_shape, generando abstract...")
        abstract_prompt = f"""
        You are a senior CAD engineer. Based on the following CAD metadata, provide a brief abstract description (1-2 sentences) of what this CAD model looks like.
        
        CAD metadata: {json.dumps(json_data, indent=2)[:500]}...
        
        Provide only a brief visual description of the overall shape.
        """
        abstract_description = generate_response(model, tokenizer, abstract_prompt)
    
    print(f"Abstract Description: {abstract_description[:100]}...")
    
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
    
    # Extraer keywords del archivo de anotación si el directorio está disponible
    keywords = ""
    if annotation_dir:
        root_id, sample_id = uid.split('/')
        annot_dir = os.path.join(annotation_dir, uid, "qwen2_vlm_annotation")
        if os.path.exists(annot_dir):
            keywords = extract_keywords_from_annotation(annot_dir, sample_id)
            print(f"Keywords extraídas: {keywords[:100]}...")
    
    # Si no pudimos extraer keywords, generamos algunas
    if not keywords:
        print("No se pudieron extraer keywords, generando...")
        keywords_prompt = f"""
        You are a senior CAD engineer. Based on this description of a CAD model, provide 5-10 relevant keywords separated by commas.
        
        Description: {abstract_description}
        
        Provide only a comma-separated list of keywords (no explanations or other text).
        """
        keywords = generate_response(model, tokenizer, keywords_prompt)
        print(f"Keywords generadas: {keywords[:100]}...")
    
    # Generar NLI (instrucciones en lenguaje natural)
    nli_prompt = create_nli_prompt(json_data)
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
    parser.add_argument('--annotation_dir', default=None, help='Directorio con las anotaciones existentes (para extraer keywords)')
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
                result = process_single_cad(uid, json_path, model, tokenizer, args.annotation_dir)
                writer.writerow(result)
                # Guardar después de cada muestra para no perder datos en caso de error
                csvfile.flush()
            except Exception as e:
                print(f"Error procesando {uid}: {e}")
                continue

if __name__ == "__main__":
    main()