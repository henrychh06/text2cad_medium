import os
import json
import csv
import re
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from glob import glob

# max_memory = {0: "20GB", 1: "20GB"}

def load_pipeline(model_name):
    print(f"Cargando pipeline del modelo {model_name}...")
    pipe = pipeline(
        "text-generation", 
        model=model_name,
        torch_dtype=torch.float16,  # Reduce el consumo de memoria
        #low_cpu_mem_usage=True,
        trust_remote_code=True)
    return pipe

def generate_response(pipe, prompt, system_message="You are Qwen, a helpful CAD design expert."):
    """
    Genera una respuesta combinando un mensaje del sistema y el prompt del usuario,
    utilizando el pipeline de text-generation.

    Parámetros:
      pipe: El pipeline de generación de texto ya inicializado.
      prompt: El prompt o mensaje del usuario.
      system_message: Mensaje del sistema que contextualiza la generación.

    Retorna:
      El texto generado por el modelo.
    """
    combined_prompt = f"{system_message}\n{prompt}"
    response = pipe(combined_prompt, max_new_tokens=1024, temperature=0.7, do_sample=True)
    generated_text = response[0]['generated_text']
    # Si la respuesta comienza con el system_message, lo removemos:
    if generated_text.startswith(system_message):
        generated_text = generated_text[len(system_message):].strip()
    return generated_text

def load_minimal_json(json_path):
    """Carga un archivo JSON minimal."""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_keywords_from_annotation(annotation_dir, sample_id):
    """
    Extrae palabras clave del archivo de anotación final_*.json.
    Busca la segunda etiqueta <KEYWORDS></KEYWORDS> en el contenido.
    """
    try:
        # Buscar el archivo final_*.json
        annotation_files = glob(os.path.join(annotation_dir, f"final_{sample_id}.json"))
        
        if not annotation_files:
            return ""  # No se encontró el archivo
        
        # Leer el contenido del archivo
        with open(annotation_files[0], 'r') as f:
            content = f.read()
        
        # Usar regex para extraer la segunda ocurrencia de palabras clave
        keyword_matches = re.findall(r'<KEYWORDS>(.*?)</KEYWORDS>', content, re.DOTALL)
        
        # Retornar la segunda ocurrencia si existe, de lo contrario una cadena vacía
        if len(keyword_matches) >= 2:
            return keyword_matches[1].strip()
        else:
            return ""  # No se encontró una segunda ocurrencia
    
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
    Crea un prompt NLI según el formato mostrado en el paper,
    e incluye tags para cada parte (<part_1>, <part_2>, etc.) en la sección de instrucciones detalladas.
    """
    prompt = """[INST]
You are a senior CAD engineer and you are tasked to provide natural language instructions to a junior CAD designer for generating a parametric CAD model.

Overview information about the CAD assembly JSON:
1. The CAD assembly JSON lists the process of constructing a CAD model.
2. Every CAD model consists of one or multiple intermediate CAD parts.
3. These intermediate CAD parts are listed in the "parts" key of the CAD assembly JSON.
4. The first intermediate CAD part is the base part and the subsequent parts build upon the previously constructed parts using the operation defined for that part.
5. All intermediate parts combine to a final CAD model.

Every intermediate CAD part is generated using the following steps:
Step 1: Draw a 2D sketch.
Step 2: Scale the 2D sketch using the sketch_scale parameter.
Step 3: Transform the scaled 2D sketch into a 3D sketch using the Euler angles and translation.
Step 4: Extrude the 2D sketch to generate the 3D model.
Step 5: Final dimensions of the 3D model are defined by the length, width, and height parameters.

Detailed CAD assembly JSON:
"""
    if "parts" in json_data:
        prompt += f"\nThis model has {len(json_data['parts'])} part(s).\n"
        for part_id, part_info in json_data["parts"].items():
            # Usamos el nombre de la parte para crear el tag, por ejemplo <part_part_1>
            prompt += f"\n<part_{part_id}>"
            prompt += f"\nPart {part_id}:"
            if "coordinate_system" in part_info:
                coords = part_info["coordinate_system"]
                prompt += f"\n- Coordinate system: Euler angles {coords.get('Euler Angles', 'N/A')}, Translation Vector {coords.get('Translation Vector', 'N/A')}"
            if "sketch" in part_info:
                prompt += "\n- Sketch details:"
                for face_id, face in part_info["sketch"].items():
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
                            else:
                                prompt += f"\n      - {primitive_id}: {primitive}"
            if "extrusion" in part_info:
                extrusion = part_info["extrusion"]
                prompt += f"\n- Extrusion details:"
                prompt += f"\n  - Extrude depth towards normal: {extrusion.get('extrude_depth_towards_normal', 'N/A')}"
                prompt += f"\n  - Extrude depth opposite normal: {extrusion.get('extrude_depth_opposite_normal', 'N/A')}"
                prompt += f"\n  - Sketch scale: {extrusion.get('sketch_scale', 'N/A')}"
                prompt += f"\n  - Operation: {extrusion.get('operation', 'N/A')}"
            if "description" in part_info:
                desc = part_info["description"]
                if "shape" in desc:
                    prompt += f"\n- Shape description: {desc['shape']}"
                if "length" in desc and "width" in desc and "height" in desc:
                    prompt += f"\n- Dimensions: length={desc.get('length', 'N/A')}, width={desc.get('width', 'N/A')}, height={desc.get('height', 'N/A')}"
            prompt += f"\n</part_{part_id}>"
    if "final_shape" in json_data:
        prompt += f"\n\nFinal shape description: {json_data['final_shape']}"
    
    prompt += """

Based on the CAD assembly JSON information provided above, write detailed natural language instructions for creating this CAD model. Your instructions should cover the complete process from initial setup to the final model, including all steps for creating and transforming each part.
[/INST]"""
    return prompt


import time
def process_single_cad(uid, json_path, pipe, annotation_dir=None):
    print(f"Procesando UID: {uid}")
    
    # Cargar datos JSON
    json_data = load_minimal_json(json_path)
    
    # Extraer abstract
    abstract_description = json_data.get("final_shape", "")
    if not abstract_description and "final_name" in json_data:
        abstract_description = json_data.get("final_name", "")
    
    if not abstract_description:
        print("No se encontró final_shape, generando abstract...")
        abstract_prompt = f"""
        You are a senior CAD engineer. Based on the following CAD metadata, provide a brief abstract description (1-2 sentences) of what this CAD model looks like.
        
        CAD metadata: {json.dumps(json_data, indent=2)[:500]}...
        
        Provide only a brief visual description of the overall shape.
        """
        abstract_description = generate_response(pipe, abstract_prompt)
    
    print(f"Abstract Description: {abstract_description[:100]}...")
    
    # Generar instrucciones de nivel principiante
    beginner_prompt = create_beginner_prompt(abstract_description, json_data)
    beginner_description = generate_response(pipe, beginner_prompt)
    # Encerrar el resultado con los tags de nivel (por ejemplo, <level1> ... </level1>)
    beginner_description = f"<level1>\n{beginner_description.strip()}\n</level1>"
    print(f"Beginner Instructions generadas: {beginner_description[:100]}...")
    
    # Generar instrucciones de nivel intermedio
    intermediate_prompt = create_intermediate_prompt(abstract_description, beginner_description, json_data)
    intermediate_description = generate_response(pipe, intermediate_prompt)
    # Encerrar el resultado con tags de nivel, por ejemplo <level2>
    intermediate_description = f"<level2>\n{intermediate_description.strip()}\n</level2>"
    print(f"Intermediate Instructions generadas: {intermediate_description[:100]}...")
    
    # Generar instrucciones de nivel experto
    expert_prompt = create_expert_prompt(abstract_description, beginner_description, intermediate_description, json_data)
    expert_description = generate_response(pipe, expert_prompt)
    # Encerrar el resultado con tags de nivel, por ejemplo <level3>
    expert_description = f"<level3>\n{expert_description.strip()}\n</level3>"
    print(f"Expert Instructions generadas: {expert_description[:100]}...")
    
    # Extraer o generar keywords
    keywords = ""
    if annotation_dir:
        root_id, sample_id = uid.split('/')
        annot_dir = os.path.join(annotation_dir, uid, "qwen2_vlm_annotation")
        if os.path.exists(annot_dir):
            keywords = extract_keywords_from_annotation(annot_dir, sample_id)
            print(f"Keywords extraídas: {keywords[:100]}...")
    if not keywords:
        print("No se pudieron extraer keywords, generando...")
        keywords_prompt = f"""
        You are a senior CAD engineer. Based on this description of a CAD model, provide 5-10 relevant keywords separated by commas.
        
        Description: {abstract_description}
        
        Provide only a comma-separated list of keywords (no explanations or other text).
        """
        keywords = generate_response(pipe, keywords_prompt)
        print(f"Keywords generadas: {keywords[:100]}...")
    
    # Generar NLI
    nli_prompt = create_nli_prompt(json_data)
    nli_data = generate_response(pipe, nli_prompt)
    print(f"NLI generado: {nli_data[:100]}...")
    
    # Crear objeto con los datos generados
    all_level_data = {
        "abstract": abstract_description,
        "beginner": beginner_description,
        "intermediate": intermediate_description,
        "expert": expert_description,
        "keywords": keywords
    }
    
    return {
        "uid": uid,
        "abstract": abstract_description,
        "beginner": beginner_description,
        "intermediate": intermediate_description,
        "expert": expert_description,
        "description": json_data.get("final_shape", ""),
        "keywords": keywords,
        "all_level_data": json.dumps(all_level_data),
        "nli_data": nli_data
    }

def main():
    parser = argparse.ArgumentParser(description='Generar anotaciones de CAD usando Qwen2.5-14B-Instruct-1M con pipeline')
    parser.add_argument('--input_dir', required=True, help='Directorio raíz con los archivos minimal_json')
    parser.add_argument('--split_json', required=True, help='Archivo JSON con los splits train/test/validation')
    parser.add_argument('--output_file', required=True, help='Archivo CSV de salida')
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-14B-Instruct-1M', help='Nombre del modelo Qwen a utilizar')
    parser.add_argument('--split', default='train', choices=['train', 'test', 'validation', 'all'], 
                        help='Split a procesar (train, test, validation, all)')
    parser.add_argument('--max_samples', type=int, default=None, help='Número máximo de muestras a procesar')
    parser.add_argument('--annotation_dir', default=None, help='Directorio con las anotaciones existentes (para extraer keywords)')
    args = parser.parse_args()
    
    # Cargar el pipeline en lugar de cargar el modelo y el tokenizador
    pipe = load_pipeline(args.model_name)
    
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
            json_path = os.path.join(args.input_dir, uid, "minimal_json", f"{sample_id}_merged_vlm.json")
            # Si usas un nombre de archivo alternativo, por ejemplo:
            # json_path = os.path.join(args.input_dir, uid, "minimal_json", f"{sample_id}_merged_vlm.json2")
            
            if not os.path.exists(json_path):
                print(f"Advertencia: Archivo no encontrado: {json_path}")
                alt_json_path = os.path.join(args.input_dir, root_id, sample_id, "minimal_json", f"{sample_id}.json")
                if os.path.exists(alt_json_path):
                    json_path = alt_json_path
                    print(f"Usando ruta alternativa: {json_path}")
                else:
                    print(f"Error: No se pudo encontrar el archivo JSON para {uid}")
                    continue
            
            try:
                result = process_single_cad(uid, json_path, pipe, args.annotation_dir)
                writer.writerow(result)
                csvfile.flush()
            except Exception as e:
                print(f"Error procesando {uid}: {e}")
                continue

if __name__ == "__main__":
    main()
