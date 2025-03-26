import os 
import json 
import csv 
import re 
import argparse
from tqdm import tqdm
from transformers import pipeline
from glob import glob
import time

def load_pipeline(model_name):
    print(f"Cargando pipeline del modelo {model_name}...")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype="auto",  # Puedes ajustar según tu entorno
        trust_remote_code=True)
    return pipe

def generate_response(pipe, prompt, system_message="You are Qwen, a helpful CAD design expert."):
    """
    Genera una respuesta combinando el mensaje del sistema y el prompt.
    Se elimina del output el system_message (si aparece al inicio) para que la respuesta "limpia" contenga únicamente la generación del LLM.
    """
    combined_prompt = f"{system_message}\n{prompt}"
    response = pipe(combined_prompt, max_new_tokens=1024, temperature=0.7, do_sample=True)
    generated_text = response[0]['generated_text']
    # Si el output empieza con el system_message, se elimina esa parte:
    if generated_text.startswith(combined_prompt):
    # Retorna solo lo que sigue después del combined_prompt
        return generated_text[len(combined_prompt):].strip()
    else:
    # Si no se detecta, retorna el texto completo
        return generated_text.strip()

def load_minimal_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_keywords_from_annotation(annotation_dir, sample_id):
    try:
        annotation_files = glob(os.path.join(annotation_dir, sample_id, "minimal_json",  f"{sample_id}_merged_vlm.json"))
        if not annotation_files:
            return ""
        with open(annotation_files[0], 'r') as f:
            content = f.read()
        keyword_matches = re.findall(r'<KEYWORDS>(.*?)</KEYWORDS>', content, re.DOTALL)
        if len(keyword_matches) >= 2:
            return keyword_matches[1].strip()
        else:
            return ""
    except Exception as e:
        print(f"Error al extraer palabras clave: {e}")
        return ""

def create_beginner_prompt(abstract_description, json_data):
    prompt = f""" 
    You are a senior CAD engineer. Based on the following information, provide simplified step-by-step instructions (Level 1) for creating this CAD model.

    Abstract shape description: {abstract_description}

    CAD metadata overview:"""
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

    Using this information, write beginner-friendly instructions for creating this CAD model. Your instructions should:

    Use simple language without technical jargon.

    Provide a general overview of the steps (no exact coordinates or parameters).

    Focus on the basic shape and structure.

    Keep the explanation under 5-6 sentences.

    Explain the process in a way that a CAD beginner could understand.
    """
    return prompt

def create_intermediate_prompt(abstract_description, beginner_description, json_data):
    prompt = f""" 
    You are a senior CAD engineer. Based on the following information, provide intermediate-level instructions (Level 2) for creating this CAD model. This level should be clearly different from both the abstract description and beginner level instructions.

    Abstract shape description: {abstract_description}

    Beginner instructions: {beginner_description}

    CAD metadata:"""
    if "parts" in json_data:
        for part_id, part_info in json_data["parts"].items():
            prompt += f"\n\nPart {part_id}:"
            if "sketch" in part_info:
                sketch = part_info["sketch"]
                prompt += "\n- Sketch contains: "
                primitives = []
                for face in sketch.values():
                    for loop in face.values():
                        for primitive_id in loop.keys():
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

    Provide a general overview of the geometric properties.

    Include approximate dimensions and relations between parts.

    Describe the main construction steps in moderate detail.

    Avoid exact coordinates but include relative positions.

    Target an audience with basic CAD knowledge.

    Keep the explanation under 10 sentences.

    Make sure this level is distinctly different from both the abstract and beginner levels.
    """
    return prompt

def create_expert_prompt(abstract_description, beginner_description, intermediate_description, json_data):
    prompt = f""" 
    You are a senior CAD engineer. Based on the following information, provide expert-level instructions (Level 3) for creating this CAD model with precise parameters.

    Abstract description: {abstract_description}

    Beginner instructions: {beginner_description}

    Intermediate instructions: {intermediate_description}

    Detailed CAD metadata:"""
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
                        prompt += f"\n - {loop_id}:"
                        for primitive_id, primitive in loop.items():
                            if "Circle" in primitive_id:
                                prompt += f"\n - {primitive_id}: Center={primitive.get('Center', 'N/A')}, Radius={primitive.get('Radius', 'N/A')}"
                            elif "Line" in primitive_id:
                                prompt += f"\n - {primitive_id}: Start={primitive.get('Start', 'N/A')}, End={primitive.get('End', 'N/A')}"
                            elif "Arc" in primitive_id:
                                prompt += f"\n - {primitive_id}: Start={primitive.get('Start', 'N/A')}, Mid={primitive.get('Mid', 'N/A')}, End={primitive.get('End', 'N/A')}"
                            else:
                                prompt += f"\n - {primitive_id}: {primitive}"
            if "extrusion" in part_info:
                extrusion = part_info["extrusion"]
                prompt += f"\n- Extrusion:"
                prompt += f"\n - Depth towards normal: {extrusion.get('extrude_depth_towards_normal', 'N/A')}"
                prompt += f"\n - Depth opposite normal: {extrusion.get('extrude_depth_opposite_normal', 'N/A')}"
                prompt += f"\n - Sketch scale: {extrusion.get('sketch_scale', 'N/A')}"
                prompt += f"\n - Operation: {extrusion.get('operation', 'N/A')}"

    prompt += """

    Based on this detailed information, write expert-level CAD modeling instructions that:

    Include precise coordinates, parameters, and measurements.

    Follow a logical step-by-step process for creating the model.

    Use proper CAD terminology.

    Provide complete details for sketch creation, scaling, and extrusion.

    Include all necessary transformations and operations.

    Target an audience with advanced CAD knowledge.

    Your instructions should be detailed enough that a CAD expert could reproduce the model exactly.
    """
    return prompt

def create_nli_prompt(json_data):
    """
    Crea un prompt NLI siguiendo el formato del paper e incluye tags para cada parte.
    """
    prompt = """[INST] You are a senior CAD engineer and you are tasked to provide natural language instructions to a junior CAD designer for generating a parametric CAD model.

    Overview information about the CAD assembly JSON:

    The CAD assembly JSON lists the process of constructing a CAD model.

    Every CAD model consists of one or multiple intermediate CAD parts.

    These intermediate CAD parts are listed in the "parts" key of the CAD assembly JSON.

    The first intermediate CAD part is the base part and the subsequent parts build upon the previously constructed parts using the operation defined for that part.

    All intermediate parts combine to a final CAD model.

    Every intermediate CAD part is generated using the following steps: Step 1: Draw a 2D sketch. Step 2: Scale the 2D sketch using the sketch_scale parameter. Step 3: Transform the scaled 2D sketch into a 3D sketch using the Euler angles and translation. Step 4: Extrude the 2D sketch to generate the 3D model. Step 5: Final dimensions of the 3D model are defined by the length, width, and height parameters.

    Detailed CAD assembly JSON: """
    if "parts" in json_data:
        prompt += f"\nThis model has {len(json_data['parts'])} part(s).\n"
        for part_id, part_info in json_data["parts"].items():
            prompt += f"\n<part_{part_id}>"
            prompt += f"\nPart {part_id}:"
            if "coordinate_system" in part_info:
                coords = part_info["coordinate_system"]
                prompt += f"\n- Coordinate system: Euler angles {coords.get('Euler Angles', 'N/A')}, Translation Vector {coords.get('Translation Vector', 'N/A')}"
            if "sketch" in part_info:
                prompt += "\n- Sketch details:"
                for face_id, face in part_info["sketch"].items():
                    prompt += f"\n - {face_id}:"
                    for loop_id, loop in face.items():
                        prompt += f"\n - {loop_id}:"
                        for primitive_id, primitive in loop.items():
                            if "Circle" in primitive_id:
                                prompt += f"\n - {primitive_id}: Center={primitive.get('Center', 'N/A')}, Radius={primitive.get('Radius', 'N/A')}"
                            elif "Line" in primitive_id:
                                prompt += f"\n - {primitive_id}: Start={primitive.get('Start', 'N/A')}, End={primitive.get('End', 'N/A')}"
                            elif "Arc" in primitive_id:
                                prompt += f"\n - {primitive_id}: Start={primitive.get('Start', 'N/A')}, Mid={primitive.get('Mid', 'N/A')}, End={primitive.get('End', 'N/A')}"
                            else:
                                prompt += f"\n - {primitive_id}: {primitive}"
            if "extrusion" in part_info:
                extrusion = part_info["extrusion"]
                prompt += f"\n- Extrusion details:"
                prompt += f"\n - Extrude depth towards normal: {extrusion.get('extrude_depth_towards_normal', 'N/A')}"
                prompt += f"\n - Extrude depth opposite normal: {extrusion.get('extrude_depth_opposite_normal', 'N/A')}"
                prompt += f"\n - Sketch scale: {extrusion.get('sketch_scale', 'N/A')}"
                prompt += f"\n - Operation: {extrusion.get('operation', 'N/A')}"
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

    Based on the CAD assembly JSON information provided above, write detailed natural language instructions for creating this CAD model. Your instructions should cover the complete process from initial setup to the final model, including all steps for creating and transforming each part. [/INST]"""
    return prompt

def process_single_cad(uid, json_path, pipe, annotation_dir=None):
    print(f"Procesando UID: {uid}")
    t0 = time.time()

    # Cargar datos JSON
    json_data = load_minimal_json(json_path)

    # Extraer abstract (y description igual al abstract, según el paper)
    abstract_description = json_data.get("final_shape", "")
    if not abstract_description and "final_name" in json_data:
        abstract_description = json_data.get("final_name", "")

    if not abstract_description:
        print("No se encontró final_shape, generando abstract...")
        abstract_prompt = f"""
        You are a senior CAD engineer. Based on the following CAD metadata, provide a brief abstract description (1-2 sentences) of what this CAD model looks like.

        CAD metadata: {json.dumps(json_data, indent=2)[:500]}...

        Provide only a brief visual description of the overall shape. """
        abstract_description = generate_response(pipe, abstract_prompt)
        print(f"Abstract Description: {abstract_description[:100]}...")

    # Para el CSV, description es igual al abstract.
    description = abstract_description

    # Generar instrucciones de nivel principiante (L1)
    beginner_prompt = create_beginner_prompt(abstract_description, json_data)
    beginner_out = generate_response(pipe, beginner_prompt)
    beginner_out = f"<level1>\n{beginner_out.strip()}\n</level1>"
    print(f"Beginner Instructions generadas: {beginner_out[:100]}...")

    # Generar instrucciones de nivel intermedio (L2)
    intermediate_prompt = create_intermediate_prompt(abstract_description, beginner_out, json_data)
    intermediate_out = generate_response(pipe, intermediate_prompt)
    intermediate_out = f"<level2>\n{intermediate_out.strip()}\n</level2>"
    print(f"Intermediate Instructions generadas: {intermediate_out[:100]}...")

    # Generar instrucciones de nivel experto (L3)
    expert_prompt = create_expert_prompt(abstract_description, beginner_out, intermediate_out, json_data)
    expert_out = generate_response(pipe, expert_prompt)
    expert_out = f"<level3>\n{expert_out.strip()}\n</level3>"
    print(f"Expert Instructions generadas: {expert_out[:100]}...")

    # Extraer o generar keywords
    keywords = ""
    t_kw = time.time()
    if annotation_dir:
        root_id, sample_id = uid.split('/')
        annot_dir = os.path.join(annotation_dir, uid, "qwen2_vlm_annotation")
        if os.path.exists(annot_dir):
            keywords = extract_keywords_from_annotation(annot_dir, sample_id)
            print(f"Keywords extraídas: {keywords[:100]}...")
    if not keywords:
        print("No se pudieron extraer keywords, generando...")
        keywords_prompt = f"""
        You are a senior CAD engineer. Based on the description of the CAD model below, provide a comma-separated list of 5 keywords. Description: {abstract_description} """
        keywords = generate_response(pipe, keywords_prompt)
        print(f"Keywords generadas: {keywords[:100]}... (en {time.time()-t_kw:.2f} s)")

    # Generar NLI (instrucciones en lenguaje natural, incluyendo tags de partes)
    t_nli = time.time()
    nli_prompt = create_nli_prompt(json_data)
    nli_out = generate_response(pipe, nli_prompt)
    print(f"NLI generado en {time.time()-t_nli:.2f} s, muestra: {nli_out[:100]}...")

    total_time = time.time() - t0
    print(f"UID {uid} procesado en {total_time:.2f} s")

    # Crear objeto con todos los niveles
    all_level_data = {
        "abstract": abstract_description,
        "beginner": beginner_out,
        "intermediate": intermediate_out,
        "expert": expert_out,
        "keywords": keywords
    }

    return {
        "uid": uid,
        "abstract": abstract_description,
        "beginner": beginner_out,
        "expert": expert_out,
        "description": description,
        "keywords": keywords,
        "intermediate": intermediate_out,
        "all_level_data": json.dumps(all_level_data, ensure_ascii=False),
        "nli_data": nli_out
    }

def main():
    parser = argparse.ArgumentParser(description='Generar anotaciones de CAD usando un LLM según Text2CAD')
    parser.add_argument('--input_dir', required=True, help='Directorio raíz con los archivos minimal_json')
    parser.add_argument('--split_json', required=True, help='Archivo JSON con los splits train/test/validation')
    parser.add_argument('--output_file', required=True, help='Archivo CSV de salida')
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-7B-Instruct-1M', help='Nombre del modelo Qwen a utilizar')
    parser.add_argument('--split', default='train', choices=['train', 'test', 'validation', 'all'], help='Split a procesar')
    parser.add_argument('--max_samples', type=int, default=None, help='Número máximo de muestras a procesar')
    parser.add_argument('--annotation_dir', default=None, help='Directorio con anotaciones existentes (para extraer keywords)')
    args = parser.parse_args()

    pipe = load_pipeline(args.model_name)

    with open(args.split_json, 'r') as f:
        split_data = json.load(f)

    if args.split == 'all':
        uids = split_data.get('train', []) + split_data.get('test', []) + split_data.get('validation', [])
    else:
        uids = split_data.get(args.split, [])

    if args.max_samples:
        uids = uids[:args.max_samples]

    print(f"Procesando {len(uids)} UIDs del split '{args.split}'")

    fieldnames = ['uid', 'abstract', 'beginner', 'expert', 'description', 'keywords', 'intermediate', 'all_level_data', 'nli_data']

    with open(args.output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for uid in tqdm(uids):
            root_id, sample_id = uid.split('/')
            # Ajusta la ruta al nombre de archivo de acuerdo a tus datos (ej. _merged_vlm.json2)
            json_path = os.path.join(args.input_dir, uid, "minimal_json", f"{sample_id}_merged_vlm.json")
            if not os.path.exists(json_path):
                print(f"Advertencia: Archivo no encontrado: {json_path}")
                alt_json_path = os.path.join(args.input_dir, root_id, sample_id, "minimal_json", f"{sample_id}_merged_vlm.json")
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