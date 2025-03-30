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
        annotation_files = glob(os.path.join(annotation_dir, sample_id, "minimal_json",  f"{sample_id}.json"))
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

def create_beginner_prompt(minimal_json, custom_instruction):
    """
    Genera el prompt para el nivel beginner usando el minimal JSON y un mensaje personalizado.
    """
    prompt = f"""{custom_instruction}

Minimal JSON data:
{json.dumps(minimal_json, indent=2)}

Based solely on the above JSON data, provide beginner-level CAD instructions that:
- Use simple language and avoid technical jargon.
- Present a clear, step-by-step overview of the basic shape and process.
- Limit the explanation to 5-6 sentences.
"""
    return prompt

def create_intermediate_prompt(minimal_json, custom_instruction):
    """
    Genera el prompt para el nivel intermediate usando el minimal JSON y un mensaje personalizado.
    """
    prompt = f"""{custom_instruction}

Minimal JSON data:
{json.dumps(minimal_json, indent=2)}

Based solely on the above JSON data, provide intermediate-level CAD instructions that:
- Include an overview of the geometry and main construction steps.
- Describe relative dimensions and spatial relationships (without exact coordinates).
- Use moderate technical detail appropriate for someone with basic CAD knowledge.
- Limit the explanation to around 7-10 sentences.
"""
    return prompt

def create_expert_prompt(minimal_json, custom_instruction):
    """
    Genera el prompt para el nivel expert usando el minimal JSON y un mensaje personalizado.
    """
    prompt = f"""{custom_instruction}

Minimal JSON data:
{json.dumps(minimal_json, indent=2)}

Based solely on the above JSON data, provide expert-level CAD instructions that:
- Include precise parameters, detailed transformation steps, and complete CAD terminology.
- Provide exact measurements and technical details that allow an expert to reproduce the model exactly.
- Follow a logical, step-by-step process with clear technical instructions.
"""
    return prompt


def create_all_level_data_prompt(minimal_json, custom_instruction):
    """
    Genera el prompt para combinar los tres niveles (beginner, intermediate y expert) en un solo bloque,
    etiquetado con <level1>, <level2> y <level3>.
    """
    prompt = f"""{custom_instruction}

Minimal JSON data:
{json.dumps(minimal_json, indent=2)}

Based solely on the above JSON data, generate a response that contains three sections:
<level1>
[Beginner-level instructions: use simple language with a step-by-step overview of the CAD process.]
</level1>

<level2>
[Intermediate-level instructions: include an overview of the geometry, construction steps and relative dimensions without exact coordinates.]
</level2>

<level3>
[Expert-level instructions: provide precise parameters, technical details, and a step-by-step process that an expert can follow to reproduce the model exactly.]
</level3>

Ensure that the output includes all three sections in the exact order, and that each section is clearly delimited by its respective tags.
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
    """
    Procesa un archivo CAD usando el minimal JSON y genera los distintos niveles de instrucciones,
    además de otros campos requeridos para el CSV.
    """
    print(f"Procesando UID: {uid}")
    # Cargar datos JSON
    json_data = load_minimal_json(json_path)
    
    # Generar abstract (puede ser extraído o generado; en este ejemplo se toma el "final_shape" o se genera)
    abstract_description = json_data.get("final_shape", "")
    if not abstract_description and "final_name" in json_data:
        abstract_description = json_data.get("final_name", "")
    if not abstract_description:
        abstract_prompt = f"""
        You are a senior CAD engineer. Based on the following CAD metadata, provide a brief abstract description (1-2 sentences) of what this CAD model looks like.

        CAD metadata: {json.dumps(json_data, indent=2)}...
        Provide only a brief visual description of the overall shape.
        """
        abstract_description = generate_response(pipe, abstract_prompt)
    
    # Para el CSV, description se iguala al abstract
    description = abstract_description

    # Configurar instrucciones personalizadas para cada nivel
    custom_instruction_beginner = (
        "You are a senior CAD engineer. Provide beginner-level instructions for creating the CAD model."
    )
    custom_instruction_intermediate = (
        "You are a senior CAD engineer. Provide intermediate-level instructions for creating the CAD model."
    )
    custom_instruction_expert = (
        "You are a senior CAD engineer. Provide expert-level instructions for creating the CAD model with precise parameters."
    )
    custom_instruction_all_levels = (
        "You are a senior CAD engineer. Combine all levels of instructions into a single structured output."
    )

    # Generar instrucciones de nivel Beginner
    beginner_prompt = create_beginner_prompt(json_data, custom_instruction_beginner)
    beginner_out = generate_response(pipe, beginner_prompt)
    beginner_out = beginner_out.strip()  # Se puede ajustar el formato si se requiere

    # Generar instrucciones de nivel Intermediate
    intermediate_prompt = create_intermediate_prompt(json_data, custom_instruction_intermediate)
    intermediate_out = generate_response(pipe, intermediate_prompt)
    intermediate_out = intermediate_out.strip()

    # Generar instrucciones de nivel Expert
    expert_prompt = create_expert_prompt(json_data, custom_instruction_expert)
    expert_out = generate_response(pipe, expert_prompt)
    expert_out = expert_out.strip()

    # Generar el campo all_level_data que agrupa los tres niveles con etiquetas
    all_level_data_prompt = create_all_level_data_prompt(json_data, custom_instruction_all_levels)
    all_level_data_out = generate_response(pipe, all_level_data_prompt)
    all_level_data_out = all_level_data_out.strip()

    # Extraer o generar keywords (en este ejemplo se asume generación si no se extraen)
    keywords = ""
    if annotation_dir:
        # Aquí se podría extraer de anotaciones existentes
        # Por simplicidad, se omite esta parte y se genera con el LLM si no hay keywords
        pass
    if not keywords:
        keywords_prompt = f"""
        You are a senior CAD engineer. Based on the following description of the CAD model, provide a comma-separated list of 5 keywordsm, in this format [keyword_1, ... , keyword_5].
        Description: {abstract_description}
        """
        keywords = generate_response(pipe, keywords_prompt)
        keywords = keywords.strip()

    # Generar NLI (instrucciones en lenguaje natural detalladas con tags de partes) - se mantiene igual según tu ejemplo original
    nli_prompt = create_nli_prompt(json_data)
    nli_out = generate_response(pipe, nli_prompt)
    
    # Construir el diccionario final para el CSV siguiendo el orden deseado
    result = {
        "uid": uid,
        "abstract": abstract_description,
        "beginner": beginner_out,
        "expert": expert_out,
        "description": description,
        "keywords": keywords,
        "intermediate": intermediate_out,
        "all_level_data": all_level_data_out,
        "nli_data": nli_out
    }
    
    return result

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