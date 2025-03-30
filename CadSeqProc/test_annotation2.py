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
        torch_dtype="auto",
        trust_remote_code=True)
    return pipe

def generate_response(pipe, prompt, system_message="You are Qwen, a helpful CAD design expert."):
    """
    Genera una respuesta combinando el mensaje del sistema y el prompt.
    Se elimina del output el system_message (si aparece al inicio) para que la respuesta "limpia" contenga únicamente la generación del LLM.
    """
    # Simplificar el mensaje del sistema para evitar confusiones 
    system_message = "You are a CAD design expert."
    
    combined_prompt = f"{system_message}\n{prompt}"
    response = pipe(combined_prompt, max_new_tokens=1024, temperature=0.7, do_sample=True)
    generated_text = response[0]['generated_text']
    
    # Limpiar posibles tokens de control en la respuesta
    clean_text = re.sub(r'#RegionBegin.*?RegionEnd[⚗️⚗]?', '', generated_text, flags=re.DOTALL)
    clean_text = re.sub(r'[⚗️⚗]', '', clean_text)
    
    # Si el output empieza con el system_message, se elimina esa parte:
    if clean_text.startswith(combined_prompt):
        clean_text = clean_text[len(combined_prompt):].strip()
    
    return clean_text.strip()

def load_minimal_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_keywords_from_annotation(uid, annotation_dir):
    try:
        root_id, sample_id = uid.split('/')
        annotation_files = glob(os.path.join(annotation_dir, root_id, sample_id, "qwen2_vlm_annotation", f"final_{sample_id}.json"))
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

For reference, here is the complete CAD data:
{json.dumps(minimal_json, indent=2, ensure_ascii=False)}

Provide beginner-level CAD instructions that:
- Use simple language and avoid technical jargon
- Present a clear, step-by-step overview of the basic shape and process
- Limit the explanation to 5-6 sentences
"""
    return prompt

def create_intermediate_prompt(minimal_json, custom_instruction):
    """
    Genera el prompt para el nivel intermediate usando el minimal JSON y un mensaje personalizado.
    """
    prompt = f"""{custom_instruction}

For reference, here is the complete CAD data:
{json.dumps(minimal_json, indent=2, ensure_ascii=False)}

Provide intermediate-level CAD instructions that:
- Include an overview of the geometry and main construction steps
- Describe relative dimensions and spatial relationships (without exact coordinates)
- Use moderate technical detail appropriate for someone with basic CAD knowledge
- Limit the explanation to around 7-10 sentences
"""
    return prompt

def create_expert_prompt(minimal_json, custom_instruction):
    """
    Genera el prompt para el nivel expert usando el minimal JSON y un mensaje personalizado.
    """
    prompt = f"""{custom_instruction}

For reference, here is the complete CAD data:
{json.dumps(minimal_json, indent=2, ensure_ascii=False)}

Provide expert-level CAD instructions that:
- Include precise parameters, detailed transformation steps, and complete CAD terminology
- Provide exact measurements and technical details that allow an expert to reproduce the model exactly
- Follow a logical, step-by-step process with clear technical instructions
"""
    return prompt

def create_all_level_data_prompt(minimal_json, custom_instruction):
    """
    Versión simplificada que solicita directamente el formato de salida deseado
    """
    prompt = f"""{custom_instruction}

For reference, here is the complete CAD data:
{json.dumps(minimal_json, ensure_ascii=False)}

Write THREE sets of CAD instructions with different detail levels:

BEGINNER LEVEL:
Create simple step-by-step instructions that use basic language and avoid technical terms (2-4 sentences).

INTERMEDIATE LEVEL:
Provide more detailed instructions with geometry overview and relative dimensions (4-6 sentences).

EXPERT LEVEL:
Provide more detailed instruction and give precise technical details with exact measurements and CAD terminology (6-8 sentences).

Format your response EXACTLY like this:

<level1>
(Your beginner instructions here)
</level1>

<level2>
(Your intermediate instructions here)
</level2>

<level3>
(Your expert instructions here)
</level3>
"""
    return prompt

def create_part_instruction_prompt(part_data, part_id, custom_instruction):
    """
    Crea un prompt para generar instrucciones para una parte específica del CAD
    """
    prompt = f"""{custom_instruction}

I need to create Part {part_id} for a CAD model. Here is the complete part data:
{json.dumps(part_data, ensure_ascii=False, indent=2)}

Provide clear, detailed instructions for creating ONLY this specific part.
Include step-by-step guidance on sketching, positioning, and extruding this part.
Be specific and focus on the practical aspects of creating this component.

These intermediate CAD parts are listed in the "parts" key of the CAD assembly JSON.

The first intermediate CAD part is the base part and the subsequent parts build upon the previously constructed parts using the operation defined for that part.

All intermediate parts combine to a final CAD model.

Every intermediate CAD part is generated using the following steps: Step 1: Draw a 2D sketch. Step 2: Scale the 2D sketch using the sketch_scale parameter. Step 3: Transform the scaled 2D sketch into a 3D sketch using the Euler angles and translation. Step 4: Extrude the 2D sketch to generate the 3D model. Step 5: Final dimensions of the 3D model are defined by the length, width, and height parameters.

"""
    return prompt

def create_nli_prompt(json_data):
    """
    Versión completa del prompt NLI que incluye todos los detalles del modelo CAD
    """
    prompt = """[INST] You are a senior CAD engineer and you are tasked to provide natural language instructions to a junior CAD designer for generating a parametric CAD model.

    Overview information about the CAD assembly JSON:

    The CAD assembly JSON lists the process of constructing a CAD model.

    Every CAD model consists of one or multiple intermediate CAD parts.

    These intermediate CAD parts are listed in the "parts" key of the CAD assembly JSON.

    The first intermediate CAD part is the base part and the subsequent parts build upon the previously constructed parts using the operation defined for that part.

    All intermediate parts combine to a final CAD model.

    Every intermediate CAD part is generated using the following steps: Step 1: Draw a 2D sketch. Step 2: Scale the 2D sketch using the sketch_scale parameter. Step 3: Transform the scaled 2D sketch into a 3D sketch using the Euler angles and translation. Step 4: Extrude the 2D sketch to generate the 3D model. Step 5: Final dimensions of the 3D model are defined by the length, width, and height parameters.

    Detailed CAD assembly JSON:
"""
    # Incluir todos los datos del JSON
    prompt += json.dumps(json_data, ensure_ascii=False, indent=2)
    
    prompt += """

Provide detailed instructions to create this CAD model. Include all necessary steps from initial setup to final model.
Based on the CAD assembly JSON information provided above, write detailed natural language instructions for creating this CAD model. Your instructions should cover the complete process from initial setup to the final model, including all steps for creating and transforming each part.

Your instructions should be practical and directly applicable in a CAD program.

"""
    
    return prompt

def process_single_cad(uid, json_path, pipe, annotation_dir=None):
    """
    Procesa un archivo CAD usando el minimal JSON y genera los distintos niveles de instrucciones,
    además de otros campos requeridos para el CSV.
    """
    print(f"Procesando UID: {uid}")
    try:
        # Cargar datos JSON
        json_data = load_minimal_json(json_path)
        
        # Generar abstract
        abstract_description = json_data.get("final_shape", "")
        if not abstract_description and "final_name" in json_data:
            abstract_description = json_data.get("final_name", "")
        if not abstract_description:
            abstract_prompt = "Based on the CAD metadata, provide a brief description (1-2 sentences) of what this CAD model looks like."
            abstract_description = generate_response(pipe, abstract_prompt)
        
        # Para el CSV, description se iguala al abstract
        description = abstract_description
        print(f"Descripción generada: {description[:50]}...")

        # Configurar instrucciones personalizadas
        custom_instruction = "You are a senior CAD engineer."

        # Generar instrucciones de nivel Beginner
        print("Generando instrucciones de nivel Beginner...")
        beginner_prompt = create_beginner_prompt(json_data, custom_instruction)
        beginner_out = generate_response(pipe, beginner_prompt)
        beginner_out = beginner_out.strip()
        print(f"Beginner generado: {len(beginner_out)} caracteres")

        # Generar instrucciones de nivel Intermediate
        print("Generando instrucciones de nivel Intermediate...")
        intermediate_prompt = create_intermediate_prompt(json_data, custom_instruction)
        intermediate_out = generate_response(pipe, intermediate_prompt)
        intermediate_out = intermediate_out.strip()
        print(f"Intermediate generado: {len(intermediate_out)} caracteres")

        # Generar instrucciones de nivel Expert
        print("Generando instrucciones de nivel Expert...")
        expert_prompt = create_expert_prompt(json_data, custom_instruction)
        expert_out = generate_response(pipe, expert_prompt)
        expert_out = expert_out.strip()
        print(f"Expert generado: {len(expert_out)} caracteres")

        # Generar el campo all_level_data
        print("Generando all_level_data...")
        all_level_data_prompt = create_all_level_data_prompt(json_data, custom_instruction)
        all_level_data_out = generate_response(pipe, all_level_data_prompt)
        all_level_data_out = all_level_data_out.strip()
        print(f"all_level_data generado: {len(all_level_data_out)} caracteres")

        # Extraer o generar keywords
        print("Generando keywords...")
        keywords = ""
        if annotation_dir:
            sample_id = uid.split('/')[-1]
            print(f"Extrayendo keywords de anotaciones existentes para {sample_id}...")
            keywords = extract_keywords_from_annotation(uid, annotation_dir)
            print(f"Keywords extraídas: {keywords[:50]}...")
        
        if not keywords:
            start_time = time.time()
            keywords_prompt = f"Based on this description: '{description}', provide a comma-separated list of 5 keywords for CAD modeling, in this format [keyword_1, keyword_2, keyword_3, keyword_4, keyword_5]."
            keywords = generate_response(pipe, keywords_prompt)
            keywords = keywords.strip()
            print(f"Keywords generadas: {keywords[:100]}... (en {time.time() - start_time:.2f} s)")

        # Generar NLI
        print(f"Generando NLI para {uid}...")
        nli_prompt = create_nli_prompt(json_data)
        nli_raw = generate_response(pipe, nli_prompt)
        print(f"NLI raw generado: {len(nli_raw)} caracteres")
        
        # Procesar NLI para asegurar que cada parte esté dentro de sus tags
        nli_out = ""
        if "parts" in json_data:
            for part_id in json_data["parts"].keys():
                print(f"Generando instrucciones para parte {part_id}...")
                part_prompt = create_part_instruction_prompt(json_data["parts"][part_id], part_id, custom_instruction)
                part_instructions = generate_response(pipe, part_prompt)
                
                # Verificar que las instrucciones no estén vacías
                if len(part_instructions.strip()) < 10:
                    print(f"Advertencia: Instrucciones para parte {part_id} demasiado cortas, usando instrucciones generales")
                    part_instructions = f"Create part {part_id} according to the specifications in the JSON data. This part should be created using the appropriate sketching tools, then extruded to the specified dimensions."
                
                nli_out += f"\n<{part_id}>\n{part_instructions.strip()}\n</{part_id}>\n"
                print(f"Instrucciones para parte {part_id} generadas: {len(part_instructions)} caracteres")
        
        # Si no se generaron instrucciones por partes o están vacías, usar el NLI raw
        if not nli_out.strip():
            print("Usando NLI raw debido a que las instrucciones por partes están vacías")
            nli_out = nli_raw

        # Procesar el all_level_data para extraer secciones
        print("Procesando all_level_data para extraer secciones...")
        
        # Intentar extraer secciones mediante etiquetas
        level1_match = re.search(r'<level1>(.*?)</level1>', all_level_data_out, re.DOTALL)
        level2_match = re.search(r'<level2>(.*?)</level2>', all_level_data_out, re.DOTALL)
        level3_match = re.search(r'<level3>(.*?)</level3>', all_level_data_out, re.DOTALL)
        
        if level1_match and level2_match and level3_match:
            # Si se encontraron todas las etiquetas, extraer el contenido
            beginner_from_all = level1_match.group(1).strip()
            intermediate_from_all = level2_match.group(1).strip()
            expert_from_all = level3_match.group(1).strip()
            
            # Usar estas versiones si no están vacías y son suficientemente largas
            if len(beginner_from_all) > 20:
                beginner_out = beginner_from_all
            if len(intermediate_from_all) > 20:
                intermediate_out = intermediate_from_all
            if len(expert_from_all) > 20:
                expert_out = expert_from_all
                
            print("Etiquetas de nivel extraídas correctamente de all_level_data")
        else:
            print("No se encontraron etiquetas en all_level_data, manteniendo generaciones individuales")
        
        # Construir el all_level_data final con etiquetas XML
        all_level_data_formatted = f"""<level1>
{beginner_out}
</level1>

<level2>
{intermediate_out}
</level2>

<level3>
{expert_out}
</level3>"""
        
        # Actualizar all_level_data_out con el formato XML
        all_level_data_out = all_level_data_formatted
        
        # Construir el diccionario final para el CSV
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
        
        # Verificar que ningún campo esté vacío
        for key, value in result.items():
            if key != "uid" and not value.strip():
                print(f"Advertencia: El campo {key} está vacío, generando contenido predeterminado")
                result[key] = f"Default {key} content for {uid}. This field was empty in the original generation."
        
        return result
        
    except Exception as e:
        print(f"Error grave procesando {uid}: {e}")
        # Crear un resultado de fallback para no interrumpir todo el proceso
        return {
            "uid": uid,
            "abstract": f"Error generating content for {uid}: {str(e)}",
            "beginner": f"Error generating beginner instructions for {uid}",
            "expert": f"Error generating expert instructions for {uid}",
            "description": f"Error generating description for {uid}",
            "keywords": "error, generation, failed, cad, model",
            "intermediate": f"Error generating intermediate instructions for {uid}",
            "all_level_data": f"<level1>Error generating level 1 content for {uid}</level1>\n\n<level2>Error generating level 2 content for {uid}</level2>\n\n<level3>Error generating level 3 content for {uid}</level3>",
            "nli_data": f"Error generating NLI data for {uid}"
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
    parser.add_argument('--start_index', type=int, default=0, help='Índice inicial para continuar procesamiento interrumpido')
    args = parser.parse_args()

    pipe = load_pipeline(args.model_name)

    with open(args.split_json, 'r') as f:
        split_data = json.load(f)

    if args.split == 'all':
        uids = split_data.get('train', []) + split_data.get('test', []) + split_data.get('validation', [])
    else:
        uids = split_data.get(args.split, [])

    # Aplicar inicio desde índice específico
    if args.start_index > 0:
        print(f"Continuando desde el índice {args.start_index}")
        uids = uids[args.start_index:]

    if args.max_samples:
        uids = uids[:args.max_samples]

    print(f"Procesando {len(uids)} UIDs del split '{args.split}'")

    fieldnames = ['uid', 'abstract', 'beginner', 'expert', 'description', 'keywords', 'intermediate', 'all_level_data', 'nli_data']

    # Verificar si el archivo ya existe para decidir si escribir encabezados
    file_exists = os.path.isfile(args.output_file) and os.path.getsize(args.output_file) > 0
    
    with open(args.output_file, 'a' if file_exists else 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Solo escribir encabezados si es un archivo nuevo
        if not file_exists:
            writer.writeheader()
        
        for i, uid in enumerate(tqdm(uids)):
            root_id, sample_id = uid.split('/')
            # Ajusta la ruta al nombre de archivo de acuerdo a tus datos
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
                csvfile.flush()  # Forzar la escritura a disco después de cada muestra
                
                # Guardar un registro del progreso
                if (i + 1) % 10 == 0 or i == len(uids) - 1:
                    with open('progress.txt', 'w') as f:
                        f.write(f"Procesadas {i + 1}/{len(uids)} muestras\n")
                        f.write(f"Última muestra procesada: {uid}\n")
                        f.write(f"Índice global: {args.start_index + i + 1}\n")
                
            except Exception as e:
                print(f"Error procesando {uid}: {e}")
                # Intentar continuar con la siguiente muestra
                continue

if __name__ == "__main__":
    main()