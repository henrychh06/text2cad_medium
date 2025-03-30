import json
import os
from transformers import pipeline

def load_pipeline(model_name):
    """
    Carga el pipeline del modelo especificado.
    """
    print(f"Cargando pipeline del modelo {model_name}...")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype="auto",
        trust_remote_code=True)
    return pipe

def generate_response(pipe, prompt, system_message="You are Qwen, a helpful CAD design expert."):
    """
    Genera una respuesta usando el pipeline y el prompt proporcionado.
    """
    combined_prompt = f"{system_message}\n{prompt}"
    response = pipe(combined_prompt, max_new_tokens=1024, temperature=0.7, do_sample=True)
    generated_text = response[0]['generated_text']
    
    # Depuración - imprimir los primeros caracteres de la respuesta
    print(f"Raw response snippet: {generated_text[:200]}")
    
    if generated_text.startswith(combined_prompt):
        return generated_text[len(combined_prompt):].strip()
    else:
        return generated_text.strip()

def create_all_level_data_prompt_v1(minimal_json, custom_instruction):
    """
    Versión 1: Usa un formato diferente para las etiquetas
    """
    prompt = f"""{custom_instruction}

Minimal JSON data:
{json.dumps(minimal_json, ensure_ascii=False, indent=2)}

Based solely on the above JSON data, generate a response in JSON format with three sections:
{{
  "beginner_level": "Write beginner-level instructions here with simple language and step-by-step overview",
  "intermediate_level": "Write intermediate-level instructions here with geometry overview and relative dimensions",
  "expert_level": "Write expert-level instructions here with precise parameters and technical details"
}}

Ensure your response is valid JSON that can be parsed without errors.
"""
    return prompt

def create_all_level_data_prompt_v2(minimal_json, custom_instruction):
    """
    Versión 2: Simplifica el prompt y solicita explícitamente formato JSON
    """
    # Limitar la longitud del JSON si es necesario
    json_str = json.dumps(minimal_json, ensure_ascii=False, indent=2)
    if len(json_str) > 2000:
        json_str = json_str[:2000] + "... (truncated for length)"
    
    prompt = f"""{custom_instruction}

Minimal JSON data:
{json_str}

Generate CAD instructions in three difficulty levels based on the JSON data above.
Return your response in valid JSON format as follows:

{{
  "beginner": "Simple language instructions with basic steps here",
  "intermediate": "More detailed instructions with geometry overview here",
  "expert": "Precise technical details for exact reproduction here"
}}

Your response must be valid JSON that can be parsed directly.
"""
    return prompt

def create_all_level_data_prompt_v3(minimal_json, custom_instruction):
    """
    Versión 3: Enfoque más directo con menos marcado especial
    """
    prompt = f"""{custom_instruction}

Based on this CAD data:
{json.dumps(minimal_json, ensure_ascii=False)}

Write three sets of CAD instructions:
1. BEGINNER: Simple step-by-step instructions (3-5 sentences)
2. INTERMEDIATE: Overview with geometry and relative dimensions (5-7 sentences)
3. EXPERT: Precise technical details for exact reproduction (7-10 sentences)

Format your response as valid JSON: {{"beginner": "...", "intermediate": "...", "expert": "..."}}
"""
    return prompt

def create_all_level_data_prompt_original(minimal_json, custom_instruction):
    """
    Versión Original: El prompt exactamente como estaba en el código original
    """
    prompt = f"""{custom_instruction}

Minimal JSON data:
{json.dumps(minimal_json, ensure_ascii=False, indent=2)}

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

def load_minimal_json(json_path):
    """
    Carga un archivo JSON.
    """
    with open(json_path, 'r') as f:
        return json.load(f)

def test_all_level_data_prompt(json_path, model_name="Qwen/Qwen2.5-7B-Instruct-1M"):
    """
    Prueba las tres versiones de la función y guarda los resultados.
    """
    # Cargar el modelo
    pipe = load_pipeline(model_name)
    
    # Cargar los datos JSON
    json_data = load_minimal_json(json_path)
    
    # Instrucción personalizada
    custom_instruction = "You are a senior CAD engineer. Generate instructions for creating this CAD model."
    
    # Probar las cuatro versiones
    results = {}
    
    # Versión Original
    print("\nProbando Versión Original...")
    prompt_original = create_all_level_data_prompt_original(json_data, custom_instruction)
    response_original = generate_response(pipe, prompt_original)
    results["version_original"] = {
        "prompt": prompt_original[:200] + "...",  # Truncado para brevedad
        "response": response_original
    }
    
    # Versión 1
    print("\nProbando Versión 1...")
    prompt_v1 = create_all_level_data_prompt_v1(json_data, custom_instruction)
    response_v1 = generate_response(pipe, prompt_v1)
    results["version_1"] = {
        "prompt": prompt_v1[:200] + "...",  # Truncado para brevedad
        "response": response_v1
    }
    
    # Versión 2
    print("\nProbando Versión 2...")
    prompt_v2 = create_all_level_data_prompt_v2(json_data, custom_instruction)
    response_v2 = generate_response(pipe, prompt_v2)
    results["version_2"] = {
        "prompt": prompt_v2[:200] + "...",
        "response": response_v2
    }
    
    # Versión 3
    print("\nProbando Versión 3...")
    prompt_v3 = create_all_level_data_prompt_v3(json_data, custom_instruction)
    response_v3 = generate_response(pipe, prompt_v3)
    results["version_3"] = {
        "prompt": prompt_v3[:200] + "...",
        "response": response_v3
    }
    
    # Intentar parsear las respuestas como JSON
    parsed_results = {}
    for version, data in results.items():
        try:
            parsed_json = json.loads(data["response"])
            parsed_results[version] = {
                "successfully_parsed": True,
                "parsed_data": parsed_json
            }
        except json.JSONDecodeError as e:
            parsed_results[version] = {
                "successfully_parsed": False,
                "error": str(e)
            }
    
    # Guardar resultados
    output = {
        "prompt_responses": results,
        "parsing_results": parsed_results,
        "recommended_version": "Determinar según resultados"
    }
    
    # Determinar recomendación
    for version, result in parsed_results.items():
        if result["successfully_parsed"]:
            output["recommended_version"] = version
            break
    
    # Guardar resultados en un archivo JSON
    with open("all_level_prompt_test_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nResultados guardados en 'all_level_prompt_test_results.json'")
    return output

def process_multiple_samples(input_dir, split_json, model_name, split='train', max_samples=None):
    """
    Procesa múltiples muestras según el archivo de split.
    """
    # Cargar el modelo
    pipe = load_pipeline(model_name)
    
    # Cargar el archivo de split
    with open(split_json, 'r') as f:
        split_data = json.load(f)
    
    if split == 'all':
        uids = split_data.get('train', []) + split_data.get('test', []) + split_data.get('validation', [])
    else:
        uids = split_data.get(split, [])
    
    if max_samples:
        uids = uids[:max_samples]
    
    print(f"Procesando {len(uids)} UIDs del split '{split}'")
    
    results = {
        "samples": [],
        "summary": {
            "version_original": {"success_count": 0, "failure_count": 0},
            "version_1": {"success_count": 0, "failure_count": 0},
            "version_2": {"success_count": 0, "failure_count": 0},
            "version_3": {"success_count": 0, "failure_count": 0}
        }
    }
    
    custom_instruction = "You are a senior CAD engineer. Generate instructions for creating this CAD model."
    
    for i, uid in enumerate(uids):
        print(f"\nProcesando muestra {i+1}/{len(uids)}: {uid}")
        
        root_id, sample_id = uid.split('/')
        # Ajusta la ruta al nombre de archivo de acuerdo a tus datos
        json_path = os.path.join(input_dir, uid, "minimal_json", f"{sample_id}_merged_vlm.json")
        
        if not os.path.exists(json_path):
            print(f"Advertencia: Archivo no encontrado: {json_path}")
            alt_json_path = os.path.join(input_dir, root_id, sample_id, "minimal_json", f"{sample_id}_merged_vlm.json")
            if os.path.exists(alt_json_path):
                json_path = alt_json_path
                print(f"Usando ruta alternativa: {json_path}")
            else:
                print(f"Error: No se pudo encontrar el archivo JSON para {uid}")
                continue
        
        try:
            # Cargar los datos JSON
            json_data = load_minimal_json(json_path)
            
            sample_results = {"uid": uid, "versions": {}}
            
            # Probar las cuatro versiones
            # Versión Original
            print(f"  Probando Versión Original para {uid}...")
            prompt_original = create_all_level_data_prompt_original(json_data, custom_instruction)
            response_original = generate_response(pipe, prompt_original)
            sample_results["versions"]["version_original"] = {"response": response_original}
            
            # Verificar si la respuesta contiene los símbolos de error
            contains_error_symbols = "⚗" in response_original
            if contains_error_symbols:
                results["summary"]["version_original"]["failure_count"] += 1
                sample_results["versions"]["version_original"]["status"] = "failure"
            else:
                results["summary"]["version_original"]["success_count"] += 1
                sample_results["versions"]["version_original"]["status"] = "success"
            
            # Versión 1
            print(f"  Probando Versión 1 para {uid}...")
            prompt_v1 = create_all_level_data_prompt_v1(json_data, custom_instruction)
            response_v1 = generate_response(pipe, prompt_v1)
            sample_results["versions"]["version_1"] = {"response": response_v1}
            
            # Verificar si la respuesta contiene los símbolos de error
            contains_error_symbols = "⚗" in response_v1
            if contains_error_symbols:
                results["summary"]["version_1"]["failure_count"] += 1
                sample_results["versions"]["version_1"]["status"] = "failure"
            else:
                results["summary"]["version_1"]["success_count"] += 1
                sample_results["versions"]["version_1"]["status"] = "success"
            
            # Versión 2
            print(f"  Probando Versión 2 para {uid}...")
            prompt_v2 = create_all_level_data_prompt_v2(json_data, custom_instruction)
            response_v2 = generate_response(pipe, prompt_v2)
            sample_results["versions"]["version_2"] = {"response": response_v2}
            
            # Verificar si la respuesta contiene los símbolos de error
            contains_error_symbols = "⚗" in response_v2
            if contains_error_symbols:
                results["summary"]["version_2"]["failure_count"] += 1
                sample_results["versions"]["version_2"]["status"] = "failure"
            else:
                results["summary"]["version_2"]["success_count"] += 1
                sample_results["versions"]["version_2"]["status"] = "success"
            
            # Versión 3
            print(f"  Probando Versión 3 para {uid}...")
            prompt_v3 = create_all_level_data_prompt_v3(json_data, custom_instruction)
            response_v3 = generate_response(pipe, prompt_v3)
            sample_results["versions"]["version_3"] = {"response": response_v3}
            
            # Verificar si la respuesta contiene los símbolos de error
            contains_error_symbols = "⚗" in response_v3
            if contains_error_symbols:
                results["summary"]["version_3"]["failure_count"] += 1
                sample_results["versions"]["version_3"]["status"] = "failure"
            else:
                results["summary"]["version_3"]["success_count"] += 1
                sample_results["versions"]["version_3"]["status"] = "success"
            
            results["samples"].append(sample_results)
            
            # Guardar resultados parciales cada 5 muestras
            if (i+1) % 5 == 0 or i == len(uids) - 1:
                with open("all_level_prompt_test_results.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Resultados parciales guardados ({i+1}/{len(uids)} muestras procesadas)")
                
        except Exception as e:
            print(f"Error procesando {uid}: {e}")
            continue
    
    # Determinar la mejor versión
    best_version = None
    max_success_count = -1
    
    for version, counts in results["summary"].items():
        success_rate = counts["success_count"] / (counts["success_count"] + counts["failure_count"]) if (counts["success_count"] + counts["failure_count"]) > 0 else 0
        counts["success_rate"] = success_rate
        
        if counts["success_count"] > max_success_count:
            max_success_count = counts["success_count"]
            best_version = version
    
    results["best_version"] = best_version
    
    # Guardar resultados finales
    with open("all_level_prompt_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResultados guardados en 'all_level_prompt_test_results.json'")
    print(f"Mejor versión: {best_version} con {max_success_count} éxitos")
    
    return results

if __name__ == "__main__":
    # El usuario debe especificar la ruta al archivo JSON
    import argparse
    parser = argparse.ArgumentParser(description='Probar diferentes versiones de la función all_level_data_prompt')
    parser.add_argument('--input_dir', required=True, help='Directorio raíz con los archivos minimal_json')
    parser.add_argument('--split_json', required=False, help='Archivo JSON con los splits train/test/validation')
    parser.add_argument('--json_path', required=False, help='Ruta a un único archivo JSON minimal para prueba')
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-7B-Instruct-1M', help='Nombre del modelo a utilizar')
    parser.add_argument('--split', default='train', choices=['train', 'test', 'validation', 'all'], help='Split a procesar')
    parser.add_argument('--max_samples', type=int, default=None, help='Número máximo de muestras a procesar')
    args = parser.parse_args()
    
    if args.json_path:
        # Procesar un solo archivo JSON
        test_all_level_data_prompt(args.json_path, args.model_name)
    elif args.input_dir and args.split_json:
        # Procesar múltiples muestras según el archivo de split
        process_multiple_samples(args.input_dir, args.split_json, args.model_name, args.split, args.max_samples)
    else:
        print("Error: Debe especificar --json_path para procesar un solo archivo o --input_dir y --split_json para procesar múltiples muestras.")
        parser.print_help()