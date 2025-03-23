import os
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
import glob

def ensure_dir(directory):
    """Ensure a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[DEBUG] Directorio creado: {directory}")
    else:
        print(f"[DEBUG] Directorio existente: {directory}")

def get_rendered_images(uid, renders_base_dir, n_views, part_id=None):
    """
    Obtiene las rutas de las imágenes multi-view ya generadas para un CAD.
    Se asume que los UIDs están en formato "0001/00010160" y que las imágenes
    están en:
    - Para modelo final: renders_base_dir/0001/00010160/views/00010160_final/00010160_final_view_X.png
    - Para partes: renders_base_dir/0001/00010160/views/00010160_intermediate_X/00010160_intermediate_X_view_Y.png
    """
    root_id, sample_id = uid.split('/')
    
    # Determinamos qué carpeta buscar en función de si es el modelo final o una parte
    if part_id is None or part_id == "final":
        view_folder = f"{sample_id}_final"
    else:
        # Asumimos que part_id es algo como "part_1", "part_2", etc.
        part_num = part_id.split('_')[-1]
        view_folder = f"{sample_id}_intermediate_{part_num}"
    
    folder = os.path.join(renders_base_dir, root_id, sample_id, "views", view_folder)
    print(f"[DEBUG] Buscando imágenes en: {folder}")
    
    if not os.path.exists(folder):
        print(f"[DEBUG] Carpeta de imágenes no encontrada: {folder}")
        return None
    
    # Buscamos las imágenes con patrón específico
    if part_id is None or part_id == "final":
        pattern = f"{sample_id}_final_view_*.png"
    else:
        part_num = part_id.split('_')[-1]
        pattern = f"{sample_id}_intermediate_{part_num}_view_*.png"
    
    image_paths = sorted(glob.glob(os.path.join(folder, pattern)))
    print(f"[DEBUG] Se encontraron {len(image_paths)} imágenes en {folder} con patrón {pattern}")
    
    if len(image_paths) < n_views:
        print(f"[DEBUG] Se esperaban {n_views} vistas pero se encontraron {len(image_paths)} en {folder}")
    
    return image_paths[:n_views] if image_paths else None

def generate_shape_description(image_paths, device, vlm_pipe, is_part=False):
    if not image_paths:
        print("[DEBUG] No hay imágenes para generar descripción.")
        return "<NAME>Unknown</NAME>\n<DESCRIPTION>Shape description not available.</DESCRIPTION>\n<KEYWORDS></KEYWORDS>"
    
    descriptions = []
    name_descriptions = []
    
    # Modificamos el prompt para generar el formato esperado
    prompt = (
        """This is an image of a Computer Aided Design (CAD) model.  You
        are a senior CAD engineer who knows the object name, where and how
        the CAD model is used. Give an accurate natural language description
        about the CAD model to a junior CAD designer who can design it from
        your simple description. Wrap the description in the following tags"""
        "You are a senior CAD engineer who needs to: "
        "1. Identify a precise name for this CAD component "
        "2. Describe its shape and structure "
        "3. List keywords related to the component "
        
        "Format your response EXACTLY as follows (including the tags):\n"
        "<NAME>Brief component name (1-3 words)</NAME>\n"
        "<DESCRIPTION>Detailed description focusing on shape, structure, and geometric features (6-12 words)</DESCRIPTION>\n"
        "<KEYWORDS>keyword1, keyword2, keyword3, ...(4-6 keywords)</KEYWORDS>\n\n"
        
        "Rules:\n"
        "- Do not use words like 'blue', 'shadow', 'transparent', 'metal', 'plastic', 'image', 'black', 'grey', 'CAD model', 'abstract', 'orange', 'purple', 'golden', 'green'\n"
        "- Focus on shape, structure, and geometric features\n"
        "- Do not mention colors, materials, or rendering aspects\n"
        "- Following are some bad examples: 1. CAD model 2. Metal object\n"
        f"- You are looking at {'a part of a larger assembly' if is_part else 'a complete CAD model'}"
    )
    
    for img_path in image_paths:
        try:
            print(f"[DEBUG] Procesando imagen: {img_path}")
        except Exception as e:
            print(f"[DEBUG] Error procesando {img_path}: {e}")
            continue
        
        try:
            # Construimos la conversación usando "url" en lugar de "image"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": img_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            # Llamamos a la pipeline con el parámetro 'text'
            output = vlm_pipe(text=messages, max_new_tokens=300)  # Aumentamos el número de tokens para capturar toda la respuesta
            # Extraemos solo el texto generado
            gen_text = output[0].get('generated_text', '')
            if isinstance(gen_text, list):
                description = " ".join(str(x) for x in gen_text).strip()
            else:
                description = gen_text.strip()
            print(f"[DEBUG] Descripción generada para {img_path}: {description[:100]}...")
            descriptions.append(description)
        except Exception as e:
            print(f"[DEBUG] Error generando descripción para {img_path}: {e}")
            continue
    
    # Tomamos la descripción más completa (la que contiene todas las etiquetas)
    valid_descriptions = [desc for desc in descriptions if "<NAME>" in desc and "<DESCRIPTION>" in desc]
    if valid_descriptions:
        best_description = max(valid_descriptions, key=len)
    else:
        # Si no tenemos una descripción válida, generamos una con placeholders
        best_description = "<NAME>Unknown object</NAME>\n<DESCRIPTION>A CAD model with geometric features</DESCRIPTION>\n<KEYWORDS>cad, model, geometry</KEYWORDS>"
    
    print(f"[DEBUG] Descripción final (longitud {len(best_description)} caracteres)")
    return best_description

def process_cad_file(uid, json_path, renders_base_dir, output_dir, device, args, vlm_pipe):
    try:
        print(f"[DEBUG] Procesando JSON: {json_path} para UID: {uid}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Creamos el directorio para las anotaciones del VLM
        root_id, sample_id = uid.split('/')
        output_vlm_dir = os.path.join(output_dir, uid, "qwen2_vlm_annotation")
        ensure_dir(output_vlm_dir)
        
        # Obtenemos las imágenes para el modelo completo (final)
        final_image_paths = get_rendered_images(uid, renders_base_dir, args.n_views, part_id="final")
        if not final_image_paths:
            print(f"[DEBUG] No se encontraron imágenes finales para {uid}")
            return None
        
        # Generamos la descripción para el modelo completo (final)
        final_description = generate_shape_description(final_image_paths, device, vlm_pipe, is_part=False)
        print(f"[DEBUG] Descripción final para {uid}: {final_description[:100]} ...")
        
        # Guardamos la descripción final en el formato esperado
        final_output_path = os.path.join(output_vlm_dir, f"final_{sample_id}.json")
        with open(final_output_path, 'w') as f:
            json.dump(final_description, f)
        print(f"[DEBUG] Guardada descripción final en: {final_output_path}")
        
        # Si el archivo JSON contiene información sobre partes, procesamos cada parte
        if 'parts' in data:
            for part_id, part_info in data['parts'].items():
                # Construimos el part_id para buscar las imágenes correspondientes
                part_num = part_id.split('_')[-1]  # Extraemos el número de la parte
                
                # Buscamos las imágenes para esta parte específica
                part_image_paths = get_rendered_images(uid, renders_base_dir, args.n_views, part_id=f"part_{part_num}")
                
                # Si no se encuentran imágenes para esta parte específica, usamos las del modelo final
                if not part_image_paths:
                    print(f"[DEBUG] No se encontraron imágenes para la parte {part_id} de {uid}, usando imágenes del modelo final")
                    part_image_paths = final_image_paths
                
                # Generamos la descripción para esta parte
                part_description = generate_shape_description(part_image_paths, device, vlm_pipe, is_part=True)
                
                # Guardamos la descripción de la parte
                part_output_path = os.path.join(output_vlm_dir, f"{part_id}_{sample_id}.json")
                with open(part_output_path, 'w') as f:
                    json.dump(part_description, f)
                print(f"[DEBUG] Guardada descripción de parte {part_id} en: {part_output_path}")
        
        return uid, "processed_successfully"
    except Exception as e:
        print(f"[DEBUG] Error procesando {json_path}: {e}")
        return None

def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[DEBUG] Usando dispositivo: {device}")
    
    print("[DEBUG] Cargando pipeline del VLM...")
    try:
        # Usamos la pipeline "image-text-to-text" con el modelo LLaVA
        # Nota: Este código utiliza el modelo LLaVA, pero en el directorio estamos 
        # guardando los archivos como "qwen2_vlm_annotation" para mantener
        # compatibilidad con merge_vlm_minimal.py
        vlm_pipe = pipeline(
            "image-text-to-text",
            model="llava-hf/llama3-llava-next-8b-hf",
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16,  # Reduce el consumo de memoria
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("[DEBUG] Pipeline cargado correctamente.")
    except Exception as e:
        print(f"[DEBUG] Error cargando el pipeline del VLM: {e}")
        return
    
    with open(args.split_json, 'r') as f:
        split_data = json.load(f)
    all_uids = split_data.get(args.split, [])
    print(f"[DEBUG] UIDs en split '{args.split}': {len(all_uids)}")
    if args.max_samples:
        all_uids = all_uids[:args.max_samples]
    
    print(f"[DEBUG] Procesando {len(all_uids)} modelos del split '{args.split}'")
    processed_uids = []
    
    for uid in tqdm(all_uids, desc="Generando descripciones"):
        # Construimos la ruta del JSON
        # Si uid es "0001/00010160", la ruta del JSON debería incluir el minimal_json
        root_id, sample_id = uid.split('/')
        json_path = os.path.join(args.input_dir, uid, "minimal_json", f"{sample_id}.json")
        if not os.path.exists(json_path):
            print(f"[DEBUG] Archivo no encontrado: {json_path}")
            continue
        
        result = process_cad_file(uid, json_path, args.renders_dir, args.output_dir, device, args, vlm_pipe)
        if result:
            processed_uid, _ = result
            processed_uids.append(processed_uid)
        else:
            print(f"[DEBUG] No se pudo generar descripción para UID: {uid}")
    
    output_file = os.path.join(args.output_dir, "processed_uids.json")
    ensure_dir(args.output_dir)
    with open(output_file, 'w') as f:
        json.dump(processed_uids, f, indent=2)
    
    print(f"[DEBUG] Generadas descripciones para {len(processed_uids)} modelos")
    print(f"[DEBUG] Guardado registro en: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar shape descriptions para modelos CAD usando VLM")
    parser.add_argument("--input_dir", required=True, help="Directorio base de archivos CAD JSON")
    parser.add_argument("--renders_dir", required=True, help="Directorio base donde se guardaron las imágenes multi-view")
    parser.add_argument("--output_dir", required=True, help="Directorio de salida para las descripciones")
    parser.add_argument("--split_json", required=True, help="Archivo JSON con el split (train/val/test)")
    parser.add_argument("--split", default="train", help="Split a procesar (train, test, validation)")
    parser.add_argument("--max_samples", type=int, default=None, help="Número máximo de muestras a procesar")
    parser.add_argument("--n_views", type=int, default=9, help="Número de vistas a usar para la descripción")
    parser.add_argument("--cpu", action="store_true", help="Forzar el uso de CPU")
    
    args = parser.parse_args()
    main(args)