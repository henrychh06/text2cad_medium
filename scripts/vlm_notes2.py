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

def get_rendered_images(uid, renders_base_dir, n_views):
    """
    Obtiene las rutas de las imágenes multi-view ya generadas para un CAD.
    Se asume que los UIDs están en formato "0001/00010160" y que las imágenes
    están en renders_base_dir/0001/00010160/.
    """
    folder = os.path.join(renders_base_dir, *uid.split('/'))
    print(f"[DEBUG] Buscando imágenes en: {folder}")
    if not os.path.exists(folder):
        print(f"[DEBUG] Carpeta de imágenes no encontrada: {folder}")
        return None
    
    image_paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    print(f"[DEBUG] Se encontraron {len(image_paths)} imágenes en {folder}")
    if len(image_paths) < n_views:
        print(f"[DEBUG] Se esperaban {n_views} vistas pero se encontraron {len(image_paths)} en {folder}")
    return image_paths[:n_views] if image_paths else None

def generate_shape_description(image_paths, device, vlm_pipe):
    if not image_paths:
        print("[DEBUG] No hay imágenes para generar descripción.")
        return "Shape description not available."
    
    descriptions = []
    
    prompt = """
[INST] This is an image of a Computer Aided Design (CAD) model.
You are a senior CAD engineer who knows the object name, where and how the CAD model is used.
Give an accurate natural language description about the CAD model to a junior CAD designer who can design it from your simple description.
Wrap the description in the following tags <OBJECT> and </OBJECT>.
Following are some bad examples:
CAD model
Metal object
Abide by the following rules.
Rules:
Do not use words like - "blue", "shadow", "transparent", "metal", "plastic", "image", "black", "grey", "CAD model", "abstract", "orange", "purple", "golden", "green"
Focus on shape, structure, and geometric features.
Do not mention colors, materials, or rendering aspects.
/INST]
    """
    
    for img_path in image_paths:
        try:
            print(f"[DEBUG] Procesando imagen: {img_path}")
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[DEBUG] Error abriendo {img_path}: {e}")
            continue
        
        try:
            # Enviamos el prompt y la imagen a la pipeline
            output = vlm_pipe({"text": prompt, "image": image})
            description = output[0].get('generated_text', "").strip()
            if ":" in description:
                description = description.split(":", 1)[1].strip()
            print(f"[DEBUG] Descripción generada para {img_path}: {description}")
            descriptions.append(description)
        except Exception as e:
            print(f"[DEBUG] Error generando descripción para {img_path}: {e}")
            continue
    
    combined = " ".join(descriptions)
    print(f"[DEBUG] Descripción combinada (longitud {len(combined)} caracteres)")
    return combined

def process_cad_file(uid, json_path, renders_base_dir, output_dir, device, args, vlm_pipe):
    try:
        print(f"[DEBUG] Procesando JSON: {json_path} para UID: {uid}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        image_paths = get_rendered_images(uid, renders_base_dir, args.n_views)
        if not image_paths:
            print(f"[DEBUG] No se encontraron imágenes para {uid}")
            return None
        
        shape_description = generate_shape_description(image_paths, device, vlm_pipe)
        print(f"[DEBUG] Descripción final para {uid}: {shape_description[:100]} ...")
        return uid, shape_description
    except Exception as e:
        print(f"[DEBUG] Error procesando {json_path}: {e}")
        return None

def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[DEBUG] Usando dispositivo: {device}")
    
    print("[DEBUG] Cargando pipeline del VLM...")
    try:
        # Usamos la pipeline "image-text-to-text" con el modelo "liuhaotian/llava-v1.5-7b-lora"
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
    print(f"[DEBUG] UIDs en split '{args.split}': {all_uids}")
    if args.max_samples:
        all_uids = all_uids[:args.max_samples]
    
    print(f"[DEBUG] Procesando {len(all_uids)} modelos del split '{args.split}'")
    results = {}
    
    for uid in tqdm(all_uids, desc="Generando descripciones"):
        # Se espera que la ruta del JSON se construya a partir del UID completo:
        # Ejemplo: si uid es "0001/00010160", se busca input_dir/0001/00010160.json
        json_path = os.path.join(args.input_dir, f"{uid}.json")
        if not os.path.exists(json_path):
            print(f"[DEBUG] Archivo no encontrado: {json_path}")
            continue
        
        result = process_cad_file(uid, json_path, args.renders_dir, args.output_dir, device, args, vlm_pipe)
        if result:
            uid, description = result
            results[uid] = description
        else:
            print(f"[DEBUG] No se pudo generar descripción para UID: {uid}")
    
    output_file = os.path.join(args.output_dir, "shape_descriptions.json")
    ensure_dir(args.output_dir)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[DEBUG] Generadas descripciones para {len(results)} modelos")
    print(f"[DEBUG] Guardado en: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar shape descriptions para modelos CAD usando VLM")
    parser.add_argument("--input_dir", required=True, help="Directorio de archivos CAD JSON")
    parser.add_argument("--renders_dir", required=True, help="Directorio base donde se guardaron las imágenes multi-view")
    parser.add_argument("--output_dir", required=True, help="Directorio de salida para las descripciones")
    parser.add_argument("--split_json", required=True, help="Archivo JSON con el split (train/val/test)")
    parser.add_argument("--split", default="train", help="Split a procesar (train, test, validation)")
    parser.add_argument("--max_samples", type=int, default=None, help="Número máximo de muestras a procesar")
    parser.add_argument("--n_views", type=int, default=9, help="Número de vistas a usar para la descripción")
    parser.add_argument("--cpu", action="store_true", help="Forzar el uso de CPU")
    
    args = parser.parse_args()
    main(args)
