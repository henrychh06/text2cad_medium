import os
import sys
import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

# Importar las bibliotecas necesarias para trabajar con STEP
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Visualization import Display3d
from OCC.Core.V3d import V3d_XposYnegZpos, V3d_XnegYnegZpos, V3d_XnegYposZpos, V3d_XposYposZpos
from OCC.Display.SimpleGui import init_display
from OCC.Core.Graphic3d import Graphic3d_Camera, Graphic3d_TypeOfBackground

# Configuración del logger
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Evitar problemas con el método de inicio de procesos en diferentes plataformas
try:
    multiprocessing.set_start_method("forkserver", force=True)
except RuntimeError:
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# Función para asegurar que un directorio existe
def ensure_dir(directory):
    """
    Crea un directorio si no existe.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# Función para obtener todos los archivos STEP de forma recursiva
def get_step_files(input_dir, max_workers=4):
    """
    Escanea el directorio de entrada y devuelve todos los archivos STEP encontrados.
    """
    step_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.step') or file.lower().endswith('.stp'):
                step_files.append(os.path.join(root, file))
    
    return step_files

# Función para generar vistas de un archivo STEP
def generate_views(step_file_path, output_dir, uid, number_views=4, img_size=(800, 600), verbose=False):
    """
    Genera múltiples vistas de un archivo STEP y las guarda como imágenes.
    """
    try:
        # Crear directorio de salida basado en la estructura UID
        view_output_dir = os.path.join(output_dir, uid, "views")
        ensure_dir(view_output_dir)
        
        # Obtener el nombre base del archivo STEP (ej: 00010001_final)
        step_basename = os.path.splitext(os.path.basename(step_file_path))[0]
        
        # Crear una subcarpeta dentro de "views" con el nombre base del archivo STEP
        step_subfolder = os.path.join(view_output_dir, step_basename)
        ensure_dir(step_subfolder)
        
        if verbose:
            logger.info(f"Generando vistas para {step_file_path}")
            logger.info(f"Guardando en {step_subfolder}")
        
        # Inicializar el display en modo offscreen para evitar requerir una interfaz gráfica
        display, start_display, _, _ = init_display(backend_str="opengl2", offscreen=True)
        
        # Cargar el archivo STEP
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(step_file_path)
        
        if status != IFSelect_RetDone:
            logger.error(f"Error al leer el archivo STEP: {step_file_path}")
            return False
        
        # Transferir el modelo
        step_reader.TransferRoot()
        shape = step_reader.Shape()
        
        # Mostrar el modelo
        display.DisplayShape(shape, update=True)
        display.FitAll()
        
        # Configurar el tamaño de la imagen
        display.View.Window().SetSize(img_size[0], img_size[1])
        
        # Definir las vistas según el número solicitado
        views = []

        if number_views >= 1:
            views.append({"name": "isometric", "dir": V3d_XposYnegZpos})
        if number_views >= 2:
            views.append({"name": "front", "dir": V3d_XposYposZpos})
        if number_views >= 3:
            views.append({"name": "side", "dir": V3d_XnegYnegZpos})
        if number_views >= 4:
            views.append({"name": "top", "dir": V3d_XnegYposZpos})

        # Vistas adicionales para llegar a 9
        if number_views >= 5:
            # Vista trasera: se puede definir como la inversa de la frontal.
            # Nota: Si los objetos V3d_* no se pueden invertir directamente, puedes definir el vector manualmente.
            views.append({"name": "back", "dir": (-1, -1, -1)})
        if number_views >= 6:
            # Vista izquierda: opuesta a la lateral (por ejemplo, manualmente definida)
            views.append({"name": "left", "dir": (-1, 1, 1)})
        if number_views >= 7:
            # Vista inferior: opuesta a la superior (invirtiendo la componente Z)
            views.append({"name": "bottom", "dir": (1, -1, -1)})
        if number_views >= 8:
            # Vista diagonal 45°: combinación entre isométrica y frontal (valores de ejemplo)
            views.append({"name": "diagonal_45", "dir": (0.5, 0.5, 0.707)})
        if number_views >= 9:
            # Vista diagonal 135°: otra orientación diagonal (valores de ejemplo)
            views.append({"name": "diagonal_135", "dir": (-0.5, 0.5, 0.707)})

        
        # Generar imágenes para cada vista
        for i, view in enumerate(views, start=1):
            # Configurar la vista
            display.View.SetProj(view["dir"])
            display.FitAll()
            
            # Nombre de salida: "00010001_final_view_1.png", "00010001_final_view_2.png", etc.
            output_path = os.path.join(step_subfolder, f"{step_basename}_view_{i}.png")
            display.View.Dump(output_path)
            
            if verbose:
                logger.info(f"Vista guardada: {output_path}")
        
        # Cerrar el display para liberar recursos
        display.Repaint()
        
        return True
        
    except Exception as e:
        logger.error(f"Error al generar vistas para {step_file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return False


# Función para procesar un archivo STEP específico
def process_step_file(step_file_path, args):
    """
    Procesa un archivo STEP y genera las vistas.
    
    Args:
        step_file_path: Ruta al archivo STEP
        args: Argumentos de línea de comandos
    
    Returns:
        success: True si se procesó correctamente, False en caso contrario
    """
    try:
        # Determinar la UID basada en la estructura de directorios
        # Ejemplo: /data/models/0003/00003121/step/model.step
        # UID sería: 0003/00003121
        
        # Extraer los componentes de la ruta
        parts = step_file_path.split(os.sep)
        
        # Buscar la posición de "step" en la ruta
        try:
            step_index = parts.index("step")
            # Los dos directorios anteriores a "step" forman la UID
            if step_index >= 2:
                uid = os.path.join(parts[step_index-2], parts[step_index-1])
            else:
                # Si no se puede determinar la UID correctamente
                uid = os.path.basename(os.path.dirname(os.path.dirname(step_file_path)))
        except ValueError:
            # Si "step" no está en la ruta, usamos los últimos dos componentes antes del archivo
            uid = os.path.join(parts[-3], parts[-2])
        
        # Generar las vistas
        success = generate_views(
            step_file_path=step_file_path,
            output_dir=args.output,
            uid=uid,
            number_views=args.number_views,
            img_size=(args.img_width, args.img_height),
            verbose=args.verbose
        )
        
        return success, uid
        
    except Exception as e:
        if args.verbose:
            logger.error(f"Error procesando {step_file_path}: {str(e)}")
            logger.error(traceback.format_exc())
        return False, ""

# Función principal
def main():
    """
    Función principal para procesar archivos STEP y generar vistas.
    """
    parser = argparse.ArgumentParser(
        description="Generador de vistas múltiples de archivos STEP"
    )
    parser.add_argument(
        "--input", required=True, help="Directorio de entrada con archivos STEP organizados por UID", type=str
    )
    parser.add_argument(
        "--output", required=True, help="Directorio de salida para las imágenes generadas", type=str
    )
    parser.add_argument(
        "--number_views", type=int, default=4, help="Número de vistas a generar por modelo"
    )
    parser.add_argument(
        "--img_width", type=int, default=800, help="Ancho de las imágenes generadas"
    )
    parser.add_argument(
        "--img_height", type=int, default=600, help="Alto de las imágenes generadas"
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Número máximo de trabajadores para procesamiento paralelo"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Mostrar información detallada durante el procesamiento"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.verbose:
        logger.info(f"Iniciando procesamiento con {args.max_workers} trabajadores")
        logger.info(f"Buscando archivos STEP en {args.input}")
    
    # Asegurar que el directorio de salida existe
    ensure_dir(args.output)
    
    # Obtener todos los archivos STEP del directorio de entrada
    all_step_files = get_step_files(args.input, args.max_workers)
    
    if args.verbose:
        logger.info(f"Encontrados {len(all_step_files)} archivos STEP")
    
    # Procesar los archivos en paralelo
    success_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Enviar tareas al ejecutor
        futures = [
            executor.submit(process_step_file, step_path, args)
            for step_path in all_step_files
        ]
        
        # Procesar resultados a medida que se completan
        for future in tqdm(as_completed(futures), desc="Procesando archivos", total=len(futures)):
            success, uid = future.result()
            if success:
                success_count += 1
            else:
                failed_count += 1
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info(f"Procesamiento completado en {processing_time:.2f} segundos")
    logger.info(f"Archivos procesados exitosamente: {success_count}")
    logger.info(f"Archivos con error: {failed_count}")

if __name__ == "__main__":
    main()