import os
import glob
import json
import h5py
import numpy as np
import time
import argparse
import sys
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Vec, gp_Trsf
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepCheck import BRepCheck_Analyzer

sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD

# Configuración de argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help="Carpeta de origen de los archivos CAD")
parser.add_argument('--form', type=str, default="h5", choices=["h5", "json"], help="Formato de archivo")
parser.add_argument('--idx', type=int, default=0, help="Índice de inicio de archivos a procesar")
parser.add_argument('--num', type=int, default=10, help="Cantidad de archivos a procesar (-1 para todos)")
parser.add_argument('--output', type=str, required=True, help="Carpeta de salida para las imágenes")
parser.add_argument('--with_gt', action="store_true", help="Procesa también la ground truth si está disponible")
parser.add_argument('--filter', action="store_true", help="Filtra modelos CAD inválidos usando el analizador")
args = parser.parse_args()

# Listado de archivos de entrada
src_dir = args.src
out_paths = sorted(glob.glob(os.path.join(src_dir, f"*.{args.form}")))
if args.num != -1:
    out_paths = out_paths[args.idx:args.idx + args.num]

def translate_shape(shape, translate):
    trans = gp_Trsf()
    trans.SetTranslation(gp_Vec(*translate))
    loc = TopLoc_Location(trans)
    shape.Move(loc)
    return shape

# Inicialización de la ventana OCC
display, start_display, add_menu, add_function_to_menu = init_display()

# Definición de las 9 vistas (rotaciones) deseadas
view_angles = [
    (1, 0, 0),    # Frontal
    (0, 1, 0),    # Superior
    (0, -1, 0),   # Inferior
    (-1, 0, 0),   # Trasera
    (0, 0, 1),    # Lateral derecha
    (0, 0, -1),   # Lateral izquierda
    (1, 1, 0),    # Frontal superior derecha
    (-1, 1, 0),   # Frontal superior izquierda
    (1, 0, 1)     # Frontal lateral derecha
]

def save_views(shape, file_id, output_dir):
    """
    Guarda 9 vistas del modelo CAD en la carpeta output_dir/file_id.
    Usa display.View.Dump para capturar la imagen.
    """
    folder = os.path.join(output_dir, file_id)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for idx, angle in enumerate(view_angles):
        # Limpiar la pantalla y mostrar el shape
        display.EraseAll()
        display.DisplayShape(shape, update=False)
        # Reiniciar a la vista ISO y ajustar
        display.View_Iso()
        display.FitAll()
        # Aplicar la rotación deseada
        display.View.Rotate(angle[0], angle[1], angle[2])
        display.View.Redraw()
        # Espera breve para que se repinte la vista
        time.sleep(0.3)
        # Definir el path de salida y capturar la imagen
        img_path = os.path.join(folder, f"{file_id}_view{idx}.png")
        display.View.Dump(img_path)
        print("Guardada imagen:", img_path)

# Procesamiento de cada archivo CAD
for path in out_paths:
    file_id = os.path.splitext(os.path.basename(path))[0]
    print("Procesando archivo:", path)
    try:
        if args.form == "h5":
            with h5py.File(path, 'r') as fp:
                vec = fp["vec"][:]
                vec[vec == -1] = 0
                shape = vec2CADsolid(vec)
                if args.with_gt:
                    if "gt_vec" in fp:
                        gt_vec = fp["gt_vec"][:]
                        gt_vec[gt_vec == -1] = 0
                        gt_shape = vec2CADsolid(gt_vec)
                    else:
                        gt_shape = None
                else:
                    gt_shape = None
        else:
            with open(path, 'r') as fp:
                data = json.load(fp)
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            shape = create_CAD(cad_seq)
            gt_shape = None
    except Exception as e:
        print(f"Error al cargar el modelo CAD de {path}: {e}")
        continue

    # Filtrado opcional de modelos inválidos
    if args.filter:
        analyzer = BRepCheck_Analyzer(shape)
        if not analyzer.IsValid():
            print("Modelo inválido en:", path)
            continue

    # Si lo deseas, puedes aplicar una traslación adicional:
    # shape = translate_shape(shape, [0, 0, 0])
    
    # Guardar vistas para el modelo
    save_views(shape, file_id, args.output)
    
    # Si se procesa ground truth, guardar sus vistas
    if args.with_gt and gt_shape is not None:
        save_views(gt_shape, f"{file_id}_gt", args.output)

# Si quieres mantener la ventana interactiva, descomenta la siguiente línea:

