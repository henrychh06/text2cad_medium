import os
import sys
import torch

def print_tensor_shapes(data, prefix=""):
    """
    Función recursiva para imprimir el tamaño de tensores dentro de estructuras anidadas.
    """
    if isinstance(data, torch.Tensor):
        print(f"{prefix}Tensor shape: {data.shape}")
    elif isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}Clave: {key}")
            print_tensor_shapes(value, prefix + "  ")
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            print(f"{prefix}Índice {i}:")
            print_tensor_shapes(item, prefix + "  ")
    else:
        print(f"{prefix}Tipo: {type(data)} (no es tensor)")

def process_pth_file(pth_file):
    print(f"Procesando archivo: {pth_file}")
    try:
        data = torch.load(pth_file, map_location="cpu")
        print_tensor_shapes(data, "  ")
    except Exception as e:
        print(f"  Error al cargar {pth_file}: {e}")
    print("-" * 50)

def process_directory(directory):
    # Recorre recursivamente la carpeta
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pth"):
                full_path = os.path.join(root, file)
                process_pth_file(full_path)

def main():
    if len(sys.argv) != 2:
        print("Uso: python process_pth_folder.py <directorio>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"El directorio {directory} no existe o no es una carpeta.")
        sys.exit(1)

    process_directory(directory)

if __name__ == '__main__':
    main()
