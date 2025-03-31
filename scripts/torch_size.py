import torch
import sys

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

def main(pth_file):
    try:
        data = torch.load(pth_file, map_location='cpu')
        print("Procesando archivo:", pth_file)
        print_tensor_shapes(data)
    except Exception as e:
        print("Error al cargar el archivo:", e)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python print_torch_size.py <archivo.pth>")
        sys.exit(1)
    main(sys.argv[1])
