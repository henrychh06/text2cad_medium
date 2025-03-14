import os

def guardar_nombres_json(directorio, archivo_salida):
    # Obtener la lista de archivos en el directorio
    archivos = [f for f in os.listdir(directorio) if f.endswith('.json')]
    
    # Guardar los nombres en un archivo
    with open(archivo_salida, 'w', encoding='utf-8') as salida:
        for archivo in archivos:
            salida.write(archivo + '\n')
    
    print(f'Se han guardado {len(archivos)} nombres en {archivo_salida}')

# Uso del script
directorio = "C:\\Users\\Henry\\Documents\\Data\\cad_json\\0001"  # Cambia esto por la ruta de la carpeta donde están los JSON
archivo_salida = "C:\\Users\\Henry\\Documents\\Data\\data_names.txt"  # Nombre del archivo donde se guardarán los nombres
guardar_nombres_json(directorio, archivo_salida)
