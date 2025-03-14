import h5py

with h5py.File("C:\\Users\\Henry\\Documents\\Data\\cad_vec\\0000\\00000007.h5", 'r') as file:
    print("File keys:", list(file.keys()))  # List all the available datasets
