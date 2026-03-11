"""
Perceptron multicapa
"""
import numpy as np

def inicializar_pesos(metodo, cantidad_entradas, cantidad_salidas):
    """
    Inicializa una matriz de pesos
    
    Parameters
    ---
    metodo: string
        Indica el metodo de inicialización a utilizar.
        - uniform_xavier: para funciones de activación sigmoid y tanh
        - normal_xavier: para funciones de activación sigmoid y tanh
        - kaiming: para funciones de activación ReLU, Leaky ReLU y ELU
    cantidad_entradas: integer
        Cantidad de neuronas de la capa de entrada.
    cantidad_salidad: integer
        Cantidad de neuronas para la capa de salida.
    
    Returns
    ---
    numpy.ndarray
        Matriz de pesos
    """
    # Uniform Xavier Initialization
    if(metodo == "uniform_xavier"):
        limite = np.sqrt(6/(cantidad_entradas + cantidad_salidas))
        pesos = np.random.uniform(-limite, limite, (cantidad_entradas,cantidad_salidas))

    # Normal Xavier Initialization
    if(metodo == "normal_xavier"):
        desviacion_estandar = np.sqrt(2/(cantidad_entradas + cantidad_salidas))
        pesos = np.random.normal(0, desviacion_estandar, (cantidad_entradas, cantidad_salidas))
    
    # Kaiming Initialization
    if(metodo == "kaiming"):
        desviacion_estandar = np.sqrt(2/(cantidad_entradas))
        pesos = np.random.normal(0, desviacion_estandar, (cantidad_entradas,cantidad_salidas))
    
    return pesos

def inicializar_sesgo(cantidad_salidas):
    """
    inicializar_sesgo

    Parameters
    ---
    cantidad_salidas: integer
        Cantidad de salidas de la red. Para definir la longitud del vector
    
    Returns
    ---
    numpy.ndarray
        Sesgo inicializado con un vector de ceros
    """
    # Genera vector de ceros
    sesgo = np.zeros(cantidad_salidas)

    return sesgo