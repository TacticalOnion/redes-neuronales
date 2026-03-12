"""
Perceptron multicapa
"""
import numpy as np

def inicializar_pesos(metodo, cantidad_entradas):
    """
    Inicializa el vector de pesos
    
    Parameters
    ---
    metodo: string
        Indica el metodo de inicialización a utilizar.
        - uniform_xavier: para funciones de activación sigmoid y tanh
        - normal_xavier: para funciones de activación sigmoid y tanh
        - kaiming: para funciones de activación ReLU, Leaky ReLU y ELU
    cantidad_entradas: integer
        Cantidad de entradas que recibe la reurona.
    
    Returns
    ---
    numpy.ndarray
        Vector de pesos
    """
    # Uniform Xavier Initialization
    if(metodo == "uniform_xavier"):
        limite = np.sqrt(6/(cantidad_entradas))
        pesos = np.random.uniform(-limite, limite, (cantidad_entradas))

    # Normal Xavier Initialization
    elif(metodo == "normal_xavier"):
        desviacion_estandar = np.sqrt(2/(cantidad_entradas))
        pesos = np.random.normal(0, desviacion_estandar, (cantidad_entradas))
    
    # Kaiming Initialization
    elif(metodo == "kaiming"):
        desviacion_estandar = np.sqrt(2/(cantidad_entradas))
        pesos = np.random.normal(0, desviacion_estandar, (cantidad_entradas))
    
    else:
        raise ValueError(f"Método no válido: {metodo}")

    return pesos

def funcion_activacion(salida_neurona, funcion):
    """
    Applica funcion de activacion a la salida de una neurona (suma ponderada).

    Parameters
    ---
    salida_neurona: float 
        Suma ponderada de la neurona
    funcion: string
        Indica que función de activación se va a utilizar.
        - linear: Función lineal (-∞ , ∞) [Capa de salida]. Aplicable a problemas de regresión.
        - escalera: Step function (0,1) [Capa de salida]. Aplicable a ...
        - sigmoid: Sigmoide (0,1) [Capa de salida]. Aplicable a clasificación binaria.
        - tanh: Tangente hiperbolica (-1, 1) [Capas ocultas]. Aplicable a redes neuronales recurrentes. 
        - relu: Rectified Linear Unit (0,∞) [Capa oculta]. Aplicable a CNN y MLP.
    Returns
    ---
    number
        Salida de la función de activación aplicada a la salida de la neurona
    """
    # Lineal
    if funcion == "linear":
        activacion = salida_neurona
    
    # Escalon
    elif funcion == "escalera":
        if salida_neurona >= 0:
            activacion = 1
        else:
            activacion = 0

    # Sigmoide
    elif funcion == "sigmoid":
        activacion = 1 / (1 + np.exp(-salida_neurona))

    # Tangente hiperbolica
    elif funcion == "tanh":
        activacion = 2 / (1 + np.exp(-2 * salida_neurona)) - 1

    # Rectified Linear Unit
    elif funcion == "relu":
        if salida_neurona > 0:
            activacion = salida_neurona
        else:
            activacion = 0

    else:
        raise ValueError(f"Función de activación no válida: {funcion}")

    return activacion

def perceptron(entradas, pesos, sesgo, funcion):
    """
    Calcula la salida del perceptron

    Paramters
    ---
    entradas: array
        Valores de entrada del perceptron
    pesos: array
        Pesos a aplicar
    sesgo: float
        Sesgo a aplicar
    funcion: string {linear, sigmoid, tanh, relu, leaky_relu, softplus, elu, selu}
        Nombre de la funcion de activacion
    
    Returns
    ---
    float
        Salida del perceptron

    """
    suma_ponderada = np.dot(entradas, pesos) + sesgo
    salida = funcion_activacion(suma_ponderada,funcion)
    
    return salida

def calcular_error(funcion, valores_reales, predicciones):
    """
    Calcula el error entre los valores reales del dataset de entrenamiento y las predicciones del perceptron
    Parameters
    ---
    funcion: string
        Indica con que funcion se debe calcular el error.
        - mse: Error Cuadratico Medio. 
        - mae: Error Absoluto Medio.
    valores_reales: array
        Arreglo de valores reales del dataset de entrenamiento.
    predicciones: array
        Arreglo de predicciones generadas por el perceptron.
    Returns
    ---
    Float
        Error del perceptron
    """
    # Validar dimensiones de valores resales y predicciones
    if valores_reales.shape != predicciones.shape:
        raise ValueError("valores_reales y predicciones deben tener las mismas dimensiones")

    # PROBLEMAS DE REGRESION
    # Error Cuadratico Medio
    if funcion == "mse":
        # Diferencia entre los valores reales y las predicciones
        diferencia = np.subtract(valores_reales, predicciones)
        # Elevar al cuadrado
        diferencia_cuadrado = np.square(diferencia) 
        # Calcular promedio
        error = np.mean(diferencia_cuadrado)

    # Error Absoluto Medio
    elif funcion == "mae":
        # Diferencia entre los valores reales y las predicciones
        diferencia = np.subtract(valores_reales, predicciones)
        # Diferencia absoluta
        diferencia_absoluta = np.abs(diferencia)
        # Calcular promedio
        error = np.mean(diferencia_absoluta)

    else:
        raise ValueError(f"Función no válida: {funcion}")

    return error
