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

def funcion_activacion(salida_neurona, funcion, pendiente = 0.01):
    """
    Applica funcion de activacion a la salida de una neurona (suma ponderada).

    Parameters
    ---
    salida_neurona: float 
        Suma ponderada de la neurona
    funcion: string
        Indica que función de activación se va a utilizar.
        - linear: Función lineal (-∞ , ∞) [Capa de salida]. Aplicable a problemas de regresión.
        - sigmoid: Sigmoide (0,1) [Capa de salida]. Aplicable a clasificación binaria.
        - tanh: Tangente hiperbolica (-1, 1) [Capas ocultas]. Aplicable a redes neuronales recurrentes. 
        - relu: Rectified Linear Unit (0,∞) [Capa oculta]. Aplicable a CNN y MLP.
        - leaky_relu: Leaky ReLU (-∞ , ∞) [Capa oculta]. Aplicable a CNN y MLP cuando hay dying ReLU.
        - softplus: Softplus (0,∞) [Capa oculta]. Aplicable cuando se necesita ReLU suave.
        - elu: Exponential Linear Unit (-a , ∞) [...]. Aplicable a redes profundas.
        - selu: Scaled ELU (-∞ , ∞) [...]. Aplicable a redes neuronales auto normalizadas.
    pendiente: float default 0.01
        Pendiente para la funcion de activacion Leaky ReLU
    
    Returns
    ---
    number
        Salida de la función de activación aplicada a la salida de la neurona
    """
    # Lineal
    if funcion == "linear":
        activacion = salida_neurona
    
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
    
    # Leaky Rectified Linear Unit
    elif funcion == "leaky_relu":
        if salida_neurona > 0:
            activacion = salida_neurona
        else:
            activacion = pendiente * salida_neurona

    # Softplus
    elif funcion == "softplus":
        activacion = np.log1p(np.exp(salida_neurona))
    
    # Exponential Linear Unit
    elif funcion == "elu":
        if salida_neurona > 0:
            activacion = salida_neurona
        else:
            activacion = pendiente * (np.exp(salida_neurona) - 1)

    # Scaled Exponential Linear Unit
    elif funcion == "selu":
        constante = 1.0507 # lambda
        pendiente = 1.67326 # alpha
        
        if salida_neurona > 0:
            activacion = constante * salida_neurona
        else:
            activacion = constante * pendiente * (np.exp(salida_neurona) - 1)

    else:
        raise ValueError(f"Función de activación no válida: {funcion}")

    return activacion

def perceptron(entradas, pesos, sesgo, funcion):
    suma_ponderada = np.dot(entradas, pesos) + sesgo
    salida = funcion_activacion(suma_ponderada,funcion)
    
    return salida

