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
    funcion: array
        Indica que con que funcion se debe calcular el error.
        - mse: Error Cuadratico Medio. 
        - mae: Error Absoluto Medio. 
        - cross_entropy: Entropia Cruzada. 
        - cross_entropy_muliclass: Entropia Cruzada Multiclase.
        - hinge_loss: Hinge loss. 
    valores_reales: array
        Arreglo de valores reales del dataset de entrenamiento.
    predicciones: array
        Arreglo de predicciones generadas por el perceptron.
    Returns
    ---
    Float
        Error del perceptron
    """
    # Diferencia entre los valores reales y las predicciones
    diferencia = np.subtract(valores_reales, predicciones)

    # PROBLEMAS DE REGRESION
    # Error Cuadratico Medio
    if funcion == "mse":
        # Elevar al cuadrado
        diferencia_cuadrado = np.square(diferencia) 
        # Calcular promedio
        error = np.mean(diferencia_cuadrado)

    # Error Absoluto Medio
    elif funcion == "mae":
        # Diferencia absoluta
        diferencia_absoluta = np.abs(diferencia)
        # Calcular promedio
        error = np.mean(diferencia_absoluta)
    
    # PROBLEMAS DE CLASIFICACION
    # Entropia cruzada
    elif funcion == "cross_entropy":
        # Clipping: en caso de que la probabilidad sea 0 o 1
        epsilon = 1e-15
        predicciones = np.clip(predicciones, epsilon, 1 - epsilon)
        # Cantidad de muestras
        cantidad_muestras = len(valores_reales)
        # Calculo de entropia cruzada binaria
        error = -np.sum(valores_reales * np.log(predicciones) + (1 - valores_reales) * np.log(1 - predicciones)) / cantidad_muestras

    elif funcion == "cross_entropy_muliclass":
        # Clipping: en caso de que la probabilidad sea 0 o 1
        epsilon = 1e-15
        predicciones = np.clip(predicciones, epsilon, 1 - epsilon)

        # Calculo de entropia cruzada multiclase
        error = -np.sum(valores_reales * np.log(predicciones)) / valores_reales.shape[0]
    
    elif funcion == "hinge_loss":
        # Calcular valor maximo
        clasificacion = np.maximum(0, 1 - valores_reales * predicciones)
        # Calcular promedio
        error = np.mean(clasificacion)

    else:
        raise ValueError(f"Función no válida: {funcion}")

    return error
