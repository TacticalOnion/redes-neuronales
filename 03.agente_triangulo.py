import numpy as np
import pickle
import sys
from os import system, name
from time import sleep

"""
RED NEURONAL
---
"""
# ----------------------------
# Archivo
# ----------------------------
def guardar_red(archivo, red_neuronal):
    """
    TODO documentar funcion
    """

    archivo = open(archivo,"wb")
    pickle.dump(red_neuronal, archivo)
    archivo.close()

def cargar_red(archivo):
    """
    TODO documentar funcion
    """

    with open(archivo,"rb") as contenido:
        red = contenido.read()

    return pickle.loads(red)


# ----------------------------
# Dataset
# ----------------------------
def leer_dataset(nombre_archivo):
    """
    TODO documentar funcion
    """

    dataset = np.loadtxt(nombre_archivo, delimiter = ",", skiprows = 1)
    dataset_entrada = dataset[:,:-1]
    salidas_esperadas = dataset[:,-1:]

    return dataset_entrada, salidas_esperadas

# ----------------------------
# Función activación
# ----------------------------
def funcion_activacion(funcion,salida_ponderada):
    """
    TODO documentar funcion
    """

    if funcion == "sigmoide":
        salida_activada = 1 / (1 + np.exp(-salida_ponderada))

    elif funcion == "tanh":
        salida_activada = np.tanh(salida_ponderada)

    # TODO agregar valueError
    # TODO agregar relu y leaky_relu

    return salida_activada

def derivada_funcion_activacion(funcion, salida_activada):
    """
    TODO documentar funcion
    """

    if funcion == "sigmoide":
        derivada = salida_activada * (1 - salida_activada)

    elif funcion == "tanh":
        derivada = 1 - salida_activada**2

    # TODO agregar valueError
    # TODO agregar relu y leaky_relu

    return derivada

# ----------------------------
# Inicialización
# ----------------------------
def inicializar_pesos(funcion, cantidad_entradas, cantidad_salidas):
    """
    TODO documentar funcion
    """

    funciones_sigmoidales = ["sigmoide", "tanh"]
    funciones_rectificadas = ["relu", "leaky_relu"]

    # Xavier Uniforme
    if funcion in funciones_sigmoidales:
        limite = np.sqrt(6/(cantidad_entradas + cantidad_salidas))
        pesos = np.random.uniform(-limite, limite, (cantidad_entradas, cantidad_salidas))

    # He Uniforme
    elif funcion in funciones_rectificadas:
        limite = np.sqrt(6 / cantidad_entradas)
        pesos = np.random.uniform(-limite, limite, (cantidad_entradas, cantidad_salidas))

    # TODO agregar valueError

    return pesos

# ----------------------------
# Monitoreo
# ----------------------------
# TODO agregar funcion para calcular y graficar coste de entrenamiento por epoca

# ----------------------------
# Red Neuronal
# ----------------------------

class RedNeuronal:
    """
    TODO documentar clase
    """

    def __init__(self, arquitectura, funcion, tasa_aprendizaje = 0.1):
        """
        Parametros
        ---
        - `arquitectura`: (array) indica cuantos nodos tiene cada capa de la red. <br><br>
        Ejemplo [2,3,4,2]: 2 nodos para la capa de entrada, 3 nodos para la capa oculta 1, 
        4 nodos para la capa oculta 2 y 2 nodos para la capa de salida.
        - `funcion`: (string) indica la funcion de activacion a utilizar. Valores disponibles:
            - "sigmoide"
            - "tanh"
            - "relu" *pendiente
            - "leaky_relu" *pendiente
        - `tasa_aprendizaje`: (float) indica la magnitud de cambios respecto al peso.
        """
        # Arquitectura 
        self.arquitectura = arquitectura

        # Función de activación
        self.funcion = funcion
        self.funcion_activacion = funcion_activacion
        self.derivada_funcion_activacion = derivada_funcion_activacion

        # Tasa de aprendizaje
        self.tasa_aprendizaje = tasa_aprendizaje

        # Incializar pesos
        self.pesos = []
        for capa in range(len(arquitectura) -1):
            self.pesos.append(
                inicializar_pesos(funcion,
                arquitectura[capa],
                arquitectura[capa+1])
                )
        
        # Incializar sesgos
        self.sesgos = []
        for capa in range(len(arquitectura)-1):
            self.sesgos.append(
                np.zeros((1, arquitectura[capa + 1]))
            )


    def propagacion_adelante(self, data_entrenamiento):
        """
        TODO documentar funcion
        """
        activacion = data_entrenamiento.copy()
        activaciones = [activacion]

        for capa in range(1, len(self.arquitectura)):
            salida_ponderada = np.dot(activacion,self.pesos[capa - 1]) + self.sesgos[capa - 1]
            salida_activada = self.funcion_activacion(self.funcion, salida_ponderada)
            activacion = salida_activada
            activaciones.append(activacion)

        return activaciones

    def propagacion_atras(self, activaciones, salidas_esperadas):
        """
        TODO documentar funcion
        """
        deltas = []

        # Delta salida
        error_salida = activaciones[-1] - salidas_esperadas
        delta = error_salida * self.derivada_funcion_activacion(self.funcion, activaciones[-1])
        deltas.append(delta)

        # Deltas ocultas
        for capa in range(len(self.pesos) - 1, 0, -1):
            pesos_transpuesta = self.pesos[capa].T
            delta = np.dot(delta, pesos_transpuesta) * self.derivada_funcion_activacion(self.funcion, activaciones[capa])

            deltas.append(delta)

        deltas.reverse()

        # Actualizar pesos y sesgos
        for capa in range(len(self.pesos)):
            activacion_transpuesta = activaciones[capa].T
            self.pesos[capa] -= self.tasa_aprendizaje * np.dot(activacion_transpuesta, deltas[capa])
            self.sesgos[capa] -= self.tasa_aprendizaje * np.sum(deltas[capa], axis = 0, keepdims = True)
    
    def entrenar(self, data_entrenamiento, salidas_esperadas, epocas):
        """
        TODO documentar funcion
        """
        # TODO agregar MSE por epoca para monitorear aprendizaje

        for epoca in range(epocas):
            activaciones = self.propagacion_adelante(data_entrenamiento)
            self.propagacion_atras(activaciones, salidas_esperadas)

    def predecir(self, entradas):
        """
        TODO documentar funcion
        """
        entradas = np.array(entradas)
        entradas = entradas.reshape(1, -1)

        activaciones = self.propagacion_adelante(entradas)
        
        return activaciones[-1]

"""
MAPA
---
"""
CASILLA_AGENTE = "  △  "
CASILLA_OBSTACULO = "  X  "
CASILLA_PREMIO = "  *  "
CASILLA_VACIA = "     "

PORCENTAJE_PREMIO = 0.20
PORCENTAJE_OBSTACULOS = 0.20

# ----------------------------
# Posiciones
# ----------------------------
def inicializar_posiciones(dimension,cantidad,posicion_agente):
    """
    TODO documentar función
    """
    posiciones = [[np.random.randint(0, dimension - 1),np.random.randint(0, dimension - 1)] for premio in range(cantidad)]
    
    for posicion in posiciones:
        while(posicion_agente == posicion):
            posicion = [np.random.randint(0, dimension - 1),np.random.randint(0, dimension - 1)]
    
    return posiciones

def corregir_posiciones(dimension, arreglo_a, arreglo_b):
    """
    Se asegura de que dos arreglos de posiciones no compartan ninguna posición. 
    TODO documentar función
    """
    for posicion_a in arreglo_a:
        for posicion_b in arreglo_b:
            while(posicion_a == posicion_b):
                posicion_b = [np.random.randint(0, dimension - 1),np.random.randint(0, dimension - 1)]

def inicializar_mapa(mapa, premios, obstaculos, posicion_agente):
    # Agente
    mapa[posicion_agente[0]],[posicion_agente[1]] = CASILLA_AGENTE

    # Premios
    for posicion in premios:
            mapa[posicion[0]][posicion[1]] = CASILLA_PREMIO

    # Obstaculos
    for posicion in obstaculos:
            mapa[posicion[0]][posicion[1]] = CASILLA_OBSTACULO

def clear():
        if name == 'nt':
            _ =system('cls')
        else:
            _=system('clear')

# ----------------------------
# Mapa
# ----------------------------
class Mapa:
    def __init__(self, dimension):
        """
        TODO documentar funcion
        Parametros
        ---
        - `dimension`: (int) indica las dimensiones del tablero (cuadrado). 
        """
        self.dimension = dimension 
        self.posicion_agente = [0,0]
        self.posicion_anterior = [-1,-1]
        self.mapa = [[CASILLA_VACIA for casilla in range(dimension)] for columna in range(dimension)]
        self.cantidad_premio = np.floor(dimension * dimension * PORCENTAJE_PREMIO)
        self.cantidad_obstaculos = np.floor(dimension * dimension * PORCENTAJE_OBSTACULOS)

        # Distribución aleatoria de premio y obstaculos
        # Inicializar posiciones
        posicion_premios = inicializar_posiciones(dimension,self.cantidad_premio,self.posicion_agente)
        posicion_obstaculos = inicializar_posiciones(dimension, self.cantidad_obstaculos,self.posicion_agente)
        
        # Asegurar que no se sobrepongan los premios y los obstaculos
        corregir_posiciones(dimension,posicion_premios,posicion_obstaculos)

        # Guardar posiciones
        self.premios = posicion_premios
        self.obstaculos = posicion_obstaculos

        # Inicializar mapa
        inicializar_mapa(self.mapa,self.premios, self.obstaculos, self.posicion_agente)

    def imprimir_mapa(self):
        
        if self.posicion_anterior[0] > -1:
            self.mapa[self.posicion_anterior[0]][self.posicion_anterior[1]] = CASILLA_AGENTE

        for fila in self.mapa:
            print("+-----" * self.dimension, end='')
            print("+")

            for valor in fila:
                print("|" + valor, end = "")
            
            print("|", end = "\n")

        print("+-----" * self.dimension)
        print("Premio   : ", self.cantidad_premio)
        print("Agente   : ", self.posicion_agente)

        self.mapa[self.posicion_anterior[0]][self.posicion_anterior[1]] = CASILLA_VACIA

    def detectar_premio(self):
        """
        TODO documentar función
        """
        if self.mapa[self.posicion_agente[0]][self.posicion_agente[1]] == CASILLA_PREMIO:
            return True
        else:
            return False

    def mover_agente(self):
        movimientos = [[1,0],
        [-1,0],
        [0,1],
        [0,-1]]

        self.posicion_anterior = self.posicion_agente
        movimientos_disponibles = [[m[0] + self.posicion_agente[0], m[1] + self.posicion_agente[1]]
            for m in movimientos
            if m[0] + self.posicion_agente[0] > -1 if m[0] + self.posicion_agente[0] < self.dimension
            and
            m[1] + self.posicion_agente[1] > -1 if m[1] + self.posicion_agente[1] < self.dimension]
        #omite las opciones donde hay obstaculos
        movimientos_disponibles = [m for m in movimientos_disponibles if m not in self.obstaculos]
        #finalmente elige un movimiento aleatoriamente
        movimiento = np.random.choice(movimientos_disponibles)
        #actualiza la posicion del agente
        self.posicion_agente = movimiento

    def ejecutar_agente(self):
        """
        TODO documentar función
        """
        if self.detectar_premio():
            self.cantidad_premio -= 1

        self.mapa[self.posicion_agente[0]][self.posicion_agente[1]] = CASILLA_AGENTE
        self.mover_agente()

    def simular(self):
        while True:
            self.ver_rejilla()
            self.ejecutar_agente()
            sleep(1)
            clear()

"""
AGENTE
---
"""
def main():
    DIMENSION_MINIMA = 5
    DIMENSION_MAXIMA = 10
    print("--------------------------------")
    print("AGENTE TRINAGULO")
    print("--------------------------------")

    while True:
        dimension = int(input(f"Ingresa la dimension del mapa [{DIMENSION_MINIMA}-{DIMENSION_MAXIMA}]"))
        dimension_valida = dimension >= DIMENSION_MINIMA and dimension <= DIMENSION_MAXIMA
    
        if dimension_valida:
            mapa = Mapa(dimension)
            mapa.simular()
        else:
            print(f"{dimension} no es valido. Debe ser entre 4 y 10.")

if __name__ == '__main__':
    main()
