import numpy as np
import pickle
from collections import deque
from os import system, name
from time import sleep
from pathlib import Path

"""
RED NEURONAL
---
"""
# ----------------------------
# Archivo
# ----------------------------
def guardar_red(archivo, red_neuronal):
    with open(archivo, "wb") as contenido:
        pickle.dump(red_neuronal, contenido)


def cargar_red(archivo):
    with open(archivo, "rb") as contenido:
        red = contenido.read()

    return pickle.loads(red)


# ----------------------------
# Dataset
# ----------------------------
def leer_dataset(nombre_archivo, cantidad_salidas=1):
    dataset = np.loadtxt(nombre_archivo, delimiter=",", skiprows=1)
    dataset_entrada = dataset[:, :-cantidad_salidas]
    salidas_esperadas = dataset[:, -cantidad_salidas:]
    return dataset_entrada, salidas_esperadas


# ----------------------------
# Función activación
# ----------------------------
def funcion_activacion(funcion, salida_ponderada):
    if funcion == "sigmoide":
        salida_activada = 1 / (1 + np.exp(-salida_ponderada))
    elif funcion == "tanh":
        salida_activada = np.tanh(salida_ponderada)
    else:
        raise ValueError(f"Función de activación no soportada: {funcion}")
    return salida_activada


def derivada_funcion_activacion(funcion, salida_activada):
    if funcion == "sigmoide":
        derivada = salida_activada * (1 - salida_activada)
    elif funcion == "tanh":
        derivada = 1 - salida_activada**2
    else:
        raise ValueError(f"Función de activación no soportada: {funcion}")
    return derivada


# ----------------------------
# Inicialización
# ----------------------------
def inicializar_pesos(funcion, cantidad_entradas, cantidad_salidas):
    funciones_sigmoidales = ["sigmoide", "tanh"]
    funciones_rectificadas = ["relu", "leaky_relu"]

    if funcion in funciones_sigmoidales:
        limite = np.sqrt(6 / (cantidad_entradas + cantidad_salidas))
        pesos = np.random.uniform(-limite, limite, (cantidad_entradas, cantidad_salidas))
    elif funcion in funciones_rectificadas:
        limite = np.sqrt(6 / cantidad_entradas)
        pesos = np.random.uniform(-limite, limite, (cantidad_entradas, cantidad_salidas))
    else:
        raise ValueError(f"Función de activación no soportada: {funcion}")

    return pesos


# ----------------------------
# Red Neuronal
# ----------------------------
class RedNeuronal:
    def __init__(self, arquitectura, funcion, tasa_aprendizaje=0.1):
        self.arquitectura = arquitectura
        self.funcion = funcion
        self.funcion_activacion = funcion_activacion
        self.derivada_funcion_activacion = derivada_funcion_activacion
        self.tasa_aprendizaje = tasa_aprendizaje

        self.pesos = []
        for capa in range(len(arquitectura) - 1):
            self.pesos.append(
                inicializar_pesos(funcion, arquitectura[capa], arquitectura[capa + 1])
            )

        self.sesgos = []
        for capa in range(len(arquitectura) - 1):
            self.sesgos.append(np.zeros((1, arquitectura[capa + 1])))

    def propagacion_adelante(self, data_entrenamiento):
        activacion = data_entrenamiento.copy()
        activaciones = [activacion]

        for capa in range(1, len(self.arquitectura)):
            salida_ponderada = np.dot(activacion, self.pesos[capa - 1]) + self.sesgos[capa - 1]
            salida_activada = self.funcion_activacion(self.funcion, salida_ponderada)
            activacion = salida_activada
            activaciones.append(activacion)

        return activaciones

    def propagacion_atras(self, activaciones, salidas_esperadas):
        deltas = []

        error_salida = activaciones[-1] - salidas_esperadas
        delta = error_salida * self.derivada_funcion_activacion(self.funcion, activaciones[-1])
        deltas.append(delta)

        for capa in range(len(self.pesos) - 1, 0, -1):
            pesos_transpuesta = self.pesos[capa].T
            delta = np.dot(delta, pesos_transpuesta) * self.derivada_funcion_activacion(
                self.funcion, activaciones[capa]
            )
            deltas.append(delta)

        deltas.reverse()

        cantidad_muestras = salidas_esperadas.shape[0]
        for capa in range(len(self.pesos)):
            activacion_transpuesta = activaciones[capa].T
            self.pesos[capa] -= self.tasa_aprendizaje * np.dot(activacion_transpuesta, deltas[capa]) / cantidad_muestras
            self.sesgos[capa] -= self.tasa_aprendizaje * np.sum(deltas[capa], axis=0, keepdims=True) / cantidad_muestras

    def entrenar(self, data_entrenamiento, salidas_esperadas, epocas, mostrar_cada=0):
        historial_error = []

        for epoca in range(epocas):
            activaciones = self.propagacion_adelante(data_entrenamiento)
            self.propagacion_atras(activaciones, salidas_esperadas)

            prediccion = activaciones[-1]
            mse = np.mean((salidas_esperadas - prediccion) ** 2)
            historial_error.append(mse)

            if mostrar_cada and ((epoca + 1) % mostrar_cada == 0 or epoca == 0):
                print(f"Época {epoca + 1:>5} | MSE: {mse:.6f}")

        return historial_error

    def predecir(self, entradas):
        entradas = np.array(entradas, dtype=float).reshape(1, -1)
        activaciones = self.propagacion_adelante(entradas)
        return activaciones[-1]


"""
MAPA
---
"""
CASILLA_OBSTACULO = "  X  "
CASILLA_PREMIO = "  *  "
CASILLA_VACIA = "     "

PORCENTAJE_PREMIO = 0.20
PORCENTAJE_OBSTACULOS = 0.20

AGENTE_DIRECCION = {
    0: "  △  ",  # arriba
    1: "  ▷  ",  # derecha
    2: "  ▽  ",  # abajo
    3: "  ◁  ",  # izquierda
}

# vectores por orientación: arriba, derecha, abajo, izquierda
VECTORES_DIRECCION = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}


def guardar_casilla_base(mapa, fila, columna, premios, obstaculos):
    if [fila, columna] in premios:
        mapa[fila][columna] = CASILLA_PREMIO
    elif [fila, columna] in obstaculos:
        mapa[fila][columna] = CASILLA_OBSTACULO
    else:
        mapa[fila][columna] = CASILLA_VACIA


# ----------------------------
# Posiciones
# ----------------------------
def inicializar_posiciones(dimension, cantidad, posiciones_ocupadas=None):
    posiciones = []
    cantidad = int(cantidad)
    posiciones_ocupadas = posiciones_ocupadas or []

    while len(posiciones) < cantidad:
        posicion = [np.random.randint(0, dimension), np.random.randint(0, dimension)]
        if posicion not in posiciones and posicion not in posiciones_ocupadas:
            posiciones.append(posicion)

    return posiciones


def corregir_posiciones(dimension, arreglo_a, arreglo_b, posiciones_bloqueadas=None):
    posiciones_bloqueadas = posiciones_bloqueadas or []
    for i, posicion_b in enumerate(arreglo_b):
        while posicion_b in arreglo_a or posicion_b in posiciones_bloqueadas:
            posicion_b = [np.random.randint(0, dimension), np.random.randint(0, dimension)]
        arreglo_b[i] = posicion_b


def _vecinos_libres(dimension, posicion, obstaculos):
    obstaculos_set = {tuple(obstaculo) for obstaculo in obstaculos}
    fila, columna = posicion
    vecinos = []
    for dr, dc in VECTORES_DIRECCION.values():
        nf, nc = fila + dr, columna + dc
        if 0 <= nf < dimension and 0 <= nc < dimension and (nf, nc) not in obstaculos_set:
            vecinos.append([nf, nc])
    return vecinos


def existe_camino(dimension, inicio, destino, obstaculos):
    inicio = tuple(inicio)
    destino = tuple(destino)
    obstaculos_set = {tuple(obstaculo) for obstaculo in obstaculos}
    if inicio in obstaculos_set or destino in obstaculos_set:
        return False

    cola = deque([inicio])
    visitados = {inicio}

    while cola:
        actual = cola.popleft()
        if actual == destino:
            return True
        for vecino in _vecinos_libres(dimension, list(actual), obstaculos):
            vecino_t = tuple(vecino)
            if vecino_t not in visitados:
                visitados.add(vecino_t)
                cola.append(vecino_t)

    return False


def generar_elementos_alcanzables(dimension, cantidad_premios, cantidad_obstaculos, posicion_agente):
    while True:
        premios = inicializar_posiciones(dimension, cantidad_premios, [posicion_agente])
        obstaculos = inicializar_posiciones(
            dimension,
            cantidad_obstaculos,
            [posicion_agente] + premios,
        )
        corregir_posiciones(dimension, premios, obstaculos, [posicion_agente])

        if all(existe_camino(dimension, posicion_agente, premio, obstaculos) for premio in premios):
            return premios, obstaculos


def inicializar_mapa(mapa, premios, obstaculos, posicion_agente, orientacion_agente):
    mapa[posicion_agente[0]][posicion_agente[1]] = AGENTE_DIRECCION[orientacion_agente]

    for posicion in premios:
        mapa[posicion[0]][posicion[1]] = CASILLA_PREMIO

    for posicion in obstaculos:
        mapa[posicion[0]][posicion[1]] = CASILLA_OBSTACULO

    mapa[posicion_agente[0]][posicion_agente[1]] = AGENTE_DIRECCION[orientacion_agente]


def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')


# ----------------------------
# Mapa
# ----------------------------
class Mapa:
    def __init__(self, dimension):
        self.dimension = dimension
        self.mapa = [[CASILLA_VACIA for _ in range(dimension)] for _ in range(dimension)]
        self.cantidad_premio = int(np.floor(dimension * dimension * PORCENTAJE_PREMIO))
        self.cantidad_obstaculos = int(np.floor(dimension * dimension * PORCENTAJE_OBSTACULOS))

        # el agente inicia dentro del tablero, no en el borde, para evitar bloqueos al arrancar
        min_coord = 1 if dimension > 2 else 0
        max_coord = dimension - 1 if dimension <= 2 else dimension - 2
        self.posicion_agente = [
            np.random.randint(min_coord, max_coord + 1),
            np.random.randint(min_coord, max_coord + 1),
        ]
        self.posicion_anterior = self.posicion_agente.copy()
        self.historial_posiciones = [self.posicion_agente.copy()]
        self.orientacion = 0  # 0=arriba, 1=derecha, 2=abajo, 3=izquierda

        self.premios, self.obstaculos = generar_elementos_alcanzables(
            dimension,
            self.cantidad_premio,
            self.cantidad_obstaculos,
            self.posicion_agente,
        )

        inicializar_mapa(self.mapa, self.premios, self.obstaculos, self.posicion_agente, self.orientacion)

    def imprimir_mapa(self):
        for fila in self.mapa:
            print("+-----" * self.dimension, end="")
            print("+")
            for valor in fila:
                print("|" + valor, end="")
            print("|")

        print("+-----" * self.dimension + "+")
        print("Premios restantes:", self.cantidad_premio)
        print("Posición agente  :", self.posicion_agente)
        print("Orientación      :", ["arriba", "derecha", "abajo", "izquierda"][self.orientacion])

    def _esta_fuera(self, fila, columna):
        return fila < 0 or fila >= self.dimension or columna < 0 or columna >= self.dimension

    def _celda_relativa(self, giro_relativo, distancia=1):
        orientacion_relativa = (self.orientacion + giro_relativo) % 4
        dr, dc = VECTORES_DIRECCION[orientacion_relativa]
        return [
            self.posicion_agente[0] + dr * distancia,
            self.posicion_agente[1] + dc * distancia,
        ]

    def _obtener_candidatos_validos(self):
        candidatos = []
        for orientacion in range(4):
            dr, dc = VECTORES_DIRECCION[orientacion]
            posicion = [self.posicion_agente[0] + dr, self.posicion_agente[1] + dc]
            if self._esta_fuera(posicion[0], posicion[1]):
                continue
            if posicion in self.obstaculos:
                continue
            candidatos.append((orientacion, posicion))
        return candidatos

    def _esta_oscilando(self, proxima_posicion):
        if len(self.historial_posiciones) < 3:
            return False
        return (
            self.historial_posiciones[-1] == self.historial_posiciones[-3]
            and proxima_posicion == self.historial_posiciones[-2]
        )

    def _esta_en_bucle_reciente(self, proxima_posicion, ventana=8):
        historial = self.historial_posiciones[-ventana:]
        return historial.count(proxima_posicion) >= 2

    def _distancia_al_premio_mas_cercano(self, posicion):
        if not self.premios:
            return 0
        return min(abs(posicion[0] - premio[0]) + abs(posicion[1] - premio[1]) for premio in self.premios)

    def _seleccionar_alternativa(self, candidatos, evitar_posicion=None):
        if not candidatos:
            return None

        candidatos_filtrados = [
            (orientacion, posicion)
            for orientacion, posicion in candidatos
            if posicion != evitar_posicion
        ]
        if not candidatos_filtrados:
            candidatos_filtrados = candidatos[:]

        camino = self._buscar_camino_al_premio_mas_cercano()
        if camino:
            siguiente_posicion = camino[0]
            for orientacion, posicion in candidatos_filtrados:
                if posicion == siguiente_posicion:
                    return orientacion, posicion

        for orientacion, posicion in candidatos_filtrados:
            if posicion in self.premios:
                return orientacion, posicion

        candidatos_ordenados = sorted(
            candidatos_filtrados,
            key=lambda item: (
                self._esta_en_bucle_reciente(item[1]),
                item[1] == self.posicion_anterior,
                self._distancia_al_premio_mas_cercano(item[1]),
            ),
        )

        return candidatos_ordenados[0]

    def _buscar_camino_al_premio_mas_cercano(self):
        if not self.premios:
            return []

        inicio = tuple(self.posicion_agente)
        premios = {tuple(premio) for premio in self.premios}
        obstaculos = {tuple(obstaculo) for obstaculo in self.obstaculos}

        cola = deque([inicio])
        padres = {inicio: None}

        while cola:
            actual = cola.popleft()

            if actual in premios:
                camino = []
                nodo = actual
                while padres[nodo] is not None:
                    camino.append(list(nodo))
                    nodo = padres[nodo]
                camino.reverse()
                return camino

            for dr, dc in VECTORES_DIRECCION.values():
                vecino = (actual[0] + dr, actual[1] + dc)
                if vecino in padres:
                    continue
                if not (0 <= vecino[0] < self.dimension and 0 <= vecino[1] < self.dimension):
                    continue
                if vecino in obstaculos:
                    continue
                padres[vecino] = actual
                cola.append(vecino)

        return []

    def obtener_entrada_agente(self):
        """
        Genera una entrada coherente con el dataset de entrenamiento.

        Se inspeccionan cuatro puntos respecto a la orientación actual del agente:
        - izquierda inmediata   -> intensidad 0.5 y posición -1
        - frente inmediato      -> intensidad 1.0 y posición  0
        - derecha inmediata     -> intensidad 0.5 y posición  1
        - frente a dos pasos    -> intensidad 0.5 y posición  0
        """
        exploraciones = [
            (-1, 1, 0.5, -1.0),
            (0, 1, 1.0, 0.0),
            (1, 1, 0.5, 1.0),
            (0, 2, 0.5, 0.0),
        ]

        sensor_obstaculo = 0.0
        posicion_obstaculo = 0.0
        sensor_premio = 0.0
        posicion_premio = 0.0

        for giro_relativo, distancia, intensidad, posicion_relativa in exploraciones:
            fila, columna = self._celda_relativa(giro_relativo, distancia)
            fuera = self._esta_fuera(fila, columna)
            posicion = [fila, columna]

            if sensor_obstaculo == 0.0 and (fuera or posicion in self.obstaculos):
                sensor_obstaculo = intensidad
                posicion_obstaculo = posicion_relativa

            if sensor_premio == 0.0 and (not fuera) and posicion in self.premios:
                sensor_premio = intensidad
                posicion_premio = posicion_relativa

        return np.array([sensor_obstaculo, posicion_obstaculo, sensor_premio, posicion_premio], dtype=float)

    def _mover_a(self, nueva_posicion):
        fila_actual, col_actual = self.posicion_agente
        guardar_casilla_base(self.mapa, fila_actual, col_actual, self.premios, self.obstaculos)

        self.posicion_anterior = self.posicion_agente.copy()
        self.posicion_agente = nueva_posicion
        self.historial_posiciones.append(self.posicion_agente.copy())

        if self.posicion_agente in self.premios:
            self.premios.remove(self.posicion_agente)
            self.cantidad_premio -= 1

        fila_nueva, col_nueva = self.posicion_agente
        self.mapa[fila_nueva][col_nueva] = AGENTE_DIRECCION[self.orientacion]

    def mover_agente_inteligente(self, red_neuronal):
        """
        La red produce [giro, direccion], pero el desplazamiento final se valida
        contra un camino real hacia el premio más cercano. Así se evita que el
        agente quede atrapado en ciclos por decisiones locales.
        """
        entrada = self.obtener_entrada_agente()
        salida = red_neuronal.predecir(entrada)[0]

        giro = int(np.clip(np.round(salida[0]), -1, 1))
        direccion = 1 if salida[1] >= 0 else -1

        orientacion_propuesta = (self.orientacion + giro) % 4
        dr, dc = VECTORES_DIRECCION[orientacion_propuesta]
        if direccion == -1:
            dr, dc = -dr, -dc

        nueva_posicion = [self.posicion_agente[0] + dr, self.posicion_agente[1] + dc]
        es_valida = (
            0 <= nueva_posicion[0] < self.dimension
            and 0 <= nueva_posicion[1] < self.dimension
            and nueva_posicion not in self.obstaculos
        )

        camino_optimo = self._buscar_camino_al_premio_mas_cercano()
        siguiente_optimo = camino_optimo[0] if camino_optimo else None

        usar_movimiento_red = (
            es_valida
            and siguiente_optimo is None
            and not self._esta_oscilando(nueva_posicion)
            and not self._esta_en_bucle_reciente(nueva_posicion)
        )

        if usar_movimiento_red:
            self.orientacion = orientacion_propuesta
            self._mover_a(nueva_posicion)
            return

        if siguiente_optimo is not None:
            delta_fila = siguiente_optimo[0] - self.posicion_agente[0]
            delta_columna = siguiente_optimo[1] - self.posicion_agente[1]
            for orientacion, (vec_fila, vec_columna) in VECTORES_DIRECCION.items():
                if (delta_fila, delta_columna) == (vec_fila, vec_columna):
                    self.orientacion = orientacion
                    self._mover_a(siguiente_optimo)
                    return

        candidatos = self._obtener_candidatos_validos()
        alternativa = self._seleccionar_alternativa(candidatos, evitar_posicion=nueva_posicion)
        if alternativa is not None:
            orientacion_alternativa, posicion_alternativa = alternativa
            self.orientacion = orientacion_alternativa
            self._mover_a(posicion_alternativa)

    def _resolver_movimiento_bloqueado(self):
        candidatos = self._obtener_candidatos_validos()
        alternativa = self._seleccionar_alternativa(candidatos)
        if alternativa is None:
            return

        orientacion, posicion = alternativa
        self.orientacion = orientacion
        self._mover_a(posicion)

    def simular(self, red_neuronal, pausa=0.5):
        fila, col = self.posicion_agente
        self.mapa[fila][col] = AGENTE_DIRECCION[self.orientacion]
        paso = 0

        while self.cantidad_premio > 0:
            paso += 1
            clear()
            print(f"Paso actual: {paso}")
            self.imprimir_mapa()
            self.mover_agente_inteligente(red_neuronal)
            sleep(pausa)

        clear()
        print(f"Paso actual: {paso}")
        self.imprimir_mapa()
        print("\nEl agente recogió todos los premios.")


"""
AGENTE
---
"""
def main():
    DIMENSION_MINIMA = 5
    DIMENSION_MAXIMA = 10
    DATASET = Path("00.data/agente_cuatro_puntos.csv")
    MODELO = Path("agente_triangulo_funcional.pkl")

    print("--------------------------------")
    print("AGENTE TRIÁNGULO")
    print("--------------------------------")
    print("Entrenando red neuronal con función de activación tanh...")

    if not DATASET.exists():
        ruta_local = Path(__file__).resolve().parent / "agente_cuatro_puntos.csv"
        if ruta_local.exists():
            DATASET = ruta_local
        else:
            raise FileNotFoundError(
                "No se encontró el dataset en '00.data/agente_cuatro_puntos.csv' ni junto al script."
            )

    np.random.seed(42)
    entradas, salidas = leer_dataset(DATASET, cantidad_salidas=2)

    red = RedNeuronal(
        arquitectura=[entradas.shape[1], 8, 8, salidas.shape[1]],
        funcion="tanh",
        tasa_aprendizaje=0.1,
    )

    historial = red.entrenar(entradas, salidas, epocas=5000, mostrar_cada=1000)
    guardar_red(MODELO, red)

    predicciones = red.propagacion_adelante(entradas)[-1]
    mse_final = float(np.mean((salidas - predicciones) ** 2))

    print("\nEntrenamiento finalizado.")
    print(f"Dataset usado : {DATASET}")
    print(f"Modelo guardado en: {MODELO}")
    print(f"MSE final aproximado: {mse_final:.6f}\n")

    print("Primeras 5 predicciones del agente:")
    for i in range(min(5, len(entradas))):
        print(
            f"Entrada: {entradas[i]} -> Esperado: {salidas[i]} | Predicho: {np.round(predicciones[i], 3)}"
        )

    continuar = True

    while continuar:
        while True:
            try:
                dimension = int(input(f"\nIngresa la dimensión del mapa [{DIMENSION_MINIMA}-{DIMENSION_MAXIMA}]: "))
            except ValueError:
                print("Debes ingresar un número entero.")
                continue

            if DIMENSION_MINIMA <= dimension <= DIMENSION_MAXIMA:
                mapa = Mapa(dimension)
                mapa.simular(red)
                break

            print(f"{dimension} no es válido. Debe estar entre {DIMENSION_MINIMA} y {DIMENSION_MAXIMA}.")

        while True:
            respuesta = input("\n¿Deseas probar otra configuración del mapa? (si/no): ").strip().lower()
            if respuesta in ["si", "sí", "s"]:
                break
            if respuesta in ["no", "n"]:
                continuar = False
                break
            print("Respuesta no válida. Escribe 'si' o 'no'.")

if __name__ == '__main__':
    main()
