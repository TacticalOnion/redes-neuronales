import numpy as np

def activacion_sigmoide(suma_ponderada):
    activacion = 1 / (1 + np.exp(-suma_ponderada))

    return activacion

def derivada_sigmoide(salida_activada):
    derivada = salida_activada * (1 - salida_activada)

    return derivada

def leer_dataset(nombre_archivo):
    dataset = np.loadtxt(nombre_archivo, delimiter = ",", skiprows = 1)
    dataset_entrada = dataset[:,:-1]
    salidas_esperadas = dataset[:,-1:]

    return dataset_entrada, salidas_esperadas

class PerceptronMulticapa:
    def __init__(self, cantidad_entradas, cantidad_ocultas, cantidad_salidas, tasa_aprendizaje = 0.1):
        self.tasa_aprendizaje = tasa_aprendizaje

        # Inicializacion de pesos
        self.pesos_entrada_oculta = np.random.randn(cantidad_entradas, cantidad_ocultas) * 0.1
        self.sesgo_oculta = np.zeros((1, cantidad_ocultas))

        self.pesos_oculta_salida = np.random.randn(cantidad_ocultas, cantidad_salidas) * 0.1
        self.sesgo_salida = np.zeros((1,cantidad_salidas))

    def propagacion(self, entradas):
        # Capa oculta
        self.suma_ponderada_oculta = np.dot(entradas, self.pesos_entrada_oculta) + self.sesgo_oculta
        self.prediccion_oculta = activacion_sigmoide(self.suma_ponderada_oculta)

        # Capa salida
        self.suma_ponderada_salida = np.dot(self.prediccion_oculta,self.pesos_oculta_salida) + self.sesgo_salida
        self.prediccion_salida = activacion_sigmoide(self.suma_ponderada_salida)

        return self.prediccion_salida

    def entrenar(self, entradas, salidas_esperadas, epocas):

        for epoca in range(epocas):
            # Propagación hacia adelante
            salida = self.propagacion(entradas)

            # Error salida
            error_salida = salida - salidas_esperadas
            delta_salida = error_salida * derivada_sigmoide(salida)

            # Error capa oculta
            pesos_oculta_salida_transpuesta = self.pesos_oculta_salida.T
            error_oculta = np.dot(delta_salida, pesos_oculta_salida_transpuesta)
            delta_oculta = error_oculta * derivada_sigmoide(self.prediccion_oculta)

            # Actualizacion de pesos y sesgos
            prediccion_oculta_transpuesta = self.prediccion_oculta.T
            self.pesos_oculta_salida -= self.tasa_aprendizaje * np.dot(prediccion_oculta_transpuesta, delta_salida)
            self.sesgo_salida -= self.tasa_aprendizaje * np.sum(delta_salida, axis = 0, keepdims = True)

            entradas_transpuesta = entradas.T
            self.pesos_entrada_oculta -= self.tasa_aprendizaje * np.dot(entradas_transpuesta, delta_oculta)
            self.sesgo_oculta -= self.tasa_aprendizaje * np.sum(delta_oculta, axis = 0, keepdims = True)

    def predecir(self, entradas):
        salida = self.propagacion(entradas)
        
        return (salida >= 0.5).astype(int)
    
def main():
    np.random.seed(21)
    archivo = "00.data/compuerta_xor.csv"
    dataset_entradas, salidas_esperadas = leer_dataset(archivo)

    modelo = PerceptronMulticapa(
        cantidad_entradas = dataset_entradas.shape[1],
        cantidad_ocultas = 2,
        cantidad_salidas = 1,
        tasa_aprendizaje = 0.5
    )

    modelo.entrenar(dataset_entradas, salidas_esperadas, epocas = 10000)

    print("Resultados")
    for entrada in dataset_entradas:
        print(entrada, "->", modelo.predecir(entrada))

if __name__ == "__main__":
    main()
        

