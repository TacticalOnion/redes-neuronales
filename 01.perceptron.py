import numpy as np

def activacion_escalon(suma_ponderada):
    """
    Funcion escalón
    """
    if suma_ponderada >= 0:
        activacion = 1
    else:
        activacion = 0
    
    return activacion

def leer_dataset(nombre_archivo):
    dataset = np.loadtxt(nombre_archivo, delimiter = ",", skiprows = 1)
    dataset_entrada = dataset[:,:-1]
    salidas_esperadas = dataset[:,-1:]

    return dataset_entrada, salidas_esperadas

class Perceptron:
    def __init__(self, cantidad_entradas, tasa_aprendizaje):
        self.pesos = np.zeros(cantidad_entradas)
        self.sesgo = 0.0
        self.tasa_aprendizaje = tasa_aprendizaje
    
    def predecir(self, entradas):
        suma_ponderada = np.dot(entradas, self.pesos) + self.sesgo
        prediccion = activacion_escalon(suma_ponderada)
        
        return prediccion

    def entrenar(self, dataset_entradas, salidas_esperadas, epocas):
        if dataset_entradas.shape[0] != salidas_esperadas.shape[0]:
            raise ValueError(f"Cada registro del dataset de entradas debe tener 1 salida esperada.")

        for epoca in range(epocas):
            for entrada, salida_esperada in zip(dataset_entradas, salidas_esperadas):
                prediccion = self.predecir(entrada)
                error = salida_esperada - prediccion

                self.pesos += self.tasa_aprendizaje * error * entrada
                self.sesgo += self.tasa_aprendizaje * error

def main():
    np.random.seed(21)
    archivo = "00.data/compuerta_or.csv"
    dataset_entradas, salidas_esperadas = leer_dataset(archivo)
    

    modelo = Perceptron(
        cantidad_entradas = dataset_entradas.shape[1], 
        tasa_aprendizaje = 0.1
    )
    modelo.entrenar(dataset_entradas, salidas_esperadas, epocas = 20)

    print("Resultados")
    for entrada in dataset_entradas:
        print(entrada, "->", modelo.predecir(entrada))

if __name__ == "__main__":
    main()