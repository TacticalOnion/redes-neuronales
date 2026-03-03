# -- coding: utf-8 --
"""
Perceptron unicapa para compuerta AND y OR

Modelo de perceptrón para los siguientes casos:
- Compuerta AND
- Compuerta OR
- Dataset de cáncer de mama
"""

import numpy as np
import matplotlib.pyplot as plt

# Uso de perceptrón entrenado para la compuerta AND (pesos adecuados)
# Y o CONJUNCION
x_entradas=[[1,1,1], [1,0,1], [0,1,1], [0,0,1]]
#x_entradas=[[1,1,0], [1,0,1], [0,1,1], [0,0,0]]
# Pesos adecuados
# perceptron unicapa: 
# Pesos ideales
# pesosP = [1,1,-1.5]
# Pesos iniciales
pesosP = [-1,-1,-1]
# clases para la CONJUNCIÓN
w_clases = [1,0,0,0]
umbral = 0.5
# Función sigmoide S(t) = ___1__
#                         1 + e^-t
def sigmoide(t):
    return (1 / (1 + pow(2.718281828,-t)))

# Función escalón E | si( y > umbral) = 1
#                   | 0
def escalon(suma_ponderada):
    if suma_ponderada > umbral:
        return 1
    else:
        return 0

# Perceptón unicapa
def perceptron(ejemplar):
    suma_ponderada = 0;
    print("DATOS:")
    print(ejemplar)
    for x,w in zip(ejemplar,pesosP):
        suma_ponderada += x*w
    sig = sigmoide(suma_ponderada)
    esc = escalon(sig)
    escalon(sigmoide(suma_ponderada))
    print("SUMA PONDERADA "+str(suma_ponderada))
    print("SIGMOIDE ="+str(sig))
    print("ESCALON ="+str(esc))
    return esc

# Perceptrón unicapa parametrizado
def perceptronP(ejemplar,pesos):
    suma_ponderada = 0;
    for x,w in zip(ejemplar,pesos):
        suma_ponderada += x*w
    sig = sigmoide(suma_ponderada)
    esc = escalon(sig)
    return esc

# Perceptrón unicapa parametrizado
def perceptronE(ejemplar,pesos):
    suma_ponderada = 0;
    for x,w in zip(ejemplar,pesos):
        suma_ponderada += x*w
    sig = sigmoide(suma_ponderada)
    esc = escalon(sig)
    return sig,esc


# Obtiene los nuevos pesos a partir de la neurona actual
# p: peso del enlace actual
# x: dato de entrada
# y: valor calculado (observado) 
def actualizaP(p,x,y):
    p1 = p + x * y
    print("p1 = p + x * y",p1,p,"+",x,"*",y)
    return p1

def entrenar(ejemplares,pesos,clases,limite):
    print("-----------------------------------")
    print(" Entrenando")
    print(" Pesos ",pesos)
    print(" Clases",clases)
    # número de ajustes por mala clasificación
    a = 0;
    ajustes = []
    for i in range(limite):
        for e,c in zip(ejemplares,clases):
            y,ye = perceptronE(e,pesos)
            # Discrepancia entre valor observado y valor esperado (clase)
            if ye != c:
                a = a + 1
                ajustes.append(a)
                print("Eureka")
                print(e," -> ",ye," vs ",c)
                if c == 0:
                    y = y * -1
                pesos = [actualizaP(p,d,y) for p,d in zip(pesos,e)]
                print(" Nuevos Pesos ",pesos)
            else:
                ajustes.append(a)

    print("Fin")
    print(" Pesos ",pesos)
    print(" # Errores de clasificación",a)
    print("-----------------------------------")
    return pesos,ajustes

# Grafica un vector de datos dado
def graficaPesos(datos1,datos2,ajustes):
    plt.subplot(1,2,1)
    plt.title("Pesos iniciales vs finales") 
    # etiqueta abcisas
    plt.xlabel("X") 
    # etiqueta ordenadas 
    plt.ylabel("Y") 
    plt.plot(datos1,color="green",label="P Iniciales") 
    plt.plot(datos2,color="orange",label="P Finales") 
    plt.legend()
    plt.subplot(1,2,2)
    plt.title("Ajuste de pesos") 
    plt.xlabel("# pruebas") 
    plt.ylabel("# ajustes x error al clasificar") 
    plt.plot(ajustes,color="blue",label="Ajustes") 
    plt.legend()
    plt.show()


def main():
    # parámetros: (ejemplares,pesos,clases,limite)
    print("======================================")
    print("== ENTRENANDO PERCEPTRÓN =============")
    print("== Función buscada: Conjunción (Y) ===")
    print("== Pesos iniciales predeterminados ===")
    print("==", pesosP,"===")
    opc = int(input('Dame los pesos iniciales o acepta los predeterminados: [1 (Dar pesos), 2 Predeterminados]>>'))
    if opc == 1:
        pesosP[0]  = int(input('Dar peso 1:]>>'))
        pesosP[1]  = int(input('Dar peso 2:]>>'))
        pesosP[2]  = int(input('Dar peso 3:]>>'))
    elif opc != 2:
        print("Opción no reconocida, adiós")
        exit(0)
    limite = 25
    opc = int(input('Dame el no. de iteraciones o acepta las predeterminadas [1 (Dar # de iteraciones), 2 Predeterminadas:25 ]>>'))
    if opc == 1:
        limite = int(input('Dar # de iteraciones ]>>'))
    elif opc != 2:
        print("Opción no reconocida, adiós")
        exit(0)

    print("=====================================")
    print("== USANDO PERCEPTRÓN ENTRENADO ======")
    pesosE,ajustes = entrenar(x_entradas,pesosP,w_clases,limite)
    for e in x_entradas:
        print("============================")
        print("CÁLCULOS");
        print("-----------------------------------")
        salida = perceptronP(e,pesosE)
        print("-----------------------------------")
        print("RESULTADO");
        print("\tDATOS \t SALIDA")
        print("\t"+str(e[0])+" Y "+str(e[1])+" = "+str(salida))
    graficaPesos(pesosP,pesosE,ajustes)


if __name__ == "__main__":
    main()

