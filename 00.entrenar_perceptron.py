# -- coding: utf-8 --
"""
Perceptron unicapa para compuerta AND y OR

Modelo de perceptrón para los siguientes casos:
- Compuerta AND
- Compuerta OR
- Dataset de cáncer de mama
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Activación y predicción
# =========================

def sigmoide(t):
    return 1.0 / (1.0 + np.exp(-t))

def escalon(prob, umbral=0.5):
    return 1 if prob >= umbral else 0

def forward(x, w):
    net = np.dot(x, w)
    return sigmoide(net)

def predict(x, w, umbral=0.5):
    return escalon(forward(x, w), umbral)


# =========================
# Datasets AND / OR
# =========================

def dataset_and():
    X = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=float)
    d = np.array([1, 0, 0, 0], dtype=float)
    return X, d

def dataset_or():
    X = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=float)
    d = np.array([1, 1, 1, 0], dtype=float)
    return X, d


# =========================
# Carga de CSV (N entradas + 1 clase)
# =========================

def cargar_dataset_csv_binario(filepath, n_entradas=None, delimiter=","):
    """
    Lee un CSV con encabezado, ignora el encabezado y asume:
      - n_entradas columnas de entrada
      - 1 columna de clase al final (0/1)
    Si n_entradas es None, se infiere como (#columnas - 1).

    Agrega bias automáticamente.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No existe el archivo: {filepath}")

    data = np.genfromtxt(filepath, delimiter=delimiter, skip_header=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_cols = data.shape[1]
    if n_cols < 2:
        raise ValueError("El CSV debe tener al menos 2 columnas (entradas + clase).")

    if n_entradas is None:
        n_entradas = n_cols - 1

    if n_cols != n_entradas + 1:
        raise ValueError(
            f"Columnas inválidas. Esperaba {n_entradas + 1} columnas "
            f"({n_entradas} entradas + 1 clase), pero encontré {n_cols}."
        )

    Xn = data[:, 0:n_entradas].astype(float)
    d = data[:, n_entradas].astype(float)  # última columna = clase

    # Bias
    bias = np.ones((Xn.shape[0], 1), dtype=float)
    X = np.hstack([Xn, bias])

    return X, d


# =========================
# Entrenamiento (Regla Delta)
# =========================

def entrenar_delta(X, d, w_init, epocas=50, eta=0.1, umbral=0.5, verbose=False):
    """
    Regla Delta:
      y = sigmoide(net)
      e = d - y
      w <- w + eta * e * x

    Costo (MSE/2): (1/N) * sum( 0.5*(d - y)^2 )
    """
    w = w_init.astype(float).copy()
    N = X.shape[0]

    historial_costo = []
    historial_acc = []

    for ep in range(epocas):
        costo_ep = 0.0
        aciertos = 0

        for i in range(N):
            x = X[i]
            yi = forward(x, w)
            e = d[i] - yi

            w = w + eta * e * x

            costo_ep += 0.5 * (e ** 2)
            if (1 if yi >= umbral else 0) == int(d[i]):
                aciertos += 1

        costo_ep /= N
        acc_ep = (aciertos / N) * 100.0

        historial_costo.append(costo_ep)
        historial_acc.append(acc_ep)

        if verbose:
            print(f"Época {ep+1:03d} | Costo={costo_ep:.6f} | Acc={acc_ep:.2f}% | w={w}")

    return w, historial_costo, historial_acc


# =========================
# Gráficas
# =========================

def graficar_metricas(hist_acc, hist_costo):
    epocas = np.arange(1, len(hist_acc) + 1)

    plt.figure(figsize=(12, 5))

    # Gráfico 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.title("% Precisión de clasificación por época")
    plt.xlabel("Época")
    plt.ylabel("Accuracy (%)")
    plt.plot(epocas, hist_acc, label="Accuracy")
    plt.legend()

    # Gráfico 2: Costo
    plt.subplot(1, 2, 2)
    plt.title("Descenso del gradiente (Costo) por época")
    plt.xlabel("Época")
    plt.ylabel("Costo (MSE/2)")
    plt.plot(epocas, hist_costo, label="Costo")
    plt.legend()

    plt.tight_layout()
    plt.show()


# =========================
# Menú principal
# =========================

def menu_dataset():
    print("\n======================================")
    print("== SELECCIÓN DE DATASET ==============")
    print("1) Dataset función AND")
    print("2) Dataset función OR")
    print("3) Cargar archivo desde /data")
    opc = int(input("Selecciona opción [1-3] >> "))

    if opc == 1:
        return dataset_and(), "AND"
    elif opc == 2:
        return dataset_or(), "OR"
    elif opc == 3:
        print("\n--- CARGA DE ARCHIVO (/data) ---")
        print('1) "cancerMamaDiscretizado.csv"')
        print("2) Especificar nombre de archivo")
        sub = input("Selecciona [1 / 2] >> ").strip()

        data_dir = "data"

        if sub == "1":
            f1 = os.path.join(data_dir, "cancerMamaDiscretizado.csv")
            filepath = f1

            # INFIERE n_entradas = (#cols - 1)
            X, d = cargar_dataset_csv_binario(filepath, n_entradas=None)
            return (X, d), os.path.basename(filepath)

        elif sub == "2":
            nombre = input("Nombre del archivo (ej: miDataset.csv) >> ").strip()
            n_entradas = int(input("Cantidad de entradas (features) >> "))

            filepath = os.path.join(data_dir, nombre)
            X, d = cargar_dataset_csv_binario(filepath, n_entradas=n_entradas)
            return (X, d), nombre

        else:
            print("Opción no reconocida. Saliendo.")
            exit(0)
    else:
        print("Opción no reconocida. Saliendo.")
        exit(0)


def main():
    (X, d), nombre_ds = menu_dataset()

    print("\n======================================")
    print("== ENTRENANDO PERCEPTRÓN (DELTA) =====")
    print(f"== Dataset: {nombre_ds}")
    print("======================================")

    # Pesos iniciales: (#entradas + bias)
    n_pesos = X.shape[1]
    w = -1.0 * np.ones(n_pesos, dtype=float)

    opc = int(input(f"Pesos iniciales: [1) Dar pesos, 2) Predeterminados ({w.tolist()})] >> "))
    if opc == 1:
        for i in range(n_pesos):
            w[i] = float(input(f"Dar peso {i+1} >> "))
    elif opc != 2:
        print("Opción no reconocida. Saliendo.")
        exit(0)

    epocas = 50
    opc = int(input("Épocas: [1) Dar #, 2) Predeterminadas (50)] >> "))
    if opc == 1:
        epocas = int(input("Dar # de épocas >> "))
    elif opc != 2:
        print("Opción no reconocida. Saliendo.")
        exit(0)

    eta = 0.1
    opc = int(input("Tasa de aprendizaje (eta): [1) Dar eta, 2) Predeterminada (0.1)] >> "))
    if opc == 1:
        eta = float(input("Dar eta >> "))
    elif opc != 2:
        print("Opción no reconocida. Saliendo.")
        exit(0)

    umbral = 0.5

    print("\n--- Entrenando ---")
    w_final, hist_costo, hist_acc = entrenar_delta(
        X=X, d=d, w_init=w, epocas=epocas, eta=eta, umbral=umbral, verbose=False
    )

    print("\n--- Resultados finales ---")
    print("Pesos iniciales:", w)
    print("Pesos finales  :", w_final)

    # Mostrar algunas predicciones (primeras 10 si es grande)
    print("\n(Mostrando hasta 10 ejemplos)")
    print("\tEntradas... \t-> y_pred \t(d)")
    max_show = min(10, X.shape[0])
    for i in range(max_show):
        y_pred = predict(X[i], w_final, umbral=umbral)
        # imprimir todas las entradas excepto bias
        entradas = [int(v) for v in X[i][:-1]]
        print(f"\t{entradas}\t-> {y_pred}\t\t({int(d[i])})")

    graficar_metricas(hist_acc, hist_costo)


if __name__ == "__main__":
    main()