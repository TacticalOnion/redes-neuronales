# Entradas
* `Sensor_Obstaculo`: 0 (no), 0.5 (cerca), 1 (muy cerca)
* `Pos_Obstaculo`: -1 (izquierda), 0 (no visible), 1 (derecha)
* `Sensor_Premio`: 0 (no), 0.5 (cerca), 1 (muy cerca)
* `Pos_Premio`: -1 (izquierda), 0 (centro/no visible), 1 (derecha)

# Salidas
* `Giro`: -1 izquierda, 0 recto, 1 derecha
* `Direccion`: 1 avanzar, -1 retroceder

# Reglas usadas
1. **Prioridad 1:** evitar obstáculo cercano.
2. **Prioridad 2:** buscar premio si no hay peligro.
3. **Si no hay nada:** avanzar.
