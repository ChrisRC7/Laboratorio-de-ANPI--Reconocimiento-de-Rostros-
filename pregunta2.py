import numpy as np
import time
import matplotlib.pyplot as plt
from svdCompact import *



# Inicializar listas para almacenar los tiempos de ejecución de svd y svdCompact
execution_times_svd = []
execution_times_svdCompact = []

# Crear un vector de valores de k
k_values = [5, 6, 7, 8, 9, 10, 11, 12]

for k in k_values:
    # Generar una matriz aleatoria A de tamaño 2^k x 2^(k-1)
    A = np.random.rand(2**k, 2**(k-1))
    
    # Medir el tiempo de ejecución de la función svd
    start_time = time.time()
    U, S, V = np.linalg.svd(A)
    time_svd = time.time() - start_time
    
    # Medir el tiempo de ejecución de la función svdCompact
    start_time = time.time()
    Ur, Sr, Vr = svdCompact(A)
    time_svdCompact = time.time() - start_time
    
    # Almacena los tiempos de ejecución en las listas
    execution_times_svd.append(time_svd)
    execution_times_svdCompact.append(time_svdCompact)

# Graficar los tiempos de ejecución de svd y svdCompact
plt.plot(k_values, execution_times_svdCompact, 'o-', label='svdCompact')
plt.plot(k_values, execution_times_svd, 'o-', label='svd')
plt.xlabel('Valor de k')
plt.ylabel('Tiempo de ejecución (segundos)')
plt.title('Tiempo de ejecución de svdCompact y svd vs. k')
plt.legend()
plt.grid(True)
plt.show()
