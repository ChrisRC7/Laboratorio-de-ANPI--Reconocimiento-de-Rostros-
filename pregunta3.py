import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from svdCompact import *

# Directorio principal
main_directory = 'training/s1'
folders = os.listdir('training')
carpetas = len(folders) - 2
imagenes = len(os.listdir(main_directory)) - 2
N = carpetas * imagenes

# Inicializar S con la primera imagen
first_img_path = os.path.join('training', 's1', '1.jpg')
first_img = io.imread(first_img_path)
S = first_img.flatten().reshape(-1, 1)

for k in range(1, carpetas + 1):
    dire = os.path.join('training', 's' + str(k))
    for j in range(1, imagenes + 1):
        imgpath = os.path.join(dire, str(j) + '.jpg')
        A = io.imread(imgpath)
        S = np.hstack((S, A.flatten().reshape(-1, 1)))

num_columnas = S.shape[1]

# Calcular la cara promedio usando (2)
mean_face = np.mean(S, axis=1)

# Formar la matriz A con la cara promedio calculada
A = S - mean_face[:, np.newaxis]

# Calcular la SVD de A
Ur, Sr, Vr = svdCompact(A)

# Obtener el rango de la matriz A
r = np.linalg.matrix_rank(Sr)

# Calcular los vectores de coordenadas para cada individuo conocido
coordinate_vectors = np.zeros((r, N))

for i in range(N):
    xi = np.dot(Ur[:, :r].T, S[:, i] - mean_face)
    coordinate_vectors[:, i] = xi

# Reconocimiento facial para cada carpeta
for k in range(1, carpetas + 1):
    newimg = os.path.join('compare', 'p' + str(k) + '.jpg')
    new_face = io.imread(newimg)

    # Suponiendo que new_face ya está en escala de grises, no es necesario convertir
    # new_face = color.rgb2gray(new_face)
    new_face_vector = new_face.flatten()

    # Calcular el vector de coordenadas x usando (13)
    x = np.dot(Ur[:, :r].T, new_face_vector - mean_face)

    # Calcular la proyección del vector fp en el espacio de caras usando (16)
    fp = np.dot(Ur[:, :r], x)

    # Calcular la distancia εf al espacio de caras usando (17)
    epsilon_f = np.linalg.norm(new_face_vector - fp) / np.linalg.norm(new_face_vector)



    # Calcular la distancia iε a cada individuo conocido usando (14)
    distances_to_known_individuals = np.zeros(N)

    for i in range(N):
        distances_to_known_individuals[i] = np.linalg.norm(x - coordinate_vectors[:, i])

    # Identificar el individuo conocido asociado con la distancia mínima
    min_index = np.argmin(distances_to_known_individuals)

    # Mostrar las imágenes
    plt.subplot(1, 2, 1)
    plt.imshow(new_face, cmap='gray')
    plt.title('Nueva Cara')

    plt.subplot(1, 2, 2)
    identified_face = np.reshape(S[:, min_index], new_face.shape)
    plt.imshow(identified_face, cmap='gray')
    plt.title('Cara Identificada')

    plt.suptitle('Error: {:.2f}%'.format(epsilon_f))
    plt.pause(1)
    plt.clf()

