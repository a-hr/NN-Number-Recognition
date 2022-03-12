# Codigo para mostrar imagenes del set, no es necesario ejecutarlo, solo imprime unos numeros :)
import matplotlib.pyplot as plt
import numpy as np

filas = 2
columnas = 8
num = filas * columnas
imagenes = X_training[0:num]
etiquetas = Y_training[0:num]
fig, axes = plt.subplots(filas, columnas, figsize=(1.5 * columnas, 2 * filas))
for i in range(num):
    ax = axes[i // columnas, i % columnas]
    ax.imshow(imagenes[i].reshape(28, 28), cmap="gray_r")
    ax.set_title("Label: {}".format(np.argmax(etiquetas[i])))
plt.tight_layout()
plt.show()


# Codigo para mostrar imagenes del set, no es necesario ejecutarlo, solo imprime como se ven antes y despues de las transformaciones
filas = 4
columnas = 8
num = filas * columnas
print("ANTES:\n")
fig1, axes1 = plt.subplots(filas, columnas, figsize=(1.5 * columnas, 2 * filas))
for i in range(num):
    ax = axes1[i // columnas, i % columnas]
    ax.imshow(X_training[i].reshape(28, 28), cmap="gray_r")
    ax.set_title("Label: {}".format(np.argmax(Y_training[i])))
plt.tight_layout()
plt.show()
print("DESPUES:\n")
fig2, axes2 = plt.subplots(filas, columnas, figsize=(1.5 * columnas, 2 * filas))
for X, Y in datagen.flow(
    X_training,
    Y_training.reshape(Y_training.shape[0], 10),
    batch_size=num,
    shuffle=False,
):
    for i in range(0, num):
        ax = axes2[i // columnas, i % columnas]
        ax.imshow(X[i].reshape(28, 28), cmap="gray_r")
        ax.set_title("Label: {}".format(int(np.argmax(Y[i]))))
    break
plt.tight_layout()
plt.show()