import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Grisetara pasatu irudia
    if len(image.shape) == 3:  # Koloretan 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Grisetan
        gray_image = image
    
    return gray_image

def apply_convolution(image, kernel):
    # Aplicar la convolución
    convolved_image = cv2.filter2D(image, -1, kernel)
    return convolved_image

def main():
    image_path = 'C:---panda.png'  #Aldatu gure dozun irudiagatik
    
    # Cargar y preprocesar la imagen
    gray_image = load_and_preprocess_image(image_path)

    # Ejercicio 1: Filtro de identidad
    identity_kernel = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])
    identity_image = apply_convolution(gray_image, identity_kernel)

    # Ejercicio 2: Filtro de desenfoque
    blur_kernel = np.full((3, 3), 1/5)
    blurred_image = apply_convolution(gray_image, blur_kernel)

    # Mostrar resultados
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title("Imagen Original")
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Ejercicio 1 - Filtro Identidad")
    plt.imshow(identity_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Ejercicio 2 - Filtro Desenfoque")
    plt.imshow(blurred_image, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()


##################### RESULTADOS
"""
Ejercicio 1: La imagen debería verse igual a la original porque el filtro de identidad no altera los valores de los píxeles.
Ejercicio 2: La imagen resultante será una versión suavizada de la original, con los detalles más finos difuminados debido al promedio de los píxeles en un vecindario de 3x3.
"""