import numpy as np

#Vector entre 7 y 66
a=np.linspace(7,66,10, dtype=int)
print(a)

#Invertir el vector
b=np.flip(a)
print(b)

#Matriz 4*4 con valores entre 0 y 15
c=np.random.randint(0,15,(4,4)) 
print(c)

#MAtriz de indentidad 5*5
d=np.identity(5).astype(int)
print(d)

#5*5 centro 1 y bordes 1 el resto 0
e=np.zeros((5,5)).astype(int)
e[2,2]=1 #modifico el elemento central
e1=np.ones(5).astype(int) #genero un array de unos para introducirlo
e[0,:]=e1 #Pongo las filas como 1
e[4,:]=e1
e[:,0]=e1 #Pongo las columnas como 1
e[:,4]=e1
print(e)

#matriz 4*4 los valores del 0 al 3 aumentando
f = np.linspace(0, 3, 4 * 4).reshape(4, 4)
print(f)

#Array de 0 2*7
g=np.zeros((2,7)).astype(int)
print(g)

#5*4 todo ceros menos la primera fila que es 1
h=np.zeros((5,4)).astype(int)
h1=np.ones(4).astype(int) #genero un array de unos para introducirlo
h[0,:]=h1 #Pongo las filas como 1
print(h)

#Array tipo tablero de ajedrez
#dimensiones del tablero 8*8
i=[]
x=8

for n in range(x):
    for m in range(x):
        print((m + n) % 2, end=' ') # indica el modulo (hondarra) de la suma de n y m (estas son las posiciones que recorro), si es impar sale uno y si es par 0
        i.append((m + n) % 2) #si ademas de visualizarlo quiero almacenar el array generado- lo guardo como una lista
    print('\n', end=' ') #nueva linea

i2=np.array(i).reshape(8,8) #Trasformo la lista en un array de las dimensiones que quiero 
#print(i2)
