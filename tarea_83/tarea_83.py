print("Este programa devuelve el area de un cuadrado según la longitud de su lado")

a=input("Introduzca el lado del cuadrado: ")
anum=float(a)
if anum>0:
    Area=anum*anum
    print(Area)
else:
    print("El lado del cuadrado debe ser un valor positivo")