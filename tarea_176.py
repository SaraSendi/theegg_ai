#Este programa tiene 6 funciones diferentes, cada una de ellas soluciona uno de los problemas propuestos en la TAREA  176.
# Es el propio usuario quien decide que problema desea resolver, e introduce el usuario los datos para resolver los problemas      

def numMax ():
    print("Este programa solicita al usuario 3 números y determina el máximo")
    num=[] #lista para almacenar los numeros solicitados al usuario
    i=1
    while i<4:
        numeroSTR=input("Introduzca el número "+str(i)+": ")
        numero=float(numeroSTR)
        i=i+1
        num.append(numero)

    maximo=max(num)
    print(maximo)
def fraseLen ():
    print("Este programa solicita al usuario una frase e indica su longitud, incluyendo los espacios y todo tipo de carácter")
    frase=input("Introduzca una frase: ")
    longitud=len(frase)
    print(longitud)
def vocal():
    print("Este programa solicita al usuario un carácter e indica si se trata de una vocal")
    caracter=input("Introduzca un carácter:")

    if caracter == "a" or caracter == "e" or caracter == "i" or caracter == "o" or caracter == "u":
        print("Se trata de una vocal")
    elif caracter == "A" or caracter == "E" or caracter == "I" or caracter == "O" or caracter == "U":
        print("Se trata de una vocal")
    else:
        print("NO se trata de una vocal")

def palindromo():
    print("Esta función determina si la paralabra introducida es un palíndromo o no")
    palabraI=input("Introduzca una palabra:") #Solicito al usuario la palabra
    palabra=palabraI.upper() #Paso el string a mayúsculas para que no detecte diferentes las mayúsculas y minúsculas
      
    palabraINV=list(reversed(palabra)) #Invierto la lista de la palabra para comparar los elementos de la lista
    for x in range(len(palabra)): #Recorro todos los elementos de la palabra (Lo recorro por posiciones)
        if palabra[x]==palabraINV[x]: #Si los elementos de la palabra y la palabra invertida sigo en el bucle y sino lo paro
            continue
        else:
            break
    if x==len(palabra)-1: #Si completo todo el blucle es un palindromo, pero si lo rompo antes no lo es (la longitud x va en todas las posiciones empieza en 0 y la longitud en 1, por eso el -1)
        print("Se trata de un palindromo")
    else:
        print("NO es un palindromo")

def suma():
    print("Esta función suma los elementos de una lista, los elementos deben ser números")
    lista=[] #lista donde se almacenan los números que el usuario desea sumar
    yn="Y"
    while yn=="Y" or yn=="y": #Bucle: Mientras el usuario quiera introducir números debe pulsar Y y seguirá el bucle
        numeroSTR=input("Introduzca un número: ")
        numero=float(numeroSTR) #EL dato introducido es str y lo paso a float para poder sumar
        lista.append(numero) #Incorporo a la lista el número
        yn=input("Desea introducir otro número Y/N?")
    suma=sum(lista)
    print(suma)

def coincidencia():
    print("Esta función compara dos listas,introducidas por el usuario, y dice que elementos están repetidos")
    lista1=[] 
    lista2=[]
    coincide=[]
    yn="Y"
    #Introduzco los elemntos de la primera lista
    print("ELEMENTOS LISTA 1")
    while yn=="Y" or yn=="y":
        elemento1=input("Introduzca un elemento: ")
        yn=input("Desea introducir otro elemento a la lista 1? Y/N ")
        lista1.append(elemento1)
    #Introduzca los elementos de la segunda lista
    yn="Y"
    print("ELEMENTOS LISTA 2")
    while yn=="Y" or yn=="y":
        elemento2=input("Introduzca un elemento: ")
        yn=input("Desea introducir otro elemento a la lista 2 Y/N? ")
        lista2.append(elemento2)
    #Comparo las listas
    elementos_lista1 = list(dict.fromkeys(lista1)) #analizo solo los elementos diferentes de cada lista, retiro los repetidos
    elementos_lista2 = list(dict.fromkeys(lista2))
    for n in elementos_lista1: #recorro los elementos de la lista 1 y de la lista 2
        for m in elementos_lista2:
            if m==n:
                coincide.append(n)
    if len(coincide)==0:
        print("NO hay ningun elemento que coincide")
    else:
        print("Los elementos que coinciden son: ")
        print(coincide)




print("A continuación se presentan los problemas que este programa es capaz de resolver")
print("     1) Calcular el máximo de 3 números")
print("     2) Determinar la longitud de una frase")
print("     3) El carácter introducido es una vocal o no")
print("     4) Sumar los valores de una lista")
print("     5) Determinar si la palabra introducida es un palíndromo")
print("     6) Calcular los valores coincidentes de una lista")
respuestaSTR=input("¿Que problema desea resolver? Indíquelo con el número: ")
respuesta=float(respuestaSTR)

if respuesta==1:
    numMax()
elif respuesta==2:
    fraseLen()
elif respuesta==3:
    vocal()  
elif respuesta==4:
    suma() 
elif respuesta==5:
    palindromo()
elif respuesta==6:
    coincidencia()
else:
    print("La tarea introducida debe estar en la lista")