#Este módulo calcula la altura que tarda un objeto en alcanzar el suelo segun el tiempo que tarda. La variable de entrada es el tiempo en todas las funciones y la salida la altura en cada planeta
#Como variable local se define la gravedad en cada caso 
def tierra(t):
    g_tierra=9.8
    y_tierra=1/2*g_tierra*t**2
    return y_tierra
def marte(t):
    g_marte=3.7
    y_marte=1/2*g_marte*t**2
    return y_marte
def jupiter(t):
    g_jupiter=27.8
    y_jupiter=1/2*g_jupiter*t**2
    return y_jupiter