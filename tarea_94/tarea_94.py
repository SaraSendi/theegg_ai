import tiempo #El modulo creado es tiempo, que calcula el tiempo que tarda en caer el objeto. Importo este Modulo para usarlo
t_str=input("Cuanto tiempo necesita el objeto para llegar al suelo? ") #El usuario introduce la altura como str
t_float=float(t_str) #como el problema dice que el tiempo sea int, el str lo paso a float y a continuación a entero (para que no de problemas si el usuario lo mete con coma)
t=int(t_float)

print('La altura en la TIERRA es: ' , round(tiempo.tierra(t),1), "m") #Imprimo las alturas en cada planeta, y lo redondeo
print('La altura en la MARTE es: ' , round(tiempo.marte(t),1), "m")
print('La altura en la JUPITER es: ' , round(tiempo.jupiter(t),1), "m")