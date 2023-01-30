import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

########## PREPARAR LOS DATOS ##########
#Importamos los datos especificando la ruta, estos datos estan como CSV
path="C:/Users/saras/Desktop/The egg/AlgoritmosML1/Tarea144_RegresionPolinimial" #Fijar la ruta de ubicación del archivo
archivo="mojarra.csv" #Nombre del archivo
path_input=path+"/"+archivo
dataset = pd.read_csv(path_input, sep=";", decimal=",")
#print(dataset)
#print()
########## DIVISION DE LOS DATOS##############
#Defino los datos correspondientes a las etiquetas
x = dataset["age"].to_numpy().reshape(-1,1)
y = dataset["length"].to_numpy().reshape(-1,1)
#Separar los datos en entrenamiento y testeo (80-20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
#Visualizacion de los datos
plt.plot(x, y, 'b.')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

################### MODELO DE REGRESION LINEAL
def linalRegresion(x_train, y_train, x_test, y_test):
    
    lr = LinearRegression() #Cargamos el modelo de regresión lienal
    model = lr.fit(x_train, y_train) #Emppleando los datos de entrenamiento ajustamos la recta, entrenamos el modelo
    #model.coef_, model.intercept_ # Obtenemos el coeficiente de regresión y el punto de corte con el eje y
    y_predicted = lr.predict(x_train) #Tras ajustar la recta se predicen los datos datos de entrenamiento
    y_test_predicted = lr.predict(x_test) #Finalmente se predice la y de testeo

    #Graficar los resultados; Graficamos mediante puntos azules los datos de entrenamiento, en magenta los de test y junto a ello en rosa se representa la linea de regresión
    plt.scatter(x_train, y_train) #los datos de entrenamiento
    plt.scatter(x_test, y_test, color="m") #datos test reales
    plt.plot(x_test, y_test_predicted, color="r") #datos predichos
    plt.xlabel("edad")
    plt.ylabel("longitud")
    plt.show()
    #Calculo los coeficientes a y b de la recta para exportarlos
    a=model.coef_
    b=model.intercept_
    return(y_test_predicted, y_predicted, a , b, model)

def polinomial (orden,x_train, y_train, x_test, y_test):
    #Tras el algoritmo de regresion lineal se plantea una regresion polinomial, y se compara el error de los dos
    #la división de datos en train y test es la misma a la empleada en el modelo de regresión lienal (por ello no se repite), se realiza fuera de las funciones
    #MODELO DE REGRESION POLINOMIAL
    poly = PolynomialFeatures (degree=orden,include_bias=True) #defino el grado del polinomio que quiero emplear, el grado lo meto bajo la variable "ORDEN"
    #trasnformamos las caracteristicas para un polinomio 
    lr = LinearRegression()
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly= poly.fit_transform(x_test)
    model_poly = lr.fit(x_train_poly, y_train) #Entreno el modelo empleando los daots transformados de x y el y
    y_train_poly_predicted = model_poly.predict (x_train_poly)
    y_test_poly_predicted = model_poly.predict(x_test_poly) 
    #Coeficientes del polinomio:
    a=model_poly.coef_
    b=model_poly.intercept_
    #Visualizo el resultado, para visualizar se deben hacer algunas operaciones previas. No es directo como en el caso de la RL
    #Genero puntos para poder generar la linea de segundo grado, del 1 al 6 con 50 puntos
    puntos = np.linspace(1, 6, 50).reshape (-1, 1)
    puntos_poly = poly.fit_transform (puntos)
    y_puntos_poly = model_poly.predict (puntos_poly)

    plt.scatter(x_train, y_train, color = "b") #Puntos empleados para train
    plt.scatter (x_test, y_test, color = "m") #Puntos empleados para test
    plt.plot (puntos, y_puntos_poly, "r") #Puntos generados siguiendo la recta de ajuste
    plt.xlabel("edad")
    plt.ylabel("longitud")
    plt.show()
    return (x_train_poly,x_test_poly, y_train_poly_predicted, y_test_poly_predicted, a,b,model_poly)

#FUNCION PARA CALCULAR LOS ERRORES, EMPLEAREMOS LA MISMA FUNCION EN TODOS LOS CASOS
def error(y_test, y_test_predicted, y_predicted, x_train, x_test, y_train, Model):

    #Comprobar errores, loc calculo e imprimo en pantalla. A continuacion, estos errores los exportaré a csv todos los errores de los modelos (Fuera de la funcion para hacerlo con todos)
    #El error R2 o mse se calcula según la diferencia que se se da entre el y predicho y el y_test, ya que este último es el resultado que deberia haber dado. 
    # print R2 error
    print('ERRORES GENERADOS:')
    print("Comprobación del error generado: ")
    print()
    r2 = metrics.r2_score(y_test, y_test_predicted)
    print("R cuadrado (R2): ", r2)
    #print MSE (Error cuadratico medio)
    mse = metrics.mean_squared_error(y_test, y_test_predicted)
    print("Error cuadratico medio (MSE): ", mse)
    rmse=np.sqrt(mse)
    print("Cuadrado del error cuadratico medio (rMSE): ", rmse)
    mae=metrics.mean_absolute_error(y_test,y_test_predicted)
    print("Error medio absoluto (MAE): ", mae)

    n = dataset["age"].size
    k = 1 #en este caso solo es una variable
    ad_r2 = (1 - (((n-1)/(n-k-1)) * (1 - r2)))
    print("Error R cuadrado ajustado: ", ad_r2)
    #las precisiones de los modelos
    print('Precisión sobre el set de entrenamiento:')
    train_set_acc=Model.score(x_train, y_train)
    print(train_set_acc)
    print('Precisión sobre el set de Test:')
    test_set_acc=Model.score(x_test, y_test)
    print(test_set_acc)
    
    return(r2,mse,rmse, mae, ad_r2, train_set_acc,test_set_acc)
    

#Llamadas a las funciones, en este caso queremos un modelo de regresion lineal, y polinomicos de los oredenes: 2,5 y 25


[y_test_predicted_Lr, y_predicted_Lr, a_lr, b_lr, lrModel]=linalRegresion(x_train, y_train, x_test, y_test)
[r2_lr,mse_lr,rmse_lr, mae_lr, ad_r2_Lr, train_set_acc_lr,test_set_acc_lr]=error(y_test, y_test_predicted_Lr, y_predicted_Lr, x_train, x_test, y_train, lrModel)
errorLr=[train_set_acc_lr,test_set_acc_lr,r2_lr,mse_lr,rmse_lr, mae_lr, ad_r2_Lr,a_lr,b_lr]

#POLINOMIAL
otr=input('¿Deseas probar una regresion polinomica?(Y/N)') #Pregunto al usuario si desea realizar el ajuste mediante este modelo
otr=otr.upper()
erroresNombre=["Precision sobre set de Train", "Precision sobre set de Test", "R cuadrado", "Error cuadratico medio (MSE)",'Raiz del error cuadratico medio (RMSE)', 'Error medio absoluto (MAE)','R cuadrado ajustado ', "Valores de coeficientes a","Valor de interseccion b"]
errorTotal=[erroresNombre,errorLr] #EN esta lista acoplare todos los elementes de errores, para a continuacion exportarlo como csv. 
ordenes=[] #Almaceno los ordenes que quiero emplear en esta lista para exportar a conticuacion los datos a CSV
while otr=='Y': #Mientras la respuesta sea afirmativa hago este ajuste, pido al usuario que orden desea probar y cuando termina le pregunto si quiere probar otro orden
    orden=input('¿Que orden deseas que tenga el polinomio de ajuste?') 
    orden=int(orden)
    print('Errores al emplear un polinomio de orden', orden,':')
    [x_train_poly, x_test_poly, y_train_poly_predicted, y_test_poly_predicted, a_p,b_p,model_poly]=polinomial(orden,x_train, y_train, x_test, y_test)
    [r2_p,mse_p,rmse_p, mae_p,ad_r2_p,train_set_acc_p,test_set_acc_p]=error(y_test, y_test_poly_predicted  ,y_train_poly_predicted, x_train_poly, x_test_poly, y_train, model_poly)
    errorPoly_Simple=[train_set_acc_p,test_set_acc_p,r2_p,mse_p,rmse_p, mae_p, ad_r2_p, a_p,b_p]
    errorTotal.append(errorPoly_Simple)
    ordenes.append(orden)
    otr=input('¿Deseas probar una regresion polinomica?(Y/N)')

#EXPORTO LOS DIFERENTES ERRORES A CSV PARA PODER COMPARARLOS

Result_pandas=pd.DataFrame(errorTotal).T

columnas=['Parámetro','Regresion Lineal']
for i in ordenes:
    i=int(i)
    istr=str(i)
    
    #strOrden=str('Polinomio Orden'+ i)
    columnas.append('Polinomio Orden'+ istr) #pongo el nombre de las columnas
Result_pandas.columns=columnas

nombre_resultados="ResultadosEvaluacionModeloPP.csv"
Result_pandas.to_csv(path+"/"+nombre_resultados, sep=";", index=False, decimal=",")

