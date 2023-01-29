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
def linalRegresion(x_train, y_train, x_test):
    lr = LinearRegression() #Cargamos el modelo de regresión lienal
    lrModel=lr.fit(x_train, y_train) #Emppleando los datos de entrenamiento ajustamos la recta

    y_predicted = lr.predict(x_test) #tras ajustar la linea de regresion empleando el X de test se predice el valor de y 
    #Graficar los resultados; Graficamos mediante puntos los datos de entrenamiento y dibujamos junto a los datos la recta de regresion generada (En rojo)
    plt.scatter(x_train, y_train) 
    plt.plot(x_test, y_predicted, "r")
    plt.show()
    a=lrModel.coef_
    b=lrModel.intercept_
    return(y_predicted, a , b, lrModel)

def polinomial (orden,x_train, y_train, x_test):
    #Tras el algoritmo de regresion lineal se plantea una regresion polinomial, y se compara el error de los dos
    #la división de datos en train y test es la misma a la empleada en el modelo de regresión lienal (por ello no se repite), se realiza fuera de las funciones
    #MODELO DE REGRESION POLINOMIAL
    poly = PolynomialFeatures (degree=orden,include_bias=True) #defino el grado del polinomio que quiero emplear, el grado lo meto bajo la variable "ORDEN"
    #trasnformamos las caracteristicas para un polinomio 
    lr = LinearRegression()
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly= poly.fit_transform(x_test)
    model_poly = lr.fit(x_train_poly, y_train) #Entreno el modelo empleando los daots transformados de x y el y
    y_poly_predicted = model_poly.predict (x_test_poly)
    #Coeficientes del polinomio:
    a=model_poly.coef_
    b=model_poly.intercept_
    #Visualizo el resultado 
    plt.scatter(x_train, y_train) 
    plt.plot(x_test, y_poly_predicted, "r")
    plt.show()
    return (x_train_poly,x_test_poly, y_poly_predicted, a,b,model_poly)

#FUNCION PARA CALCULAR LOS ERRORES, EMPLEAREMOS LA MISMA FUNCION EN TODOS LOS CASOS
def error(y_test,y_predicted, x_train, x_test, y_train, Model):

    #Comprobar errores, loc calculo e imprimo en pantalla. A continuacion, estos errores los exportaré a csv todos los errores de los modelos (Fuera de la funcion para hacerlo con todos)
    #El error R2 o mse se calcula según la diferencia que se se da entre el y predicho y el y_test, ya que este último es el resultado que deberia haber dado. 
    # print R2 error
    print('ERRORES GENERADOS:')
    print("Comprobación del error generado: ")
    print()
    r2 = metrics.r2_score(y_test, y_predicted)
    print("R cuadrado (R2): ", r2)
    #print MSE (Error cuadratico medio)
    mse = metrics.mean_squared_log_error(y_test, y_predicted)
    print("Error cuadratico medio (MSE): ", mse)
    rmse=np.sqrt(mse)
    print("Error cuadratico medio (MSE): ", rmse)
    mae=metrics.mean_absolute_error(y_test,y_predicted)
    print("Error medio absoluto (MAE): ", mae)

    #las precisiones de los modelos
    print('Precisión sobre el set de entrenamiento:')
    train_set_acc=Model.score(x_train, y_train)
    print(train_set_acc)
    print('Precisión sobre el set de Test:')
    test_set_acc=Model.score(x_test, y_test)
    print(test_set_acc)
    return(r2,mse,rmse, mae, train_set_acc,test_set_acc)
#Llamadas a las funciones, en este caso queremos un modelo de regresion lineal, y polinomicos de los oredenes: 2,5 y 25

[y_predicted_Lr, a_lr, b_lr, lrModel]=linalRegresion(x_train, y_train, x_test)
[r2_lr,mse_lr,rmse_lr, mae_lr,train_set_acc_lr,test_set_acc_lr]=error(y_test,y_predicted_Lr, x_train, x_test, y_train, lrModel)
errorLr=[train_set_acc_lr,test_set_acc_lr,r2_lr,mse_lr,rmse_lr, mae_lr,a_lr,b_lr]

[x_train_poly2, x_test_poly2, y_poly2_predicted, a_p2,b_p2,model_poly2]=polinomial(2,x_train, y_train, x_test)
[r2_p2,mse_p2,rmse_p2, mae_p2,train_set_acc_p2,test_set_acc_p2]=error(y_test,y_poly2_predicted, x_train_poly2, x_test_poly2, y_train, model_poly2)
errorPoly2=[train_set_acc_p2,test_set_acc_p2,r2_p2,mse_p2,rmse_p2, mae_p2,a_p2,b_p2]

[x_train_poly5, x_test_poly5, y_poly5_predicted, a_p5,b_p5,model_poly5]=polinomial(5,x_train, y_train, x_test)
[r2_p5,mse_p5,rmse_p5, mae_p5,train_set_acc_p5,test_set_acc_p5]=error(y_test,y_poly5_predicted, x_train_poly5, x_test_poly5, y_train, model_poly5)
errorPoly5=[train_set_acc_p5,test_set_acc_p5,r2_p5,mse_p5,rmse_p5, mae_p5,a_p5,b_p5]

[x_train_poly25, x_test_poly25, y_poly25_predicted, a_p25,b_p25,model_poly25]=polinomial(25,x_train, y_train, x_test)
[r2_p25,mse_p25,rmse_p25, mae_p25,train_set_acc_p25,test_set_acc_p25]=error(y_test,y_poly25_predicted, x_train_poly25, x_test_poly25, y_train, model_poly25)
errorPoly25=[train_set_acc_p25,test_set_acc_p25,r2_p25,mse_p25,rmse_p25, mae_p25,a_p25,b_p25]


#EXPORTO LOS DIFERENTES ERRORES A CSV PARA PODER COMPARARLOS
erroresNombre=["Precision sobre set de Train", "Precision sobre set de Test", "R cuadrado", "Error cuadratico medio (MSE)",'Raiz del error cuadratico medio (RMSE)', 'Error medio absoluto (MAE)', "Valores de coeficientes a","Valor de interseccion b"]

ResultError=[erroresNombre, errorLr, errorPoly2, errorPoly5, errorPoly25]
Result_pandas=pd.DataFrame(ResultError).T
Result_pandas.columns=['Parámetro','Regresion Lineal','Polinomio Orden 2', 'Polinomio Orden 5','Polinomio orden 25'] #pongo el nombre de las columnas
nombre_resultados="ResultadosEvaluacionModeloLr.csv"
Result_pandas.to_csv(path+"/"+nombre_resultados, sep=";", index=False, decimal=",")

