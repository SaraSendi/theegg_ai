# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split


#Lectura de datos
df = pd.read_csv("C:/Users/saras/Documents/Codigos_cursos/GIT/theegg_ai/tarea_146/ejemplo_dataset.csv")
df.head()

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(df.X1, df.X2, c=df.y)
ax.set_title("Datos ESL.mixture")
plt.show()

X = df[["X1","X2"]]
Y = df["y"]


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=13)

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(x_train, y_train)

y_pred = logistic_regression_model.predict(x_test)
print(y_pred)