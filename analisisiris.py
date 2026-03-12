import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Cargar dataset
iris = load_iris()

df_iris = pd.DataFrame(iris.data, columns = iris.feature_names)
df_iris['especies'] = iris.target
df_iris['nombre_especie'] = df_iris['especies'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"Numero de instancias: {iris.data.shape[0]}")
print(f"Numero de atributos: {iris.data.shape[1]}")
print(f"Clases existentes: {iris.target_names}")
print("\nTipos de atributos (pandas info): ")
df_iris.info()

# Estilo visual
sns.set_style(style = "whitegrid")

sns.pairplot(df_iris.drop('especies', axis = 1), hue = 'nombre_especie', palette = 'bright')
plt.suptitle("Matriz de Dispersion - Iris Dataset", y = 1.02)
plt.show()

# Clasificacion Supervisada
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state = 42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

exactitud = accuracy_score(y_test, y_pred)
print(f"Exactitud del modelo: {exactitud * 100:.2f}%\n")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = iris.target_names)
disp.plot(cmap = plt.cm.Blues)
plt.title("Matriz de Confusion - Decision Tree Classifier")
plt.show()

#Ejemplos
# 3 ejemplos nuevos: [sepal_length, sepal_width, petal_length, petal_width]
# Ejemplo 1: Pétalos pequeños (Típico de Setosa)
# Ejemplo 2: Medidas intermedias (Típico de Versicolor)
# Ejemplo 3: Pétalos muy grandes (Típico de Virginica)
nuevos_datos = [
    [5.0, 3.4, 1.5, 0.2], 
    [6.2, 2.9, 4.3, 1.3], 
    [7.1, 3.0, 5.9, 2.1]  
]

predicciones = clf.predict(nuevos_datos)

for i, pred in enumerate(predicciones):
    print(f"Flor nueva {i+1} clasificada como: {iris.target_names[pred]}")