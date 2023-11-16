from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd
import os
import matplotlib.pyplot as plt
from implicit.als import AlternatingLeastSquares
import pickle



# Obtener la ruta del directorio actual
ruta_actual = os.getcwd()

df = pd.read_csv( ruta_actual +'/recos_supermercados/supermercados.csv')

# Definir el mínimo de pedidos para considerar a un producto como popular
min_pedidos = 20  # Ajusta este valor según tus necesidades

# Definir el mínimo de interacciones para considerar a un usuario como activo
min_interacciones = 10  # Ajusta este valor según tus necesidades

# Crear un DataFrame con el conteo de interacciones por usuario
interacciones_usuario = df.groupby('CodCliente').size()



# Eliminar columnas que no necesitas
df = df.drop(['Tienda', 'UnidadMedida','NombreCliente', 'Categoria','PorcDescuento', 'Subcategoria','fecha','descripcion','PesoNeto','PrecioUnitario','ImpDescuento', 'ImporteLineaBs', 'CostoUnitario'], axis=1)

df['rating_implicito'] = df.groupby(['CodCliente', 'codigosap'])['Cantidad'].transform('sum')

# Filtrar para mantener solo a los usuarios activos
usuarios_activos = interacciones_usuario[interacciones_usuario >= min_interacciones].index

df_activos = df[df['CodCliente'].isin(usuarios_activos)]


# Crear un DataFrame con el conteo de pedidos por producto
pedidos_producto = df_activos.groupby('codigosap').size()

# Filtrar para mantener solo los productos populares
productos_populares = pedidos_producto[pedidos_producto >= min_pedidos].index
df_activos_populares = df_activos[df_activos['codigosap'].isin(productos_populares)]


# Tienda,CodCliente,NombreCliente,Categoria,Subcategoria,codigosap,descripcion,fecha,Cantidad,UnidadMedida,PesoNeto,PrecioUnitario,PorcDescuento,ImporteLinea,ImpDescuento,ImporteLineaBs,CostoUnitario
# Supongamos que df es tu DataFrame con las columnas ['CodCliente', 'codigosap', 'rating']
reader = Reader(rating_scale=(1, df_activos_populares['rating_implicito'].max()))  # Ajusta la escala según tus datos
data = Dataset.load_from_df(df_activos_populares[['CodCliente', 'codigosap', 'rating_implicito']], reader)


# Dividir los datos en conjunto de entrenamiento y de prueba
trainset, testset = train_test_split(data, test_size=0.25)



# Usar KNNWithMeans para filtrado colaborativo basado en usuarios
algo = KNNWithMeans(sim_options={'name': 'cosine', 'user_based': True})
algo.fit(trainset)

# Guardar el modelo entrenado
filename = 'modelo_recomendacion.pkl'
with open(filename, 'wb') as file:
    pickle.dump(algo, file)

# Cargar el modelo entrenado
with open('modelo_recomendacion.pkl', 'rb') as file:
    modelo_cargado = pickle.load(file)

# Realizar predicciones y evaluar el modelo
predictions = modelo_cargado.test(testset)
accuracy.rmse(predictions)


#User_id
user_id = 647917


def get_unseen_items(df, user_id):
    # Obtiene todos los ítems únicos en el conjunto de datos
    unique_items = set(df['codigosap'].unique())
    
    # Obtiene los ítems con los que el usuario ya ha interactuado
    user_items = set(df[df['CodCliente'] == user_id]['codigosap'].unique())

    # Filtra los ítems que no han sido interactuados por el usuario
    unseen_items = list(unique_items - user_items)

    return unseen_items

unseen_items = get_unseen_items(df_activos_populares, user_id)  # Reemplaza 'user_id' con el ID del usuario real


def predict_ratings(user_id, unseen_items, algorithm):
    predictions = []
    for item_id in unseen_items:
        # Predecir la preferencia (o valoración) para cada ítem no visto
        predictions.append(algorithm.predict(user_id, item_id))

    return predictions

user_predictions = predict_ratings(user_id, unseen_items, modelo_cargado)  # 'algo' es tu modelo entrenado


def get_top_n_recommendations(predictions, n=10):
    # Ordenar las predicciones por valoración estimada
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Devolver los 'n' ítems con mayor valoración estimada
    top_n_items = [pred.iid for pred in predictions[:n]]
    return top_n_items

top_recommendations = get_top_n_recommendations(user_predictions, n=10)

