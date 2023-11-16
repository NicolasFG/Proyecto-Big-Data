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
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd


# Obtener la ruta del directorio actual
ruta_actual = os.getcwd()

df = pd.read_csv( ruta_actual +'/recos_supermercados/supermercados.csv')

# Definir el mínimo de pedidos para considerar a un producto como popular
min_pedidos = 20  # Ajusta este valor según tus necesidades

# Definir el mínimo de interacciones para considerar a un usuario como activo
min_interacciones = 10  # Ajusta este valor según tus necesidades

# Crear un DataFrame con el conteo de interacciones por usuario
interacciones_usuario = df.groupby('CodCliente').size()

# Crear un mapeo de codigosap a descripcion
codigo_a_descripcion = df.set_index('codigosap')['descripcion'].to_dict()

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





# Crear un mapeo de IDs a índices
user_id_map = {user_id: index for index, user_id in enumerate(df_activos_populares['CodCliente'].unique())}
item_id_map = {item_id: index for index, item_id in enumerate(df_activos_populares['codigosap'].unique())}


# Convertir los IDs a índices
df_activos_populares['user_index'] = df_activos_populares['CodCliente'].map(user_id_map)
df_activos_populares['item_index'] = df_activos_populares['codigosap'].map(item_id_map)

# Número total de usuarios e ítems
n_users = len(user_id_map)
n_items = len(item_id_map)

print("Número total de usuarios:", n_users)
print("Número total de ítems:", n_items)

# Crear la matriz de interacciones usuario-item como COO
user_item_matrix = coo_matrix(
    (df_activos_populares['rating_implicito'], 
     (df_activos_populares['user_index'], df_activos_populares['item_index'])),
    shape=(n_users, n_items)
)

user_item_matrix_csr = user_item_matrix.tocsr()

print("Dimensiones de user_item_matrix_csr:", user_item_matrix_csr.shape)


# Entrenar el modelo ALS
model = AlternatingLeastSquares(factors=50, use_gpu=False)  # Ajusta los parámetros según necesites
model.fit(user_item_matrix_csr.T * 40)  # Multiplicar por un factor de confianza si es necesario


# Guardar el modelo entrenado
filename = 'als_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

# Cargar el modelo entrenado
with open('als_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)




#User_id
#user_id = 647917

# Seleccionar aleatoriamente algunos user_id de df_activos_populares
user_ids_prueba = df_activos_populares['CodCliente'].sample(n=5).unique()

for user_id in user_ids_prueba:
    if user_id in user_id_map:
        user_index = user_id_map[user_id]
        if user_index < user_item_matrix_csr.shape[0]:
            recommended_indices = model.recommend(user_index, user_item_matrix_csr[user_index], N=10)
            indices_recomendados, _ = recommended_indices
            # Convertir índices de recomendaciones a IDs de productos
            top_recommendations = []
        
            
            # Iterar sobre los índices de recomendaciones
            for idx in indices_recomendados:
                if idx >= n_items:
                    print(f"Índice {idx} está fuera de rango para los ítems.")
                    continue
                
                if idx in item_id_map:
                    product_id = item_id_map[idx]
                    top_recommendations.append(product_id)

                    # Obtener la descripción del producto
                    descripcion_producto = codigo_a_descripcion.get(product_id, "Descripción no disponible")
                    print(f"Producto Recomendado: ID {product_id}, Descripción: {descripcion_producto}")
                else:
                    print(f"Índice {idx} no encontrado en item_id_map.")
        else:
            print(f"user_index {user_index} está fuera del rango de la matriz para user_id {user_id}.")
    else:
        print(f"El user_id {user_id} no se encuentra en el conjunto de datos.")




