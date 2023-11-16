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
min_pedidos = 20  

# Definir el mínimo de interacciones para considerar a un usuario como activo
min_interacciones = 10 

# Crear un DataFrame con el conteo de interacciones por usuario
interacciones_usuario = df.groupby('CodCliente').size()

# Crear un mapeo de CodCliente a NombreCliente
cliente_a_nombre = df.set_index('CodCliente')['NombreCliente'].drop_duplicates().to_dict()


# Eliminar columnas que no necesitas
df = df.drop(['Tienda', 'UnidadMedida','NombreCliente', 'Categoria','PorcDescuento','fecha','descripcion','PesoNeto','PrecioUnitario','ImpDescuento', 'ImporteLineaBs', 'CostoUnitario'], axis=1)
# Filtrar y excluir usuarios con CodCliente igual a 0
df = df[df['CodCliente'] != 0]

df['rating_implicito'] = df.groupby(['CodCliente', 'codigosap','Subcategoria'])['Cantidad'].transform('sum')

# Filtrar para mantener solo a los usuarios activos
usuarios_activos = interacciones_usuario[interacciones_usuario >= min_interacciones].index

df_activos = df[df['CodCliente'].isin(usuarios_activos)]


# Crear un DataFrame con el conteo de pedidos por producto
pedidos_producto = df_activos.groupby('codigosap').size()

# Filtrar para mantener solo los productos populares
productos_populares = pedidos_producto[pedidos_producto >= min_pedidos].index
df_activos_populares = df_activos[df_activos['codigosap'].isin(productos_populares)]


# Crear un mapeo de codigosap a descripcion
codigo_a_descripcion = df_activos_populares.set_index('codigosap')['Subcategoria'].to_dict()


# Crear un mapeo de IDs a índices
user_id_map = {user_id: index for index, user_id in enumerate(df_activos_populares['CodCliente'].unique())}
item_id_map = {item_id: index for index, item_id in enumerate(df_activos_populares['codigosap'].unique())}


# Convertir los IDs a índices
df_activos_populares['user_index'] = df_activos_populares['CodCliente'].map(user_id_map)
df_activos_populares['item_index'] = df_activos_populares['codigosap'].map(item_id_map)

print("DATAFRAME: ")
print(df_activos_populares)


# Número total de usuarios e ítems
n_users = len(user_id_map)
n_items = len(item_id_map)

#print("Imprimiendo los users_id")
#for user_id in user_id_map:
#    print(user_id)

#print("Imprimiendo los items_id")
#for item_id in item_id_map:
#    print(item_id)

#print("Número total de usuarios:", n_users)
#print("Número total de ítems:", n_items)



# Crear la matriz de interacciones usuario-item como COO
user_item_matrix = coo_matrix(
    (df_activos_populares['rating_implicito'], 
     (df_activos_populares['user_index'], df_activos_populares['item_index'])),
    shape=(n_users, n_items)
)

print(user_item_matrix)

user_item_matrix_csr = user_item_matrix.tocsr()
"""
print("Dimensiones de user_item_matrix_csr:", user_item_matrix_csr.shape)


# Entrenar el modelo ALS
model = AlternatingLeastSquares(factors=50, use_gpu=False)  # Ajusta los parámetros según necesites
model.fit(user_item_matrix_csr.T * 40)  # Multiplicar por un factor de confianza si es necesario


# Guardar el modelo entrenado
filename = 'als_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

"""

# Cargar el modelo entrenado
with open('als_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)




# Seleccionar aleatoriamente algunos user_id de df_activos_populares
user_ids_prueba = df_activos_populares['CodCliente'].sample(n=5).unique()
#print("Imprimiendo IDs de prueba: ")
#print(user_ids_prueba)
#print(user_item_matrix_csr)

# [4877886 4312435 2610189 9903713 6926311]

# Crear un mapeo inverso de item_id_map
item_index_to_codigosap = {index: codigosap for codigosap, index in item_id_map.items()}

# Asumiendo que tienes un user_id específico para probar
user_id_prueba = 2459801  # Reemplaza con un user_id real de tu conjunto de datos

# Desempaquetar los arrays de índices recomendados y sus puntuaciones


# Verificar si el user_id está en el user_id_map
if user_id_prueba in user_id_map:
    user_index = user_id_map[user_id_prueba]
    # Hacer recomendaciones para este usuario
    recommended = loaded_model.recommend(user_index, user_item_matrix_csr[user_index], N=5)

    indices_recomendados, puntuaciones = recommended
    print(f"Recomendaciones para el usuario {user_id_prueba} ({cliente_a_nombre.get(user_id_prueba, 'Nombre no disponible')}):")
    
    # Iterar sobre los índices y puntuaciones
    for item_idx, score in zip(indices_recomendados, puntuaciones):
        # Comprobar si el índice del ítem está en item_index_to_codigosap
        if item_idx in item_index_to_codigosap:
            # Encontrar el codigosap correspondiente
            codigosap = item_index_to_codigosap[item_idx]
            # Obtener la descripción del producto
            descripcion = codigo_a_descripcion.get(codigosap, "Descripción no disponible")
            print(f"Producto Recomendado: ID {codigosap}, Score: {score}, Descripción: {descripcion}")
        else:
            print(f"Índice de ítem {item_idx} no encontrado en item_index_to_codigosap.")
else:
    print(f"user_id {user_id_prueba} no encontrado en user_id_map.")


"""
# Seleccionar una muestra de tu DataFrame
sample = df_activos_populares.sample(5)

# Iterar sobre la muestra y verificar en la matriz CSR
for index, row in sample.iterrows():
    user_idx = user_id_map[row['CodCliente']]
    item_idx = item_id_map[row['codigosap']]
    rating = row['rating_implicito']
    
    # Comprobar si el valor en la matriz CSR coincide
    if user_item_matrix_csr[user_idx, item_idx] != rating:
        print(f"Discrepancia encontrada en usuario {row['CodCliente']} e ítem {row['codigosap']}")

# Elegir un user_id para explorar
user_id_ejemplo = df_activos_populares['CodCliente'].sample(1).iloc[0]
user_index = user_id_map[user_id_ejemplo]

# Obtener todas las interacciones para ese usuario
interacciones_usuario = user_item_matrix_csr[user_index]

# Explorar las interacciones
print(f"Interacciones para el usuario {user_id_ejemplo} (índice {user_index}):")
print(interacciones_usuario)
"""


"""
for user_id in user_ids_prueba:
    if user_id in user_id_map:
        user_index = user_id_map[user_id]
        print(user_index)
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

"""