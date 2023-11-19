import pandas as pd
import os
import matplotlib.pyplot as plt
from implicit.als import AlternatingLeastSquares
import pickle
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pdh
import pandas as pd

import pandas as pd
import os

import pandas as pd
import os

def preprocesamiento():
    # Obtener la ruta del directorio actual y cargar el DataFrame
    ruta_actual = os.getcwd()
    df = pd.read_csv(ruta_actual + '/recos_supermercados/cleaned_data.csv')

    # Calcular el promedio de interacciones por cliente
    interacciones_por_cliente = df.groupby('CodCliente').size()
    promedio_interacciones = interacciones_por_cliente.mean()

    # Filtrar para mantener solo los clientes activos
    min_interacciones = promedio_interacciones
    clientes_activos = interacciones_por_cliente[interacciones_por_cliente >= min_interacciones].index
    df_clientes_activos = df[df['CodCliente'].isin(clientes_activos)]

    # Crear un DataFrame con el conteo de pedidos por producto para clientes activos
    pedidos_producto = df_clientes_activos.groupby('codigosap').size()

    # Calcular el promedio de pedidos por producto
    promedio_pedidos = pedidos_producto.mean()

    # Filtrar para mantener solo los productos populares
    min_pedidos = promedio_pedidos
    productos_populares = pedidos_producto[pedidos_producto >= min_pedidos].index
    df_clientes_activos_y_productos_populares = df_clientes_activos

    # Calcular la preferencia de subcategoría
    conteo_subcategoria = df_clientes_activos_y_productos_populares.groupby(['CodCliente', 'Subcategoria']).size().reset_index(name='Conteo')
    conteo_maximo = conteo_subcategoria.groupby('CodCliente')['Conteo'].transform('max')
    conteo_subcategoria['PreferenciaSubcategoria'] = conteo_subcategoria['Conteo'] / conteo_maximo

    # Unir con el DataFrame principal
    df_clientes_activos_y_productos_populares = pd.merge(df_clientes_activos_y_productos_populares, conteo_subcategoria[['CodCliente', 'Subcategoria', 'PreferenciaSubcategoria']], on=['CodCliente', 'Subcategoria'], how='left')

    # Asumiendo que tienes una columna 'PrecioUnitario' en tu DataFrame
    # Normaliza 'PrecioUnitario' (puedes ajustar este paso según tus datos)
    df_clientes_activos_y_productos_populares['PrecioUnitarioNormalized'] = (df_clientes_activos_y_productos_populares['PrecioUnitario'] - df_clientes_activos_y_productos_populares['PrecioUnitario'].min()) / (df_clientes_activos_y_productos_populares['PrecioUnitario'].max() - df_clientes_activos_y_productos_populares['PrecioUnitario'].min())

    # Calcular un rating ponderado
    df_clientes_activos_y_productos_populares['rating_implicito'] = df_clientes_activos_y_productos_populares['Cantidad'] * df_clientes_activos_y_productos_populares['PrecioUnitarioNormalized'] * df_clientes_activos_y_productos_populares['PreferenciaSubcategoria']

    # Crear un mapeo de CodCliente a NombreCliente
    cliente_a_nombre = df_clientes_activos_y_productos_populares.set_index('CodCliente')['NombreCliente'].drop_duplicates().to_dict()

    # Crear un mapeo de codigosap a descripcion
    codigo_a_descripcion = df_clientes_activos_y_productos_populares.set_index('codigosap')['Concatenated'].to_dict()

    # Crear mapeos de IDs a índices
    user_id_map = {user_id: index for index, user_id in enumerate(df_clientes_activos_y_productos_populares['CodCliente'].unique())}
    item_id_map = {item_id: index for index, item_id in enumerate(df_clientes_activos_y_productos_populares['codigosap'].unique())}

    print("Pasando a la verificación de los mapeos")
    # Verificar la longitud de los mapeos
    if len(user_id_map) != df_clientes_activos_y_productos_populares['CodCliente'].nunique():
        print("Hay una discrepancia en los mapeos de user_id_map.")
    if len(item_id_map) != df_clientes_activos_y_productos_populares['codigosap'].nunique():
        print("Hay una discrepancia en los mapeos de item_id_map.")

    # Verificar la unicidad de los índices en los mapeos
    if len(set(user_id_map.values())) != len(user_id_map.values()):
        print("Hay índices duplicados en user_id_map.")
    if len(set(item_id_map.values())) != len(item_id_map.values()):
        print("Hay índices duplicados en item_id_map.")

    # Crear un mapeo de item a codigo sap
    item_index_to_codigosap = {index: codigosap for codigosap, index in item_id_map.items()}

    # Dividir en conjuntos de entrenamiento y prueba - Esto depende de cómo planees hacerlo
    #df_train, df_test = train_test_split(df_clientes_activos_y_productos_populares, test_size=0.2, random_state=42)

    return df_clientes_activos_y_productos_populares, user_id_map, item_id_map, cliente_a_nombre, item_index_to_codigosap, codigo_a_descripcion


 

def entrenamiento(df_clientes_activos_y_productos_populares, user_id_map, item_id_map):
    # Número total de usuarios e ítems

    # Convertir los IDs a índices
    df_clientes_activos_y_productos_populares['user_index'] = df_clientes_activos_y_productos_populares['CodCliente'].map(user_id_map)
    df_clientes_activos_y_productos_populares['item_index'] = df_clientes_activos_y_productos_populares['codigosap'].map(item_id_map)

    n_users = len(user_id_map)
    n_items = len(item_id_map)

    # Crear la matriz de interacciones usuario-item como COO
    user_item_matrix = coo_matrix(
        (df_clientes_activos_y_productos_populares['rating_implicito'], 
        (df_clientes_activos_y_productos_populares['user_index'], df_clientes_activos_y_productos_populares['item_index'])),
        shape=(n_users, n_items)
    )

    # Convertir a CSR para el entrenamiento
    user_item_matrix_csr = user_item_matrix.tocsr()

    # Entrenar el modelo ALS
    model = AlternatingLeastSquares(factors=100, use_gpu=False)
    model.fit(user_item_matrix_csr * 70)  # Multiplicar por un factor de confianza si es necesario

    # Guardar el modelo entrenado
    with open('ultimos_resultados/als_model_mejorado.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Guardar user_id_map y item_id_map
    with open('ultimos_resultados/user_id_map.pkl', 'wb') as file:
        pickle.dump(user_id_map, file)
    with open('ultimos_resultados/item_id_map.pkl', 'wb') as file:
        pickle.dump(item_id_map, file)

    # Guardar cliente_a_nombre
    with open('ultimos_resultados/cliente_a_nombre.pkl', 'wb') as file:
        pickle.dump(cliente_a_nombre, file)

    # Guardar item_index_to_codigosap
    with open('ultimos_resultados/item_index_to_codigosap.pkl', 'wb') as file:
        pickle.dump(item_index_to_codigosap, file)

    # Guardar codigo_a_descripcion
    with open('ultimos_resultados/codigo_a_descripcion.pkl', 'wb') as file:
        pickle.dump(codigo_a_descripcion, file)


def validation(df):
    user_id_prueba_list = df['CodCliente'].drop_duplicates().head(2)

    user_id_1 = 647917
    #user_id_2 = 3321832

    # Filtrar el DataFrame para obtener solo los registros con los user_id especificados
    filtered_df = df.loc[df['CodCliente'].isin([user_id_1])]

    # Agrupar por características del producto y calcular el rating promedio
    grouped = filtered_df.groupby(['CodCliente', 'NombreCliente', 'Subcategoria', 'codigosap'])['rating_implicito'].mean().reset_index()

    # Ordenar por rating promedio y seleccionar los 10 mejores
    top_rated = grouped.nlargest(10, 'rating_implicito')


    #top_rated = filtered_df.nlargest(10, 'rating_implicito')


    # Seleccionar solamente las columnas especificadas
    columns_to_select = ['CodCliente', 'NombreCliente','Subcategoria','codigosap','rating_implicito']
    top_rated_selected_columns = top_rated[columns_to_select]

    print(top_rated_selected_columns)


def testing(df, model_file, user_id_map_file, item_id_map_file, cliente_a_nombre_file, item_index_to_codigosap_file, codigo_a_descripcion_file):

    # Cargar el modelo entrenado
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)

    # Cargar los mapeos
    with open(user_id_map_file, 'rb') as file:
        user_id_map = pickle.load(file)
    with open(item_id_map_file, 'rb') as file:
        item_id_map = pickle.load(file)
    
    
    with open(cliente_a_nombre_file, 'rb') as file:
        cliente_a_nombre = pickle.load(file)
    with open(item_index_to_codigosap_file, 'rb') as file:
        item_index_to_codigosap = pickle.load(file)
    with open(codigo_a_descripcion_file, 'rb') as file:
        codigo_a_descripcion = pickle.load(file)

    df['user_index'] = df['CodCliente'].map(user_id_map)
    df['item_index'] = df['codigosap'].map(item_id_map)
        
    # Número total de usuarios e ítems
    n_users = len(user_id_map)
    n_items = len(item_id_map)

    user_item_matrix = coo_matrix(
    (df['rating_implicito'], 
        (df['user_index'], df['item_index'])),
        shape=(n_users, n_items)
    )

    user_item_matrix_csr = user_item_matrix.tocsr()

    # Asumiendo que tienes un user_id específico para probar
    user_id_prueba = 3518866

    #Genero una lista de 10 ids de usuarios de prueba.
    user_id_prueba_list = df['CodCliente'].drop_duplicates().sample(2)


    if user_id_prueba in user_id_map:
        user_index = user_id_map[user_id_prueba]
        recommended = loaded_model.recommend(user_index, user_item_matrix_csr[user_index], N=10)

        indices_recomendados, puntuaciones = recommended
        print(f"Recomendaciones para el usuario {user_id_prueba} ({cliente_a_nombre.get(user_id_prueba, 'Nombre no disponible')}):")
        for item_idx, score in zip(indices_recomendados, puntuaciones):
            if item_idx in item_index_to_codigosap:
                codigosap = item_index_to_codigosap[item_idx]
                descripcion = codigo_a_descripcion.get(codigosap, "Descripción no disponible")
                print(f"Producto Recomendado: ID {codigosap}, Score: {score}, Descripción: {descripcion}")
            else:
                print(f"Índice de ítem {item_idx} no encontrado en item_index_to_codigosap.") 
    else:
        print(f"user_id {user_id_prueba} no encontrado en user_id_map.")
        
"""
    for user_id_prueba in user_id_prueba_list:
        if user_id_prueba in user_id_map:
            user_index = user_id_map[user_id_prueba]
            recommended = loaded_model.recommend(user_index, user_item_matrix_csr[user_index], N=5)

            indices_recomendados, puntuaciones = recommended
            print(f"Recomendaciones para el usuario {user_id_prueba} ({cliente_a_nombre.get(user_id_prueba, 'Nombre no disponible')}):")
            
            for item_idx, score in zip(indices_recomendados, puntuaciones):
                if item_idx in item_index_to_codigosap:
                    codigosap = item_index_to_codigosap[item_idx]
                    descripcion = codigo_a_descripcion.get(codigosap, "Descripción no disponible")
                    print(f"Producto Recomendado: ID {codigosap}, Score: {score}, Descripción: {descripcion}")
                else:
                    print(f"Índice de ítem {item_idx} no encontrado en item_index_to_codigosap.")
        else:
            print(f"user_id {user_id_prueba} no encontrado en user_id_map.")
    """

df, user_id_map, item_id_map, cliente_a_nombre, item_index_to_codigosap, codigo_a_descripcion = preprocesamiento()

entrenamiento(df,user_id_map,item_id_map)

ruta_actual = os.getcwd()

modelo = ruta_actual + "/ultimos_resultados/als_model_mejorado.pkl"
cliente_a_nombre = ruta_actual + "/ultimos_resultados/cliente_a_nombre.pkl"
codigo_a_descripcion = ruta_actual + "/ultimos_resultados/codigo_a_descripcion.pkl"
item_id_map = ruta_actual + "/ultimos_resultados/item_id_map.pkl"
item_index_to_codigosap = ruta_actual + "/ultimos_resultados/item_index_to_codigosap.pkl"
user_id_map = ruta_actual + "/ultimos_resultados/user_id_map.pkl"

#validation(df)


testing(df, modelo,user_id_map,item_id_map,cliente_a_nombre,item_index_to_codigosap,codigo_a_descripcion)


#Se toma en cuenta los meses para recomendar un producto
def preprocesamiento_por_meses():
    # Obtener la ruta del directorio actual y cargar el DataFrame
    ruta_actual = os.getcwd()
    df = pd.read_csv(ruta_actual + '/recos_supermercados/supermercados.csv')

    # Filtro los clientes que tienen un codigo de clientes igual a 0, porque no son clientes reales.
    df = df[df['CodCliente'] != 0]

    # Calcular el promedio de interacciones por cliente
    interacciones_por_cliente = df.groupby('CodCliente').size()
    promedio_interacciones = interacciones_por_cliente.mean()

    # Filtrar para mantener solo los clientes activos
    min_interacciones = promedio_interacciones
    clientes_activos = interacciones_por_cliente[interacciones_por_cliente >= min_interacciones].index
    df_clientes_activos = df[df['CodCliente'].isin(clientes_activos)]

    # Crear un DataFrame con el conteo de pedidos por producto para clientes activos
    pedidos_producto = df_clientes_activos.groupby('codigosap').size()

    # Calcular el promedio de pedidos por producto
    promedio_pedidos = pedidos_producto.mean()

    # Filtrar para mantener solo los productos populares
    min_pedidos = promedio_pedidos
    productos_populares = pedidos_producto[pedidos_producto >= min_pedidos].index
    df_clientes_activos_y_productos_populares = df_clientes_activos[df_clientes_activos['codigosap'].isin(productos_populares)]

    # Convertir la columna 'fecha' a datetime y extraer el mes y el año
    df_clientes_activos_y_productos_populares['fecha'] = pd.to_datetime(df_clientes_activos_y_productos_populares['fecha'])
    df_clientes_activos_y_productos_populares['Mes'] = df_clientes_activos_y_productos_populares['fecha'].dt.month
    df_clientes_activos_y_productos_populares['Ano'] = df_clientes_activos_y_productos_populares['fecha'].dt.year

    # Ahora, agrupamos por cliente, código de producto, mes y año para calcular el rating_implicito
    df_clientes_activos_y_productos_populares['rating_implicito'] = df_clientes_activos_y_productos_populares.groupby(['CodCliente', 'codigosap', 'Mes', 'Ano'])['Cantidad'].transform('sum')


    # Crear un mapeo de CodCliente a NombreCliente
    cliente_a_nombre = df_clientes_activos_y_productos_populares.set_index('CodCliente')['NombreCliente'].drop_duplicates().to_dict()
    
    # Crear un mapeo de codigosap a descripcion
    codigo_a_descripcion = df_clientes_activos_y_productos_populares.set_index('codigosap')['Concatenated'].to_dict()

    # Crear mapeos de IDs a índices
    user_id_map = {user_id: index for index, user_id in enumerate(df_clientes_activos_y_productos_populares['CodCliente'].unique())}
    item_id_map = {item_id: index for index, item_id in enumerate(df_clientes_activos_y_productos_populares['codigosap'].unique())}

    # Crear un mapeo de item a codigo sap
    item_index_to_codigosap = {index: codigosap for codigosap, index in item_id_map.items()}

    # Dividir en conjuntos de entrenamiento y prueba
    #df_train, df_test = train_test_split(df_clientes_activos_y_productos_populares, test_size=0.2, random_state=42)

    return df_clientes_activos_y_productos_populares, user_id_map, item_id_map, cliente_a_nombre, item_index_to_codigosap, codigo_a_descripcion



























"""

# Testing 


# Dividir los datos
df_train, df_test = train_test_split(df_clientes_activos_y_productos_populares, test_size=0.2, random_state=42)

# Crear mapeos de IDs a índices para el conjunto de entrenamiento
user_id_map_train = {user_id: index for index, user_id in enumerate(df_train['CodCliente'].unique())}
item_id_map_train = {item_id: index for index, item_id in enumerate(df_train['codigosap'].unique())}

# Convertir IDs a índices para el conjunto de entrenamiento
df_train['user_index'] = df_train['CodCliente'].map(user_id_map_train)
df_train['item_index'] = df_train['codigosap'].map(item_id_map_train)

# Número total de usuarios e ítems
n_users_train = len(user_id_map_train)
n_items_train = len(item_id_map_train)



# Crear la matriz de interacciones usuario-item como COO
user_item_matrix_test = coo_matrix(
    (df_train['rating_implicito'], 
     (df_train['user_index'], df_train['item_index'])),
    shape=(n_users_train, n_items_train)
)


user_item_matrix_train_csr = user_item_matrix_test.tocsr()

model.fit(user_item_matrix_train_csr.T * 40)




predictions = model.predict(user_item_matrix_test)

# Evaluar el rendimiento
precision = accuracy.precision(predictions)
recall = accuracy.recall(predictions)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

"""



"""         
matriz_usuario_producto = df_clientes_activos_y_productos_populares.pivot_table(
    index='CodCliente', 
    columns='codigosap', 
    values='algún_valor_de_interacción', 
    fill_value=0
)
"""






"""

# Convertir la columna 'fecha' a formato de fecha
df['fecha'] = pd.to_datetime(df['fecha'])

# Extraer el año y crear una nueva columna
df['Año'] = df['fecha'].dt.year

# Agrupar por año
interacciones_por_año = df.groupby('Año')

# Calcular el total de interacciones por año
total_interacciones_por_año = interacciones_por_año.size()

# Calcular el número de clientes únicos por año
clientes_unicos_por_año = interacciones_por_año['CodCliente'].nunique()

# Calcular el promedio de interacciones por año
promedio_interacciones_por_año = total_interacciones_por_año / clientes_unicos_por_año

# Imprimir el promedio de interacciones por año
print("Promedio de interacciones por año:")
print(promedio_interacciones_por_año)
"""

