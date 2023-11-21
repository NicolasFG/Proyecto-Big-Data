import os
import matplotlib.pyplot as plt
from implicit.als import AlternatingLeastSquares
import pickle
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pdh
import pandas as pd

#Primer: Preprocesamiento y entrenamiento de la data para recomendacion general

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

    """     # Crear un DataFrame con el conteo de pedidos por producto para clientes activos
        pedidos_producto = df_clientes_activos.groupby('codigosap').size()

        # Calcular el promedio de pedidos por producto
        promedio_pedidos = pedidos_producto.mean()

        # Filtrar para mantener solo los productos populares
        min_pedidos = promedio_pedidos
        productos_populares = pedidos_producto[pedidos_producto >= min_pedidos].index 
    """
    df_clientes_activos_y_productos_populares = df_clientes_activos

    # Calcular la preferencia de subcategoría
    conteo_subcategoria = df_clientes_activos_y_productos_populares.groupby(['CodCliente', 'Subcategoria']).size().reset_index(name='Conteo')
    conteo_maximo = conteo_subcategoria.groupby('CodCliente')['Conteo'].transform('max')
    conteo_subcategoria['PreferenciaSubcategoria'] = conteo_subcategoria['Conteo'] / conteo_maximo
    print(conteo_subcategoria)

    # Unir con el DataFrame principal
    df_clientes_activos_y_productos_populares = pd.merge(df_clientes_activos_y_productos_populares, conteo_subcategoria[['CodCliente', 'Subcategoria', 'PreferenciaSubcategoria']], on=['CodCliente', 'Subcategoria'], how='left')
    print(df_clientes_activos_y_productos_populares)

    # Normaliza 'PrecioUnitario'
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

    random_clients = df['CodCliente'].drop_duplicates().sample(10)

    for user_id_prueba in random_clients:
        print(f"\n---- Procesando Usuario ID: {user_id_prueba} ----")

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

        print(f"\nTop 10 productos más comprados por el usuario {user_id_prueba}:")
        user_purchases = df[df['CodCliente'] == user_id_prueba]
        product_totals = user_purchases.groupby('codigosap')['Cantidad'].sum().sort_values(ascending=False)
        top_products = product_totals.head(10)
        for codigosap, total_quantity in top_products.items():
            descripcion = codigo_a_descripcion.get(codigosap, "Descripción no disponible")
            print(f"Producto: ID {codigosap}, Cantidad Total Comprada: {total_quantity}, Descripción: {descripcion}")

        print(f"\nTop 10 productos con el mayor rating implícito comprados por el usuario {user_id_prueba}:")
        product_ratings = user_purchases.groupby('codigosap')['rating_implicito'].sum().sort_values(ascending=False)
        top_10_products = product_ratings.head(10)
        for codigosap, total_rating in top_10_products.items():
            descripcion = codigo_a_descripcion.get(codigosap, "Descripción no disponible")
            print(f"Producto: ID {codigosap}, Rating Implícito Total: {total_rating}, Descripción: {descripcion}")


    """     
    user_id_prueba = 3518866

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


    print(f"\nTop 10 productos más comprados por el usuario {user_id_prueba}:")
    user_purchases = df[df['CodCliente'] == user_id_prueba]

    product_totals = user_purchases.groupby('codigosap')['Cantidad'].sum().sort_values(ascending=False)

    top_products = product_totals.head(10)

    for codigosap, total_quantity in top_products.items():
        descripcion = codigo_a_descripcion.get(codigosap, "Descripción no disponible")
        print(f"Producto: ID {codigosap}, Cantidad Total Comprada: {total_quantity}, Descripción: {descripcion}") 


    print(f"\nTop 10 productos con el mayor rating implícito comprados por el usuario {user_id_prueba}:")
    user_purchases = df[df['CodCliente'] == user_id_prueba]

    product_ratings = user_purchases.groupby('codigosap')['rating_implicito'].sum().sort_values(ascending=False)

    top_10_products = product_ratings.head(10)

    for codigosap, total_rating in top_10_products.items():
        descripcion = codigo_a_descripcion.get(codigosap, "Descripción no disponible")
        print(f"Producto: ID {codigosap}, Rating Implícito Total: {total_rating}, Descripción: {descripcion}") 
    """


#Segundo: Preprocesamiento y entrenamiento de la data para recomendacion de los productos en sus meses respectivos

def preprocesamiento_por_meses():
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

    # Estandarización del campo fecha a datetime
    df_clientes_activos['fecha'] = pd.to_datetime(df_clientes_activos['fecha'])
    df_clientes_activos['mes'] = df_clientes_activos['fecha'].dt.month

    # Calcular el conteo de productos y determinar el primer cuartil
    conteo_productos = df_clientes_activos.groupby('codigosap').size()
    N = conteo_productos.quantile(0.25)

    # Calcular la frecuencia de cada producto por mes
    conteo_mes_producto = df_clientes_activos.groupby(['mes', 'codigosap']).size().reset_index(name='conteo_mes')

    # Identificar los productos populares por mes
    productos_populares_por_mes = conteo_mes_producto.groupby('mes').apply(lambda x: x[x['conteo_mes'] >= N]).reset_index(drop=True)

    # Unir los productos populares por mes al DataFrame principal
    df_clientes_activos_y_productos_populares = pd.merge(df_clientes_activos, productos_populares_por_mes, on=['mes', 'codigosap'], how='left')

    # Calcular la preferencia de subcategoría
    conteo_subcategoria = df_clientes_activos_y_productos_populares.groupby(['CodCliente', 'Subcategoria']).size().reset_index(name='Conteo')
    conteo_maximo = conteo_subcategoria.groupby('CodCliente')['Conteo'].transform('max')
    conteo_subcategoria['PreferenciaSubcategoria'] = conteo_subcategoria['Conteo'] / conteo_maximo

    # Unir con el DataFrame principal
    df_clientes_activos_y_productos_populares = pd.merge(df_clientes_activos_y_productos_populares, conteo_subcategoria[['CodCliente', 'Subcategoria', 'PreferenciaSubcategoria']], on=['CodCliente', 'Subcategoria'], how='left')

    # Normalizar 'PrecioUnitario'
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

    # Crear un mapeo de índice de ítem a (codigosap, mes)
    item_index_to_codigosap_mes = {index: (row['codigosap'], row['mes']) for index, row in productos_populares_por_mes.iterrows()}
    

    return df_clientes_activos_y_productos_populares, user_id_map, item_id_map, cliente_a_nombre, item_index_to_codigosap_mes, codigo_a_descripcion, productos_populares_por_mes

def entrenamiento2(df_clientes_activos_y_productos_populares, user_id_map, item_id_map, productos_populares_por_mes):
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

    # Crear el mapeo de índice de ítem a (codigosap, mes)
    item_index_to_codigosap_mes = {index: (row['codigosap'], row['mes']) for index, row in productos_populares_por_mes.iterrows()}

    # Guardar el mapeo de índice de ítem a (codigosap, mes)
    with open('ultimos_resultados/item_index_to_codigosap_mes.pkl', 'wb') as file:
        pickle.dump(item_index_to_codigosap_mes, file)

    # Guardar el modelo entrenado
    with open('ultimos_resultados/als_model_mejorado_mes.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Guardar user_id_map y item_id_map
    with open('ultimos_resultados/user_id_map_mes.pkl', 'wb') as file:
        pickle.dump(user_id_map, file)
    with open('ultimos_resultados/item_id_map_mes.pkl', 'wb') as file:
        pickle.dump(item_id_map, file)

    # Guardar cliente_a_nombre
    with open('ultimos_resultados/cliente_a_nombre_mes.pkl', 'wb') as file:
        pickle.dump(cliente_a_nombre2, file)

    # Guardar item_index_to_codigosap
    with open('ultimos_resultados/item_index_to_codigosap_mes.pkl', 'wb') as file:
        pickle.dump(item_index_to_codigosap2, file)

    # Guardar codigo_a_descripcion
    with open('ultimos_resultados/codigo_a_descripcion_mes.pkl', 'wb') as file:
        pickle.dump(codigo_a_descripcion2, file)

    # Guardar codigo_a_descripcion
    with open('ultimos_resultados/productos_populares_por_mes.pkl', 'wb') as file:
        pickle.dump(productos_populares_por_mes, file)



def convertir_mes_a_numero(mes_nombre):
    meses = {
        "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
        "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
    }
    return meses.get(mes_nombre.capitalize(), 0)  


def testing_meses(mes_input, user_id_prueba, df, model_file, user_id_map_file, item_id_map_file, cliente_a_nombre_file, item_index_to_codigosap_mes_file, codigo_a_descripcion_file, productos_por_mes):

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
    
    with open(codigo_a_descripcion_file, 'rb') as file:
        codigo_a_descripcion = pickle.load(file)

    # Cargar el mapeo de índice de ítem a (codigosap, mes)
    with open(item_index_to_codigosap_mes_file, 'rb') as file:
        item_index_to_codigosap_mes = pickle.load(file)

    print("Tipo de productos_por_mes:", type(productos_por_mes))
    print("Ejemplo de productos_por_mes:", productos_por_mes.head())
    print("mes_input:", mes_input)

    
    mes_numero = convertir_mes_a_numero(mes_input)

    # Filtrar los productos populares para el mes dado
    #productos_populares_mes_actual = productos_por_mes[productos_por_mes['mes'] == mes_input]
    
    productos_populares_mes_actual = productos_por_mes[productos_por_mes['mes'] == mes_numero]

    # Obtener solo los codigosap de los productos populares en ese mes
    codigosap_populares_mes = set(productos_populares_mes_actual['codigosap'])

    

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


    # Procesar recomendaciones como antes
    if user_id_prueba in user_id_map:
        user_index = user_id_map[user_id_prueba]

        # Añadir lógica para manejar productos no populares
        recomendaciones_totales = 10
        recomendaciones = []

        # Obtener e imprimir el nombre del usuario
        nombre_usuario = cliente_a_nombre.get(user_id_prueba, "Nombre no disponible")
        print(f"Nombre del usuario: {nombre_usuario}")



        while len(recomendaciones) < recomendaciones_totales:
            indices_productos_populares = [item_id_map[codigosap] for codigosap in codigosap_populares_mes if codigosap in item_id_map]
            user_item_matrix_filtrada = user_item_matrix_csr[user_index, indices_productos_populares]

            recommended = loaded_model.recommend(user_index, user_item_matrix_filtrada, N=recomendaciones_totales)
            indices_recomendados, puntuaciones = recommended

            for item_idx, score in zip(indices_recomendados, puntuaciones):
                if item_idx in item_index_to_codigosap_mes:
                    codigosap, mes_recomendado = item_index_to_codigosap_mes[item_idx]
                    if codigosap in codigosap_populares_mes:
                        descripcion = codigo_a_descripcion.get(codigosap, "Descripción no disponible")
                        recomendaciones.append((codigosap, score, descripcion))
                        if len(recomendaciones) == recomendaciones_totales:
                            break
                else:
                    print(f"Índice de ítem {item_idx} no encontrado en item_index_to_codigosap.")

        for codigosap, score, descripcion in recomendaciones:
            print(f"Producto Recomendado para Mes {mes_input}: ID {codigosap}, Score: {score}, Descripción: {descripcion}")

    else:
        print(f"user_id {user_id_prueba} no encontrado en user_id_map.")


ruta_actual = os.getcwd()


#Recomendacion general
df, user_id_map, item_id_map, cliente_a_nombre, item_index_to_codigosap, codigo_a_descripcion = preprocesamiento()

entrenamiento(df,user_id_map,item_id_map)

modelo = ruta_actual + "/ultimos_resultados/als_model_mejorado.pkl"
cliente_a_nombre = ruta_actual + "/ultimos_resultados/cliente_a_nombre.pkl"
codigo_a_descripcion = ruta_actual + "/ultimos_resultados/codigo_a_descripcion.pkl"
item_id_map = ruta_actual + "/ultimos_resultados/item_id_map.pkl"
item_index_to_codigosap = ruta_actual + "/ultimos_resultados/item_index_to_codigosap.pkl"
user_id_map = ruta_actual + "/ultimos_resultados/user_id_map.pkl"


testing(df, modelo,user_id_map,item_id_map,cliente_a_nombre,item_index_to_codigosap,codigo_a_descripcion)

""" #Recomendacion por meses

df2, user_id_map2, item_id_map2, cliente_a_nombre2, item_index_to_codigosap2, codigo_a_descripcion2, productos_populares_por_mes = preprocesamiento_por_meses()

productos_populares_por_mes_sin_modificar = productos_populares_por_mes

#entrenamiento2(df2,user_id_map2,item_id_map2,productos_populares_por_mes)

modelo2 = ruta_actual + "/ultimos_resultados/als_model_mejorado_mes.pkl"
cliente_a_nombre2 = ruta_actual + "/ultimos_resultados/cliente_a_nombre_mes.pkl"
codigo_a_descripcion2 = ruta_actual + "/ultimos_resultados/codigo_a_descripcion_mes.pkl"
item_id_map2 = ruta_actual + "/ultimos_resultados/item_id_map_mes.pkl"
item_index_to_codigosap2 = ruta_actual + "/ultimos_resultados/item_index_to_codigosap_mes.pkl"
user_id_map2 = ruta_actual + "/ultimos_resultados/user_id_map_mes.pkl"
productos_populares_por_mes2 = ruta_actual + "/ultimos_resultados/productos_populares_por_mes.pkl"

testing_meses("Febrero","3518866",df2, modelo2,user_id_map2,item_id_map2,cliente_a_nombre2,item_index_to_codigosap2,codigo_a_descripcion2,productos_populares_por_mes_sin_modificar) """
