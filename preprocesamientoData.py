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
import math
import pandas as pd
import os

# Obtener la ruta del directorio actual y cargar el DataFrame
ruta_actual = os.getcwd()
df = pd.read_csv(ruta_actual + '/recos_supermercados/supermercados.csv')

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

print(df_clientes_activos_y_productos_populares)


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

