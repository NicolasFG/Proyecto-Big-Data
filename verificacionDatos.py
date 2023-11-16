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
codigo_a_Subcategoria = df.set_index('codigosap')['Subcategoria'].to_dict()

# Eliminar columnas que no necesitas
#df = df.drop(['Tienda', 'UnidadMedida','NombreCliente', 'Categoria','PorcDescuento', 'Subcategoria','fecha','descripcion','PesoNeto','PrecioUnitario','ImpDescuento', 'ImporteLineaBs', 'CostoUnitario'], axis=1)

"""
df['rating_implicito'] = df.groupby(['CodCliente', 'codigosap'])['Cantidad'].transform('sum')

# Filtrar para mantener solo a los usuarios activos
usuarios_activos = interacciones_usuario[interacciones_usuario >= min_interacciones].index

df_activos = df[df['CodCliente'].isin(usuarios_activos)]


# Crear un DataFrame con el conteo de pedidos por producto
pedidos_producto = df_activos.groupby('codigosap').size()

# Filtrar para mantener solo los productos populares
productos_populares = pedidos_producto[pedidos_producto >= min_pedidos].index
df_activos_populares = df_activos[df_activos['codigosap'].isin(productos_populares)]
"""

# Verificar si hay valores nulos en 'descripcion'
#print("Valores nulos en 'descripcion':", df['descripcion'].isnull().sum())

# Verificar la unicidad de las descripciones para cada 'codigosap'
descripciones_unicas = df.groupby('codigosap')['descripcion'].nunique()
#print("Descripciones únicas por 'codigosap':\n", descripciones_unicas)

# Identificar códigos de producto con más de una descripción
codigos_con_multiples_descripciones = descripciones_unicas[descripciones_unicas > 1]
#print("Códigos con múltiples descripciones:\n", codigos_con_multiples_descripciones)

# Mostrar ejemplos de códigos de producto con descripciones inconsistentes
#for codigo in codigos_con_multiples_descripciones.index:
#    print(df[df['codigosap'] == codigo][['codigosap', 'descripcion']].drop_duplicates())

"""
# Verificar si todas las descripciones en 'codigo_a_descripcion' son consistentes con 'df'
for codigo, descripcion in codigo_a_descripcion.items():
    if codigo in df['codigosap'].values:
        descripcion_df = df[df['codigosap'] == codigo]['descripcion'].iloc[0].strip().lower()
        if descripcion.strip().lower() != descripcion_df:
            print(f"Inconsistencia encontrada para {codigo}: {descripcion} vs {descripcion_df}")
"""
#Usar el campo Subcategoria

# Inicializar la bandera como False
inconsistencia_encontrada = False

for codigo, Subcategoria in codigo_a_Subcategoria.items():
    if codigo in df['codigosap'].values:
        Subcategoria_df = df[df['codigosap'] == codigo]['Subcategoria'].iloc[0].strip().lower()
        if Subcategoria.strip().lower() != Subcategoria_df:
            print(f"Inconsistencia encontrada para {codigo}: {Subcategoria} vs {Subcategoria_df}")

# Verificar la bandera al final del bucle
if not inconsistencia_encontrada:
    print("Se revisaron todas las inconsistencias y no hubo ninguna.")