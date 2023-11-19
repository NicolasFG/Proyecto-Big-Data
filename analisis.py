import os
import matplotlib.pyplot as plt
from implicit.als import AlternatingLeastSquares
import pickle
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pdh
import pandas as pd

ruta_actual = os.getcwd()
df = pd.read_csv(ruta_actual + '/recos_supermercados/cleaned_data.csv')

conteo_productos = df.groupby('codigosap')['descripcion'].count()

# Crear un nuevo DataFrame que incluya los nombres de los productos
df_conteo = pd.DataFrame({'conteo': conteo_productos}).reset_index()
df_conteo = df_conteo.merge(df[['codigosap', 'descripcion']].drop_duplicates(), on='codigosap')

# Ordenar y seleccionar los top 10
top_10_productos = df_conteo.sort_values(by='conteo', ascending=False).head(10)

plt.figure(figsize=(12,8))
plt.barh(top_10_productos['descripcion'], top_10_productos['conteo'])
plt.title('Top 10 Productos con más Transacciones')
plt.ylabel('Producto')
plt.xlabel('Cantidad de Transacciones')
plt.tight_layout()
plt.show()

top_10_productos = df_conteo.sort_values(by='conteo', ascending=True).head(10)

# Crear un gráfico de barras horizontal
plt.figure(figsize=(12, 8))
plt.barh(top_10_productos['descripcion'], top_10_productos['conteo'])
plt.title('Top 10 Productos con menos Transacciones')
plt.xlabel('Cantidad de Transacciones')
plt.ylabel('Producto')
plt.tight_layout()
plt.show()

# Calcular ingresos por producto
df['ingresos'] = df['Cantidad'] * df['PrecioUnitario']

# Agrupar por código de producto y sumar los ingresos
ingresos_por_producto = df.groupby('codigosap')['ingresos'].sum()

# Crear un nuevo DataFrame que incluya los nombres de los productos
df_ingresos = pd.DataFrame({'ingresos': ingresos_por_producto}).reset_index()
df_ingresos = df_ingresos.merge(df[['codigosap', 'descripcion']].drop_duplicates(), on='codigosap')

# Ordenar y seleccionar los top 10
top_10_productos = df_ingresos.sort_values(by='ingresos', ascending=False).head(10)

# Crear un gráfico de barras
plt.figure(figsize=(12,8))
plt.barh(top_10_productos['descripcion'], top_10_productos['ingresos'])
plt.title('Top 10 Productos con más Ingresos Generados')
plt.ylabel('Producto')
plt.xlabel('Ingresos Generados')
plt.tight_layout()
plt.show()

top_10_productos = df_ingresos.sort_values(by='ingresos', ascending=True).head(10)

# Crear un gráfico de barras
plt.figure(figsize=(12,8))
plt.barh(top_10_productos['descripcion'], top_10_productos['ingresos'])
plt.title('Top 10 Productos con menos Ingresos Generados')
plt.ylabel('Producto')
plt.xlabel('Ingresos Generados')
plt.tight_layout()
plt.show()

conteo_clientes = df.groupby('CodCliente')['NombreCliente'].count()

# Crear un nuevo DataFrame que incluya los nombres de los clientes
df_conteo = pd.DataFrame({'conteo': conteo_clientes}).reset_index()
df_conteo = df_conteo.merge(df[['CodCliente', 'NombreCliente']].drop_duplicates(), on='CodCliente')

# Ordenar y seleccionar los top 10
top_10_clientes = df_conteo.sort_values(by='conteo', ascending=False).head(10)

plt.figure(figsize=(12,8))
plt.barh(top_10_clientes['NombreCliente'], top_10_clientes['conteo'])
plt.title('Top 10 Clientes con más Transacciones')
plt.ylabel('Cliente')
plt.xlabel('Cantidad de Transacciones')
plt.tight_layout()
plt.show()

# Ordenar y seleccionar los top 10
top_10_clientes = df_conteo.sort_values(by='conteo', ascending=True).head(10)

# Crear un gráfico de barras horizontal
plt.figure(figsize=(12, 8))
plt.barh(top_10_clientes['NombreCliente'], top_10_clientes['conteo'])
plt.title('Top 10 Clientes con menos Transacciones')
plt.xlabel('Cantidad de Transacciones')
plt.ylabel('Cliente')
plt.tight_layout()
plt.show()

# Agrupar por código de cliente y sumar los ingresos
ingresos_por_producto = df.groupby('CodCliente')['ingresos'].sum()

# Crear un nuevo DataFrame que incluya los nombres de los productos
df_ingresos = pd.DataFrame({'ingresos': ingresos_por_producto}).reset_index()
df_ingresos = df_ingresos.merge(df[['CodCliente', 'NombreCliente']].drop_duplicates(), on='CodCliente')

# Ordenar y seleccionar los top 10
top_10_clientes = df_ingresos.sort_values(by='ingresos', ascending=False).head(10)

# Crear un gráfico de barras
plt.figure(figsize=(12,8))
plt.barh(top_10_clientes['NombreCliente'], top_10_clientes['ingresos'])
plt.title('Top 10 Clientes con más Ingresos Generados')
plt.ylabel('Cliente')
plt.xlabel('Ingresos Generados')
plt.tight_layout()
plt.show()

top_10_clientes = df_ingresos.sort_values(by='ingresos', ascending=True).head(10)

# Crear un gráfico de barras
plt.figure(figsize=(12,8))
plt.barh(top_10_clientes['NombreCliente'], top_10_clientes['ingresos'])
plt.title('Top 10 Clientes con menos Ingresos Generados')
plt.ylabel('Cliente')
plt.xlabel('Ingresos Generados')
plt.tight_layout()
plt.show()

# Convertir la columna 'fecha' a formato de fecha
df['fecha'] = pd.to_datetime(df['fecha'])

# Extraer el mes y el año de la fecha
df['mes_año'] = df['fecha'].dt.to_period('M')

# Agrupar por mes y año
agrupado_por_mes = df.groupby('mes_año').agg({'codigosap': 'count', 'ingresos': 'sum'})

# Graficar transacciones por mes
plt.figure(figsize=(12, 6))
agrupado_por_mes['codigosap'].plot(kind='bar')
plt.title('Transacciones por Mes')
plt.xlabel('Mes')
plt.ylabel('Número de Transacciones')
plt.xticks(rotation=45)
plt.show()

# Graficar ingresos por mes
plt.figure(figsize=(12, 6))
agrupado_por_mes['ingresos'].plot(kind='bar', color='orange')
plt.title('Ingresos por Mes')
plt.xlabel('Mes')
plt.ylabel('Ingresos Totales')
plt.xticks(rotation=45)
plt.show()
