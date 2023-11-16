import pandas as pd
import os
import matplotlib.pyplot as plt


# Obtener la ruta del directorio actual
ruta_actual = os.getcwd()

df = pd.read_csv( ruta_actual +'/recos_supermercados/supermercados.csv')

# Ver las primeras filas
print(df.head())

# Eliminar columnas que no necesitas
df = df.drop(['Tienda', 'UnidadMedida', 'PorcDescuento', 'ImpDescuento', 'ImporteLineaBs', 'CostoUnitario'], axis=1)

df['fecha'] = pd.to_datetime(df['fecha'])



"""

print("#############################################")

productos_vendidos = df.groupby('descripcion')['Cantidad'].sum()

productos_ordenados = productos_vendidos.sort_values(ascending=False)

top_productos = productos_ordenados.head(10)

print(top_productos)

top_productos.plot(kind='bar')
plt.title('Productos Más Vendidos')
plt.xlabel('Producto')
plt.ylabel('Cantidad Vendida')
plt.xticks(rotation=45)  # Rota los nombres de los productos para mejor visualización
plt.show()
"""
#productos_mas_vendidos.plot(kind='bar')
#plt.show()

#ventas_por_mes.plot(kind='line')
#plt.show()

#Voy a filtrar el codigo cliente igual a 0, ya que no tiene ningun cliente asignado y no me importa en mi objetivo final.
df = df[df['CodCliente'] != 0]

"""
#Tengo los clientes recurrentes (clientes fieles)
print("Clientes fieles")

conteo_compras = df.groupby('CodCliente').size()
clientes_recurrentes = conteo_compras[conteo_compras > 1]
df_clientes_recurrentes = df[df['CodCliente'].isin(clientes_recurrentes.index)]

conteo_compras_recurrentes = df_clientes_recurrentes.groupby(['CodCliente', 'NombreCliente']).size()

top_10_clientes_recurrentes = conteo_compras_recurrentes.sort_values(ascending=False).head(10)

top_10_clientes_recurrentes = top_10_clientes_recurrentes.reset_index()
top_10_clientes_recurrentes['Etiqueta'] = top_10_clientes_recurrentes['CodCliente'].astype(str) + ' - ' + top_10_clientes_recurrentes['NombreCliente']


print(top_10_clientes_recurrentes)
"""


#Tengo los productos que mas se venden
print("Productos mas vendidos")


productos_mas_vendidos = df.groupby(['descripcion', 'Categoria'])['Cantidad'].sum().sort_values(ascending=False)
top_10_productos = productos_mas_vendidos.head(10)
print(top_10_productos)




#categorias_mas_vendidas = df.groupby('Categoria')['Cantidad'].sum().sort_values(ascending=False)

#ventas_por_mes = df.resample('M', on='fecha')['Cantidad'].sum()



"""
# Asumiendo que 'descripcion' es la columna que identifica los productos
productos_unicos = df['descripcion'].unique()
print(productos_unicos)
print(len(productos_unicos))

print("#############################################")

print(productos_mas_vendidos)
print(len(productos_mas_vendidos))
"""







"""
# Crear un gráfico de barras
top_10_clientes_recurrentes.plot(kind='bar')
plt.title('Top 10 Clientes por Número de Compras')
plt.xlabel('Código del Cliente')
plt.ylabel('Número de Compras')
plt.xticks(rotation=45)
plt.show()
"""

# O puedes querer rellenar los valores faltantes en lugar de eliminarlos
#df = df.fillna(0) # Por ejemplo, rellenar con ceros

# Ver información general (tipos de datos, valores no nulos, etc.)
#print(df.info())


#print(df.describe())


# Eliminar filas donde hay al menos un elemento faltante
#df = df.dropna()

# O puedes querer rellenar los valores faltantes en lugar de eliminarlos
#df = df.fillna(0) # Por ejemplo, rellenar con ceros




#print(df.describe())


# Eliminar filas donde hay al menos un elemento faltante
#df = df.dropna()

# O puedes querer rellenar los valores faltantes en lugar de eliminarlos
#df = df.fillna(0) # Por ejemplo, rellenar con ceros




# Filtrar filas basado en ciertos criterios
#df = df[df['Cantidad'] > 0]  # Por ejemplo, conservar solo filas con cantidad positiva


#Aplicando Normalizacion

