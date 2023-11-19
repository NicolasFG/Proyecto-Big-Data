import pandas as pd
import os


ruta_actual = os.getcwd()

df = pd.read_csv( ruta_actual +'/recos_supermercados/supermercados.csv')

# Filtro los clientes que tienen un codigo de clientes igual a 0, porque no son clientes reales.
df = df[df['CodCliente'] != 0]

df['Concatenated'] = df['Categoria'] + ' / ' + df['Subcategoria'] + ' / ' + df['descripcion']

df['Concatenated'] = df['Concatenated'].str.upper()

first_codigosap = {}

for index, row in df.iterrows():
    concatenated_value = row['Concatenated']
    if concatenated_value in first_codigosap:
        df.at[index, 'codigosap'] = first_codigosap[concatenated_value]
    else:
        first_codigosap[concatenated_value] = row['codigosap']

interacciones_por_cliente = df.groupby('CodCliente').size()
promedio_interacciones = interacciones_por_cliente.mean()

df.to_csv(ruta_actual + '/recos_supermercados/cleaned_data.csv', index=False)
