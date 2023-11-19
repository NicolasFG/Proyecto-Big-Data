import pandas as pd
import os

# Load your CSV file

ruta_actual = os.getcwd()

df = pd.read_csv( ruta_actual +'/recos_supermercados/supermercados.csv')

# Filtro los clientes que tienen un codigo de clientes igual a 0, porque no son clientes reales.
df = df[df['CodCliente'] != 0]

# Step 1: Concatenate 'Categoria', 'Subcategoria', and 'Descripcion' into a single column
df['Concatenated'] = df['Categoria'] + ' / ' + df['Subcategoria'] + ' / ' + df['descripcion']

# Step 2: Convert this new column to uppercase
df['Concatenated'] = df['Concatenated'].str.upper()

# Step 3: Ensure no duplicate 'codigosap' in the new column
# Create a dictionary to track the first occurrence of each concatenated value
first_codigosap = {}

for index, row in df.iterrows():
    concatenated_value = row['Concatenated']
    if concatenated_value in first_codigosap:
        # If the concatenated value is already seen, replace the 'codigosap'
        df.at[index, 'codigosap'] = first_codigosap[concatenated_value]
    else:
        # Store the first 'codigosap' for this concatenated value
        first_codigosap[concatenated_value] = row['codigosap']

# Calcular el promedio de interacciones por cliente
interacciones_por_cliente = df.groupby('CodCliente').size()
promedio_interacciones = interacciones_por_cliente.mean()

# Save the cleaned DataFrame to a CSV file
df.to_csv(ruta_actual + '/recos_supermercados/cleaned_data.csv', index=False)
