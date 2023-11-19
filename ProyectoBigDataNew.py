from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
import pandas as pd
import os
import matplotlib.pyplot as plt
from implicit.als import AlternatingLeastSquares
import pickle
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pdh
import pandas as pd


def preprocesamiento():
    # Obtener la ruta del directorio actual y cargar el DataFrame
    ruta_actual = os.getcwd()
    df = pd.read_csv(ruta_actual + '/recos_supermercados/supermercados.csv')

    # Filtro los clientes que tienen un codigo de clientes igual a 0, porque no son clientes reales.
    df = df[df['CodCliente'] != 0]

    # Calcular el promedio de interacciones por cliente
    interacciones_por_cliente = df.groupby(['codigosap']).size()
    promedio_interacciones = interacciones_por_cliente.mean()



    # Filtrar para mantener solo los clientes activos
    min_interacciones = promedio_interacciones
    
    clientes_activos = interacciones_por_cliente[interacciones_por_cliente >= min_interacciones].index
    df_clientes_activos = df[df['CodCliente'].isin(clientes_activos)]

    # Crear un DataFrame con el conteo de pedidos por producto para clientes activos
    pedidos_producto_mayor_ganancia = df.groupby('codigosap').size()

    # Calcular el promedio de interacciones por cliente
    interacciones_por_cliente = df.groupby('CodCliente').size()
    promedio_interacciones = interacciones_por_cliente.mean()

    # Filtrar para mantener solo los clientes activos
    min_interacciones = promedio_interacciones
    
    clientes_activos = interacciones_por_cliente[interacciones_por_cliente >= min_interacciones].index
    df_clientes_activos = df[df['CodCliente'].isin(clientes_activos)]

    #Clientes Fieles (df todos los clientes fieles)

    print(df_clientes_activos)


preprocesamiento()

#Productos que mas ingresos generan PrecioUnitario*Cantidad
