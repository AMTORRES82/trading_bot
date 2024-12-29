import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from estrategia_trading import EstrategiaTrading  # Asegúrate de importar tu clase correctamente

# Configuración inicial de la página
st.set_page_config(page_title="Estrategia de Trading", layout="wide")

# Título principal
st.title("📈 Estrategia de Trading con Bandas de Bollinger y RSI")

# Barra lateral para parámetros
st.sidebar.header("Configuración de la Estrategia")

# Definir el par predeterminado
pair = st.sidebar.text_input("Par de Trading (Ej: BTC/USD)", value="ETH/USDT")

# Diccionario con parámetros predeterminados por par
default_params_by_pair = {
   "ETH/USDT": {
        'buy_threshold': -0.20997609155869645,
         'sell_threshold': 0.33720546489308373,
         'weight_rsi': 0.21966916415306803,
         'margen_bb_up': -0.09043144177267018,
         'margen_bb_down': -0.005258409961895008,
         'bias': 0.07329623454969729}
            },
    "BTC/USDT": {
        'buy_threshold': -0.18449622279468214,
        'sell_threshold': 0.08081585009697578,
        'weight_rsi': 0.27502739017465827,
        'margen_bb_up': -0.01678456481588446,
        'margen_bb_down': 0.0053315001317940924,
        'bias': 0.0350977151128202
    }
}

# Obtener los parámetros por defecto basados en el par seleccionado
if pair in default_params_by_pair:
    default_params = default_params_by_pair[pair]
else:
    # Parámetros predeterminados genéricos si el par no está definido
    default_params = {
        'buy_threshold': -0.3,
        'sell_threshold': 0.3,
        'weight_rsi': 0.5,
        'margen_bb_up': 0.05,
        'margen_bb_down': 0.05,
        'bias': 0.0
    }

# Configurar entradas para las fechas
since = st.sidebar.date_input("Fecha Inicio", value=pd.to_datetime("2008-01-01"))
to = st.sidebar.date_input("Fecha Fin", value=pd.to_datetime("2024-11-19"))

# Configurar deslizadores con los valores predeterminados seleccionados
buy_threshold = st.sidebar.slider("Umbral de Compra", min_value=-1.0, max_value=0.0, 
                                   value=default_params['buy_threshold'], step=0.001)
sell_threshold = st.sidebar.slider("Umbral de Venta", min_value=0.0, max_value=1.0, 
                                    value=default_params['sell_threshold'], step=0.001)
weight_rsi = st.sidebar.slider("Peso del RSI", min_value=0.0, max_value=1.0, 
                                value=default_params['weight_rsi'], step=0.001)
margen_bb_up = st.sidebar.slider("Margen Superior Bollinger", min_value=-0.5, max_value=0.5, 
                                  value=default_params['margen_bb_up'], step=0.001)
margen_bb_down = st.sidebar.slider("Margen Inferior Bollinger", min_value=-0.5, max_value=0.5, 
                                    value=default_params['margen_bb_down'], step=0.001)
capital_inicial = st.sidebar.number_input("Capital Inicial", min_value=100.0, max_value=100000.0, 
                                          value=1000.0, step=1.0)
bias = st.sidebar.slider("Bias", min_value=-1.0, max_value=1.0, 
                         value=default_params['bias'], step=0.001)

# Convertir las fechas seleccionadas a formato 'YYYY-MM-DD'
since_str = since.strftime("%Y-%m-%d")
to_str = to.strftime("%Y-%m-%d")

# Botón para iniciar la estrategia
if st.sidebar.button("Iniciar Estrategia"):
    # Crear instancia de la estrategia
    estrategia = EstrategiaTrading(
        pair=pair,
        since=since_str,
        to=to_str,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        weight_rsi=weight_rsi,
        margen_bb_up=margen_bb_up,
        margen_bb_down=margen_bb_down,
        capital_inicial=capital_inicial,
        bias=bias,
    )

    # Descargar datos
    st.subheader("📉 Datos de Kraken")
    with st.spinner("Descargando datos desde Kraken..."):
        estrategia.download_kraken_data()

    if estrategia.data is not None:
        st.write("Datos descargados exitosamente:")
        st.dataframe(estrategia.data.head())
        estrategia.download_kraken_data(graph=1)
        estrategia.calculate_bollinger_bands(graph=1)
        estrategia.calculate_rsi(graph=1)

        # Generar señales y backtest
        with st.spinner("Generando señales y realizando backtest..."):
            resultados = estrategia.objetive_function()

        if resultados is not None:
            # Mostrar métricas clave
            st.subheader("📊 Resultados del Backtest")
            st.write(resultados)

            # Gráficos
            with st.spinner("Generando gráficos..."):
                st.subheader("📈 Gráficos de la Estrategia")
                estrategia.graph()

        else:
            st.error("Error al calcular los resultados de la estrategia.")
    else:
        st.error("Error al descargar los datos. Revisa los parámetros ingresados.")

else:
    st.info("Configura los parámetros de la estrategia en la barra lateral y haz clic en 'Iniciar Estrategia'.")
