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
pair = st.sidebar.text_input("Par de Trading (Ej: BTC/USD)", value="BTCUSD")
since = st.sidebar.date_input("Fecha Inicio", value=pd.to_datetime("2023-01-01"))
to = st.sidebar.date_input("Fecha Fin", value=pd.to_datetime("2023-12-31"))
buy_threshold = st.sidebar.slider("Umbral de Compra", min_value=-1.0, max_value=0.0, value=-0.3, step=0.1)
sell_threshold = st.sidebar.slider("Umbral de Venta", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
weight_rsi = st.sidebar.slider("Peso del RSI", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
margen_bb_up = st.sidebar.slider("Margen Superior Bollinger", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
margen_bb_down = st.sidebar.slider("Margen Inferior Bollinger", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
capital_inicial = st.sidebar.number_input("Capital Inicial", min_value=100.0, max_value=100000.0, value=1000.0, step=100.0)
bias = st.sidebar.slider("Bias", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)

# Convertir las fechas seleccionadas a formato 'YYYY-MM-DD'
since_str = since.strftime("%Y-%m-%d")
to_str = to.strftime("%Y-%m-%d")

# Mostrar las fechas convertidas para depuración
#st.write("Fecha Inicio (formato correcto):", since_str)
#st.write("Fecha Fin (formato correcto):", to_str)

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
        estrategia.download_kraken_data(graph=1)
    
    if estrategia.data is not None:
        st.write("Datos descargados exitosamente:")
        st.dataframe(estrategia.data.head())

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

