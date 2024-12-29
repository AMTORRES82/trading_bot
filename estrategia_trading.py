import pandas as pd
import numpy as np
import krakenex
import streamlit as st
import matplotlib as plt
from plotly import graph_objects as go
from utils import calcular_sharpe_ratio, calcular_rentabilidad_acumulada_anualizada, cum_return_list

class EstrategiaTrading:

    def __init__(self, pair, since, to, buy_threshold=-0.3, sell_threshold=0.3, weight_rsi=0.5,
                 margen_bb_up=0.05, margen_bb_down=0.05, capital_inicial=1000, bias=0):
        self.pair = pair
        self.since = since
        self.to = to
        self.bias=bias
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.weight_rsi = weight_rsi
        self.margen_bb_up = margen_bb_up
        self.margen_bb_down = margen_bb_down
        self.capital_inicial = capital_inicial
        self.data = None  # Inicializar data como None
        self.resultados = None  # Inicializar resultados como None

    def download_kraken_data(self, feature='close', graph=0):
        """
        Descarga datos históricos OHLC para un par dado desde Kraken.

        Args:
            feature (str): Nombre de la columna a visualizar ('open', 'high', 'low', 'close', etc.).
            graph (int): 0 para no graficar, 1 para mostrar la gráfica.

        Returns:
            None: Los datos se almacenan en self.data.
        """
        try:
            since_time = int(pd.to_datetime(self.since).timestamp())
            to_time = int(pd.to_datetime(self.to).timestamp())
            # Mostrar los timestamps para depuración
            #st.write("Timestamp Fecha Inicio kraken:", since_time)
            #st.write("Timestamp Fecha Fin kraken:", to_time)
        
            if since_time >= to_time:
                raise ValueError("El parámetro 'since' debe ser anterior al parámetro 'to'.")

            k = krakenex.API()
            ohlc_data = k.query_public('OHLC', {'pair': self.pair, 'interval': 1440, 'since': since_time})
            if ohlc_data.get('error'):
                raise Exception(f"Error al descargar datos de Kraken: {ohlc_data['error']}")

            df = pd.DataFrame(ohlc_data['result'][self.pair])
            df.columns = ['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df = df[df.index < pd.to_datetime(self.to)]
            df = df.apply(pd.to_numeric, errors='coerce')

            if feature not in df.columns:
                raise ValueError(f"La columna '{feature}' no existe en los datos descargados.")

            self.data = df[[feature]].copy()  # Asegurar que data sea un DataFrame con una columna

            if graph == 1:
                fig = go.Figure(data=[go.Scatter(x=df.index, y=df[feature])])
                fig.update_layout(title=f"{feature} de {self.pair}",
                                  xaxis_title="Fecha",
                                  yaxis_title=feature)
                fig.show()
        #except Exception as e:
        #    print(f"Error al descargar datos de Kraken: {e}")
        except Exception as e:
            st.error(f"Error al descargar datos de Kraken: {e}")

    def calculate_bollinger_bands(self, window=20, num_std=2, graph=0):
        """
        Calcula las Bandas de Bollinger.

        Args:
            window (int): Tamaño de la ventana para calcular la media móvil.
            num_std (float): Número de desviaciones estándar para las bandas.
            graph (int): 0 para no graficar, 1 para mostrar la gráfica.

        Returns:
            tuple: Media móvil, banda superior, banda inferior.
        """
        try:
            rolling_mean = self.data['close'].rolling(window=window).mean()
            rolling_std = self.data['close'].rolling(window=window).std()
            upper_band = (rolling_mean + (rolling_std * num_std)).bfill()
            lower_band = (rolling_mean - (rolling_std * num_std)).bfill()

            if graph == 1:
    
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=self.data.index, y=self.data['close'], name='Precio'))
                fig.add_trace(go.Scatter(x=self.data.index, y=upper_band, name='Banda Superior', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=self.data.index, y=lower_band, name='Banda Inferior', line=dict(color='red')))
                fig.update_layout(title="Bollinger Bands",
                                  xaxis_title="Fecha",
                                  yaxis_title="Precio")
                return fig.show()
            else:
                return rolling_mean, upper_band, lower_band
                

            
        except Exception as e:
            #print(f"Error al calcular las Bandas de Bollinger: {e}")
            st.error(f"Error al calcular las Bandas de Bollinger: {e}") 

    def calculate_rsi(self, window=14, graph=0):
        """
        Calcula el Índice de Fuerza Relativa (RSI).

        Args:
            window (int): Tamaño de la ventana para calcular el RSI.
            graph (int): 0 para no graficar, 1 para mostrar la gráfica.

        Returns:
            pandas.Series: RSI calculado.
        """
        try:
            delta = self.data['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            rsi = (100 - (100 / (1 + rs))).bfill()

            if graph == 1:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=self.data.index, y=self.data['close'], name='Precio'), secondary_y=False)
                fig.add_trace(go.Scatter(x=self.data.index, y=rsi, name='RSI'), secondary_y=True)
                fig.update_yaxes(title_text="Precio", secondary_y=False)
                fig.update_yaxes(title_text="RSI", secondary_y=True)
                fig.update_layout(title="RSI y Precio",
                                  xaxis_title="Fecha",
                                  legend_title="Leyenda")
                return fig.show()
            else:
                return rsi

                
        except Exception as e:                       
          #  print(f"Error al calcular el RSI: {e}")
            st.error(f"Error al calcular el RSI: {e}")

    def generate_signal(self,rsi_sobrecompra=70, rsi_sobreventa=30):
        """
        Genera señales de compra y venta basadas en RSI y Bandas de Bollinger.

        Args:
            rsi_sobrecompra (int): Umbral de sobrecompra para RSI.
            rsi_sobreventa (int): Umbral de sobreventa para RSI.

        Returns:
            pandas.Series: Señales generadas.
        """
        try:
            self.data['rsi'] = self.calculate_rsi(window=14, graph=0)
            rolling_mean, upper_band, lower_band = self.calculate_bollinger_bands(window=20, num_std=2, graph=0)
            self.data['upper_band'] = upper_band
            self.data['lower_band'] = lower_band
            self.data=self.data.copy()
            signal_rsi = np.where(self.data['rsi'] <= rsi_sobreventa, 1,  # Señal de compra
                                               np.where(self.data['rsi'] >= rsi_sobrecompra, -1, 0))  # Señal de venta

            signal_bb = np.where(self.data['close'] <= self.data['lower_band'] * (1 + self.margen_bb_down), 1,
                                              np.where(self.data['close'] >= self.data['upper_band'] * (1 + self.margen_bb_up), -1, 0))


            return signal_rsi, signal_bb
        except Exception as e:
            st.error(f"Error al generar señales: {e}")
            return None


    def apply_thresholds(self,combined_signal, buy_threshold, sell_threshold, bias=0):
        """
        Applies thresholds to a combined signal to generate trading signals.

        Args:
            combined_signal (numpy.ndarray or pandas.Series): The combined signal data.
            buy_threshold (float): The threshold below which a buy signal (-1) is triggered.
            sell_threshold (float): The threshold above which a sell signal (1) is triggered.
            bias (float): A bias to add to the combined signal before applying the thresholds.

        Returns:
            numpy.ndarray: An array of trading signals (-1 for buy, 1 for sell, 0 for hold).
        """
        try:
            # Validate inputs
            if not isinstance(combined_signal, (np.ndarray, pd.Series)):
                raise ValueError("El parámetro 'combined_signal' debe ser un numpy.ndarray o un pandas.Series.")

            if not isinstance(buy_threshold, (int, float)):
                raise ValueError("El parámetro 'buy_threshold' debe ser un número.")

            if not isinstance(sell_threshold, (int, float)):
                raise ValueError("El parámetro 'sell_threshold' debe ser un número.")

            if not isinstance(bias, (int, float)):
                raise ValueError("El parámetro 'bias' debe ser un número.")

            if buy_threshold >= sell_threshold:
                raise ValueError("El 'buy_threshold' debe ser menor que el 'sell_threshold'.")

            # Apply bias and thresholds
            combined_signal_tan = np.tanh(combined_signal + bias)
            trading_signal = np.where(
                combined_signal_tan <= buy_threshold, -1,  # Buy signal
                np.where(combined_signal_tan >= sell_threshold, 1, 0)  # Sell or hold signal
            )

            return trading_signal

        except Exception as e:
            st.error(f"Error apply_thresholds: {e}")
            return None


    def backtest(self):
        """
        Calcula el rendimiento de una estrategia de trading basada en señales.

        Args:
            data (pandas.DataFrame): DataFrame que debe contener las columnas 'close' y 'trading_signal'.
            capital_inicial (float): Capital inicial de la estrategia.

        Returns:
            dict: Un diccionario con métricas de rendimiento, incluyendo Sharpe Ratio, VaR, retornos acumulados y estadísticas adicionales.
        """
        try: 
            #st.write("Contenido de self.data antes del backtest:", self.data.head())
            #st.write("Valores nulos en self.data:", self.data.isnull().sum())
            # Validar entradas
            if not isinstance(self.data, pd.DataFrame):
              raise ValueError("El parámetro 'data' debe ser un pandas.DataFrame.")

            if 'close' not in self.data.columns or 'trading_signal' not in self.data.columns:
              raise ValueError("El DataFrame debe contener las columnas 'close' y 'trading_signal'.")

            if not isinstance(self.capital_inicial, (int, float)) or self.capital_inicial <= 0:
              raise ValueError("El parámetro 'capital_inicial' debe ser un número positivo.")

            # Inicializar variables
            ################
            posicion = 0  # 0: sin posición, 1: comprado
            cash = self.capital_inicial  # Capital en efectivo
            capital = cash
            acciones = 0
            #################

            # Listas de retornos y capital
            #################
            retornos = []  # Lista para almacenar los retornos diarios
            retorno_total_mercado = []  # Lista para almacenar todos los retornos diarios con independencia de que tengamos o no posición
            retorno_tot_acum_estrategia=[] # Lista para almacenar los retornos tot acum
            retorno_tot_acum_mercado=[] # Lista para almacenar los retornos tot acum
            capital_historial = []  # Para registrar la evolución del capital total
            cash_historial = [] # Para registrar la evolución del cash
            acciones_historial = []
            #################

            self.data['trading_signal_ult'] = self.data['trading_signal'].shift(1).fillna(0)
            self.data['close_ult'] = self.data['close'].shift(1).fillna(0)
            retorno_diario = self.data['close'].pct_change().fillna(0)
            retorno_total_mercado=retorno_diario.to_list()
            #cash_historial.append(cash)
            #capital_historial.append(capital)
            #acciones_historial.append(acciones)
        
            # Depuración: Validar columnas antes del bucle
            #st.write("Contenido de self.data antes del bucle:")
            #st.write(self.data.head())

            # Iterar sobre los datos
            for i in range(len(self.data)):
              if i==0:
                cash_historial.append(cash)
                capital_historial.append(capital)
                acciones_historial.append(acciones)


              precio_ult = self.data['close_ult'][i]
              precio = self.data['close'][i]
              signal = self.data['trading_signal_ult'][i]

              if acciones_historial[-1]>0:
                posicion=1
              else:
                posicion=0

              # Si hay señal de compra y no tenemos posición comprada (todo es cash):
              if signal == 1 and posicion == 0 and i > 0:
                acciones = cash_historial[-1] / precio_ult
                posicion = 1
                cash = 0  # Todo el capital está en acciones
                capital = acciones * precio + cash
                cash_historial.append(cash)
                capital_historial.append(capital)
                acciones_historial.append(acciones)

              # Si hay señal de compra y tenemos una posición comprada:
              elif signal == 1 and posicion == 1 and i > 0:
                cash_historial.append(0)
                capital_historial.append(capital_historial[-1])
                acciones_historial.append(acciones_historial[-1])

              # Si hay señal de venta y tenemos una posición comprada:
              elif signal == -1 and posicion == 1  and i > 0:
                acciones = 0
                posicion = 0
                cash = acciones_historial[-1] * precio_ult
                capital = acciones * precio + cash  # Todo el capital está en efectivo
                cash_historial.append(cash)
                capital_historial.append(capital)
                acciones_historial.append(acciones)

              # Si hay señal de venta y NO tenemos una posición comprada:
              elif signal == -1 and posicion == 0  and i > 0:
                cash_historial.append(cash_historial[-1])
                capital_historial.append(capital_historial[-1])
                acciones_historial.append(0)

              elif i > 0:
                cash_historial.append(cash_historial[-1])
                capital_historial.append(capital_historial[-1])
                acciones_historial.append(acciones_historial[-1])


              # Calcular el retorno diario

              if posicion == 1:
                retornos.append(retorno_total_mercado[i])
              else:
                retornos.append(0)


            retorno_tot_acum_estrategia=cum_return_list(retornos)
            retorno_tot_acum_mercado=cum_return_list(retorno_total_mercado)



            self.data['retorno_total']=retorno_diario
            self.data['retornos']=retornos

            self.data['cum_return']=retorno_tot_acum_estrategia
            self.data['cum_return_mercado']=retorno_tot_acum_mercado

            self.data['cantidad_acciones']=acciones_historial
            self.data['cash']=cash_historial
            #st.write("Contenido final del bucle de self.data:")
            #st.write(self.data.head())

            # Cálculo del Sharpe Ratio
            sharpe_ratio = calcular_sharpe_ratio(retornos)
            sharpe_ratio_mercado = calcular_sharpe_ratio(retorno_total_mercado)

            #Cálculo del VaR
            VaR95_mercado = np.percentile(retorno_total_mercado, 5)
            VaR99_mercado = np.percentile(retorno_total_mercado, 1)
            VaR95_estrategia = np.percentile(retornos, 5)
            VaR99_estrategia = np.percentile(retornos, 1)


            rentabilidad_anualizada = calcular_rentabilidad_acumulada_anualizada(retornos, retorno_tot_acum_estrategia[-1])
            rentabilidad_anualizada_mercado = calcular_rentabilidad_acumulada_anualizada(retorno_total_mercado, retorno_tot_acum_mercado[-1])
            self.data=self.data.copy()
            #st.write("Contenido final de self.data:")
            #st.write(self.data.head())
            return {
              "Sharpe_Ratio_anualizado_estrategia": round(sharpe_ratio,4),
              "Retorno_Acumulado_estrategia": round(retorno_tot_acum_estrategia[-1],4),
              "Retorno_Acumulado_Anualizado_estrategia": round(rentabilidad_anualizada,4),
              "Retorno_bis":round(capital_historial[-1]/self.capital_inicial-1,4),
              "Sharpe_Ratio_anualizado_mercado": round(sharpe_ratio_mercado,4),
              "Retorno_Acumulado_mercado": round(retorno_tot_acum_mercado[-1],4),
              "Retorno_Acumulado_Anualizado_mercado": round(rentabilidad_anualizada_mercado,4),
              "VaR_95_estrategia": round(VaR95_estrategia,4)*self.capital_inicial,
              "VaR_95_mercado": round(VaR95_mercado,4)*self.capital_inicial,
              "VaR_99_estrategia": round(VaR99_estrategia,4)*self.capital_inicial,
              "VaR_99_mercado": round(VaR99_mercado,4)*self.capital_inicial,
              "Capital_Inicial": round(self.capital_inicial,2),
              "Capital_Final": round(capital_historial[-1],2),
              "Num_acciones": acciones_historial[-1],
              "Cash_final": round(cash_historial[-1],2),
              "Fecha_inicio":self.data.index.min().strftime("%d/%m/%Y"),
              "Fecha_fin":self.data.index.max().strftime("%d/%m/%Y"),
              "Num_días": len(self.data),
              "Num_días_liquidez": len(self.data[self.data['retornos'] == 0]),
              "Num_dias_trading": len(self.data[self.data['retornos'] != 0]),
              "Num_señales_de_compra": len(self.data[self.data['trading_signal'] == 1]),
              "Num_señales_de_venta": len(self.data[self.data['trading_signal'] == -1])
            }
    
        except Exception as e:
            st.error(f"Error backtest: {e}")
            return None

    def objetive_function(self):
        """
        Calcula el rendimiento de una estrategia de trading basada en parámetros ajustables.

        Args:
            data (pandas.DataFrame): DataFrame que debe contener los datos de precios necesarios.
            capital_inicial (float): Capital inicial de la estrategia.
            buy_threshold (float): Umbral para generar señales de compra.
            sell_threshold (float): Umbral para generar señales de venta.
            weight_rsi (float): Peso del RSI en la combinación de señales.
            margen_bb_up (float): Margen superior para las bandas de Bollinger.
            margen_bb_down (float): Margen inferior para las bandas de Bollinger.
            bias (float): Sesgo adicional para ajustar las señales combinadas.

        Returns:
            dict: Un diccionario con métricas de rendimiento calculadas.
        """
        try:
            # Validaciones iniciales
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError("El parámetro 'data' debe ser un pandas.DataFrame.")

            if 'close' not in self.data.columns:
                raise ValueError("El DataFrame debe contener una columna 'close'.")

            if not (0 <= self.weight_rsi <= 1):
                raise ValueError("El parámetro 'weight_rsi' debe estar entre 0 y 1.")

            if not isinstance(self.capital_inicial, (int, float)) or self.capital_inicial <= 0:
                raise ValueError("El parámetro 'capital_inicial' debe ser un número positivo.")

            if self.buy_threshold >= self.sell_threshold:
                raise ValueError("El 'buy_threshold' debe ser menor que el 'sell_threshold'.")

            #print(f"Parámetros: capital_inicial={self.capital_inicial}, buy_threshold={self.buy_threshold}, sell_threshold={self.sell_threshold},"
            #      f"weight_rsi={self.weight_rsi}, margen_bb_up={self.margen_bb_up}, margen_bb_down={self.margen_bb_down}, bias={self.bias}")
            
            # Generar señales
            signal_rsi, signal_bb = self.generate_signal()

            # Combinar señales
            combined_signal = self.weight_rsi * signal_rsi + (1 - self.weight_rsi) * signal_bb

            # Añadir señales al DataFrame
            self.data['signal_rsi'] = signal_rsi
            self.data['signal_bb'] = signal_bb
            self.data['suma_signal'] = signal_rsi + signal_bb
            self.data['combined_signal'] = combined_signal

            # Aplicar umbrales para generar señales de trading
            self.data['trading_signal'] = self.apply_thresholds(combined_signal, self.buy_threshold, self.sell_threshold, self.bias)
            
            #Actualizamos data
            self.data=self.data.copy()

            # Calcular el rendimiento
            resultados = self.backtest()

            return resultados

        except Exception as e:
            st.error(f"Error func.obj.: {e}")
            return None

    def graph(self):
        """
        Genera gráficos para analizar la estrategia de trading.
        """
        try:
            if self.data is None:
                raise ValueError("Los datos no están disponibles. Ejecuta primero los métodos necesarios para generar los datos.")

            # Aumentar el tamaño de la figura para que los gráficos tengan más espacio
            fig, axes = plt.subplots(5, 1, figsize=(12, 30))  # 5 gráficos en 1 columna
            fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3)  # Ajustar el espacio entre los gráficos

            # Título general de la figura
            fig.suptitle('Análisis de Retornos y Cantidad de Acciones', fontsize=20)

            # Graficar retornos en el primer eje
            ax1 = axes[0]
            ax1.plot(self.data.index, self.data['retornos'], label='Retornos', color='blue')
            ax1.set_xlabel('Tiempo')
            ax1.set_ylabel('Retornos', color='blue')
            ax1.set_title('Retornos Diarios')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.legend(loc='upper left')

            # Graficar cantidad de acciones en el segundo eje, con eje secundario para cash
            ax2 = axes[1]
            ax2.plot(self.data.index, self.data['cantidad_acciones'], label='Cantidad de Acciones', color='red')
            ax2.set_ylabel('Cantidad de Acciones', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_title('Cantidad de Acciones y Cash')
            ax2.legend(loc='upper left')

            # Eje secundario para cash
            ax2_cash = ax2.twinx()
            ax2_cash.plot(self.data.index, self.data['cash'], label='Cash', color='green')
            ax2_cash.set_ylabel('Cash', color='green')
            ax2_cash.tick_params(axis='y', labelcolor='green')
            ax2_cash.legend(loc='upper right')

            # Histograma de retornos (Estrategia vs Mercado)
            ax3 = axes[2]
            ax3.hist(self.data['retornos'], bins=30, color='blue', alpha=0.7, label='Retornos Estrategia')
            ax3.hist(self.data['retorno_total'], bins=30, color='red', alpha=0.5, label='Retornos Mercado')
            ax3.set_xlabel('Retornos')
            ax3.set_ylabel('Frecuencia')
            ax3.set_title('Distribución de Retornos: Estrategia vs Mercado')
            ax3.legend(loc='upper right')

            # Graficar retorno acumulado vs retorno acumulado del mercado
            ax4 = axes[3]
            ax4.plot(self.data.index, self.data['cum_return'], label='Retorno Acumulado Estrategia', color='red')
            ax4.plot(self.data.index, self.data['cum_return_mercado'], label='Retorno Acumulado Mercado', color='blue', linestyle='--')
            ax4.set_xlabel('Tiempo')
            ax4.set_ylabel('Retorno Acumulado')
            ax4.set_title('Retorno Acumulado Estrategia vs Mercado')
            ax4.legend(loc='upper left')

            # Graficar señales de trading y precio del activo
            ax5 = axes[4]
            ax5.plot(self.data.index, self.data['trading_signal_ult'], label='Trading Signal', color='blue', linestyle='--')
            ax5.set_xlabel('Tiempo')
            ax5.set_ylabel('Señales de Trading', color='blue')
            ax5.tick_params(axis='y', labelcolor='blue')
            ax5.set_title('Señales de Trading y Precio del Activo')
            ax5.legend(loc='upper left')

            # Eje secundario para el precio del activo
            ax5_price = ax5.twinx()
            ax5_price.plot(self.data.index, self.data['close'], label='Precio', color='red')
            ax5_price.set_ylabel('Precio', color='red')
            ax5_price.tick_params(axis='y', labelcolor='red')
            ax5_price.legend(loc='upper right')

            # Mostrar la gráfica
            plt.subplots_adjust(top=0.92)
            plt.show()

        except Exception as e:
            st.error(f"Error al generar los gráficos: {e}")
