import numpy as np

def cum_return_list(returns):
    """
    Calculates the cumulative returns from a list or array of returns.

    Args:
        returns (list, numpy.ndarray, or pandas.Series): A sequence of returns.

    Returns:
        list: A list of cumulative returns.
    """
    try:
        # Validate input
        if not isinstance(returns, (list, np.ndarray, pd.Series)):
            raise ValueError("El parámetro 'returns' debe ser un list, numpy.ndarray o pandas.Series.")

        # Ensure all elements are numeric
        returns = np.array(returns, dtype=float)

        # Calculate cumulative returns
        cum_return = []
        acumulado = 1
        for retorno in returns:
            if retorno != 0:
                acumulado *= (retorno + 1)
            cum_return.append(acumulado - 1)

        return cum_return

    except Exception as e:
        print(f"Error: {e}")
        return None

def calcular_sharpe_ratio(retornos_trading):
    """
    Calcula el Sharpe Ratio considerando solo los días de trading o todos los días.

    Args:
        data (pandas.DataFrame): DataFrame que debe contener la columna 'retornos'.
        mercado (int): 0 para considerar solo los días de trading (retornos != 0),
                       1 para considerar todos los días.

    Returns:
        float: Sharpe Ratio anualizado.
    """
    try:
        # Validar entradas

        #if 'retornos' not in data.columns:
            #raise ValueError("El DataFrame debe contener una columna llamada 'retornos'.")


        # Filtrar los retornos según sea para calcular sharpe de mercado o de nuestra estrategia
        #if mercado == 0:
        #    retornos_trading = data[data['retornos'] != 0]['retornos']
        #else:  # mercado == 1
        #    retornos_trading = data['retornos']

        retornos_trading_filtrados = [r for r in retornos_trading if r != 0]

        # Verificar si hay suficientes datos para calcular el Sharpe Ratio
        if len(retornos_trading_filtrados) < 15:
            raise ValueError("No hay suficientes datos para calcular el Sharpe Ratio.")

        # Calcular la media y desviación estándar de los retornos
        media_exceso_retorno = np.mean(retornos_trading_filtrados)
        desviacion_exceso_retorno = np.std(retornos_trading_filtrados)

        if desviacion_exceso_retorno == 0:
            raise ValueError("La desviación estándar de los retornos es cero, el Sharpe Ratio no se puede calcular.")

        # Calcular el Sharpe Ratio anualizado
        sharpe_ratio = (media_exceso_retorno / desviacion_exceso_retorno) * np.sqrt(365)

        return sharpe_ratio

    except Exception as e:
        print(f"Error: {e}")
        return None

def calcular_rentabilidad_acumulada_anualizada(retornos_trading, rentabilidad_acumulada):
    """
    Calcula la rentabilidad acumulada anualizada considerando solo los días de trading.

    Args:
        data (pandas.DataFrame): DataFrame que debe contener la columna 'retornos'.
        rentabilidad_acumulada (float): Rentabilidad acumulada en el periodo analizado.

    Returns:
        float: Rentabilidad acumulada anualizada considerando solo los días de trading.
    """
    try:
        # Validar entradas
        if not isinstance(rentabilidad_acumulada, (int, float)):
            raise ValueError("El parámetro 'rentabilidad_acumulada' debe ser un número.")

        # Filtrar los días de trading (retornos != 0)

        retornos_trading_filtrados = [r for r in retornos_trading if r != 0]

        # Calcular el número de días de trading
        numero_dias_trading = len(retornos_trading_filtrados)

        # Si no hay días de trading, la rentabilidad acumulada anualizada es 0
        if numero_dias_trading == 0:
            return 0

        # Calcular la rentabilidad acumulada anualizada
        rentabilidad_acumulada_anualizada = (1 + rentabilidad_acumulada) ** (365 / numero_dias_trading) - 1

        return rentabilidad_acumulada_anualizada

    except Exception as e:
        print(f"Error: {e}")
        return None
