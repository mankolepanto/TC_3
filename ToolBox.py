import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def describe_df(df):
    df_resultado = pd.DataFrame(columns=['DATA_TYPE', 'MISSINGS (%)', 'UNIQUE_VALUES', 'CARDIN (%)'], index=df.columns)
    total = len(df) 
    
    for column in df.columns:

        #Data type
        df_resultado.at[column, 'DATA_TYPE'] = df[column].dtype
        
        # Missings
        missing = df[column].isnull().sum()
        df_resultado.at[column, 'MISSINGS (%)'] = missing/total*100
        
        # Unique values
        df_resultado.at[column, 'UNIQUE_VALUES'] = df[column].nunique()

        # Cardinalidad
        cardinalidad = df[column].nunique()
        porcentaje_cardinalidad = (cardinalidad / total) * 100
        df_resultado.at[column, 'CARDIN (%)'] = round(porcentaje_cardinalidad, 2)

     
    return df_resultado.T

def tipifica_variables(df:pd.DataFrame, umbral_categoria:int, umbral_continua:float):
    """
    Clasifica las columnas de un DataFrame según su tipo sugerido basándose en cardinalidad y porcentaje de cardinalidad.
    Los tipos posibles son Binaria, Categórica, Numérica Continua y Numérica Discreta.

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos cuyas variables se desean tipificar.
    umbral_categoria (int): Umbral de cardinalidad para determinar si una variable es categórica.
    umbral_continua (float): Umbral de porcentaje de cardinalidad para diferenciar entre numérica continua y discreta.

    Retorna:
    pd.DataFrame: DataFrame con dos columnas: 'nombre_variable' que contiene los nombres de las columnas originales,
                  y 'tipo_sugerido' que indica el tipo de dato sugerido para cada columna basado en su cardinalidad y 
                  porcentaje de cardinalidad.
    """
    resultado = pd.DataFrame()
    resultado['nombre_variable']=df.columns
    resultado['tipo_sugerido']=pd.Series()
    
    total_filas = len(df)
    
    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_cardinalidad = (cardinalidad / total_filas) * 100
        
        if cardinalidad == 2:
            tipo_sugerido = 'Binaria'
        elif cardinalidad < umbral_categoria:
            tipo_sugerido = 'Categórica'
        else:
            if porcentaje_cardinalidad >= umbral_continua:
                tipo_sugerido = 'Numérica Continua'
            else:
                tipo_sugerido = 'Numérica Discreta'
        
        resultado.loc[resultado['nombre_variable'] == col, 'tipo_sugerido'] = tipo_sugerido
    
    return resultado


def get_features_num_regression(dataframe, target_col, umbral_corr, pvalue=None):
    # Comprobación de que dataframe es un DataFrame de pandas
    if not isinstance(dataframe, pd.DataFrame):
        print("Error: El argumento 'dataframe' debe ser un DataFrame de pandas.")
        return None
    
    # Comprobación de que target_col es una columna válida en dataframe
    if target_col not in dataframe.columns:
        print("Error: El argumento 'target_col' no es una columna válida en el DataFrame.")
        return None
    
    # Comprobación de que target_col es numérica
    if not np.issubdtype(dataframe[target_col].dtype, np.number):
        print("Error: El argumento 'target_col' debe ser una variable numérica continua.")
        return None
    
    # Comprobación de que umbral_corr está entre 0 y 1
    if not 0 <= umbral_corr <= 1:
        print("Error: El argumento 'umbral_corr' debe estar entre 0 y 1.")
        return None
    
    # Comprobación de que pvalue, si se proporciona, es un float entre 0 y 1
    if pvalue is not None:
        if not isinstance(pvalue, float) or not 0 <= pvalue <= 1:
            print("Error: El argumento 'pvalue' debe ser un número flotante entre 0 y 1.")
            return None
    
    # Calcular las correlaciones entre todas las características y target_col
    correlations = dataframe.corr()[target_col].abs()
    
    # Seleccionar las características con correlación superior a umbral_corr
    selected_features = correlations[correlations > umbral_corr].index.tolist()
    
    # Si se proporciona un pvalue, realizar un test de hipótesis para cada característica seleccionada
    if pvalue is not None:
        p_values = []
        for feature in selected_features:
            # Calcular el p-value utilizando una prueba de correlación de Pearson
            p_val = pearsonr(dataframe[feature], dataframe[target_col])[1]
            # Si el p-value es menor o igual a 1-pvalue, conservar la característica
            if p_val <= 1 - pvalue:
                p_values.append(p_val)
            else:
                selected_features.remove(feature)
        
        print("P-values:", p_values)
    
    return selected_features




def get_features_cat_regression(df:pd.DataFrame, target_col:str, pvalue:float=0.05):
    """
    Identifica columnas categóricas en un DataFrame que están significativamente relacionadas con
    una columna objetivo numérica, utilizando el análisis de varianza (ANOVA).

    Argumentos:
    df (pd.DataFrame): DataFrame que contiene los datos a analizar.
    target_col (str): Nombre de la columna objetivo que debe ser numérica.
    pvalue (float, opcional): Umbral de p-valor para considerar significativas las columnas categóricas.

    Retorna:
    List[str]: Lista de nombres de columnas categóricas que tienen una relación estadísticamente significativa
               con la columna objetivo, basada en un p-valor menor o igual al umbral especificado.
    """
    # Verificar que p-valor es una probabilidad
    if not 0 < pvalue < 1:
        print("Error: El p-valor debe estar en [0,1].")
        return None
    # Verificar que target_col está en el DataFrame
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no se encuentra en el DataFrame.")
        return None

    # Verificar que target_col es numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: La columna '{target_col}' debe ser numérica.")
        return None
    cardinalidad = df[target_col].nunique()
    total_filas = len(df)
    porcentaje_cardinalidad = (cardinalidad / total_filas) * 100

    # Verificar que target_col tiene cardinalidad suficiente para ser continua o discreta
    if porcentaje_cardinalidad<=10:
        print(f"Error: La columna '{target_col}' no tiene suficiente cardinalidad para ser considerada númerica continua o discreta.")
        return None

    
    # Lista para guardar las columnas categóricas significativas
    significant_cats = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        grouped = df.groupby(col, observed=True)[target_col].apply(list)
        _, p_val = stats.f_oneway(*grouped)
        if p_val <= pvalue:
            significant_cats.append(col)

    return significant_cats


def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):

#Comprobaciones necesarias para no dar error como consecuencia de los valores de entrada

    # Si la lista está vacía, entonces la función igualará "columns" a las variables numéricas del dataframe y se comportará como se describe en el párrafo anterior.
    if len(columns) == 0:
        columns = df.select_dtypes(include=['float', 'int']).columns.tolist() #modo generico
        #columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)'] #modo df_iris
  
    # Comprobar si las columnas existen en el dataframe
    for col in columns:
        if col not in df.columns:
            print(f"Error: La columna '{col}' no existe en el dataframe.")
            return None

    # Si target_col está vacío, igualar a la primera columna numérica continua del dataframe
    if target_col == "":
        target_col = df.select_dtypes(include=['float', 'int']).columns[0]
        #target_col = ['sepal length (cm)'] #modo df_iris

    # Comprobar si umbral_corr es un número válido entre 0 y 1
    if not 0 <= umbral_corr <= 1:
        print("Error: 'umbral_corr' debe ser un número entre 0 y 1.")
        return None

    # Comprobar si pvalue es un número válido entre 0 y 1 
    if pvalue is not None:
        if pvalue > 1 or pvalue < 0:
            print("Error: 'pvalue' debe ser un número entre 0 y 1.")
            return None

    # Comprobar si target_col es una columna numérica continua del dataframe
        if tipifica_variables(df, 0, umbral_corr) != 'Numérica Continua':
            print("Error: 'target_col' debe ser una variable numérica continua del dataframe.")
            return None


#[...] la función pintará una pairplot del dataframe considerando la columna designada por "target_col" y aquellas incluidas en "column" que 
#cumplan que su correlación con "target_col" es superior en valor absoluto a "umbral_corr", y que, en el caso de ser pvalue diferente de "None", además cumplan el 
#test de correlación para el nivel 1-pvalue de significación estadística. La función devolverá los valores de "columns" que cumplan con las condiciones anteriores.

    # Crear una lista para almacenar las columnas que cumplen con las condiciones
    valid_columns = []
    
    # Filtrar las columnas que cumplen con el umbral de correlación y el test de p-value si se proporciona
    for col in columns:
        if col != target_col:
            correlation, p_value = pearsonr(df[target_col], df[col])
            if abs(correlation) > umbral_corr and (pvalue is None or p_value < pvalue):
                valid_columns.append(col)
    
    # Si no hay columnas válidas, imprimir mensaje y retornar None
    if len(valid_columns) == 0:
        print("No hay columnas que cumplan los criterios especificados.")
        return None
    
    # Dividir las columnas válidas en grupos de máximo 5 para plotear
    for i in range(0, len(valid_columns), 5):
        subset_columns = valid_columns[i:i+5]
        subset_columns.append(target_col)  # Asegurar que la columna target también está incluida
        sns.pairplot(df[subset_columns], diag_kind='kde', kind='reg', height=3)
        plt.show()
    
    return valid_columns



def plot_features_cat_regression(dataframe, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    # Comprobación de que dataframe es un DataFrame de pandas
    if not isinstance(dataframe, pd.DataFrame):
        print("Error: El argumento 'dataframe' debe ser un DataFrame de pandas.")
        return None
    
    # Comprobación de que target_col es una columna válida en dataframe
    if target_col != "" and target_col not in dataframe.columns:
        print("Error: El argumento 'target_col' no es una columna válida en el DataFrame.")
        return None
    
    # Comprobación de que columns es una lista de strings
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        print("Error: El argumento 'columns' debe ser una lista de strings.")
        return None
    
    # Comprobación de que pvalue es un float
    if not isinstance(pvalue, float):
        print("Error: El argumento 'pvalue' debe ser un número flotante.")
        return None
    
    # Comprobación de que pvalue está entre 0 y 1
    if not 0 <= pvalue <= 1:
        print("Error: El argumento 'pvalue' debe estar entre 0 y 1.")
        return None
    
    # Comprobación de que with_individual_plot es booleano
    if not isinstance(with_individual_plot, bool):
        print("Error: El argumento 'with_individual_plot' debe ser booleano.")
        return None
    
    # Si la lista de columnas está vacía, selecciona todas las columnas categóricas del DataFrame
    if len(columns) == 0:
        columns = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Almacenará las columnas que cumplan con las condiciones
    selected_columns = []
    
    # Iterar sobre las columnas categóricas seleccionadas
    for col in columns:
        # Calcular el p-value utilizando una prueba de chi-cuadrado
        contingency_table = pd.crosstab(dataframe[col], dataframe[target_col])
        _, p_val, _, _ = chi2_contingency(contingency_table)
        
        # Si el p-valor es menor que 1-pvalue,  se hace histograma agrupado
        if p_val < 1 - pvalue:
            selected_columns.append(col)
            if with_individual_plot:
                sns.histplot(data=dataframe, x=col, hue=target_col, multiple="stack")
                plt.title(f"{col} vs {target_col}")
                plt.show()
            else:
                sns.histplot(data=dataframe, x=col, hue=target_col, multiple="stack", stat="count", common_norm=False)
                plt.title(f"{col} vs {target_col}")
                plt.show()
    
    return selected_columns



'''ToolBox II'''


def eval_model(target, predictions, problem_type, metrics):
    """
    Evalúa el rendimiento de un modelo de Machine Learning según las métricas especificadas.
    
    Argumentos:
    target: Lista de valores reales del target.
    predictions: Lista de valores predichos por el modelo.
    problem_type (str): Tipo de problema ('regression' o 'classification').
    metrics: Lista de métricas a calcular.
    
    Retorna:
    tuple: Tupla con los valores de las métricas en el orden de aparición en la lista de entrada.
    """
    results = []
    
    # Funciones auxiliares para las métricas de regresión
    def calculate_rmse(target, predictions):
        return np.sqrt(mean_squared_error(target, predictions))
    
    def calculate_mae(target, predictions):
        return mean_absolute_error(target, predictions)
    
    def calculate_mape(target, predictions):
        if np.any(target == 0):
            raise ValueError("MAPE no se puede calcular con valores de target igual a 0")
        return np.mean(np.abs((target - predictions) / target)) * 100
    
    # Funciones auxiliares para métricas de clasificación
    def calculate_accuracy(target, predictions):
        return accuracy_score(target, predictions)
    
    def calculate_precision(target, predictions, average='macro'):
        return precision_score(target, predictions, average=average)
    
    def calculate_recall(target, predictions, average='macro'):
        return recall_score(target, predictions, average=average)
    
    # Procesamiento de métricas según el tipo de problema
    if problem_type == 'regression':
        for metric in metrics:
            if metric == 'RMSE':
                rmse = calculate_rmse(target, predictions)
                print(f"RMSE: {rmse}")
                results.append(rmse)
            elif metric == 'MAE':
                mae = calculate_mae(target, predictions)
                print(f"MAE: {mae}")
                results.append(mae)
            elif metric == 'MAPE':
                try:
                    mape = calculate_mape(target, predictions)
                    print(f"MAPE: {mape}")
                    results.append(mape)
                except ValueError as e:
                    print(e)
            elif metric == 'GRAPH':
                plt.scatter(target, predictions)
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('Actual vs Predicted')
                plt.show()
    
    elif problem_type == 'classification':
        for metric in metrics:
            if metric == 'ACCURACY':
                accuracy = calculate_accuracy(target, predictions)
                print(f"Accuracy: {accuracy}")
                results.append(accuracy)
            elif metric == 'PRECISION':
                precision = calculate_precision(target, predictions)
                print(f"Precision: {precision}")
                results.append(precision)
            elif metric == 'RECALL':
                recall = calculate_recall(target, predictions)
                print(f"Recall: {recall}")
                results.append(recall)
            elif metric == 'CLASS_REPORT':
                report = classification_report(target, predictions)
                print("Classification Report:\n", report)
            elif metric == 'MATRIX':
                matrix = confusion_matrix(target, predictions)
                display = ConfusionMatrixDisplay(confusion_matrix=matrix)
                display.plot()
                plt.show()
            elif metric == 'MATRIX_RECALL':
                display = ConfusionMatrixDisplay.from_predictions(target, predictions, normalize='true')
                display.plot()
                plt.show()
            elif metric == 'MATRIX_PRED':
                display = ConfusionMatrixDisplay.from_predictions(target, predictions, normalize='pred')
                display.plot()
                plt.show()
            elif 'PRECISION_' in metric:
                class_label = metric.split('_')[1]
                precision = precision_score(target, predictions, labels=[class_label], average='macro', zero_division=0)
                if precision == 0:
                    raise ValueError(f"Etiqueta {class_label} no encontrada en target")
                print(f"Precision for class {class_label}: {precision}")
                results.append(precision)
            elif 'RECALL_' in metric:
                class_label = metric.split('_')[1]
                recall = recall_score(target, predictions, labels=[class_label], average='macro', zero_division=0)
                if recall == 0:
                    raise ValueError(f"Etiqueta {class_label} no encontrada en target")
                print(f"Recall for class {class_label}: {recall}")
                results.append(recall)
    
    return tuple(results)



def get_features_num_classification(df, target_col, pvalue=0.05):
    """
    Selecciona columnas numéricas cuyo ANOVA con la columna target supere un nivel de significación.
    
    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    target_col (str): Nombre de la columna target.
    pvalue (float): Nivel de significación para el test de ANOVA. Por defecto 0.05.
    
    Retorna:
    list: Lista de columnas numéricas que cumplen con el criterio de ANOVA.
    """
    # Comprobaciones de los argumentos de entrada
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'df' no es un DataFrame.")
        return None
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no existe en el DataFrame.")
        return None
    if not pd.api.types.is_categorical_dtype(df[target_col]) and not pd.api.types.is_object_dtype(df[target_col]):
        print(f"La columna '{target_col}' no es categórica.")
        return None
    if not isinstance(pvalue, float) or not (0 < pvalue < 1):
        print("El argumento 'pvalue' debe ser un float entre 0 y 1.")
        return None
    
    # Convertir target_col a categórica si no lo es
    if not pd.api.types.is_categorical_dtype(df[target_col]):
        df[target_col] = df[target_col].astype('category')
    
    # Filtrar columnas numéricas
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    significant_features = []
    
    # Realizar el test ANOVA
    for col in numeric_cols:
        groups = [df[df[target_col] == cat][col].dropna() for cat in df[target_col].cat.categories]
        if all(len(group) > 0 for group in groups):
            _, p_val = f_oneway(*groups)
            if p_val < (1 - pvalue):
                significant_features.append(col)
    
    return significant_features

