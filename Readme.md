

<div style="text-align:center;">
    <h1 style="border-radius: 10px; padding: 10px; width: 50%; margin: 0 auto; background-color: lightgray; text-transform: uppercase;">TEAM CHALLENGE</h1>
    <img src="./Imagenes/portadaa.png" alt="Portada" style="border-radius: 25px; width: 50%;">
</div>


### Contenido del Repositorio:

-Hemos creado una caja de herramientas para hacer mas sencilla la léctura de los datos que nos encontremos en un futuro. Las funciones a las que me refiero las podremos encontrar en `ToolBox.py`.

## Funciones y explicación de las mismas:

> **Describe**

>> Esta función nos sirve para obtener información util de un df como el tipado de sus columnas, los missings, los valores únicos y la cardinalidad.

> **Tipifica_Variables**

>> Clasifica las columnas de un DataFrame según su tipo sugerido basándose en cardinalidad y porcentaje de cardinalidad. Los tipos posibles son Binaria, Categórica, Numérica Continua y Numérica Discreta.

> **Get_features_num_regression**

>> Devuelve una lista de columnas numéricas del df seleccionado cuya correlación con el target supere la correlación deseada.

> **Get_features_cat_regression**

>> Identifica columnas categóricas en un DataFrame que están significativamente relacionadas con una columna objetivo numérica, utilizando el análisis de varianza (ANOVA).

> **Plot_features_num_regression**

>> La función pintará una pairplot del dataframe considerando la columna designada por "target" y aquellas incluidas en "column" que cumplan que su correlación con "target_col" es superior en valor absoluto a "umbral_corr" y que en el caso de ser pvalue diferente de "None", además cumplan el test de correlación para el nivel 1-pvalue de significación estadística. La función devolverá los valores de "columns" que cumplan con las condiciones anteriores.

> **Plot_features_num_regression**

>> Esta función plot_features_cat_regression toma un DataFrame, una columna objetivo (target_col), y una lista de columnas categóricas (columns). Utiliza pruebas de chi-cuadrado para evaluar la relación entre cada columna categórica y la columna objetivo. Si la relación es significativa (según un valor pvalue dado), muestra un histograma agrupado para visualizar la relación. La función devuelve las columnas categóricas que cumplen con los criterios de significancia.

## Data

> Columnas de nuestro DataFrame

>> Col: Colesterol
>> Gluc-Basal: Glucosa en reposo
>> HDL: High Density Lipoprotein	
>> Glyhb: Hemoglobina Glucosilada
>> Ciudad: Dos ciudades, Lousia (Kentucky) y Buckingham (Buckinghamshire)

>> Edad: 

>> Sexo: Hombre / Mujer

>> Peso:

>> bp.1s: Presión arterial sistólica.

>> bp.1d: Presión arterial diastólica.


## TEST

- Se va a utilizar un Dataset obtenido de Kaggle en el que podemos encontrar varios marcadores de salud. Con este dataset vamos a mostrar el buen funcionamiento de nuestras funciones y podremos observar el desarrollo de las mismas.



