"""Todas las funciones y catálogos reutilizables del proyecto (caso Clearview Properties).

Los scripts (`eda_exploracion.py`, `preparacion_matriz_modelo.py`,
`feature_selection_genetico.py`) son bitácoras de lectura lineal: qué se hizo, qué se
encontró, en qué orden. Toda la lógica reutilizable — limpieza, catálogos de códigos válidos,
filtros de variables, operadores del algoritmo genético — vive acá, para no mezclar ambas
cosas. Los scripts solo importan y llaman.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ============================================================================
# Constantes y catálogos (Data_Dictionary.txt + hallazgos de calidad de datos)
# ============================================================================

# Valor de SalePrice que aparece una sola vez con forma de placeholder de "sin dato"
# (9,999,999), no de precio real de una vivienda en Cedar Falls.
CENTINELA_SOSPECHOSO = 9_999_999

# Mismo patrón de centinela, encontrado en Garage Cars al preparar la matriz de modelado:
# nadie tiene un garaje de 99 autos. Aparece en 2 filas.
SENTINELA_GARAGE_CARS = 99

# Columnas donde Data_Dictionary.txt dice explícitamente que NaN == "no tiene esa
# característica" (no es un dato faltante real).
NAN_ESTRUCTURAL = {
    "Pool QC": "no tiene piscina",
    "Misc Feature": "no tiene característica adicional",
    "Alley": "no tiene acceso por callejón",
    "Fence": "no tiene cerca",
    "Fireplace Qu": "no tiene chimenea",
    "Mas Vnr Type": "no tiene revestimiento de mampostería",
    "Garage Type": "no tiene garaje",
    "Garage Yr Blt": "no tiene garaje",
    "Garage Finish": "no tiene garaje",
    "Garage Qual": "no tiene garaje",
    "Garage Cond": "no tiene garaje",
    "Bsmt Qual": "no tiene sótano",
    "Bsmt Cond": "no tiene sótano",
    "Bsmt Exposure": "no tiene sótano",
    "BsmtFin Type 1": "no tiene sótano",
    "BsmtFin Type 2": "no tiene sótano",
}

# Códigos válidos por columna categórica, transcritos de Data_Dictionary.txt.
CODIGOS_VALIDOS = {
    "MS Zoning": ["A (agr)", "C (all)", "FV", "I (all)", "RH", "RL", "RM"],
    "Street": ["Grvl", "Pave"],
    "Alley": ["Grvl", "Pave"],
    "Lot Shape": ["IR1", "IR2", "IR3", "Reg"],
    "Land Contour": ["Bnk", "HLS", "Low", "Lvl"],
    "Utilities": ["AllPub", "NoSeWa", "NoSewr"],
    "Lot Config": ["Corner", "CulDSac", "FR2", "FR3", "Inside"],
    "Land Slope": ["Gtl", "Mod", "Sev"],
    "Neighborhood": [
        "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor",
        "Edwards", "Gilbert", "Greens", "GrnHill", "IDOTRR", "Landmrk", "MeadowV",
        "Mitchel", "NAmes", "NPkVill", "NWAmes", "NoRidge", "NridgHt", "OldTown",
        "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
    ],
    "Condition 1": ["Artery", "Feedr", "Norm", "PosA", "PosN", "RRAe", "RRAn", "RRNe", "RRNn"],
    "Condition 2": ["Artery", "Feedr", "Norm", "PosA", "PosN", "RRAe", "RRAn", "RRNn"],
    "Bldg Type": ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"],
    "House Style": ["1.5Fin", "1.5Unf", "1Story", "2.5Fin", "2.5Unf", "2Story", "SFoyer", "SLvl"],
    "Roof Style": ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"],
    "Roof Matl": ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"],
    "Exterior 1st": [
        "AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard",
        "ImStucc", "MetalSd", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd",
        "Wd Sdng", "WdShing",
    ],
    "Exterior 2nd": [
        "AsbShng", "AsphShn", "Brk Cmn", "BrkFace", "CBlock", "CmentBd", "HdBoard",
        "ImStucc", "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco",
        "VinylSd", "Wd Sdng", "Wd Shng",
    ],
    "Mas Vnr Type": ["BrkCmn", "BrkFace", "CBlock", "Stone"],
    "Exter Qual": ["Ex", "Fa", "Gd", "TA"],
    "Exter Cond": ["Ex", "Fa", "Gd", "Po", "TA"],
    "Foundation": ["BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"],
    "Bsmt Qual": ["Ex", "Fa", "Gd", "Po", "TA"],
    "Bsmt Cond": ["Ex", "Fa", "Gd", "Po", "TA"],
    "Bsmt Exposure": ["Av", "Gd", "Mn", "No"],
    "BsmtFin Type 1": ["ALQ", "BLQ", "GLQ", "LwQ", "Rec", "Unf"],
    "BsmtFin Type 2": ["ALQ", "BLQ", "GLQ", "LwQ", "Rec", "Unf"],
    "Heating": ["Floor", "GasA", "GasW", "Grav", "OthW", "Wall"],
    "Heating QC": ["Ex", "Fa", "Gd", "Po", "TA"],
    "Central Air": ["N", "Y"],
    "Electrical": ["FuseA", "FuseF", "FuseP", "Mix", "SBrkr"],
    "Kitchen Qual": ["Ex", "Fa", "Gd", "Po", "TA"],
    "Functional": ["Maj1", "Maj2", "Min1", "Min2", "Mod", "Sal", "Sev", "Typ"],
    "Fireplace Qu": ["Ex", "Fa", "Gd", "Po", "TA"],
    "Garage Type": ["2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd"],
    "Garage Finish": ["Fin", "RFn", "Unf"],
    "Garage Qual": ["Ex", "Fa", "Gd", "Po", "TA"],
    "Garage Cond": ["Ex", "Fa", "Gd", "Po", "TA"],
    "Paved Drive": ["N", "P", "Y"],
    "Pool QC": ["Ex", "Fa", "Gd", "TA"],
    "Fence": ["GdPrv", "GdWo", "MnPrv", "MnWw"],
    "Misc Feature": ["Elev", "Gar2", "Othr", "Shed", "TenC"],
    "Sale Type": ["COD", "CWD", "Con", "ConLD", "ConLI", "ConLw", "New", "Oth", "VWD", "WD"],
    "Sale Condition": ["Abnorml", "AdjLand", "Alloca", "Family", "Normal", "Partial"],
}


# ============================================================================
# Limpieza y auditoría de calidad (usadas por eda_exploracion.py y
# preparacion_matriz_modelo.py)
# ============================================================================

def limpiar_numero_texto(serie: pd.Series) -> pd.Series:
    """Convierte a numérico columnas que llegaron como texto con $, comas o unidades (' sqft')."""
    s = serie.astype(str).str.strip()
    s = s.str.replace(r"[\$,]", "", regex=True)
    s = s.str.replace(r"\s*sqft\s*$", "", regex=True, case=False)
    s = s.replace({"nan": np.nan, "": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


def auditar_categoricas(data: pd.DataFrame, mapa_codigos: dict) -> pd.DataFrame:
    """Compara cada columna categórica contra sus códigos válidos (Data_Dictionary.txt).

    Separa dos problemas distintos: variantes corregibles (mismo valor, ruido de mayúsculas
    o espacios) de valores realmente inválidos (sin equivalente en el diccionario, p. ej.
    placeholders como "??" o "Unknown").
    """
    filas = []
    for col, validos in mapa_codigos.items():
        if col not in data.columns:
            continue
        validos_norm = {v.casefold(): v for v in validos}
        crudos = data[col].dropna().astype(str)
        normalizados = crudos.str.strip()
        corregibles = normalizados[
            normalizados.str.casefold().isin(validos_norm) & ~normalizados.isin(validos)
        ]
        invalidos = normalizados[~normalizados.str.casefold().isin(validos_norm)]
        if len(corregibles) or len(invalidos):
            filas.append(
                {
                    "columna": col,
                    "variantes_corregibles": sorted(corregibles.unique().tolist()),
                    "n_corregibles": len(corregibles),
                    "valores_invalidos": sorted(invalidos.unique().tolist()),
                    "n_invalidos": len(invalidos),
                }
            )
    return pd.DataFrame(filas)


def normalizar_categoricas(data: pd.DataFrame, mapa_codigos: dict) -> pd.DataFrame:
    """Aplica la corrección que `auditar_categoricas` solo diagnostica.

    Usa exactamente la misma regla de matching (casefold + strip): las variantes de
    formato (" RL", "rl", "Rl") se reescriben a su forma canónica ("RL"); los valores sin
    equivalente válido (p. ej. "??", "Unknown", "paved") se convierten en NaN real, porque
    no hay forma honesta de adivinar cuál era el valor correcto.
    """
    data = data.copy()
    for col, validos in mapa_codigos.items():
        if col not in data.columns:
            continue
        canon_por_casefold = {v.casefold(): v for v in validos}
        es_nulo = data[col].isnull()
        normalizado = data[col].astype(str).str.strip()
        mapeado = normalizado.str.casefold().map(canon_por_casefold)
        data[col] = mapeado.where(~es_nulo, data[col])
    return data


def seleccionar_columnas_para_log1p(
    data: pd.DataFrame,
    columnas: list,
    umbral_skew: float = 0.75,
    umbral_aceptable: float = 1.0,
) -> tuple[list, list, pd.DataFrame]:
    """Decide con evidencia (no de memoria) qué columnas se benefician de log1p.

    Muchas variables de este dataset están dominadas por ceros (Pool Area, Misc Val,
    Low Qual Fin SF, etc.): tienen skew crudo alto, pero log1p NO las arregla — a veces las
    empeora — porque la masa en cero no se puede volver simétrica con una transformación
    monótona. Aplicar log1p "porque el skew es alto" sin verificar el resultado es un error.

    Reglas: una columna es candidata si abs(skew crudo) > umbral_skew; se acepta aplicarle
    log1p solo si el skew resultante mejora Y queda por debajo de umbral_aceptable. Si no,
    se dejan sin transformar y se marcan como candidatas a un indicador binario en una
    iteración futura (p. ej. "Tiene_Piscina"), no a log1p.

    Devuelve (columnas_a_transformar, columnas_zero_inflated_sin_arreglo, reporte).
    """
    filas = []
    for col in columnas:
        serie = data[col].dropna()
        if serie.empty:
            continue
        skew_crudo = serie.skew()
        if abs(skew_crudo) <= umbral_skew:
            continue
        skew_log1p = np.log1p(serie.clip(lower=0)).skew()
        mejora = abs(skew_crudo) - abs(skew_log1p)
        aplica = bool(mejora > 0 and abs(skew_log1p) < umbral_aceptable)
        filas.append(
            {
                "columna": col,
                "skew_crudo": round(skew_crudo, 3),
                "skew_log1p": round(skew_log1p, 3),
                "mejora": round(mejora, 3),
                "aplicar_log1p": aplica,
            }
        )
    reporte = pd.DataFrame(filas).sort_values("skew_crudo", ascending=False, key=abs)
    a_transformar = reporte.loc[reporte["aplicar_log1p"], "columna"].tolist()
    zero_inflated_sin_arreglo = reporte.loc[~reporte["aplicar_log1p"], "columna"].tolist()
    return a_transformar, zero_inflated_sin_arreglo, reporte


# ============================================================================
# Filtros de preselección de variables (usados por feature_selection_genetico.py)
# ============================================================================

def filtrar_varianza_casi_nula(X: pd.DataFrame, umbral: float = 0.01) -> tuple[list, pd.DataFrame]:
    """Descarta columnas binarias (dummies/indicadores) casi constantes.

    Una dummy presente en <umbral (o >1-umbral) de las filas no tiene suficientes
    observaciones del lado minoritario para aportar señal generalizable — solo infla el
    espacio de búsqueda. Devuelve (columnas_descartadas, X_filtrado).
    """
    es_binaria = X.isin([0, 1]).all()
    prop_unos = X.loc[:, es_binaria].mean()
    descartadas = prop_unos[(prop_unos < umbral) | (prop_unos > 1 - umbral)].index.tolist()
    return descartadas, X.drop(columns=descartadas)


def rankear_por_correlacion(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Correlación de Pearson absoluta de cada columna de X con el target, descendente.

    Funciona igual para numéricas y para dummies 0/1 (ahí es la correlación punto-biserial,
    matemáticamente el mismo cálculo que un Pearson normal).
    """
    return X.apply(lambda c: np.corrcoef(c, y)[0, 1]).abs().sort_values(ascending=False)


def podar_multicolinealidad(
    X: pd.DataFrame, candidatas_ordenadas: list, umbral: float = 0.85
) -> tuple[list, list]:
    """Poda greedy de redundancia: recorre `candidatas_ordenadas` (se espera de mayor a menor
    relevancia con el target) y descarta cualquiera que corr. > `umbral` con una ya elegida.

    Devuelve (columnas_finales, podadas) donde `podadas` es una lista de tuplas
    (columna_descartada, columna_sobreviviente_redundante, correlación).
    """
    seleccionadas: list = []
    podadas: list = []
    for col in candidatas_ordenadas:
        redundante_con = None
        for ya_elegida in seleccionadas:
            r = abs(X[col].corr(X[ya_elegida]))
            if r > umbral:
                redundante_con = (ya_elegida, round(r, 2))
                break
        if redundante_con is None:
            seleccionadas.append(col)
        else:
            podadas.append((col, redundante_con[0], redundante_con[1]))
    return seleccionadas, podadas


# ============================================================================
# Algoritmo genético de feature selection (usado por feature_selection_genetico.py)
# ============================================================================

def mae_precio_score(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Scorer en dólares de precio (no en escala log) para RandomizedSearchCV/make_scorer.

    Mismo criterio de negocio que `evaluar_individuo`: no se optimiza en una escala que
    nadie en el negocio lee directamente.
    """
    return -float(np.mean(np.abs(np.expm1(y_pred_log) - np.expm1(y_true_log))))


def reemplazar_bsmt_full_bath_por_tot_bath(columnas: list) -> list:
    """Ajuste de sentido de negocio sobre el resultado del genético: prioriza `Tot_Bath`
    (baños totales equivalentes) sobre `Bsmt Full Bath` (solo baños del sótano).

    `Bsmt Full Bath` y `Tot_Bath` correlacionan apenas 0.60 entre sí — por debajo del umbral
    de poda de multicolinealidad (0.85) — así que ninguna elimina automáticamente a la otra en
    los filtros, y el genético puede seleccionar las DOS en una misma corrida. Un simple
    "reemplazar el nombre" en ese caso duplicaría `Tot_Bath` en la lista (columna repetida,
    error de XGBoost al fittear) — por eso se maneja explícito: si ya están las dos, se
    descarta `Bsmt Full Bath` sin duplicar; si solo está `Bsmt Full Bath`, se renombra.
    """
    if "Bsmt Full Bath" not in columnas:
        return list(columnas)
    if "Tot_Bath" in columnas:
        return [v for v in columnas if v != "Bsmt Full Bath"]
    return [v if v != "Bsmt Full Bath" else "Tot_Bath" for v in columnas]


def evaluar_individuo(
    mascara: np.ndarray,
    X_train: pd.DataFrame,
    y_log_train: np.ndarray,
    y_price_train: np.ndarray,
    modelo_params: dict,
    cv_folds: int,
    peso_penalizacion_tamano: float,
    mae_referencia: float,
    random_state: int,
) -> dict:
    """Fitness de un subconjunto de variables: `(1 - R²) + MAE/MAE_referencia`, más una
    penalización liviana por tamaño — las tres, sin unidades, en escala 0-1 comparable.

    No se usa MAPE: tiene un sesgo asimétrico documentado en la literatura (penaliza más
    fuerte cuando el modelo sobre-predice que cuando sub-predice — Makridakis; Hyndman &
    Koehler, 2006), lo que sesgaría al genético hacia subconjuntos que sub-estiman el precio
    sin que eso sea deseable.

    Tampoco se usa RMSE junto a R²: verificado con datos reales de este proyecto, R² y RMSE
    correlacionan a -0.999 sobre el mismo holdout (`R² = 1 - MSE/Var(y)`, y `Var(y)` es
    constante para un conjunto fijo) — sumarlos no aporta una segunda mirada, duplica el peso
    de la misma. RMSE se puede seguir reportando aparte si hace falta, pero no entra al fitness.

    `MAE` sí aporta una mirada genuinamente distinta (correlación alta pero no perfecta con
    R², -0.97 verificado): es más robusta a outliers y es la métrica que ya se usa para
    comunicar el error al negocio en dólares. Se normaliza dividiendo por `mae_referencia`
    (el MAE de predecir el precio promedio para todos, ~$57,861 en este dataset) para que
    quede en la misma escala 0-1 que `(1 - R²)`, sin mezclar dólares con un número
    adimensional.
    """
    n_sel = int(mascara.sum())
    n_features_total = X_train.shape[1]
    if n_sel < 5:
        # individuo degenerado (casi sin variables): fitness muy malo, no se entrena nada
        return {"r2": -10.0, "rmse": np.inf, "mape": 1.0, "mae": np.inf, "n_features": n_sel, "fitness": 10.0}

    cols = X_train.columns[mascara]
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    r2s, rmses, mapes, maes = [], [], [], []
    for idx_tr, idx_val in kf.split(X_train):
        modelo = XGBRegressor(**modelo_params)
        modelo.fit(X_train.iloc[idx_tr][cols], y_log_train[idx_tr])
        pred_precio = np.expm1(modelo.predict(X_train.iloc[idx_val][cols]))
        real_precio = y_price_train[idx_val]
        r2s.append(r2_score(real_precio, pred_precio))
        rmses.append(np.sqrt(np.mean((pred_precio - real_precio) ** 2)))
        mapes.append(np.mean(np.abs(pred_precio - real_precio) / real_precio))
        maes.append(np.mean(np.abs(pred_precio - real_precio)))

    r2_prom = float(np.mean(r2s))
    rmse_prom = float(np.mean(rmses))
    mape_prom = float(np.mean(mapes))
    mae_prom = float(np.mean(maes))
    penalizacion = peso_penalizacion_tamano * (n_sel / n_features_total)
    fitness = (1 - r2_prom) + (mae_prom / mae_referencia) + penalizacion
    return {
        "r2": r2_prom,
        "rmse": rmse_prom,
        "mape": mape_prom,
        "mae": mae_prom,
        "n_features": n_sel,
        "fitness": fitness,
    }


def reparar(individuo: np.ndarray, tope: int, rng: np.random.Generator) -> np.ndarray:
    """Fuerza el tope duro de variables activas, apagando genes al azar si se excede."""
    activos = np.flatnonzero(individuo)
    if len(activos) > tope:
        apagar = rng.choice(activos, size=len(activos) - tope, replace=False)
        individuo = individuo.copy()
        individuo[apagar] = False
    return individuo


def poblacion_inicial(
    tam_poblacion: int, n_features: int, tope: int, rng: np.random.Generator, prob_inicial: float = 0.3
) -> np.ndarray:
    poblacion = rng.random((tam_poblacion, n_features)) < prob_inicial
    return np.array([reparar(ind, tope, rng) for ind in poblacion])


def seleccion_torneo(poblacion: np.ndarray, fitness: np.ndarray, rng: np.random.Generator, k: int = 3) -> np.ndarray:
    """Selección por torneo: funciona igual para minimizar que para maximizar (no necesita
    que el fitness sea "más grande es mejor")."""
    idx_torneo = rng.integers(0, len(poblacion), size=k)
    ganador = idx_torneo[np.argmin(fitness[idx_torneo])]
    return poblacion[ganador]


def cruce_uniforme(padre_a: np.ndarray, padre_b: np.ndarray, n_features: int, rng: np.random.Generator):
    """Cada gen se hereda de un padre u otro al azar. No hay un orden espacial significativo
    entre columnas, así que un cruce de un punto no tendría sentido aquí."""
    mascara_cruce = rng.random(n_features) < 0.5
    hijo1 = np.where(mascara_cruce, padre_a, padre_b)
    hijo2 = np.where(mascara_cruce, padre_b, padre_a)
    return hijo1, hijo2


def mutar(individuo: np.ndarray, prob_mutacion: float, n_features: int, rng: np.random.Generator) -> np.ndarray:
    mascara_mutacion = rng.random(n_features) < prob_mutacion
    return np.where(mascara_mutacion, ~individuo, individuo)


# ============================================================================
# Segmentación (PCA + KMeans, usado por segmentacion_pca_kmeans.py)
# ============================================================================

def barrer_k_optimo(X_pca: np.ndarray, k_min: int, k_max: int, random_state: int) -> pd.DataFrame:
    """Corre KMeans para cada K en [k_min, k_max] y devuelve inercia (codo) y silueta.

    No decide el K por sí sola — devuelve el reporte para que la decisión quede documentada
    con los dos criterios lado a lado, no con uno solo.
    """
    filas = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        etiquetas = modelo.fit_predict(X_pca)
        filas.append(
            {
                "k": k,
                "inercia": modelo.inertia_,
                "silueta": silhouette_score(X_pca, etiquetas),
            }
        )
    return pd.DataFrame(filas)


def pca_kmeans_silueta(
    df: pd.DataFrame, columnas: list, n_componentes: int, k_min: int, k_max: int, random_state: int
) -> pd.DataFrame:
    """Estandariza `df[columnas]`, reduce a `n_componentes` con PCA, y barre K con
    `barrer_k_optimo`.

    Empaqueta el patrón repetido de "probar un subconjunto de variables con PCA+KMeans" que
    aparece varias veces en la comparación de candidatos de segmentación, para no repetir el
    mismo pipeline de tres líneas a mano en cada comparación.
    """
    X_z = StandardScaler().fit_transform(df[columnas])
    componentes = PCA(n_components=n_componentes, random_state=random_state).fit_transform(X_z)
    return barrer_k_optimo(componentes, k_min, k_max, random_state)


def perfil_segmentos(
    X_original: pd.DataFrame, etiquetas: np.ndarray, y_price: np.ndarray
) -> pd.DataFrame:
    """Describe cada segmento en las variables originales (no en componentes PCA) + precio.

    El precio no se usa para formar los clusters (evita el circularismo de "clusters = tiers
    de precio por construcción"), pero sí se reporta acá como validación: si los segmentos
    tienen perfiles de precio distintos, es evidencia de que la segmentación encontró algo
    con sentido de negocio, no solo ruido.
    """
    perfil = X_original.copy()
    perfil["_segmento"] = etiquetas
    perfil["_precio"] = y_price
    resumen = perfil.groupby("_segmento").mean(numeric_only=True)
    resumen["n_propiedades"] = perfil.groupby("_segmento").size()
    resumen = resumen.rename(columns={"_precio": "precio_promedio"})
    return resumen.sort_values("precio_promedio", ascending=False)


# ============================================================================
# Q1 — Rango de listado e intervalos de incertidumbre (Clearview Properties)
# ============================================================================

def calcular_multiplicadores_residuos(
    y_price_real: np.ndarray, y_price_pred: np.ndarray, p_bajo: int = 10, p_alto: int = 90
) -> tuple[float, float, np.ndarray]:
    """Calcula los multiplicadores empíricos de incertidumbre a partir de los residuos en escala log.

    Parámetros
    ----------
    y_price_real : np.ndarray
        Precios reales de venta (en USD) del conjunto de prueba (holdout).
    y_price_pred : np.ndarray
        Precios predichos (en USD) por el modelo sobre el conjunto de prueba.
    p_bajo : int, default=10
        Percentil inferior para el piso de negociación.
    p_alto : int, default=90
        Percentil superior para el techo de salida optimista.

    Retorna
    -------
    tuple[float, float, np.ndarray]
        (mult_bajo, mult_alto, residuos_log)
    """
    residuos_log = np.log1p(y_price_real) - np.log1p(y_price_pred)
    p_inf, p_sup = np.percentile(residuos_log, [p_bajo, p_alto])
    mult_bajo = float(np.exp(p_inf))
    mult_alto = float(np.exp(p_sup))
    return mult_bajo, mult_alto, residuos_log


def calcular_rango_listado(
    precio_estimado_usd: float | np.ndarray, mult_bajo: float, mult_alto: float
) -> dict:
    """Genera la estrategia de precios en 3 niveles (Piso, Mercado, Techo) para un bróker.

    Parámetros
    ----------
    precio_estimado_usd : float o np.ndarray
        Precio estimado por el modelo final (en USD).
    mult_bajo : float
        Multiplicador inferior (ej. 0.881 para -11.9%).
    mult_alto : float
        Multiplicador superior (ej. 1.168 para +16.8%).

    Retorna
    -------
    dict
        Diccionario con 'piso_negociacion', 'precio_mercado' y 'salida_optimista' redondeados a centenas.
    """
    return {
        "piso_negociacion": np.round(precio_estimado_usd * mult_bajo, -2),
        "precio_mercado": np.round(precio_estimado_usd, -2),
        "salida_optimista": np.round(precio_estimado_usd * mult_alto, -2),
    }


def graficar_banda_cotizacion(
    y_price_real: np.ndarray,
    y_price_pred: np.ndarray,
    residuos_log: np.ndarray,
    mult_bajo: float,
    mult_alto: float,
    figsize: tuple = (14, 5),
):
    """Genera la visualización ejecutiva en dos paneles de la banda de cotización y los residuos."""
    import matplotlib.pyplot as plt

    p10_log = np.log(mult_bajo)
    p90_log = np.log(mult_alto)

    orden = np.argsort(y_price_pred)
    x_vals = y_price_pred[orden]
    y_vals = y_price_real[orden]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Histograma de residuos
    axes[0].hist(residuos_log, bins=40, color="#4C72B0", alpha=0.7, edgecolor="white")
    axes[0].axvline(p10_log, color="#C44E52", linestyle="--", linewidth=2, label=f"P10 ({mult_bajo-1:+.1%})")
    axes[0].axvline(p90_log, color="#55A868", linestyle="--", linewidth=2, label=f"P90 ({mult_alto-1:+.1%})")
    axes[0].axvspan(p10_log, p90_log, color="#4C72B0", alpha=0.15, label="80% Central del Mercado")
    axes[0].set_title("1. Margen de Incertidumbre Real (Holdout)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Error Relativo (log)")
    axes[0].set_ylabel("Cantidad de Propiedades")
    axes[0].legend()

    # Panel 2: Banda de negociación
    axes[1].scatter(x_vals, y_vals, alpha=0.35, s=15, color="#4C72B0", label="Ventas Reales")
    axes[1].plot(x_vals, x_vals, color="black", linestyle="--", linewidth=1.5, label="Estimación Central (100%)")
    axes[1].plot(x_vals, x_vals * mult_alto, color="#55A868", linewidth=1.8, label=f"Techo Salida ({mult_alto-1:+.1%})")
    axes[1].plot(x_vals, x_vals * mult_bajo, color="#C44E52", linewidth=1.8, label=f"Piso Negociación ({mult_bajo-1:+.1%})")
    axes[1].fill_between(x_vals, x_vals * mult_bajo, x_vals * mult_alto, color="#55A868", alpha=0.08)
    axes[1].set_title("2. Banda de Precios Recomendada para Brókers", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Precio Estimado por el Modelo (USD)")
    axes[1].set_ylabel("Precio Real de Venta (USD)")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.show()


# ============================================================================
# Q2 — Explicabilidad y descomposición SHAP (Estructural vs Modificable)
# ============================================================================

def calcular_ranking_shap(
    modelo, X: pd.DataFrame, clasificacion_tipos: dict
) -> tuple[pd.DataFrame, np.ndarray]:
    """Calcula la importancia SHAP media y etiqueta el tipo de variable (Estructural vs Modificable).

    Parámetros
    ----------
    modelo : estimator
        Modelo ajustado (ej. XGBRegressor).
    X : pd.DataFrame
        Matriz de características (las 12 variables seleccionadas).
    clasificacion_tipos : dict
        Diccionario que mapea cada nombre de variable con su tipo ('Estructural', 'Modificable', etc.).

    Retorna
    -------
    tuple[pd.DataFrame, np.ndarray]
        (df_ranking_ordenado, shap_valores_completos)
    """
    import shap

    explainer = shap.TreeExplainer(modelo)
    shap_valores = explainer.shap_values(X)
    shap_media = np.abs(shap_valores).mean(axis=0)

    df_ranking = pd.DataFrame(
        {
            "Variable": X.columns,
            "Impacto_SHAP": shap_media,
            "Tipo": [clasificacion_tipos.get(col, "No definido") for col in X.columns],
        }
    ).sort_values("Impacto_SHAP", ascending=False).reset_index(drop=True)

    return df_ranking, shap_valores


def graficar_ranking_shap(df_ranking: pd.DataFrame, figsize: tuple = (10, 5)):
    """Genera un gráfico de barras horizontal comparando el impacto SHAP entre Estructurales y Modificables."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    colores = [
        "#4C72B0" if "Estructural" in str(t) else "#55A868"
        for t in df_ranking["Tipo"]
    ]

    ax.barh(
        df_ranking["Variable"][::-1],
        df_ranking["Impacto_SHAP"][::-1],
        color=colores[::-1],
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_xlabel("Impacto Promedio Absoluto |SHAP| (escala log precio)", fontsize=10)
    ax.set_title("Q2: Importancia de Variables — Estructurales (Azul) vs. Modificables (Verde)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()
