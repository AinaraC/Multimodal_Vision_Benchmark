import pandas as pd
import matplotlib.pyplot as plt


modelos = ["virconv", "mvxnet", "bevfusion"]

corrupciones = ["darkness", "sunlight", "snow","fog", "rain"]
traducir_c = {"darkness": "noche", "sunlight": "sol intenso", "snow": "nieve","fog": "niebla", "rain": "lluvia"}
traducir_cls = {"Car":"Coche","Pedestrian":"Peatón", "Cyclist": "Ciclista"}

metrica = "3d"

clase = "Cyclist"

dificultad = "moderate"  # SIEMPRE

resultados = []

for modelo in modelos:
    # Leer CSV solo una vez por modelo
    df = pd.read_csv(f"{modelo}/{modelo}_corruption_eval.csv")    

    # Filtrar condiciones
    df_filtrado = df[
        (df["metric"] == metrica) &
        (df["difficulty"] == dificultad) &
        (df["class"] == clase) &
        (df["corruption"].isin(["darkness", "sunlight", "snow","fog", "rain", "original"]))
    ].copy()


    df_filtrado[ "model"] = modelo 
    
    resultados.append(df_filtrado)

# Concatenar todo
df_resultado = pd.concat(resultados, ignore_index=True)

######################## TODO EL RANGO EVOLUCION AP GRAFICA LINEAS ########################
fig, axs = plt.subplots(1, len(corrupciones), figsize=(18, 5), sharey=True)

for idx, corruption in enumerate(corrupciones):
    ax = axs[idx]
    for modelo in modelos:
        # Original (sin corrupción)
        df_original = df_resultado[
            (df_resultado["model"] == modelo) &
            (df_resultado["corruption"] == "original") &
            (df_resultado["severity"] == 0) &
            (df_resultado["min dist"] == 0) &
            (df_resultado["max dist"] == 80) 
        ]["result"]

        # Corrupción específica con severidades 1-5
        df_plot = df_resultado[
            (df_resultado["model"] == modelo) &
            (df_resultado["corruption"] == corruption) &
            (df_resultado["severity"] > 0) &
            (df_resultado["min dist"] == 0) &
            (df_resultado["max dist"] == 80) 
        ].sort_values("severity")["result"]

        # Combinar: primero original, luego las severidades
        valores = pd.concat([df_original.reset_index(drop=True), df_plot.reset_index(drop=True)], ignore_index=True)

        # Crear eje x: 0 (original) + severidades [1, 2, 3, 4, 5]
        x = list(range(len(valores)))

        ax.plot(x, valores, marker="o", label=modelo.upper())
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

    ax.set_title(f"Corrupción: {traducir_c[corruption]}", fontsize = 15)
    ax.set_xlabel("Severidad", fontsize = 15)
    if idx == 0:
        ax.set_ylabel(f"AP{metrica.upper()} {traducir_cls[clase].upper()}", fontsize = 15)
    ax.grid(True)
    ax.legend(loc='lower left', prop={'size': 8})


plt.tight_layout()
plt.savefig(f"AP_{metrica}_{clase}_{dificultad}_(0,80)_severidades.png")  # Guarda la figura como PNG

######################## TODOS LOS RANGOS AR HISTOGRAMA ########################


# Calculo el AR para todos os rangos y mAR
df_AR_total = pd.DataFrame()
rangos = [(0,30), (30,50), (50,80), (0,80)]

for modelo in modelos:
    for rango in rangos:
        df_model = df_resultado[
            (df_resultado["model"] == modelo) &
            (df_resultado["min dist"] == rango[0]) &
            (df_resultado["max dist"] == rango[1])
        ]

        base = df_model[df_model["corruption"] == "original"]["result"]
        # if base.empty or base.iloc[0] == 0:
        #     continue  # Evita errores por división entre 0

        df_AR = {
            "model": modelo,
            "rango": f"{rango[0]}-{rango[1]}"
        }

        for c in corrupciones:
            s1 = df_model[(df_model["corruption"] == c) & (df_model["severity"] == 1)]["result"]
            s2 = df_model[(df_model["corruption"] == c) & (df_model["severity"] == 3)]["result"]
            s3 = df_model[(df_model["corruption"] == c) & (df_model["severity"] == 5)]["result"]

            # if s1.empty or s2.empty or s3.empty:
            #     continue

            # Calcular RA por severidad
            df_AR[f"RA_{traducir_c[c]}_s1"] = s1.iloc[0] / base.iloc[0]
            df_AR[f"RA_{traducir_c[c]}_s2"] = s2.iloc[0] / base.iloc[0]
            df_AR[f"RA_{traducir_c[c]}_s3"] = s3.iloc[0] / base.iloc[0]

            # RA promedio
            df_AR[f"RA_{traducir_c[c]}"] = (
                df_AR[f"RA_{traducir_c[c]}_s1"] +
                df_AR[f"RA_{traducir_c[c]}_s2"] +
                df_AR[f"RA_{traducir_c[c]}_s3"]
            ) / 3

        # Calcular mRA (promedio entre las RA por corrupción)
        ra_cols = [k for k in df_AR if k.startswith("RA_") and "_s" not in k]
        if ra_cols:
            df_AR["mRA"] = pd.Series(df_AR)[ra_cols].mean()
            df_AR_total = pd.concat([df_AR_total, pd.DataFrame([df_AR])], ignore_index=True)
            
df_AR_total.to_csv(f"AR_{metrica}_{clase}_{dificultad}.csv")
df_AR_total.dropna()

# Filtrar columnas RA que no tengan sufijos como _s1, _s2, _s3
ra_columns = [col for col in df_AR_total.columns if col.startswith("RA_") and not any(s in col for s in ['_s1', '_s2', '_s3'])]
df_AR_total[ra_columns] = df_AR_total[ra_columns].apply(pd.to_numeric, errors='coerce').clip(upper=1.0)

# Convertir a formato largo
df_long = df_AR_total.melt(
    id_vars=['model', 'rango'],
    value_vars=ra_columns,
    var_name='Corrupcion',
    value_name='RA'
)

# Limpiar el nombre de las condiciones (quitar el prefijo RA_)
df_long['Corrupcion'] = df_long['Corrupcion'].str.replace('RA_', '', regex=False)

# Obtener los rangos únicos para crear una fila por cada uno
rangos = df_long['rango'].unique()

# Crear subplots
fig, axes = plt.subplots(len(rangos), 1, figsize=(12, 7 * len(rangos)), sharey=True)

# Dibujar los gráficos por cada rango
for i, rango in enumerate(rangos):
    df_rango = df_long[df_long['rango'] == rango]
    
    # Crear tabla para plot: Clima como índice, columnas los modelos
    pivot = df_rango.pivot_table(index='Corrupcion', columns='model', values='RA', aggfunc='mean')
    
    
    # Saltar si no hay datos válidos
    if pivot.empty or pivot.isnull().all().all():
        print(f"[AVISO] Sin datos válidos para el rango: {rango}")
        continue
    
    # Graficar
    pivot.plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'Comparación de RA por Corrupcion - Rango {rango}', fontsize = 15)
    axes[i].set_ylabel(f'RA_{metrica.upper()} {traducir_cls[clase].upper()}', fontsize = 15)
    axes[i].set_xlabel('Corrupcion', fontsize = 15)
    axes[i].grid(True)
    axes[i].legend(title='Modelo',loc='lower left')
    axes[i].tick_params(axis='x', labelsize=15)
    axes[i].tick_params(axis='y', labelsize=15)
    axes[i].legend(loc='lower right', prop={'size': 15})

plt.tight_layout()
plt.savefig(f"RA_{metrica}_{clase}_{dificultad}_por_rango.png")  # Guarda la figura como PNG
