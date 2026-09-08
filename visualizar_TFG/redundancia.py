import pandas as pd
import matplotlib.pyplot as plt

modelos = ["virconv", "mvxnet", "bevfusion"]
traducir_cls = {"Car":"Coche","Pedestrian":"Peatón", "Cyclist": "Ciclista"}

metricas = ['aos', '3d']
rangos = [(0,30), (30,50), (50,80)]
clases = ['Pedestrian', 'Car', 'Cyclist']

######################## GRAFICA LINEAS NO CAMARA RANGOS ########################


# Crear figura con subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12), sharex=True)

# Recorrer por clase y métrica
for i, clase in enumerate(clases):
    for j, metrica in enumerate(metricas):
        ax = axes[i, j]

        for modelo in modelos:
            df = pd.read_csv(f"{modelo}/{modelo}_corruption_eval.csv")
            df = df[df['difficulty'] == 'moderate']

            rce_vals = []
            etiquetas_rango = []

            for min_d, max_d in rangos:
                # Resultado original
                val_original = df[
                    (df['corruption'] == 'original') &
                    (df['class'] == clase) &
                    (df['metric'] == metrica) &
                    (df['min dist'] == min_d) &
                    (df['max dist'] == max_d) 
                ]['result'].values

                # Resultado con corrupción sin cámara
                val_corrup = df[
                    (df['corruption'] == 'no_camera') &
                    (df['class'] == clase) &
                    (df['metric'] == metrica) &
                    (df['min dist'] == min_d) &
                    (df['max dist'] == max_d) 
                ]['result'].values

                if len(val_original) > 0 and len(val_corrup) > 0 and val_original[0] > 0:
                    rce = 100 * (val_original[0] - val_corrup[0]) / val_original[0]
                    rce_vals.append(rce)
                    etiquetas_rango.append(f"{min_d}-{max_d}")

            if rce_vals:
                ax.plot(etiquetas_rango, rce_vals, marker='o', label=modelo)

        ax.set_title(f'{clase.upper()} - {metrica.upper()}')
        ax.set_ylabel('RCE (%)')
        ax.grid(True)
        if i == 2:
            ax.set_xlabel('Rango de distancia')
        if i == 0 and j == 1:
            ax.legend(title='Modelo', loc='upper right')

plt.tight_layout()
plt.suptitle('RCE por clase, métrica y modelo\nCorrupción: Sin cámara - Dificultad: Moderada', fontsize=16, y=1.03)
plt.subplots_adjust(top=0.92)
plt.savefig('redundancia')



metricas = ['aos', '3d']
rangos = [(0,30), (30,50), (50,80)]
clases = ['Pedestrian', 'Car', 'Cyclist']

# Filtrar dificultad 'moderate'

resultado = []

# Recorrer por clase y métrica
for i, clase in enumerate(clases):
    for j, metrica in enumerate(metricas):
        ax = axes[i, j]

        for modelo in modelos:
            df = pd.read_csv(f"{modelo}/{modelo}_corruption_eval.csv")
            df = df[df['difficulty'] == 'moderate']

            rce_vals = []
            etiquetas_rango = []

            # Resultado original
            val_original = df[
                (df['corruption'] == 'original') &
                (df['class'] == clase) &
                (df['metric'] == metrica) &
                (df['min dist'] == 0) &
                (df['max dist'] == 80) 
            ]['result'].values

            # Resultado con corrupción sin cámara
            val_corrup = df[
                (df['corruption'] == 'no_camera') &
                (df['class'] == clase) &
                (df['metric'] == metrica) &
                (df['min dist'] == 0) &
                (df['max dist'] == 80) 
            ]['result'].values

            if len(val_original) > 0 and len(val_corrup) > 0 and val_original[0] > 0:
                rce = 100 * (val_original[0] - val_corrup[0]) / val_original[0]
                resultado.append({
                    'modelo': modelo,
                    'clase': clase,
                    'metrica': metrica,
                    'rce': rce
                })

df_redundancia = pd.DataFrame(resultado)
df_redundancia.to_csv('redundancia(0,80).csv')