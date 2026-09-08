import matplotlib.pyplot as plt
import pandas as pd


fig, axs = plt.subplots(4, 2, figsize=(20, 20))  # 2 filas, 3 columnas
axs = axs.flatten()  # para indexar como una lista

fig.delaxes(axs[7])  # elimina el subplot vacío

modelos = {
    'MVX-Net': pd.read_csv('mvxnet/mvxnet_orin.csv').iloc[50:],
    'BEVFusion': pd.read_csv('bevfusion/bevfusion_orin.csv').iloc[50:],
    'VirConv': pd.read_csv('virconv/virconv_orin.csv').iloc[25:]
}

for clave, df in modelos.items():
    df['elapsed_time'] = df['elapsed_time'] - df['elapsed_time'].iloc[0]
    # CPU Usage (%)
    avg_cpu = 0
    for i in range (1,8):
        avg_cpu += df[f'cpu_{i}']
        avg_cpu /= 8
    df['cpu_avg'] = avg_cpu

# GPU Usage (%)
for nombre, df in modelos.items():
    axs[0].plot(df['elapsed_time'], df['gpu_usage'], label=nombre)
axs[0].set_title("Uso GPU (%)")
axs[0].set_xlabel("Tiempo (s)")
axs[0].set_ylabel("Uso (%)")
axs[0].legend()
axs[0].grid(True)

for nombre, df in modelos.items():
    axs[1].plot(df['elapsed_time'], df['cpu_avg'], label=nombre)
axs[1].set_title("Uso CPU (%)")
axs[1].set_xlabel("Tiempo (s)")
axs[1].set_ylabel("Uso (%)")
axs[1].legend()
axs[1].grid(True)

# Temp GPU (°C)
for nombre, df in modelos.items():
    axs[2].plot(df['elapsed_time'], df['gpu_temp'], label=nombre)
axs[2].set_title("Temperatura GPU (°C)")
axs[2].set_xlabel("Tiempo (s)")
axs[2].set_ylabel("Temperatura °C")
axs[2].legend()
axs[2].grid(True)

# Temp CPU (°C)
for nombre, df in modelos.items():
    axs[3].plot(df['elapsed_time'], df['cpu_temp'], label=nombre)
axs[3].set_title("Temperatura CPU (°C)")
axs[3].set_xlabel("Tiempo (s)")
axs[3].set_ylabel("°C")
axs[3].legend()
axs[3].grid(True)

# Consumo total (mW → W)
for nombre, df in modelos.items():
    axs[4].plot(df['elapsed_time'], df['power_tot'] / 1000, label=nombre)
axs[4].set_title("Consumo total (W)")
axs[4].set_xlabel("Tiempo (s)")
axs[4].set_ylabel("Watts")
axs[4].legend()
axs[4].grid(True)

# Consumo modelo (W)
for nombre, df in modelos.items():
    axs[5].plot(df['elapsed_time'], df['power_model'] / 1000, label=nombre)
axs[5].set_title("Consumo GPU+CPU+MV (W)")
axs[5].set_xlabel("Tiempo (s)")
axs[5].set_ylabel("Watts")
axs[5].legend()
axs[5].grid(True)

# RAM (% sobre total)
for nombre, df in modelos.items():
    axs[6].plot(df['elapsed_time'], df['RAM'] * 100, label=nombre)
axs[6].set_title("Uso RAM (%)")
axs[6].set_xlabel("Tiempo (s)")
axs[6].set_ylabel("RAM (%)")
axs[6].legend()
axs[6].grid(True)

plt.savefig('Orin/todos_orin')


# Calcular métricas por modelo
resumen = []

for nombre, df in modelos.items():
    prom_gpu = df['gpu_usage'].mean()
    prom_cpu = df['cpu_avg'].mean()
    prom_ram = (df['RAM'] * 100).mean()
    prom_power_tot = (df['power_tot'] / 1000).mean()
    prom_power_model = (df['power_model'] / 1000).mean()

    delta_gpu_temp = df['gpu_temp'].max() - df['gpu_temp'].iloc[0]
    delta_cpu_temp = df['cpu_temp'].max() - df['cpu_temp'].iloc[0]
    delta_ram = (df['RAM'].max() - df['RAM'].iloc[0]) * 100

    resumen.append({
        'Modelo': nombre,
        'Prom. GPU (%)': round(prom_gpu, 2),
        'Prom. CPU (%)': round(prom_cpu, 2),
        'Prom. RAM (%)': round(prom_ram, 2),
        'Incremento Temp. GPU (°C)': round(delta_gpu_temp, 2),
        'Incremento Temp. CPU (°C)': round(delta_cpu_temp, 2),
        'Incremento RAM (%)': round(delta_ram, 2),
        'Consumo total prom. (W)': round(prom_power_tot, 2),
        'Consumo modelo prom. (W)': round(prom_power_model, 2),
    })

# Guardar resumen en CSV
df_resumen = pd.DataFrame(resumen)
df_resumen.to_csv('Orin/resumen_orin.csv', index=False)

print(df_resumen)