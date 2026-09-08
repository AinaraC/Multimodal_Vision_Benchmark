import pandas as pd
import matplotlib.pyplot as plt

classes = ['Coche', 'Peaton', 'Ciclista']
metrics = ['3D', 'AOS']
difficulty = ['Facil', 'Moderado','Dificil']

modelo = 'bevfusion'

path = f'{modelo}/training'

fig, ax = plt.subplots(2, 3, figsize=(15, 10))

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 16,          # tamaño general (títulos, etiquetas, leyenda)
    'axes.titlesize': 18,     # título del gráfico
    'axes.labelsize': 16,     # etiquetas de ejes
    'xtick.labelsize': 14,    # números del eje X
    'ytick.labelsize': 14,    # números del eje Y
    'legend.fontsize': 8     # leyenda
})

fig.suptitle("BEVFusion", fontsize=20)
for i, cls in enumerate(classes):
    for j, m in enumerate(metrics):
        for d in difficulty:
            df = pd.read_csv(f'{path}/{cls}_{m}_{d}.csv')# o pd.read_excel('archivo.xlsx')
            ax[j,i].plot(df['Step'], df['Value'].ewm(span=10).mean() , label=f'{d}')
        ax[j,i].set_xlabel('Epoch')
        ax[j,i].set_ylabel(f'AP{m}')
        ax[j,i].legend(loc='lower right')
        ax[j,i].set_title(f'{cls}')
        ax[j,i].grid(True)
                         

    
plt.tight_layout() 

plt.savefig(f"{modelo}/{modelo}_training_horizontal.png", dpi=300, bbox_inches='tight', transparent=True)