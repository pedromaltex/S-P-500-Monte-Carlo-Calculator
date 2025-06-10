import matplotlib.pyplot as plt
import numpy as np

def improve_draw():
    # Melhorando visualmente
    plt.xticks(rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

def plot1(x, y, name):
    # Plotando os gráficos
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Points', linestyle='solid', color='black')
    plt.title(f'{name} Historical Data', fontsize=20)
    plt.xticks(rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

def plot2(x, y1, y2, name, coef_log1, coef_log2):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y1, label='Exponential Growth', linestyle='dashdot', color='red')
    plt.plot(x, y2, label= name, linestyle='solid', color='black')
    plt.title(f'Exponential vs {name} (Log Scale)', fontsize=20)
    x_pos = x.iloc[-10]
    y_pos = min(y1) * 1.05  # um pouco abaixo do topo
    plt.text(x_pos, y_pos, rf'$y ={{{coef_log1:.4f} + {coef_log2:.4f} \cdot x}}$ (Exponential Growth)',
             fontsize=20,
             ha='right', va='bottom',
             color='red',
             bbox=dict(facecolor='white', alpha=0.6))
    plt.xticks(rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

def plot3(x, y1, y2, name, coef_log1, coef_log2):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y1, label='Exponential Growth', linestyle='dashdot', color='red')
    plt.plot(x, y2, label=name, linestyle='solid', color='black')
    plt.title(f'Exponential vs {name}', fontsize=20)

    x_pos = x.iloc[-10]
    y_pos = min(y2) * 1.05  # um pouco abaixo do topo
    plt.text(x_pos,y_pos , rf'$y = e^{{{coef_log1:.4f} + {coef_log2:.4f} \cdot x}}$ (Exponential Growth)',
            ha='right', va='bottom',
            color='red',
            fontsize=20,

            bbox=dict(facecolor='white', alpha=0.6))

    improve_draw()

def plot4(x, y, name):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label=f'{name}', linestyle='solid', color='black')
    plt.title(f'Exponential vs {name} (Diference)', fontsize=20)
    improve_draw()

def plot5(x, y1, y2):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y1, label='Standart Investment', linestyle='solid', color='red')
    plt.plot(x, y2, label="Weighted Investment", linestyle='dotted', color='blue')
    plt.title("Standart Investment vs Weighted Investment", fontsize=20)
    improve_draw()

def plot6(x, y1, y2):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y1, label='Standart Investment (Allocation)', linestyle='solid', color='red')
    plt.plot(x, y2, label="Weighted Investment (Allocation)", linestyle='dotted', color='blue')
    plt.title("Allocation", fontsize=20)
    improve_draw()

def plot7(df, x, y):
    # Plot com datas reais no eixo X
    plt.figure(figsize=(12, 6))
    plt.plot(df, alpha=0.6)
    plt.plot(x, y, label='Exponential Growth', linestyle='dashdot', color='black')
    plt.title("Monte Carlo Simulation - Geometric Brownian Motion (with dates)", fontsize=20)
    plt.xlabel("Data")
    plt.ylabel("Preço simulado")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot8(f1, f2):
    # Define a largura dos bins
    bin_width = f2.max()/100

    # Define os limites globais
    min_val = min(f1.min(), f2.min())
    max_val = max(f1.max(), f2.max())

    # Gera os bins com mesma largura
    bins = np.arange(np.floor(min_val), np.ceil(max_val) + bin_width, bin_width)

    plt.figure(figsize=(10, 6))

    # Histograma do primeiro portfólio
    plt.hist(f1, bins=bins, edgecolor='black', color='skyblue', alpha=1, label='Buy and Hold')

    # Histograma do segundo portfólio
    plt.hist(f2, bins=bins, edgecolor='black', color='red', alpha=0.5, label="Weighted Investment")

    # Plot mean lines
    plt.axvline(np.mean(f1), color='blue', linestyle='dashed', linewidth=2, label=f'Mean Buy & Hold: {np.mean(f1):.2f}')
    plt.axvline(np.mean(f2), color='darkred', linestyle='dashed', linewidth=2, label=f"Mean Weighted Investment: {np.mean(f2):.2f}")

    plt.title('Final values of portfolio (distribution)', fontsize=15)
    plt.xlabel('Value of portfolio (€)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot9(tot_alloc, fin_alloc):
    plt.figure(figsize=(10, 6))


    plt.hist(fin_alloc, bins=30, edgecolor='black', color='skyblue', alpha=1, label='Buy and Hold')

    plt.title('Total allocation in Weighted Investment', fontsize=15)
    plt.xlabel('Final allocation (€)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot10(h1, h2, bins):
    plt.figure(figsize=(10, 6))
    # Histograma do primeiro portfólio
    plt.hist(h1, bins=bins, edgecolor='black', color='skyblue', alpha=1, label='Buy and Hold')
    # Histograma do segundo portfólio
    plt.hist(h2, bins=bins, edgecolor='black', color='red', alpha=0.5, label="Weighted Investment")

    # Plot mean lines
    plt.axvline(np.mean(h1), color='blue', linestyle='dashed', linewidth=2, label=f'Mean Buy & Hold: {np.mean(h1):.2f}%')
    plt.axvline(np.mean(h2), color='darkred', linestyle='dashed', linewidth=2, label=f"Mean Weighted Investment: {np.mean(h2):.2f}%")

    plt.title('ROI (Return over investment)', fontsize=15)
    plt.xlabel('ROI (Return over investment) (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()













