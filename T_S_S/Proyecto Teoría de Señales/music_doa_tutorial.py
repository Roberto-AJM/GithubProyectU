# -*- coding: utf-8 -*-
"""
Tutorial Educativo: Dirección de Llegada (DOA) mediante el Algoritmo MUSIC
Este script es una guía interactiva y progresiva para comprender los conceptos
de Procesamiento de Señales y Arreglos de Antenas, específicamente la estimación
de Ángulo de Llegada (AoA) o Dirección de Llegada (DOA) usando el algoritmo MUSIC.

Desarrollado de manera didáctica con explicaciones detalladas paso a paso.
"""

import numpy as np
import matplotlib.pyplot as plt

# Usar estilo Agg para evitar problemas de visualización en el sandbox
import matplotlib
matplotlib.use('Agg')

def generar_señal_ula(M, d_over_lambda, angulos_deg, snr_db, L):
    """
    Simula las señales recibidas por un Arreglo Lineal Uniforme (ULA).
    
    Parámetros:
    - M: Número de elementos/antenas en el arreglo.
    - d_over_lambda: Relación d/lambda (distancia entre elementos / longitud de onda).
    - angulos_deg: Lista o array con los ángulos de llegada verdaderos (en grados).
    - snr_db: Relación Señal-Ruido (SNR) en decibelios.
    - L: Número de snapshots o muestras temporales.
    
    Retorna:
    - Y: Matriz de datos recibidos de tamaño (M, L).
    """
    angulos_rad = np.radians(angulos_deg)
    num_fuentes = len(angulos_deg)
    
    # 1. Generar señales de fuentes independientes como procesos gaussianos complejos
    # s(t) de tamaño (num_fuentes, L)
    S = (np.random.randn(num_fuentes, L) + 1j * np.random.randn(num_fuentes, L)) / np.sqrt(2)
    
    # 2. Construir la Matriz de Direccionamiento (Steering Matrix) 'A'
    # Cada columna es el Steering Vector 'a(theta)' de una fuente.
    # a(theta) = [1, e^(-j*2*pi*(d/lambda)*sin(theta)), ..., e^(-j*2*pi*(M-1)*(d/lambda)*sin(theta))]^T
    A = np.zeros((M, num_fuentes), dtype=complex)
    for i, theta in enumerate(angulos_rad):
        # Fase relativa para cada elemento m de 0 a M-1
        fases = -2 * np.pi * d_over_lambda * np.arange(M) * np.sin(theta)
        A[:, i] = np.exp(1j * fases)
        
    # 3. Señal recibida sin ruido: X = A * S
    X = np.dot(A, S)
    
    # 4. Agregar Ruido Blanco Gaussiano Aditivo Complejo (AWGN)
    # Potencia de la señal promedio por canal
    potencia_señal = np.mean(np.abs(X)**2)
    potencia_ruido = potencia_señal / (10**(snr_db / 10.0))
    
    # Generar ruido complejo con la potencia deseada
    ruido = (np.random.randn(M, L) + 1j * np.random.randn(M, L)) * np.sqrt(potencia_ruido / 2)
    
    # Señal recibida final: Y = A*S + N
    Y = X + ruido
    return Y

def algoritmo_music(Y, p, d_over_lambda, angulos_escaneo_deg):
    """
    Implementa el Algoritmo MUSIC (MUltiple SIgnal Classification) estándar.
    
    Parámetros:
    - Y: Matriz de datos recibidos (M, L).
    - p: Número de fuentes de señal incidentes estimadas o conocidas.
    - d_over_lambda: Relación d/lambda de diseño.
    - angulos_escaneo_deg: Array de ángulos sobre los cuales escanear el espectro.
    
    Retorna:
    - espectro_db: Espectro de MUSIC normalizado en dB.
    """
    M, L = Y.shape
    
    # 1. Calcular la Matriz de Covarianza Espacial Empírica (Ryy)
    # Ryy = (1/L) * Y * Y^H
    Ryy = np.dot(Y, np.conj(Y).T) / L
    
    # 2. Realizar la Descomposición en Autovalores (Eigendecomposition)
    # Como Ryy es Hermítica, usamos eigh para mayor estabilidad numérica y autovalores reales.
    autovalores, autovectores = np.linalg.eigh(Ryy)
    
    # Ordenar los autovalores de menor a mayor
    indices_ordenados = np.argsort(autovalores)
    autovectores_ordenados = autovectores[:, indices_ordenados]
    
    # 3. Separar el Subespacio de Ruido (Un)
    # Los primeros M - p autovectores corresponden a los autovalores más pequeños (ruido)
    Un = autovectores_ordenados[:, :M - p]
    
    # 4. Escaneo del Pseudo-espectro MUSIC
    espectro = np.zeros(len(angulos_escaneo_deg))
    
    # Pre-proyectar el subespacio de ruido para acelerar cálculo: Un * Un^H
    Un_UnH = np.dot(Un, np.conj(Un).T)
    
    for i, theta_deg in enumerate(angulos_escaneo_deg):
        theta_rad = np.radians(theta_deg)
        # Generar el Steering Vector 'a(theta)' para el ángulo de escaneo actual
        fases = -2 * np.pi * d_over_lambda * np.arange(M) * np.sin(theta_rad)
        a = np.exp(1j * fases)
        
        # Calcular el denominador: a(theta)^H * Un * Un^H * a(theta)
        # Debe ser cercano a cero si theta coincide con una DOA real debido a la ortogonalidad.
        denominador = np.real(np.dot(np.conj(a).T, np.dot(Un_UnH, a)))
        
        # Pseudo-espectro MUSIC es el recíproco
        espectro[i] = 1.0 / denominador
        
    # Normalizar el espectro y pasarlo a escala logarítmica (dB)
    espectro_normalizado = espectro / np.max(espectro)
    espectro_db = 10 * np.log10(espectro_normalizado)
    
    return espectro_db

# =============================================================================
# EXPERIMENTOS xPROGRESIVOS Y VISUALIZACIÓN
# =============================================================================

# Definir parámetros base del sistema
d_over_lambda = 0.5  # Espaciado ideal de media longitud de onda para evitar alias espacial
angulos_verdaderos = [-25.0, 15.0]  # Dos fuentes que inciden a -25 y 15 grados
p = len(angulos_verdaderos)
angulos_escaneo = np.linspace(-90, 90, 361)  # Resolución de 0.5 grados

# Crear figura con 3 subplots para mostrar el análisis de manera progresiva
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=False)
fig.suptitle('Estudio Progresivo de la Estimación de Dirección de Llegada (DOA) usando MUSIC', 
             fontsize=16, fontweight='bold', y=0.96)

# -----------------------------------------------------------------------------
# Experimento 1: Caso Base - ¿Cómo funciona MUSIC bajo condiciones estándar?
# -----------------------------------------------------------------------------
M_base = 8
L_base = 200
snr_base = 10.0  # SNR moderada (10 dB)

Y_base = generar_señal_ula(M_base, d_over_lambda, angulos_verdaderos, snr_base, L_base)
espectro_base = algoritmo_music(Y_base, p, d_over_lambda, angulos_escaneo)

ax1.plot(angulos_escaneo, espectro_base, color='blue', linewidth=2, label='Espectro MUSIC')
# Graficar líneas verticales para los ángulos verdaderos
for angulo in angulos_verdaderos:
    ax1.axvline(x=angulo, color='red', linestyle='--', linewidth=1.5, 
                label=f'DOA Real: {angulo}°' if angulo == angulos_verdaderos[0] else "")
    
ax1.set_title(f'Experimento 1: Espectro MUSIC Base ($M={M_base}$ antenas, $SNR={snr_base}$ dB, $L={L_base}$ muestras)', 
             fontsize=12, fontweight='bold')
ax1.set_ylabel('Amplitud Normalizada (dB)', fontsize=11)
ax1.set_xlabel('Ángulo de Incidencia (grados)', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower center', ncol=2)
ax1.set_ylim([-40, 2])

# -----------------------------------------------------------------------------
# Experimento 2: Influencia de la Relación Señal-Ruido (SNR)
# -----------------------------------------------------------------------------
# Compararemos el desempeño de MUSIC con SNR alta (20 dB), baja (0 dB) y extremadamente baja (-10 dB)
snrs_comparar = [20.0, 0.0, -10.0]
colores_snr = ['darkgreen', 'orange', 'purple']

for snr, color in zip(snrs_comparar, colores_snr):
    Y_snr = generar_señal_ula(M_base, d_over_lambda, angulos_verdaderos, snr, L_base)
    espectro_snr = algoritmo_music(Y_snr, p, d_over_lambda, angulos_escaneo)
    ax2.plot(angulos_escaneo, espectro_snr, color=color, linewidth=1.8, 
             label=f'SNR = {snr} dB')

for angulo in angulos_verdaderos:
    ax2.axvline(x=angulo, color='red', linestyle='--', linewidth=1.5)

ax2.set_title('Experimento 2: Efecto del Ruido (SNR) en la Resolución de Picos', 
             fontsize=12, fontweight='bold')
ax2.set_ylabel('Amplitud Normalizada (dB)', fontsize=11)
ax2.set_xlabel('Ángulo de Incidencia (grados)', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower center', ncol=3)
ax2.set_ylim([-40, 2])

# -----------------------------------------------------------------------------
# Experimento 3: Influencia del Número de Elementos del Arreglo (M)
# -----------------------------------------------------------------------------
# Compararemos arreglos pequeños (M=4), medianos (M=8) y grandes (M=16) a una SNR fija
M_comparar = [4, 8, 16]
colores_M = ['brown', 'blue', 'magenta']
snr_fijo = 5.0  # SNR de 5 dB (un poco ruidosa)

for M_val, color in zip(M_comparar, colores_M):
    Y_M = generar_señal_ula(M_val, d_over_lambda, angulos_verdaderos, snr_fijo, L_base)
    espectro_M = algoritmo_music(Y_M, p, d_over_lambda, angulos_escaneo)
    ax3.plot(angulos_escaneo, espectro_M, color=color, linewidth=1.8, 
             label=f'{M_val} antenas (M={M_val})')

for angulo in angulos_verdaderos:
    ax3.axvline(x=angulo, color='red', linestyle='--', linewidth=1.5)

ax3.set_title(f'Experimento 3: Efecto del Número de Antennas ($M$) en la Resolución Angular (SNR = {snr_fijo} dB)', 
             fontsize=12, fontweight='bold')
ax3.set_ylabel('Amplitud Normalizada (dB)', fontsize=11)
ax3.set_xlabel('Ángulo de Incidencia (grados)', fontsize=11)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='lower center', ncol=3)
ax3.set_ylim([-40, 2])

# Ajustar diseño de la figura
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Guardar la figura resultante en el directorio de scratch
plot_path = 'C:\\Users\\Rober\\OneDrive\\Documents\\piton\\project_TSS\\music_spectrum.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"[EXITOSO] El script se ejecutó correctamente. Se guardó la visualización en {plot_path}")
