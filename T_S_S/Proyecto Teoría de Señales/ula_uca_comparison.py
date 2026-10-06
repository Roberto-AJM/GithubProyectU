import numpy as np
import matplotlib.pyplot as plt

# Usar el backend 'Agg' para entorno sin servidor gráfico
import matplotlib
matplotlib.use('Agg')

def generar_array_linear(M, d_over_lambda, theta_deg):
    """
    Genera el Steering Vector para un Arreglo Lineal Uniforme (ULA)
    M: Número de antenas
    d_over_lambda: Relación d/lambda (típicamente 0.5)
    theta_deg: Ángulo de llegada de la señal en grados
    """
    theta_rad = np.radians(theta_deg)
    # Ecuación de retraso relativo para ULA: m * d * sin(theta)
    # a(theta) = exp(-j * 2 * pi * (d/lambda) * m * sin(theta))
    m = np.arange(M)
    steering_vector = np.exp(-1j * 2 * np.pi * d_over_lambda * m * np.sin(theta_rad))
    return steering_vector

def generar_array_circular(M, r_over_lambda, theta_deg):
    """
    Genera el Steering Vector para un Arreglo Circular Uniforme (UCA)
    M: Número de antenas
    r_over_lambda: Radio del círculo en términos de lambda (r/lambda)
    theta_deg: Ángulo de llegada de la señal en grados (azimut)
    """
    theta_rad = np.radians(theta_deg)
    # Ángulos de posición física de cada antena en el círculo (de 0 a 2*pi)
    phi_m = np.linspace(0, 2 * np.pi, M, endpoint=False)
    # Ecuación de retraso relativo para UCA: r * cos(theta - phi_m)
    # a(theta) = exp(j * 2 * pi * (r/lambda) * cos(theta - phi_m))
    steering_vector = np.exp(1j * 2 * np.pi * r_over_lambda * np.cos(theta_rad - phi_m))
    return steering_vector

# --- SIMULACIÓN Y COMPARACIÓN DE PATRONES DE RADIACIÓN (BEAMPATTERN) ---
# Frecuencia de portadora para control de drones: 2.4 GHz
fc = 2.4e9
c = 3e8
lambd = c / fc

# Parámetros del Arreglo
M = 8  # 8 antenas
d_over_lambda = 0.5  # d = lambda / 2 para evitar alias espacial

# Para la UCA, seleccionamos el radio 'r' para que la distancia de arco entre
# antenas adyacentes sea aproximadamente lambda/2.
# Arco = r * (2*pi/M) = lambda/2 => r/lambda = M / (4*pi)
r_over_lambda = M / (4 * np.pi)

# Ángulo real del transmisor (controlador del dron)
theta_real = 30.0  # El transmisor está a 30 grados

# Generar los vectores de direccionamiento reales (la señal física que llega)
a_ula_real = generar_array_linear(M, d_over_lambda, theta_real)
a_uca_real = generar_array_circular(M, r_over_lambda, theta_real)

# Escaneo del espacio angular (-180 a 180 grados para ver el círculo completo)
angulos_escaneo = np.linspace(-180, 180, 360)
patron_ula = []
patron_uca = []

for angulo in angulos_escaneo:
    # 1. Caso ULA
    a_ula_scan = generar_array_linear(M, d_over_lambda, angulo)
    # El patrón de radiación es la correlación (proyección) entre el vector real y el de escaneo
    potencia_ula = np.abs(np.dot(np.conj(a_ula_scan), a_ula_real))**2 / M**2
    patron_ula.append(potencia_ula)
    
    # 2. Caso UCA
    a_uca_scan = generar_array_circular(M, r_over_lambda, angulo)
    potencia_uca = np.abs(np.dot(np.conj(a_uca_scan), a_uca_real))**2 / M**2
    patron_uca.append(potencia_uca)

# Convertir a decibelios (dB) con límite inferior de -30 dB para visualización limpia
patron_ula_db = 10 * np.log10(np.array(patron_ula) + 1e-10)
patron_uca_db = 10 * np.log10(np.array(patron_uca) + 1e-10)
patron_ula_db = np.maximum(patron_ula_db, -30)
patron_uca_db = np.maximum(patron_uca_db, -30)

# --- GRAFICAR EN COORDENADAS POLARES ---
fig = plt.figure(figsize=(12, 6))

# Subplot 1: Patrón de ULA
ax1 = fig.add_subplot(121, projection='polar')
ax1.plot(np.radians(angulos_escaneo), patron_ula_db, color='blue', linewidth=2, label='Patrón ULA')
ax1.plot(np.radians(theta_real), 0, 'ro', markersize=8, label='Tx Real (30°)')
ax1.set_theta_zero_location('N')  # El norte es 0 grados
ax1.set_theta_direction(-1)       # Sentido horario
ax1.set_rmax(0)
ax1.set_rmin(-25)
ax1.set_title("Arreglo Lineal Uniforme (ULA)\nNota la simetría/ambigüedad (haz espejo a 150°)", va='bottom', fontsize=11)
ax1.legend(loc='lower left')

# Subplot 2: Patrón de UCA
ax2 = fig.add_subplot(122, projection='polar')
ax2.plot(np.radians(angulos_escaneo), patron_uca_db, color='green', linewidth=2, label='Patrón UCA')
ax2.plot(np.radians(theta_real), 0, 'ro', markersize=8, label='Tx Real (30°)')
ax2.set_theta_zero_location('N')
ax2.set_theta_direction(-1)
ax2.set_rmax(0)
ax2.set_rmin(-25)
ax2.set_title("Arreglo Circular Uniforme (UCA)\nEscaneo de 360° sin ambigüedad trasera", va='bottom', fontsize=11)
ax2.legend(loc='lower left')

plt.tight_layout()
plt.savefig('/workspace/scratch/ula_uca_comparison.png', dpi=150, bbox_inches='tight')
print("[EXITOSO] Gráfico generado de comparación ULA vs UCA")
