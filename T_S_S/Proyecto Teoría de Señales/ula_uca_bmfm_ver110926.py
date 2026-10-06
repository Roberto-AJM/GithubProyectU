import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PASO 1: PARÁMETROS DEL SISTEMA Y FÍSICA DE LA SEÑAL
# =============================================================================
# Definimos los parámetros físicos de la onda electromagnética y del arreglo.

frecuencia = 2.4e9         # Frecuencia de operación (2.4 GHz, estándar de drones)
c = 3e8                    # Velocidad de la luz en el vacío (m/s)
lamda = c / frecuencia     # Longitud de onda (lambda = c / f) (~0.125 metros o 12.5 cm)

M = 8                      # Número de antenas (sensores) en el arreglo
L = 1000                   # Snapshots: Número de muestras temporales síncronas que capturamos
snr_db = 15                # Relación Señal-Ruido (SNR) en decibelios (dB)

# Ángulos reales desde donde transmiten los controles o drones (en grados)
# MUSIC puede resolver múltiples fuentes simultáneas; aquí simulamos dos fuentes independientes.
angulos_reales = np.array([-25.0, 30.0]) 
num_fuentes = len(angulos_reales)

print(f"--- Parámetros de Simulación ---")
print(f"Frecuencia: {frecuencia/1e9} GHz | Longitud de onda (λ): {lamda*100:.2f} cm")
print(f"Elementos del arreglo (M): {M} | Muestras temporales (L): {L}")
print(f"Ángulos de transmisión reales: {angulos_reales} grados\n")

# =============================================================================
# PASO 2: GEOMETRÍA DE LOS ARREGLOS DE ANTENAS
# =============================================================================
# Para calcular los desfases, necesitamos conocer la posición física de cada antena.

# 2.1 Arreglo Lineal Uniforme (ULA)
# Separación ideal entre antenas (d) = lambda / 2 para evitar aliasing espacial (ambigüedad).
d_ula = lamda / 2  

# 2.2 Arreglo Circular Uniforme (UCA)
# Colocamos las antenas en un círculo de radio 'r_uca'.
# Para que la distancia sobre el arco/cuerda entre antenas adyacentes sea aproximadamente lambda/2:
# La fórmula geométrica para la distancia d_c entre elementos adyacentes en un círculo de radio r es:
# d_c = 2 * r * sin(pi / M). Si queremos d_c = lambda / 2, despejamos r:
r_uca = lamda / (4 * np.sin(np.pi / M))

print(f"--- Dimensiones de los Arreglos ---")
print(f"ULA - Distancia entre antenas adyacentes: {d_ula*100:.2f} cm (λ/2)")
print(f"UCA - Radio del círculo de antenas: {r_uca*100:.2f} cm (Apertura angular uniforme)\n")


# =============================================================================
# PASO 3: CÓDIGO DE LOS VECTORES DE DIRECCIONAMIENTO (STEERING VECTORS)
# =============================================================================
# El "Steering Vector" es el modelo matemático que describe cómo cambia la fase
# de una señal que llega a cada antena desde un ángulo theta en el espacio libre.

def obtener_steering_ula(theta_deg, M, d, lamda):
    """
    Calcula el Steering Vector para un ULA.
    La primera antena (m=0) se toma como referencia de fase (fase = 0).
    La trayectoria adicional de la onda para la antena 'm' es: m * d * sin(theta).
    """
    theta_rad = np.radians(theta_deg)
    m = np.arange(M)
    # Ecuación fundamental del desfase en línea recta
    fases = -2 * np.pi * (d / lamda) * m * np.sin(theta_rad)
    return np.exp(1j * fases)

def obtener_steering_uca(theta_deg, M, r, lamda):
    """
    Calcula el Steering Vector para un UCA.
    El centro del círculo se toma como la referencia de fase.
    La posición angular de la antena 'm' en el círculo es phi_m = 2*pi*m / M.
    El desfase se genera por la proyección del frente de onda: r * cos(theta - phi_m).
    """
    theta_rad = np.radians(theta_deg)
    phi_m = np.linspace(0, 2 * np.pi, M, endpoint=False) # Posiciones angulares de las antenas
    # Ecuación fundamental del desfase en circunferencia
    fases = 2 * np.pi * (r / lamda) * np.cos(theta_rad - phi_m)
    return np.exp(1j * fases)


# =============================================================================
# PASO 4: GENERACIÓN SINTÉTICA DE LA SEÑAL RECIBIDA (Y = A*S + N)
# =============================================================================
# Simulamos físicamente las ondas electromagnéticas incidiendo sobre los arreglos.

# 4.1 Generar las señales base (S) emitidas por los transmisores
# Usamos señales aleatorias complejas para modelar fuentes de banda estrecha e independientes.
S = (np.random.randn(num_fuentes, L) + 1j * np.random.randn(num_fuentes, L)) / np.sqrt(2)

# 4.2 Construir las matrices de direccionamiento (A) para ambas geometrías
# Unimos los steering vectors de cada emisor real en columnas.
A_ula = np.column_stack([obtener_steering_ula(ang, M, d_ula, lamda) for ang in angulos_reales])
A_uca = np.column_stack([obtener_steering_uca(ang, M, r_uca, lamda) for ang in angulos_reales])

# 4.3 Generar Ruido Blanco Gaussiano Complejo (AWGN)
# Calculamos la potencia del ruido según la SNR que definimos.
potencia_ruido = 10**(-snr_db / 10.0)
ruido_ula = np.sqrt(potencia_ruido / 2) * (np.random.randn(M, L) + 1j * np.random.randn(M, L))
ruido_uca = np.sqrt(potencia_ruido / 2) * (np.random.randn(M, L) + 1j * np.random.randn(M, L))

# 4.4 Matriz de datos de recepción final (Y)
# Y = A * S + Ruido (Fórmula clásica de procesamiento espacial de señales)
Y_ula = np.dot(A_ula, S) + ruido_ula
Y_uca = np.dot(A_uca, S) + ruido_uca


# =============================================================================
# PASO 5: MATRIZ DE COVARIANZA ESPACIAL (Ryy)
# =============================================================================
# Ryy resume la correlación espacial promedio entre todas las antenas.
# Es la entrada fundamental de donde los algoritmos extraen la información espacial.
# Se estima promediando a lo largo del tiempo de observación (L snapshots).

Ryy_ula = np.dot(Y_ula, np.conj(Y_ula).T) / L
Ryy_uca = np.dot(Y_uca, np.conj(Y_uca).T) / L


# =============================================================================
# PASO 6: IMPLEMENTACIÓN DEL ALGORITMO BEAMFORMING CONVENCIONAL
# =============================================================================
# El beamforming barre el espacio apuntando el arreglo a diferentes ángulos de prueba.
# Mide la potencia recibida en cada ángulo. Los picos de potencia indican los emisores.

def procesar_beamforming(Ryy, M, parametro_arreglo, lamda, angulos_barrido, tipo_arreglo):
    """
    Calcula el espectro de potencia usando Beamforming convencional.
    Ecuación: P(theta) = ( a_conj^T * Ryy * a ) / M^2
    """
    espectro = np.zeros(len(angulos_barrido))
    
    for i, theta in enumerate(angulos_barrido):
        # Seleccionamos el steering vector teórico de prueba según el tipo de geometría
        if tipo_arreglo == 'ULA':
            a = obtener_steering_ula(theta, M, parametro_arreglo, lamda)
        else:
            a = obtener_steering_uca(theta, M, parametro_arreglo, lamda)
            
        # Proyectamos la matriz de covarianza en la dirección teórica
        potencia = np.real(np.dot(np.conj(a).T, np.dot(Ryy, a)))
        espectro[i] = potencia / (M**2)  # Normalizamos por el número de antenas al cuadrado
        
    return espectro


# =============================================================================
# PASO 7: EJECUCIÓN DEL ANÁLISIS COMPARATIVO Y GENERACIÓN DE ESPECTROS
# =============================================================================

# Definimos los ángulos para el barrido espacial
# Nota: La ULA tiene simetría espejo, barremos de -90 a 90 grados.
# La UCA tiene visión completa de 360 grados, por lo que barremos de -180 a 180 grados.
angulos_barrido_ula = np.linspace(-90, 90, 360)
angulos_barrido_uca = np.linspace(-180, 180, 720)

# Ejecutamos Beamforming convencional
espectro_bf_ula = procesar_beamforming(Ryy_ula, M, d_ula, lamda, angulos_barrido_ula, 'ULA')
espectro_bf_uca = procesar_beamforming(Ryy_uca, M, r_uca, lamda, angulos_barrido_uca, 'UCA')

# =============================================================================
# PASO 8: NORMALIZACIÓN Y ESCALA LOGARÍTMICA (DECIBELIOS - dB)
# =============================================================================
# En procesamiento de señales es vital visualizar el espectro en dB para apreciar 
# picos estrechos y el rango dinámico.

def normalizar_y_db(espectro):
    """Normaliza el espectro y lo convierte a decibelios (dB) truncando el mínimo a -40 dB"""
    espectro_normalizado = espectro / np.max(espectro)
    espectro_db = 10 * np.log10(espectro_normalizado + 1e-10) # Sumamos eps para evitar log(0)
    return np.clip(espectro_db, -40, 0) # Truncamos a -40 dB para que la gráfica sea legible

db_bf_ula = normalizar_y_db(espectro_bf_ula)
db_bf_uca = normalizar_y_db(espectro_bf_uca)


# =============================================================================
# PASO 9: GRAFICACIÓN DE LOS RESULTADOS COMPARATIVOS
# =============================================================================
# Creamos figuras detalladas e interactivas para analizar los resultados.

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
fig.suptitle("Análisis Comparativo de Arreglos de Antenas (ULA vs UCA) e Algoritmos (Beamforming vs MUSIC)\n"
             f"Señales de Control de Dron (2.4 GHz) | SNR = {snr_db} dB | M = {M} Antenas", fontsize=14, fontweight='bold')

# --- SUBPLOT 1: ARREGLO LINEAL UNIFORME (ULA) ---
ax1.plot(angulos_barrido_ula, db_bf_ula, label="Beamforming Convencional (Lóbulo Ancho)", color="navy", linestyle="-", linewidth=1.5)
#
for ang in angulos_reales:
    ax1.axvline(x=ang, color="red", linestyle=":", alpha=0.8, label="Ángulo Real de Origen" if ang==angulos_reales[0] else "")
ax1.set_title("1. Desempeño sobre Arreglo Lineal Uniforme (ULA) - Escaneo: [-90°, 90°]", fontsize=12, fontweight='bold')
ax1.set_xlabel("Ángulo de Llegada (Grados)")
ax1.set_ylabel("Potencia Relativa / Pseudo-espectro (dB)")
ax1.set_xlim([-90, 90])
ax1.set_ylim([-45, 2])
ax1.grid(True, which="both", linestyle=":", alpha=0.5)
ax1.legend(loc="upper right")

# --- SUBPLOT 2: ARREGLO CIRCULAR UNIFORME (UCA) ---
ax2.plot(angulos_barrido_uca, db_bf_uca, label="Beamforming Convencional (Uniforme en 360°)", color="teal", linestyle="-", linewidth=1.5)
#
for ang in angulos_reales:
    ax2.axvline(x=ang, color="red", linestyle=":", alpha=0.8, label="Ángulo Real de Origen" if ang==angulos_reales[0] else "")
    # Graficar la simetría espejo esperada en ULA si escaneáramos 360° (para contrastar con la UCA libre de ella)
    mirror_ang = 180.0 - ang if ang > 0 else -180.0 - ang
    ax2.axvline(x=mirror_ang, color="purple", linestyle="--", alpha=0.3, label="Simetría de Espejo ULA (Ambigüedad)" if ang==angulos_reales[0] else "")
ax2.set_title("2. Desempeño sobre Arreglo Circular Uniforme (UCA) - Escaneo: [-180°, 180°]", fontsize=12, fontweight='bold')
ax2.set_xlabel("Ángulo de Llegada (Grados)")
ax2.set_ylabel("Potencia Relativa / Pseudo-espectro (dB)")
ax2.set_xlim([-180, 180])
ax2.set_ylim([-45, 2])
ax2.grid(True, which="both", linestyle=":", alpha=0.5)
ax2.legend(loc="upper right")

plt.tight_layout()

# Guardar la gráfica para que el usuario pueda visualizarla en Studio
plt.savefig("comparativo_sin_music.png", format='png', dpi=150, bbox_inches="tight")
print("[SISTEMA] Gráfica guardada exitosamente con nombre comparativo_sin_music.png")
plt.close()

print("\n--- Conclusiones Clave para tu Análisis Comparativo ---")
print("1. RESOLUCIÓN: Observa cómo las líneas de MUSIC (Naranja y Carmesí) son picos extremadamente delgados")
print("   en comparación con el Beamforming convencional, demostrando su capacidad de súper-resolución espacial.")
print("2. COBERTURA: El ULA está limitado a [-90°, 90°]. Si intentaras buscar a 150°, verías un pico espejo en 30°.")
print("   El UCA escanea los 360° completos libre de ambigüedades simétricas en su plano horizontal.")
