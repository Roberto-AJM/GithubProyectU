"""
ula_uca_bmfm_v2.py
Simulación de arreglos ULA y UCA con Beamforming convencional (línea base).

Cambios respecto a ula_uca_bmfm.py:
  - Semilla del generador aleatorio: los resultados son reproducibles.
  - Parámetros del caso agrupados en un diccionario (CASO_BASE).
  - Generación de la señal separada en dos funciones: generar_fuentes y generar_senal.
  - Programa principal dentro de main(), para poder importar las funciones desde otros scripts.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURACIÓN DEL CASO
# =============================================================================
# Todos los parámetros que definen una simulación viven en un único diccionario.
# Para probar otro escenario basta con pasar otro diccionario a main().

C_LUZ = 3e8  # Velocidad de la luz en el vacío (m/s)

CASO_BASE = {
    "semilla": 42,                    # None -> resultados distintos en cada ejecución
    "frecuencia": 2.4e9,              # Frecuencia de operación (Hz)
    "M": 8,                           # Número de antenas del arreglo
    "n_muestras": 1000,               # Muestras temporales síncronas (antes llamada L)
    "snr_db": 15,                     # Relación señal-ruido por antena (dB)
    "angulos_reales": [-25.0, 30.0],  # Ángulos de llegada de las fuentes (grados)
}


# =============================================================================
# FUNCIONES: VECTORES DE DIRECCIONAMIENTO (STEERING VECTORS)
# =============================================================================

def obtener_steering_ula(theta_deg, M, d, lamda):
    """
    Calcula el Steering Vector para un ULA.
    La primera antena (m=0) se toma como referencia de fase (fase = 0).
    La trayectoria adicional de la onda para la antena 'm' es: m * d * sin(theta).
    """
    theta_rad = np.radians(theta_deg)
    m = np.arange(M)
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
    phi_m = np.linspace(0, 2 * np.pi, M, endpoint=False)
    fases = 2 * np.pi * (r / lamda) * np.cos(theta_rad - phi_m)
    return np.exp(1j * fases)


# =============================================================================
# FUNCIONES: GENERACIÓN SINTÉTICA DE LA SEÑAL RECIBIDA (Y = A*S + N)
# =============================================================================

def generar_fuentes(caso, rng):
    """
    Genera las señales emitidas por las fuentes (matriz S).
    Son variables complejas gaussianas independientes de varianza unitaria,
    que modelan fuentes de banda estrecha estadísticamente independientes.

    Devuelve S con forma (num_fuentes, n_muestras).

    Se genera una sola vez por simulación para que ambas geometrías (ULA y UCA)
    reciban exactamente las mismas fuentes y la comparación sea directa.
    """
    num_fuentes = len(caso["angulos_reales"])
    n_muestras = caso["n_muestras"]
    parte_real = rng.standard_normal((num_fuentes, n_muestras))
    parte_imag = rng.standard_normal((num_fuentes, n_muestras))
    return (parte_real + 1j * parte_imag) / np.sqrt(2)


def generar_senal(S, caso, steering_fn, parametro_arreglo, lamda, rng):
    """
    Genera la señal recibida por un arreglo: Y = A*S + N.

    S                 : fuentes, salida de generar_fuentes.
    steering_fn       : obtener_steering_ula u obtener_steering_uca.
    parametro_arreglo : separación d (ULA) o radio r (UCA), en metros.
    rng               : generador aleatorio (para el ruido de este arreglo).

    Devuelve (Y, A): la señal recibida (M x n_muestras) y la matriz de
    direccionamiento (M x num_fuentes).
    """
    M = caso["M"]
    n_muestras = caso["n_muestras"]

    # Matriz de direccionamiento A: un steering vector por cada ángulo real
    A = np.column_stack([steering_fn(ang, M, parametro_arreglo, lamda)
                         for ang in caso["angulos_reales"]])

    # Ruido blanco gaussiano complejo (AWGN), independiente por antena y por muestra.
    # La potencia del ruido sale de la SNR con fuentes de potencia unitaria.
    potencia_ruido = 10 ** (-caso["snr_db"] / 10.0)
    ruido = np.sqrt(potencia_ruido / 2) * (rng.standard_normal((M, n_muestras))
                                           + 1j * rng.standard_normal((M, n_muestras)))

    Y = np.dot(A, S) + ruido
    return Y, A


# =============================================================================
# FUNCIONES: BEAMFORMING CONVENCIONAL Y ESCALA EN dB
# =============================================================================

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


def normalizar_y_db(espectro):
    """Normaliza el espectro y lo convierte a decibelios (dB) truncando el mínimo a -40 dB"""
    espectro_normalizado = espectro / np.max(espectro)
    espectro_db = 10 * np.log10(espectro_normalizado + 1e-10)  # Sumamos eps para evitar log(0)
    return np.clip(espectro_db, -40, 0)  # Truncamos a -40 dB para que la gráfica sea legible


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main(caso=CASO_BASE, graficar=True):
    # Un único generador aleatorio gobierna toda la simulación: misma semilla, mismos resultados.
    rng = np.random.default_rng(caso["semilla"])

    # -------------------------------------------------------------------------
    # PASO 1: PARÁMETROS DEL SISTEMA Y FÍSICA DE LA SEÑAL
    # -------------------------------------------------------------------------
    frecuencia = caso["frecuencia"]
    M = caso["M"]
    n_muestras = caso["n_muestras"]
    snr_db = caso["snr_db"]
    angulos_reales = np.array(caso["angulos_reales"])
    lamda = C_LUZ / frecuencia  # Longitud de onda (~0.125 m a 2.4 GHz)

    print("--- Parámetros de Simulación ---")
    print(f"Semilla: {caso['semilla']}")
    print(f"Frecuencia: {frecuencia/1e9} GHz | Longitud de onda (λ): {lamda*100:.2f} cm")
    print(f"Elementos del arreglo (M): {M} | Muestras temporales: {n_muestras} | SNR: {snr_db} dB")
    print(f"Ángulos de transmisión reales: {angulos_reales} grados\n")

    # -------------------------------------------------------------------------
    # PASO 2: GEOMETRÍA DE LOS ARREGLOS DE ANTENAS
    # -------------------------------------------------------------------------
    # ULA: separación lambda/2 (condición de Nyquist espacial).
    d_ula = lamda / 2

    # UCA: radio tal que la cuerda entre antenas adyacentes sea lambda/2.
    # d_c = 2 * r * sin(pi / M) = lambda / 2  ->  r = lambda / (4 * sin(pi / M))
    r_uca = lamda / (4 * np.sin(np.pi / M))

    print("--- Dimensiones de los Arreglos ---")
    print(f"ULA - Distancia entre antenas adyacentes: {d_ula*100:.2f} cm (λ/2)")
    print(f"UCA - Radio del círculo de antenas: {r_uca*100:.2f} cm\n")

    # -------------------------------------------------------------------------
    # PASO 3: GENERACIÓN DE LA SEÑAL RECIBIDA (Y = A*S + N)
    # -------------------------------------------------------------------------
    # Mismas fuentes S para ambos arreglos; el ruido es independiente en cada uno.
    S = generar_fuentes(caso, rng)
    Y_ula, A_ula = generar_senal(S, caso, obtener_steering_ula, d_ula, lamda, rng)
    Y_uca, A_uca = generar_senal(S, caso, obtener_steering_uca, r_uca, lamda, rng)

    # -------------------------------------------------------------------------
    # PASO 4: MATRIZ DE COVARIANZA ESPACIAL (Ryy)
    # -------------------------------------------------------------------------
    Ryy_ula = np.dot(Y_ula, np.conj(Y_ula).T) / n_muestras
    Ryy_uca = np.dot(Y_uca, np.conj(Y_uca).T) / n_muestras

    # -------------------------------------------------------------------------
    # PASO 5: BEAMFORMING CONVENCIONAL Y ESCALA EN dB
    # -------------------------------------------------------------------------
    # ULA: simetría espejo, se barre de -90 a 90 grados.
    # UCA: visión de 360 grados, se barre de -180 a 180 grados.
    angulos_barrido_ula = np.linspace(-90, 90, 360)
    angulos_barrido_uca = np.linspace(-180, 180, 720)

    espectro_bf_ula = procesar_beamforming(Ryy_ula, M, d_ula, lamda, angulos_barrido_ula, 'ULA')
    espectro_bf_uca = procesar_beamforming(Ryy_uca, M, r_uca, lamda, angulos_barrido_uca, 'UCA')

    db_bf_ula = normalizar_y_db(espectro_bf_ula)
    db_bf_uca = normalizar_y_db(espectro_bf_uca)

    # -------------------------------------------------------------------------
    # PASO 6: GRAFICACIÓN DE LOS RESULTADOS COMPARATIVOS
    # -------------------------------------------------------------------------
    if graficar:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
        fig.suptitle("Análisis Comparativo de Arreglos de Antenas (ULA vs UCA) con Beamforming Convencional\n"
                     f"Señales de Control de Dron (2.4 GHz) | SNR = {snr_db} dB | M = {M} Antenas",
                     fontsize=14, fontweight='bold')

        # --- SUBPLOT 1: ARREGLO LINEAL UNIFORME (ULA) ---
        ax1.plot(angulos_barrido_ula, db_bf_ula, label="Beamforming Convencional (ULA)",
                 color="navy", linestyle="-", linewidth=1.5)
        for ang in angulos_reales:
            ax1.axvline(x=ang, color="red", linestyle=":", alpha=0.8,
                        label="Ángulo Real de Origen" if ang == angulos_reales[0] else "")
        ax1.set_title("1. Desempeño sobre Arreglo Lineal Uniforme (ULA) - Escaneo: [-90°, 90°]",
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel("Ángulo de Llegada (Grados)")
        ax1.set_ylabel("Potencia Relativa / Pseudo-espectro (dB)")
        ax1.set_xlim([-90, 90])
        ax1.set_ylim([-45, 2])
        ax1.grid(True, which="both", linestyle=":", alpha=0.5)
        ax1.legend(loc="upper right")

        # --- SUBPLOT 2: ARREGLO CIRCULAR UNIFORME (UCA) ---
        ax2.plot(angulos_barrido_uca, db_bf_uca, label="Beamforming Convencional (UCA)",
                 color="teal", linestyle="-", linewidth=1.5)
        for ang in angulos_reales:
            ax2.axvline(x=ang, color="red", linestyle=":", alpha=0.8,
                        label="Ángulo Real de Origen" if ang == angulos_reales[0] else "")
            # Posición de la imagen especular que tendría un ULA al barrer 360° (ambigüedad)
            mirror_ang = 180.0 - ang if ang > 0 else -180.0 - ang
            ax2.axvline(x=mirror_ang, color="purple", linestyle="--", alpha=0.3,
                        label="Simetría de Espejo ULA (Ambigüedad)" if ang == angulos_reales[0] else "")
        ax2.set_title("2. Desempeño sobre Arreglo Circular Uniforme (UCA) - Escaneo: [-180°, 180°]",
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel("Ángulo de Llegada (Grados)")
        ax2.set_ylabel("Potencia Relativa / Pseudo-espectro (dB)")
        ax2.set_xlim([-180, 180])
        ax2.set_ylim([-45, 2])
        ax2.grid(True, which="both", linestyle=":", alpha=0.5)
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig("comparativo_sin_music.png", format='png', dpi=150, bbox_inches="tight")
        print("[SISTEMA] Gráfica guardada exitosamente con nombre comparativo_sin_music.png")
        plt.close()

    print("\n--- Observaciones del Análisis Comparativo (solo Beamforming convencional) ---")
    print("1. RESOLUCIÓN: el Beamforming convencional produce lóbulos anchos; la mejora de")
    print("   resolución con MUSIC corresponde a la siguiente etapa del proyecto.")
    print("2. COBERTURA: el ULA está limitado a [-90°, 90°]; una fuente en 150° aparecería como")
    print("   su imagen espejo en 30°. El UCA escanea los 360° sin esa ambigüedad.")

    return {
        "lamda": lamda, "d_ula": d_ula, "r_uca": r_uca,
        "Ryy_ula": Ryy_ula, "Ryy_uca": Ryy_uca,
        "angulos_barrido_ula": angulos_barrido_ula, "angulos_barrido_uca": angulos_barrido_uca,
        "db_bf_ula": db_bf_ula, "db_bf_uca": db_bf_uca,
    }


if __name__ == "__main__":
    main()
