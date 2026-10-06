
import numpy as np
import matplotlib.pyplot as plt

def generar_senal(fs, duracion, componentes):
    """
    Genera una señal compuesta por varias frecuencias.
    :param fs: Frecuencia de muestreo (Hz)
    :param duracion: Duración de la señal (segundos)
    :param componentes: Lista de tuplas (frecuencia, amplitud)
    :return: vector tiempo, señal
    """
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)
    senal = np.zeros_like(t)
    for f, A in componentes:
        senal += A * np.sin(2 * np.pi * f * t)
    return t, senal

def calcular_ancho_banda(senal, fs, umbral_db=-20):
    """
    Calcula el ancho de banda aproximado de una señal.
    :param senal: Señal en el dominio del tiempo
    :param fs: Frecuencia de muestreo
    :param umbral_db: Nivel en dB para considerar el ancho de banda
    :return: ancho de banda en Hz
    """
    N = len(senal)
    espectro = np.fft.fft(senal)
    freqs = np.fft.fftfreq(N, 1/fs)
    magnitud_db = 20 * np.log10(np.abs(espectro) / np.max(np.abs(espectro)) + 1e-12)

    # Filtrar solo frecuencias positivas
    freqs_pos = freqs[:N//2]
    magnitud_pos = magnitud_db[:N//2]

    # Determinar rango donde la magnitud está por encima del umbral
    indices = np.where(magnitud_pos > umbral_db)[0]
    if len(indices) == 0:
        return 0
    f_min, f_max = freqs_pos[indices[0]], freqs_pos[indices[-1]]
    return f_max - f_min, freqs_pos, magnitud_pos

# Parámetros de simulación
fs = 500  # Hz
duracion = 1.0  # segundos
componentes = [(200, 1.0), (600, 0.5), (1200, 0.3)]  # (frecuencia, amplitud)

# Generar señal+
t, senal = generar_senal(fs, duracion, componentes)

# Calcular ancho de banda
bw, freqs_pos, magnitud_pos = calcular_ancho_banda(senal, fs, umbral_db=-20)

# Mostrar resultados
print(f"Ancho de banda estimado: {bw:.2f} Hz")

# Graficar señal y espectro
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(t, senal)
plt.title("Señal en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")

plt.subplot(1,2,2)
plt.plot(freqs_pos, magnitud_pos)
plt.title("Espectro de la señal")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)

plt.tight_layout()
plt.show()