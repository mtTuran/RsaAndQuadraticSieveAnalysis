import numpy as np
import matplotlib.pyplot as plt

# --- 1. VERİLER ---
# Key 1, 2, 3 için bit boyutları
bits = np.array([30, 35, 37])

# --- 2. MATEMATİKSEL MODEL ---
def get_complexity(b):
    # N = 2^b -> ln(N) = b * ln(2)
    ln_N = b * np.log(2)
    ln_ln_N = np.log(ln_N)
    return np.exp(np.sqrt(ln_N * ln_ln_N))

# Kalibrasyon: 37 bitlik anahtar 0.15 saniye
comp_37 = get_complexity(37)
C = 0.15 / comp_37
times = C * get_complexity(bits)

# --- 3. 2048 BIT TAHMİNİ ---
target_bit = 2048
comp_2048 = get_complexity(target_bit)
time_2048 = C * comp_2048
years_2048 = time_2048 / (365 * 24 * 3600)

# --- 4. GRAFİK ÇİZİMİ ---
x_range = np.linspace(30, 2048, 500)
y_predicted = C * get_complexity(x_range)

plt.figure(figsize=(10, 6))

# Mavi Tahmin Eğrisi
plt.plot(x_range, y_predicted, 'b--', label='Theoretical Prediction')

# Kırmızı Ölçüm Noktaları (Küçük boyutta)
plt.scatter(bits, times, color='red', s=15, zorder=5, label='Measured Times (Keys 1-3)')

# Yeşil 2048-bit Tahmin Noktası
plt.scatter([2048], [time_2048], color='green', marker='*', s=200, zorder=5, label='2048-bit Extrapolation')

# Ayarlar
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel('Modulus Size (Bits)', fontsize=12)
plt.ylabel('Time (Seconds) [Log Scale]', fontsize=12)
plt.title('Quadratic Sieve Performance Extrapolation\n(30-bit to 2048-bit)', fontsize=14)

# 2048-bit Açıklaması
text_str = f"2048-bit Estimate:\n~{years_2048:.1e} Years"
plt.annotate(text_str, xy=(2048, time_2048), xytext=(1400, time_2048 * 1e-10),
             arrowprops=dict(facecolor='black', shrink=0.05),
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

# Measured Keys Açıklaması (Sağ Üst Konum)
plt.annotate("Measured Keys\n(30, 35, 37 bits)", xy=(37, 0.15), xytext=(200, 10),
             arrowprops=dict(facecolor='black', shrink=0.05),
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plt.legend(loc='upper left')
plt.tight_layout()

# Bilgisayarına kaydeder
plt.savefig('extrapolation_plot_final.png', dpi=300)
print("Grafik 'extrapolation_plot_final.png' adıyla kaydedildi.")