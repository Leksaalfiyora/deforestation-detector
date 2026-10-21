import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Definisi Label Kategori (Penyebab Deforestasi)
CLASSES = [
    'Pertanian / Perkebunan', 
    'Pertambangan', 
    'Kebakaran Hutan', 
    'Penebangan Selektif'
]

print("--- Mempersiapkan Data Simulasi Satelit ---")
# 2. Simulasi Data Fitur (Misal: 100 sampel data numerik fitur citra/indeks vegetasi)
X_train = np.random.rand(100, 10) # 100 sampel, 10 fitur per sampel
y_train = np.random.randint(0, 4, size=(100,)) # 100 label target (0-3)

# 3. Melatih Model (Menggunakan Random Forest Classifier)
print("\n--- Melatih Model Klasifikasi (Machine Learning) ---")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 4. Simulasi Prediksi Data Satelit Baru
print("\n--- Simulasi Prediksi Sampel Satelit Baru ---")
sample_satellite_data = np.random.rand(1, 10) # 1 data sampel baru
prediction_idx = model.predict(sample_satellite_data)[0]
predicted_class = CLASSES[prediction_idx]

print(f"\nHasil Deteksi Penyebab Utama Deforestasi: >> {predicted_class} <<")