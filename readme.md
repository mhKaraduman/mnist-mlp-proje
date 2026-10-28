# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma
# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma

Bu projede, MNIST veri seti kullanılarak sıfırdan bir **Multi-Layer Perceptron (MLP)** modeli geliştirilmiş, eğitilmiş ve kendi el yazımla yazdığım öğrenci numaram (`26120205084`) üzerinde test edilmiştir.

## 1. Model Mimarisi ve Hiperparametreler
* **Mimari:** Flatten (784) -> Linear (256) + ReLU -> Linear (128) + ReLU -> Linear (10)
* **Loss Fonksiyonu:** CrossEntropyLoss
* **Optimizer:** Adam (`lr = 0.001`)
* **Batch Size:** 64
* **Epoch Sayısı:** 5
* **Veri Ayrımı:** 48.000 Train (%80), 12.000 Validation (%20), 10.000 Test

## 2. Deney 1 Sonuçları

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.2599 | %92.16 (0.9216) | 0.1270 | %96.04 (0.9604) |
| 2 | 0.1012 | %96.86 (0.9686) | 0.0986 | %96.94 (0.9694) |
| 3 | 0.0706 | %97.78 (0.9778) | 0.1049 | %96.85 (0.9685) |
| 4 | 0.0527 | %98.33 (0.9833) | 0.0978 | %97.22 (0.9722) |
| 5 | **0.0421** | **%98.59 (0.9859)** | **0.1050** | **%97.21 (0.9721)** |

* **Test Accuracy:** %97.41 (`0.9741`)
* **Eğitim Süresi:** `213.63` saniye

### Eğitim Grafikleri
![Loss ve Accuracy Grafikleri](training_metrics.png)

## 3. Kendi El Yazısı Öğrenci Numaramın (`26120205084`) Test Edilmesi
Kağıt üzerine yazılan öğrenci numarası OpenCV ile adaptif eşikleme (adaptive thresholding) kullanılarak gölgelerden arındırılmış, her bir rakam kontur tespitiyle ayrıştırılıp en-boy oranı korunarak `28x28` piksel boyutunda MNIST formatına (siyah arka plan, beyaz rakam) dönüştürülmüştür.

### Tahmin Sonuçları
![Öğrenci Numarası Tahminleri](okul_numarasi_mnist_sonuc.png)

## 4. Genel Değerlendirme (Distribution Shift)
MNIST test verisinde **%97.41** gibi yüksek bir başarı elde edilirken, gerçek dünyadaki kendi el yazım üzerinde bazı rakamlarda modelin daha düşük güven skoru ürettiği veya bazı rakamları karıştırabildiği gözlemlenmiştir. Bunun temel sebebi **Distribution Shift** (eğitim verisi ile gerçek dünya verisi arasındaki dağılım farkı; kalem kalınlığı, yazı stili, kağıt dokusu) ve MLP mimarisinin pikselleri düzleştirerek (`Flatten`) işlemesi nedeniyle 2 boyutlu uzamsal ilişkileri koruyamamasıdır.


## 3. Kendi El Yazısı Öğrenci Numaramın (`26120205084`) Test Edilmesi
Kağıt üzerine kurşun kalemle yazılan öğrenci numarası (`26120205084`), OpenCV kütüphanesi kullanılarak şu ön işleme adımlarından geçirilmiştir:
1. **Adaptive Thresholding:** Kağıt üzerindeki ışık/gölge farkları temizlenerek arka plan tamamen siyah (`0`), rakamlar beyaz (`255`) yapılmıştır.
2. **Morfolojik İşlemler (Closing & Dilation):** İnce kurşun kalem çizgilerindeki kopukluklar birleştirilmiş ve çizgi kalınlığı MNIST standardına yaklaştırılmıştır.
3. **Orantılı Boyutlandırma ve Kütle Merkezi (Center of Mass):** Tespit edilen 11 rakamın her biri en-boy oranı korunarak `20x20` piksele küçültülmüş ve piksellerin ağırlık merkezine göre `28x28` siyah tuvalin merkezine yerleştirilmiştir.

### Tahmin Sonuçları (11/11 Doğru Tahmin)
![Öğrenci Numarası Tahminleri](okul_numarasi_mnist_sonuc.png)

## 4. Genel Değerlendirme (Distribution Shift ve Genelleme)
* Model, MNIST test veri setinde **%97.41** doğruluk elde etmiştir.
* Kendi el yazımla yazdığım `26120205084` öğrenci numarasındaki **11 rakamın tamamını (11/11) doğru tahmin etmeyi başarmıştır.**
* İlk denemelerde görüntüye morfolojik kalınlaştırma ve kütle merkezi hizalaması uygulanmadığında, kurşun kalem çizgilerinin inceliği nedeniyle modelin rakamları tanıyamadığı (`1` ve `7` olarak hatalı tahmin ettiği) gözlemlenmiştir.
* Doğru ön işleme sonrasında `2, 6, 1, 0, 5, 8` rakamları **%97 - %100** güven skoruyla tahmin edilirken, sondaki **`4` rakamı %50 güven skoruyla** doğru tahmin edilmiştir. Bunun temel sebebi **Distribution Shift** (veri dağılımı farkı) olgusudur: Kendi el yazımdaki `4` karakterinin yazılış stili, MNIST eğitim setindeki standart `4` örneklerinden farklılık göstermektedir.
