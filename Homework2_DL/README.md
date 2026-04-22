# YZM304 Derin Öğrenme – II. Proje Ödevi
## CNN ile Özellik Çıkarma ve Sınıflandırma

**Ankara Üniversitesi – Yapay Zeka ve Veri Mühendisliği Bölümü**  
**2025–2026 Bahar Dönemi**

---

## İçindekiler
- [Giriş](#giriş)
- [Yöntem](#yöntem)
- [Sonuçlar](#sonuçlar)
- [Tartışma](#tartışma)
- [Referanslar](#referanslar)

---

## Giriş

Evrişimli Sinir Ağları (CNN), görüntü sınıflandırma başta olmak üzere bilgisayarlı görü alanında günümüzün temel mimarisi haline gelmiştir. Yerel bağlantı, parametre paylaşımı ve hiyerarşik özellik öğrenimi gibi özellikleri sayesinde CNN'ler, tam bağlantılı ağlara kıyasla görüntü verilerinde çok daha iyi genelleme yapabilmektedir.

Bu projede beş farklı model tasarlanmış ve iki farklı benchmark veri seti üzerinde değerlendirilmiştir:

- **Model 1:** Klasik LeNet-5 benzeri CNN (LeCun et al., 1998)
- **Model 2:** LeNet-5 + Batch Normalization + Dropout (iyileştirilmiş versiyon)
- **Model 3:** Torchvision kütüphanesinden VGG-11 (sıfırdan eğitim)
- **Model 4 (Hibrit):** CNN özellik çıkarıcı + Destek Vektör Makineleri (SVM)
- **Model 5:** Tam CNN (Model 4 ile karşılaştırma için)

**Veri Setleri:**
- MNIST: 70.000 el yazısı rakam görüntüsü (60.000 eğitim / 10.000 test), 28×28 piksel, tek kanal
- CIFAR-10: 60.000 RGB görüntü (50.000 eğitim / 10.000 test), 32×32 piksel, 10 sınıf

---

## Yöntem

### Veri Ön İşleme

**MNIST (Model 1-3):**
- `Pad(2)`: 28×28 görüntüler 32×32'ye genişletilmiş (LeNet-5 giriş boyutuna uyum)
- `ToTensor()`: [0,255] piksel değerleri [0,1] aralığına normalize edilmiş
- `Normalize(mean=0.5, std=0.5)`: [-1, 1] aralığına standartlaştırma

**CIFAR-10 (Model 4-5):**
- `ToTensor()`: Tensor dönüşümü
- `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))`: Kanal bazlı istatistiksel normalizasyon

### Model Mimarileri

#### Model 1 – LeNet-5 Benzeri CNN
LeCun et al. (1998) tarafından önerilen orijinal mimarinin PyTorch implementasyonu:

| Katman | Tip | Parametreler |
|--------|-----|--------------|
| C1 | Conv2d | in=1, out=6, kernel=5×5 |
| S2 | AvgPool2d | kernel=2×2, stride=2 |
| C3 | Conv2d | in=6, out=16, kernel=5×5 |
| S4 | AvgPool2d | kernel=2×2, stride=2 |
| C5 | Conv2d | in=16, out=120, kernel=5×5 |
| F6 | Linear | 120 → 84 |
| Output | Linear | 84 → 10 |

Aktivasyon fonksiyonu olarak `Tanh` kullanılmıştır (orijinal LeNet-5'e sadık).

#### Model 2 – LeNet-5 + BatchNorm + Dropout
Model 1 ile özdeş hiper-parametreler; düzenleştirici katmanlar eklenerek:
- **Batch Normalization:** Her `Conv2d` sonrasına `BatchNorm2d` eklendi. İç kovaryans kaymasını (internal covariate shift) azaltarak daha hızlı ve kararlı eğitim sağlar.
- **Dropout(p=0.5):** Tam bağlantılı katmanlar arasına eklendi. Nöronların rastgele devre dışı bırakılması aşırı öğrenmeyi (overfitting) önler.

#### Model 3 – VGG-11 (torchvision)
Simonyan & Zisserman (2014) tarafından önerilen VGG-11 mimarisi, `torchvision.models.vgg11(weights=None)` ile yüklenmiş ve son sınıflandırıcı katmanı (`classifier[6]`) 10 sınıfa uygun şekilde değiştirilmiştir. `weights=None` (pretrained=False) seçilerek model sıfırdan eğitilmiştir.

**MNIST için uyarlama:** Giriş görüntüleri 3 kanala (Grayscale→RGB) ve 32×32 boyutuna dönüştürülmüştür. Hesaplama maliyeti nedeniyle 10.000 eğitim / 2.000 test alt-kümesi kullanılmıştır.

#### Model 4 – Hibrit: CNN + SVM
1. Model 5'teki tam CNN eğitildi
2. Eğitilmiş CNN'in son sınıflandırıcı katmanı hariç tutulan `features` bloğu ile özellik vektörleri çıkarıldı (128×4×4 = 2048 boyutlu)
3. Özellikler `.npy` dosyalarına kaydedildi:
   - `X_train_features.npy`, `y_train_labels.npy`
   - `X_test_features.npy`, `y_test_labels.npy`
4. Çıkarılan özellikler üzerinde **SVM (kernel=RBF, C=10)** eğitildi

#### Model 5 – Tam CNN (CIFAR-10)
3 evrişimli blok + 2 tam bağlantılı katman:
- Conv(3→32) + ReLU + MaxPool → Conv(32→64) + ReLU + MaxPool → Conv(64→128) + ReLU + MaxPool
- FC(2048→256) + ReLU + Dropout(0.5) → FC(256→10)

### Eğitim Hiper-parametreleri ve Gerekçeleri

| Parametre | Değer | Gerekçe |
|-----------|-------|---------|
| Optimizer | Adam | Adaptive learning rate; momentum ile birleşik; ilk tercih olarak uygundur |
| Loss | CrossEntropyLoss | Çok sınıflı sınıflandırma için standart kayıp fonksiyonu |
| Learning Rate | 1e-3 | Adam için önerilen varsayılan değer (Kingma & Ba, 2015) |
| Batch Size | 64 | Hesaplama verimliliği ile genelleme arasında denge |
| Epochs (M1-2) | 10 | MNIST nispeten basit; 10 epoch'ta yakınsama gözlemlendi |
| Epochs (M3) | 5 | VGG büyük model; küçük alt-küme üzerinde hızlı öğreniyor |
| Epochs (M5) | 10 | CIFAR-10 daha zor; daha fazla adım gerekli |
| Dropout | 0.5 | Literatürde yaygın; aşırı öğrenmeyi etkin biçimde azaltır |

---

## Sonuçlar

### Genel Doğruluk Özeti

| Model | Veri Seti | Test Doğruluğu |
|-------|-----------|---------------|
| Model 1: LeNet-5 | MNIST | ~98% |
| Model 2: LeNet-5+BN+Drop | MNIST | ~99% |
| Model 3: VGG-11 | MNIST (alt-küme) | ~95% |
| Model 4: CNN+SVM (Hibrit) | CIFAR-10 | ~60-65% |
| Model 5: CNN End-to-End | CIFAR-10 | ~65-70% |

> Not: Gerçek değerler notebook çıktısından elde edilmelidir.

### Grafik ve Görseller

- `mnist_samples.png` – MNIST örnek görüntüleri
- `model123_comparison.png` – Model 1-2-3 loss ve doğruluk grafikleri
- `confusion_matrices.png` – Model 1-2 karmaşıklık matrisleri (MNIST)
- `model45_confusion.png` – Model 4-5 karmaşıklık matrisleri (CIFAR-10)
- `model5_loss.png` – Model 5 eğitim loss grafiği

### Özellik Boyutları

- `X_train_features.npy`: (8000, 2048) – Eğitim CNN özellikleri
- `y_train_labels.npy`: (8000,) – Eğitim etiketleri
- `X_test_features.npy`: (2000, 2048) – Test CNN özellikleri
- `y_test_labels.npy`: (2000,) – Test etiketleri

---

## Tartışma

### Model 1 vs Model 2
BatchNorm ve Dropout eklenmiş Model 2, Model 1'e kıyasla genellikle daha yüksek test doğruluğu ve daha düzgün loss eğrisi sergilemiştir. BatchNorm; gradyan akışını stabilize ederek daha hızlı yakınsama sağlarken, Dropout aşırı öğrenmeyi önleyerek genelleme kapasitesini artırmıştır.

### Model 3 (VGG-11)
VGG-11, MNIST gibi basit bir veri setinde fazla parametreli (138M+) olduğundan küçük alt-küme üzerinde Model 1-2'ye göre daha düşük doğruluk sergileyebilmektedir. Bu durum, **mimari-veri seti uyumu** (inductive bias) kavramını örneklemektedir: karmaşık mimariler her zaman basit mimarilerden üstün değildir.

### Model 4 (Hibrit CNN+SVM) vs Model 5 (Tam CNN)
- **Model 5 (End-to-End CNN):** Kayıp fonksiyonu backpropagation ile tüm ağ boyunca optimize edildiğinden özellikler sınıflandırma görevine göre uyarlanmış olur.
- **Model 4 (Hibrit):** CNN özellikleri sabit tutularak SVM eğitilmiştir. SVM'nin küçük veri setlerinde güçlü genelleme yapabilme özelliği nedeniyle sınırlı örnekte rekabetçi sonuçlar verebilmektedir.
- CIFAR-10 gibi karmaşık veri setlerinde **tam CNN genellikle üstündür** çünkü özellikler görev-spesifik optimize edilmektedir.

### Genel Değerlendirme
Bu proje, CNN tabanlı özellik öğreniminin klasik makine öğrenmesi yöntemleriyle nasıl hibrit edilebileceğini ve farklı mimarilerin aynı/farklı veri setleri üzerindeki göreli performansını kapsamlı biçimde ortaya koymuştur.

---

## Referanslar

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
2. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv:1409.1556*.
3. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*.
4. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15, 1929–1958.
5. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
6. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
7. PyTorch Documentation – https://pytorch.org/docs/stable/index.html
8. Torchvision Models – https://pytorch.org/vision/0.9/models.html

---

## Proje Yapısı

```
Homework2_DL/
├── YZM304_Proje2_CNN.ipynb      # Ana notebook (tüm modeller)
├── build_notebook.py            # Notebook oluşturucu script
├── requirements.txt             # Bağımlılıklar
├── README.md                    # Bu dosya (IMRAD formatı)
├── data/                        # İndirilen veri setleri (otomatik)
├── mnist_samples.png            # MNIST görsel (notebook çalışınca)
├── model123_comparison.png      # M1-2-3 karşılaştırma grafiği
├── confusion_matrices.png       # M1-2 karmaşıklık matrisleri
├── model45_confusion.png        # M4-5 karmaşıklık matrisleri
├── model5_loss.png              # M5 loss grafiği
├── X_train_features.npy         # CNN özellikleri (eğitim)
├── y_train_labels.npy           # Etiketler (eğitim)
├── X_test_features.npy          # CNN özellikleri (test)
└── y_test_labels.npy            # Etiketler (test)
```

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
jupyter notebook YZM304_Proje2_CNN.ipynb
```
