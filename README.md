# Cifar Diffusion 

Bu repo, PyTorch kullanarak Diffusion Model (DDPM) eğitimini ve örnek üretimini gösteren açık ve sade bir implementasyondur. CIFAR-10 veri seti ile çalışmaktadır.

---

## Diffusion Model Nedir?

Diffusion modelleri, veriye adım adım **gürültü ekleyerek** (noising) ve bu gürültüyü geri kaldırarak (denoising) veri dağılımını modelleyen olasılıksal yöntemlerdir. Model, her adımda verilen zamandaki gürültüyü tahmin etmeyi öğrenir.

Bu yaklaşım, karmaşık veri dağılımlarını yakalamada güçlüdür ve özellikle görsel üretimde son yıllarda büyük başarı göstermiştir.

---

## Kodda Neler Var?

- **Sinüzoidal zaman gömme (Sinusoidal Embedding)**: Zaman bilgisini periyodik fonksiyonlarla matematiksel olarak kodlar.
- **Residual bloklar (ResBlock)**: Derin sinir ağlarında bilgi akışını kolaylaştıran "skip connection" yapıları.
- **Küçük U-Net mimarisi**: Görüntüleri çok katmanlı olarak küçültüp büyüterek detayları yakalar.
- **DDPM sınıfı**: Gürültü ekleme ve çıkarma işlemlerini olasılıksal kurallara göre gerçekleştirir.
- **Eğitim döngüsü**: CIFAR-10 verisi üzerinde modeli optimize eder, adım adım öğrenmesini sağlar.

---

## Matematiksel Temeller

- **Sinüzoidal Zaman Gömmesi**:  
  Modelin her zaman adımını \(t\) olarak düşünürüz.  
  \[
  PE(t)_{2i} = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad PE(t)_{2i+1} = \cos\left(\frac{t}{10000^{2i/d}}\right)
  \]  
  Burada \(d\), embedding boyutudur.

- **Gürültü Eklemek (Forward Process)**:  
  Temiz görüntü \(x_0\)’a zaman \(t\) adımında gürültü eklenir:  
  \[
  x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
  \]

- **Gürültüyü Tahmin Etmek (Reverse Process)**:  
  Model, \(x_t\) ve \(t\) verildiğinde \(\epsilon\)’i tahmin etmeye çalışır.

---

## Kurulum ve Kullanım

1. Repoyu klonlayın:
   ```bash
   git clone https://github.com/kullanici/simple-ddpm.git
   cd simple-ddpm

Gerekli kütüphaneleri yükleyin ve scripti çalıştırın.
python main.py
Eğitim sırasında samples/ klasöründe modelin ürettiği örnekler kaydedilir.
