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
  Modelin her zaman adımını "t" olarak düşünürsek :


<img width="193" height="41" alt="sin" src="https://github.com/user-attachments/assets/3f34a05c-c780-4a2a-a57f-20adaf7f843f" />



Burada 
𝑡 zaman adımı, 
𝑑 embedding boyutu.

  

- **Gürültü Eklemek (Forward Process)**:

  <img width="134" height="22" alt="den" src="https://github.com/user-attachments/assets/11bab066-5e1a-4d21-99d7-326e28ea9934" />


Burada e rastgele gürültü, a ise kümülatif çarpımdır.


## Kurulum ve Kullanım

1. Repoyu klonlayın:
   ```bash
   git clone https://github.com/kullanici/simple-ddpm.git
   cd simple-ddpm

2. Gerekli kütüphaneleri yükleyin ve scripti çalıştırın.
python main.py
Eğitim sırasında samples/ klasöründe modelin ürettiği örnekler kaydedilir.

![Adsız tasarım (3)](https://github.com/user-attachments/assets/3ff1678f-3bae-4e27-9886-570eede71476)

