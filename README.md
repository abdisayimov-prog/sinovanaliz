# 5-Modul: CNN Ko'krak Qafasi Rentgen Tasvirlari Tahlili Loyihasi

Ushbu amaliy loyiha ko'krak qafasi rentgen tasvirlari (Chest X-Ray) asosida bemorning o'pkasi holatini aniqlash uchun mo'ljallangan.

---

## 🎯 Model Natijasi
Yangi rentgen tasviri taqdim etilganda CNN modeli quyidagi xulosalardan birini chiqaradi:
1. **Sog'lom** (ishonchlilik darajasi bilan)
2. **O'pka yallig'lanishi aniqlandi** (ishonchlilik darajasi bilan)

---

## 📂 Loyiha Strukturasi
```
D:\5_modul\
│
├── train/                      # 80% Train to'plami (145 ta rasm)
│   ├── NORMAL/                 # 72 ta sog'lom rentgen tasvirlari
│   └── PNEUMONIA/              # 73 ta yallig'langan rentgen tasvirlari
│
├── test/                       # 20% Test to'plami (38 ta rasm)
│   ├── NORMAL/                 # 19 ta test rentgen tasviri
│   └── PNEUMONIA/              # 19 ta test rentgen tasviri
│
├── pneumonia_cnn.pth           # O'qitilgan eng yaxshi CNN modeli vaznlari
├── predict.py                  # Yangi rentgen tasvirini terminal orqali tekshirish dasturi
├── cnn.ipynb                   # To'liq interaktiv Jupyter Notebook
└── README.md                   # Loyiha qo'llanmasi
```

---

## 🚀 Foydalanish Usullari

### 1-usul: Terminal (Buyruqlar satri) orqali tekshirish
Ixtiyoriy rentgen tasvirini tekshirish uchun:
```bash
py predict.py "rasm_fayli.jpeg"
```

**Misollar:**
```bash
# Sog'lom tasvirni tekshirish:
py predict.py test/NORMAL/NORMAL2-IM-0052-0001.jpeg

# O'pka yallig'lanishi tasvirini tekshirish:
py predict.py test/PNEUMONIA/ryct.2020200034.fig2.jpeg
```

**Natija namunasi:**
```
=================================================================
           KO'KRAK QAFASI RENTGEN TAHLILI (CNN)
=================================================================
 Fayl nomi             : ryct.2020200034.fig2.jpeg
 Xulosa                : O'PKA YALLIG'LANISHI ANIQLANDI
 Ishonchlilik darajasi : 99.76%
-----------------------------------------------------------------
 [Sog'lom]: 0.24%   |   [O'pka yallig'lanishi]: 99.76%
=================================================================
```

---

### 2-usul: Jupyter Notebook (`cnn.ipynb`) orqali
1. VS Code yoki Jupyter Lab da `cnn.ipynb` faylini oching.
2. Istalgan rasmni tekshirish uchun:
```python
predict_xray("yangi_rentgen_rasmi.jpeg")
```
Ushbu funksiya rentgen tasvirini vizual tarzda ko'rsatib, sarlavhada uning natijasini yashil yoki qizil rangda ehtimolligi bilan chiqaradi.

---

## 🧠 CNN Arxitekturasi
* **4 ta Conv Blok:** `Conv2d` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d` (Filtrlar: 32 -> 64 -> 128 -> 256)
* **Pooling:** `AdaptiveAvgPool2d((2, 2))`
* **Classifier:** `Flatten` -> `Dropout(0.4)` -> `Linear(1024, 128)` -> `ReLU` -> `Dropout(0.3)` -> `Linear(128, 2)`
* **Eng yaxshi Test aniqligi:** ~84.21%
