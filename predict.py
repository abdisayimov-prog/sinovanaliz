"""
Rentgen Tasvirlari Tahlili (CNN Modeli)
Muallif: CNN Amaliy Loyihasi (5-Modul)
Vazifasi: Berilgan ko'krak qafasi rentgen tasviri bo'yicha:
         - "Sog'lom" yoki
         - "O'pka yallig'lanishi aniqlandi"
         natijasini ishonchlilik darajasi bilan chiqarish.
"""

import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Tkinter orqali fayl tanlash oynasi ochish
def select_file_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw() # Asosiy bo'sh oynani yashirish
        root.attributes('-topmost', True) # Oynani eng oldinga chiqarish
        file_path = filedialog.askopenfilename(
            title="Rentgen tasvirini tanlang",
            filetypes=[
                ("Rentgen rasmlari", "*.jpeg;*.jpg;*.png;*.webp;*.bmp"),
                ("Barcha fayllar", "*.*")
            ]
        )
        root.destroy()
        return file_path
    except Exception:
        return ""

# 1. CNN Model Arxitekturasi
class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 1-blok: 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 2-blok: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 3-blok: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 4-blok: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(self.features(x))

# 2. Modelni yuklash
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pneumonia_cnn.pth')

if not os.path.exists(MODEL_PATH):
    print(f"Xatolik: Model fayli '{MODEL_PATH}' topilmadi!")
    sys.exit(1)

model = PneumoniaCNN(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 3. Transformatsiya
infer_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_xray(image_path, show_image=True):
    if not image_path or not os.path.exists(image_path):
        print(f"\n[!] Xatolik: '{image_path}' fayli topilmadi!")
        return None
    
    try:
        raw_img = Image.open(image_path)
    except Exception as e:
        print(f"\n[!] Rasmni ochishda xatolik yuz berdi: {e}")
        return None
    
    input_tensor = infer_transform(raw_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()
        confidence = probabilities[pred_idx].item() * 100
        
        prob_normal = probabilities[0].item() * 100
        prob_pneumonia = probabilities[1].item() * 100
        
    if pred_idx == 0:
        result_text = "Sog'lom"
        result_color = "green"
    else:
        result_text = "O'pka yallig'lanishi aniqlandi"
        result_color = "red"
        
    print("\n" + "=" * 65)
    print("           KO'KRAK QAFASI RENTGEN TAHLILI (CNN)")
    print("=" * 65)
    print(f" Fayl nomi             : {os.path.basename(image_path)}")
    print(f" To'liq yo'l           : {os.path.abspath(image_path)}")
    print(f" NATIJA                : {result_text.upper()}")
    print(f" Ishonchlilik darajasi : {confidence:.2f}%")
    print("-" * 65)
    print(f" [Sog'lom]: {prob_normal:.2f}%   |   [O'pka yallig'lanishi]: {prob_pneumonia:.2f}%")
    print("=" * 65 + "\n")
    
    if show_image:
        try:
            plt.figure(figsize=(6, 6))
            plt.imshow(raw_img, cmap='gray')
            plt.title(f"Natija: {result_text}\n(Ishonchlilik: {confidence:.1f}%)", 
                      fontsize=14, color=result_color, weight='bold', pad=12)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception:
            pass
            
    return {
        'result': result_text,
        'confidence': confidence,
        'prob_normal': prob_normal,
        'prob_pneumonia': prob_pneumonia
    }

def main():
    print("\n" + "#" * 65)
    print("       CNN KO'KRAK QAFASI RENTGEN TASHXIS TIZIMI")
    print("#" * 65)
    
    # 1. Agar buyruqlar satrida argument berilgan bo'lsa
    if len(sys.argv) > 1:
        img_path = sys.argv[1].strip('"\'')
        predict_xray(img_path, show_image=True)
        return

    # 2. Siklda yangi rasmlarni tekshirish
    while True:
        print("\nQuyidagi usullardan birini tanlang:")
        print("  1 - Kompyuterdan rasm tanlash (Fayl tanlash oynasi ochiladi)")
        print("  2 - Rentgen rasmini terminalga sudrab tashlash (Drag & Drop)")
        print("  0 - Chiqish")
        
        choice = input("\nTanlovingizni kiriting (1/2/0): ").strip()
        
        if choice == '1':
            print("Fayl tanlash oynasi ochilmoqda...")
            img_path = select_file_dialog()
            if not img_path:
                print("Hech qanday fayl tanlanmadi.")
                continue
            predict_xray(img_path, show_image=True)
            
        elif choice == '2':
            img_path = input("\nRentgen rasmini ushbu oynaga sudrab tashlang (yoki yo'lini kiriting): ").strip().strip('"\'')
            if not img_path:
                print("Rasm yo'li kiritilmadi.")
                continue
            predict_xray(img_path, show_image=True)
            
        elif choice == '0':
            print("Dastur yakunlandi. Salomat bo'ling!")
            break
        else:
            # Agar foydalanuvchi to'g'ridan-to'g'ri rasm yo'lini tashlab yuborgan bo'lsa
            if os.path.exists(choice.strip('"\'')):
                predict_xray(choice.strip('"\''), show_image=True)
            else:
                print("Noto'g'ri buyruq kiritildi. Qaytadan urinib ko'ring.")

if __name__ == '__main__':
    main()
