"""
Ko'krak Qafasi Rentgen Tahlili - Grafik Interfeysli Dastur (GUI)
Muallif: CNN Amaliy Loyihasi (5-Modul)
Ishga tushirish: py app.py
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms

# 1. Model arxitekturasi
class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
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

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class XRayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ko'krak Qafasi Rentgen Tahlili (CNN)")
        self.root.geometry("650x720")
        self.root.minsize(550, 650)
        self.root.configure(bg="#f4f6f9")
        
        self.model = None
        self.load_model()
        self.setup_ui()
        
    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Xatolik", f"'{MODEL_PATH}' fayli topilmadi!")
            sys.exit(1)
        self.model = PneumoniaCNN(num_classes=2).to(device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        self.model.eval()
        
    def setup_ui(self):
        # Sarlavha
        header_frame = tk.Frame(self.root, bg="#1e3d59", padx=15, pady=15)
        header_frame.pack(fill=tk.X)
        
        lbl_title = tk.Label(header_frame, text="KO'KRAK QAFASI RENTGEN TAHLILI", 
                             font=("Arial", 16, "bold"), fg="white", bg="#1e3d59")
        lbl_title.pack()
        
        lbl_sub = tk.Label(header_frame, text="CNN (Convolutional Neural Network) Tashxis Modeli", 
                           font=("Arial", 10), fg="#e8ecef", bg="#1e3d59")
        lbl_sub.pack(pady=3)
        
        # Tugma
        btn_frame = tk.Frame(self.root, bg="#f4f6f9", pady=15)
        btn_frame.pack()
        
        self.btn_select = tk.Button(btn_frame, text="📁 Rentgen Rasmini Tanlash (Browse)", 
                                    font=("Arial", 12, "bold"), bg="#17b978", fg="white",
                                    padx=20, pady=8, cursor="hand2", command=self.select_image,
                                    relief=tk.FLAT)
        self.btn_select.pack()
        
        # Rasm ko'rsatish hududi
        self.img_frame = tk.Frame(self.root, bg="white", width=350, height=350, relief=tk.SOLID, bd=1)
        self.img_frame.pack(pady=10)
        self.img_frame.pack_propagate(False)
        
        self.lbl_image = tk.Label(self.img_frame, text="Bu yerda tanlangan rentgen\ntasviri aks etadi", 
                                  font=("Arial", 11), fg="#888", bg="white")
        self.lbl_image.pack(expand=True)
        
        # Natija hududi
        self.result_frame = tk.Frame(self.root, bg="#f4f6f9", pady=10)
        self.result_frame.pack(fill=tk.X, padx=30)
        
        self.lbl_result = tk.Label(self.result_frame, text="Natija: Kutilyapti...", 
                                   font=("Arial", 15, "bold"), bg="#f4f6f9", fg="#555")
        self.lbl_result.pack()
        
        self.lbl_confidence = tk.Label(self.result_frame, text="", 
                                       font=("Arial", 12), bg="#f4f6f9", fg="#333")
        self.lbl_confidence.pack(pady=4)
        
        self.lbl_details = tk.Label(self.result_frame, text="", 
                                    font=("Arial", 10), bg="#f4f6f9", fg="#666")
        self.lbl_details.pack()

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Rentgen tasvirini tanlang",
            filetypes=[
                ("Rentgen rasmlari", "*.jpeg;*.jpg;*.png;*.webp;*.bmp"),
                ("Barcha fayllar", "*.*")
            ]
        )
        if not file_path:
            return
            
        self.analyze_image(file_path)

    def analyze_image(self, file_path):
        try:
            raw_img = Image.open(file_path)
            
            # Tasvirni oynada ko'rsatish uchun resize
            display_img = raw_img.copy()
            display_img.thumbnail((340, 340))
            self.tk_img = ImageTk.PhotoImage(display_img)
            self.lbl_image.configure(image=self.tk_img, text="")
            
            # Inference
            input_tensor = transform(raw_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                conf = probs[pred_idx].item() * 100
                prob_normal = probs[0].item() * 100
                prob_pneumonia = probs[1].item() * 100
                
            if pred_idx == 0:
                result_text = "NATIJA: SOG'LOM"
                result_color = "#28a745" # Yashil
            else:
                result_text = "NATIJA: O'PKA YALLIG'LANISHI ANIQLANDI"
                result_color = "#dc3545" # Qizil
                
            self.lbl_result.configure(text=result_text, fg=result_color)
            self.lbl_confidence.configure(text=f"Ishonchlilik darajasi: {conf:.2f}%", fg=result_color)
            self.lbl_details.configure(
                text=f"Sog'lom: {prob_normal:.1f}%   |   O'pka yallig'lanishi: {prob_pneumonia:.1f}%\nFayl: {os.path.basename(file_path)}"
            )
            
        except Exception as e:
            messagebox.showerror("Xatolik", f"Tasvirni tahlil qilishda xatolik: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = XRayApp(root)
    root.mainloop()
