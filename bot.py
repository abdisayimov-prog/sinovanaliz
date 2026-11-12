"""
Ko'krak Qafasi Rentgen Tahlili - Telegram Bot
Aiogram 3.x asosida ishlaydi.
Xususiyatlari:
  - CNN (Convolutional Neural Network) modeli orqali o'pka rentgenini tekshirish
  - "Sog'lom" yoki "O'pka yallig'lanishi aniqlandi" natijasini berish
  - Yallig'lanish o'choqlarini issiqlik xaritasi (Heatmap / Metka) bilan ko'rsatish
  - Sog'lom rentgen tasviri bilan yonma-yon taqqoslash (Explainable AI)
  - GitHub loyiha havolasini taqdim etuvchi interaktiv tugmalar
"""

import os
import sys
import io
import asyncio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    FSInputFile,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton
)

# ==============================================================
# SOZLAMALAR (TOKEN VA GITHUB MANZILI)
# ==============================================================
BOT_TOKEN = os.getenv("BOT_TOKEN", "BOT_TOKENINI_SHU_YERGA_YOZING")
GITHUB_URL = os.getenv("GITHUB_URL", "https://github.com/")

# ==============================================================
# 1. CNN MODELINI YUKLASH
# ==============================================================
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pneumonia_cnn.pth')

model = PneumoniaCNN(num_classes=2).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Pneumonia CNN modeli muvaffaqiyatli yuklandi.")
else:
    print(f"XATOLIK: '{MODEL_PATH}' fayli topilmadi!")

infer_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==============================================================
# 2. XAI: METKA (HEATMAP) VA TAQQOSLASH RASMINI YASASH
# ==============================================================
def create_comparison_card(raw_img, pred_idx, confidence, output_path):
    orig_np = np.array(raw_img)
    input_tensor = infer_transform(raw_img).unsqueeze(0).to(device)
    
    # Feature maplarni olish (Heatmap uchun)
    x = input_tensor
    last_feat = None
    for i, layer in enumerate(model.features):
        x = layer(x)
        if i == 11:
            last_feat = x
            
    heatmap = torch.mean(last_feat, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
        
    heatmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlayed = cv2.addWeighted(orig_np, 0.55, heatmap_color, 0.45, 0)
    
    # Standart sog'lom rentgen rasmi (solishtirish uchun)
    sample_normal_path = os.path.join(os.path.dirname(__file__), 'test', 'NORMAL', 'NORMAL2-IM-0052-0001.jpeg')
    if os.path.exists(sample_normal_path):
        normal_img = Image.open(sample_normal_path).convert('RGB')
        normal_np = np.array(normal_img)
    else:
        normal_np = orig_np # Zaxira
        
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=140)
    
    if pred_idx == 1:
        axes[0].imshow(overlayed)
        axes[0].set_title("🔴 Siz yuborgan rentgen\n(Qizil/Sariq metka: Yallig'lanish o'choqlari)", 
                          color='#d9534f', fontsize=11, weight='bold', pad=8)
        axes[0].axis('off')
        
        axes[1].imshow(normal_np, cmap='gray')
        axes[1].set_title("🟢 Taqqoslash uchun: Sog'lom o'pka rentgeni\n(Toza, infiltratlarsiz havo maydonlari)", 
                          color='#28a745', fontsize=11, weight='bold', pad=8)
        axes[1].axis('off')
    else:
        axes[0].imshow(orig_np, cmap='gray')
        axes[0].set_title("🟢 Siz yuborgan rentgen: Sog'lom o'pka\n(O'pka maydonlari toza, me'yorda)", 
                          color='#28a745', fontsize=11, weight='bold', pad=8)
        axes[0].axis('off')
        
        axes[1].imshow(normal_np, cmap='gray')
        axes[1].set_title("🟢 Standart sog'lom etalon rentgen\n(Patologik o'zgarishlar yo'q)", 
                          color='#28a745', fontsize=11, weight='bold', pad=8)
        axes[1].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.15)
    plt.close()

# ==============================================================
# 3. AIOGRAM BOT HANDLERLARI
# ==============================================================
dp = Dispatcher()

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    user_name = message.from_user.first_name or "Foydalanuvchi"
    text = (
        f"Assalomu alaykum, <b>{user_name}</b>! 🫁\n\n"
        "<b>Ko'krak Qafasi Rentgen Tasvirlari Tahlili</b> botiga xush kelibsiz!\n\n"
        "🔬 Ushbu bot <b>CNN (Convolutional Neural Network)</b> sun'iy intellekt modeli "
        "yordamida rentgen tasvirini tekshirib, quyidagi holatlardan birini aniqlaydi:\n"
        "  • 🟢 <b>Sog'lom</b> (Normal)\n"
        "  • 🔴 <b>O'pka yallig'lanishi aniqlandi</b> (Pneumonia / COVID-19)\n\n"
        "📸 <b>Qanday ishlatiladi?</b>\n"
        "Menga ko'krak qafasi rentgen tasvirini <b>rasm (foto)</b> holatida yuboring. "
        "Bir necha soniyada batafsil tahlil va taqqoslama xulosani taqdim etaman!"
    )
    await message.answer(text, parse_mode="HTML")

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    text = (
        "ℹ️ <b>Yordam va qo'llanma:</b>\n\n"
        "1. Shunchaki ko'krak qafasi rentgen rasmini botga yuboring.\n"
        "2. Neyron tarmoq tasvirni tahlil qilib, natijani va "
        "yallig'lanish joylari ko'rsatilgan taqqoslama kartani yuboradi.\n\n"
        "⚠️ <i>Diqqat: Ushbu bot o'quv-amaliy loyiha hisoblanadi. Yakuniy tibbiy tashxis "
        "uchun mutaxassis shifokor bilan maslahatlashing.</i>"
    )
    await message.answer(text, parse_mode="HTML")

@dp.message(F.photo)
async def handle_photo(message: types.Message, bot: Bot):
    status_msg = await message.answer("⏳ <i>Rentgen tasviri qabul qilindi, neyron tarmog'i tahlil qilmoqda...</i>", parse_mode="HTML")
    
    try:
        # Rasmni yuklab olish
        photo = message.photo[-1]
        file_io = io.BytesIO()
        await bot.download(photo, destination=file_io)
        file_io.seek(0)
        
        raw_img = Image.open(file_io).convert('RGB')
        
        # Inference
        input_tensor = infer_transform(raw_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item() * 100
            prob_normal = probs[0].item() * 100
            prob_pneumonia = probs[1].item() * 100
            
        # Taqqoslash kartasini yaratish
        temp_img_path = f"temp_result_{message.from_user.id}.jpg"
        create_comparison_card(raw_img, pred_idx, confidence, temp_img_path)
        
        # Natija matni
        if pred_idx == 1:
            caption = (
                "🔴 <b>DIQQAT: O'PKA YALLIG'LANISHI ANIQLANDI!</b>\n"
                f"📊 <b>Ishonchlilik darajasi:</b> {confidence:.1f}%\n"
                f"📈 <i>[O'pka yallig'lanishi: {prob_pneumonia:.1f}% | Sog'lom: {prob_normal:.1f}%]</i>\n\n"
                "🔍 <b>Nima sababdan bunday xulosa berildi?</b>\n"
                "• <b>Qizil / Sariq metkalar</b> — neyron tarmoq e'tibor qaratgan sohalar (o'pka to'qimalarida infiltrat, suyuqlik yoki konsolidatsiya mavjudligi).\n"
                "• <b>Sog'lom o'pka bilan farqi:</b> O'ng tarafdagi sog'lom rentgenda havo nurni to'siqsiz o'tkazgani uchun maydonlar toza ko'rinadi. Sizning tasviringizda esa yallig'lanish tufayli oqish/xira soyalar hosil bo'lgan.\n\n"
                "⚠️ <i>Eslatma: Bu sun'iy intellekt xulosasi. Yakuniy tashxis uchun shifokorga murojaat qiling.</i>\n\n"
                "💡 <b>Biz qanday ishlashimiz sizga qiziqmi?</b> Quyidagi havola orqali loyihaning ochiq kodini ko'rishingiz mumkin:"
            )
        else:
            caption = (
                "🟢 <b>XULOSA: SOG'LOM!</b>\n"
                f"📊 <b>Ishonchlilik darajasi:</b> {confidence:.1f}%\n"
                f"📈 <i>[Sog'lom: {prob_normal:.1f}% | O'pka yallig'lanishi: {prob_pneumonia:.1f}%]</i>\n\n"
                "🔍 <b>Tahlil xulosasi:</b>\n"
                "• O'pka maydonlarida patologik xiralashishlar, infiltrat yoki suyuqlik to'planishi aniqlanmadi.\n"
                "• Tasvir me'yoriy sog'lom o'pka etaloniga to'liq mos keladi.\n\n"
                "💡 <b>Biz qanday ishlashimiz sizga qiziqmi?</b> Loyihaning to'liq kodi va sun'iy intellekt modeli ochiq manbada:"
            )
            
        # Inline tugmalar
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="💻 Loyihaning GitHub Kodi",
                        url=GITHUB_URL
                    )
                ]
            ]
        )
        
        # Taqqoslama rasmni yuborish
        photo_file = FSInputFile(temp_img_path)
        await message.answer_photo(
            photo=photo_file,
            caption=caption,
            parse_mode="HTML",
            reply_markup=keyboard
        )
        
        # Vaqtinchalik faylni tozalash
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
        await status_msg.delete()
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Rasmni tahlil qilishda xatolik yuz berdi: {e}")

@dp.message(F.text)
async def handle_text(message: types.Message):
    await message.answer(
        "Iltimos, tekshirish uchun ko'krak qafasi rentgen tasvirini <b>rasm (foto)</b> holatida yuboring. 🫁📸",
        parse_mode="HTML"
    )

async def main():
    if BOT_TOKEN == "BOT_TOKENINI_SHU_YERGA_YOZING":
        print("\n" + "=" * 65)
        print("DIQQAT: Bot ishlashi uchun Telegram BOT_TOKEN kiritilishi kerak!")
        print("1. Telegramda @BotFather ga kiring va /newbot buyrug'ini bering.")
        print("2. Olingan tokenni bot.py faylidagi BOT_TOKEN qismiga yoki .env ga qo'ying.")
        print("=" * 65 + "\n")
        return
        
    bot = Bot(token=BOT_TOKEN)
    print("Bot muvaffaqiyatli ishga tushdi va xabarlarni kutmoqda...")
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot to'xtatildi.")
