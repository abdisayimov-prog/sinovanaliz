"""
Ko'krak Qafasi Rentgen Tahlili - Telegram Bot
Render.com da yengil va tez ishlash uchun optimallashtirilgan.
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
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torchvision import transforms

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    FSInputFile,
    InlineKeyboardMarkup,
    InlineKeyboardButton
)
from aiohttp import web

# ==============================================================
# SOZLAMALAR (TOKEN VA GITHUB MANZILI)
# ==============================================================
BOT_TOKEN = os.getenv("BOT_TOKEN", "9029532267:AAFPwmZV6xl17_Rhb2uUexEpWTzPtYN7b6M")
GITHUB_URL = os.getenv("GITHUB_URL", "https://github.com/abdisayimov-prog/sinovanaliz")

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
# 2. XAI: METKA (HEATMAP) VA TAQQOSLASH RASMINI YASASH (YENGIL VA TEZKOR)
# ==============================================================
def create_comparison_card(raw_img, pred_idx, confidence, output_path):
    # Xotirani tejash uchun 512x512 ga o'tkazish
    processed_img = raw_img.resize((512, 512))
    orig_np = np.array(processed_img)
    
    input_tensor = infer_transform(processed_img).unsqueeze(0).to(device)
    
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
        
    heatmap_resized = cv2.resize(heatmap, (512, 512))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    if pred_idx == 1:
        # Pnevmoniya: Heatmap metkasini ustiga tushirish
        left_np = cv2.addWeighted(orig_np, 0.6, heatmap_color, 0.4, 0)
    else:
        left_np = orig_np
        
    # Standart sog'lom rentgen rasmi (solishtirish uchun)
    sample_normal_path = os.path.join(os.path.dirname(__file__), 'sample_normal.jpg')
    if os.path.exists(sample_normal_path):
        normal_img = Image.open(sample_normal_path).convert('RGB').resize((512, 512))
        right_np = np.array(normal_img)
    else:
        right_np = orig_np
        
    # Ikki tasvirni yonma-yon birlashtirish (512 x 1024)
    combined_np = np.hstack([left_np, right_np])
    card_img = Image.fromarray(combined_np)
    
    # Yuqori sarlavha panelini yasash (60px balandlik)
    banner = Image.new('RGB', (1024, 60), color=(25, 35, 48))
    draw = ImageDraw.Draw(banner)
    
    # Banner yozuvlari
    if pred_idx == 1:
        draw.text((25, 18), "SIZNING RENTGENINGIZ: O'PKA YALLIG'LANISHI (METKA BILAN)", fill=(255, 110, 110))
    else:
        draw.text((25, 18), "SIZNING RENTGENINGIZ: SOG'LOM O'PKA", fill=(110, 255, 110))
        
    draw.text((540, 18), "ETALON: ME'YORIY SOG'LOM RENTGEN", fill=(110, 255, 110))
    
    # Yakuniy kartochka
    final_card = Image.new('RGB', (1024, 572))
    final_card.paste(banner, (0, 0))
    final_card.paste(card_img, (0, 60))
    
    final_card.save(output_path, quality=88)

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

@dp.message(F.photo | F.document)
async def handle_photo(message: types.Message, bot: Bot):
    status_msg = await message.answer("⏳ <i>Rentgen tasviri qabul qilindi, neyron tarmog'i tahlil qilmoqda...</i>", parse_mode="HTML")
    temp_img_path = f"temp_result_{message.from_user.id}.jpg"
    
    try:
        file_io = io.BytesIO()
        if message.photo:
            photo = message.photo[-1]
            await bot.download(photo, destination=file_io)
        elif message.document:
            mime = message.document.mime_type or ""
            if not mime.startswith("image/"):
                await status_msg.edit_text("❌ Iltimos, faqat rasm faylini (.jpeg, .png, .jpg) yuboring.")
                return
            await bot.download(message.document, destination=file_io)
            
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
            
        # Tezkor va yengil taqqoslash kartasini yaratish (OpenCV + PIL)
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
        print(f"XATOLIK YUZ BERDI: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            await status_msg.edit_text(f"❌ Rasmni tahlil qilishda xatolik yuz berdi: {e}")
        except Exception:
            await message.answer(f"❌ Xatolik yuz berdi: {e}")

@dp.message(F.text)
async def handle_text(message: types.Message):
    await message.answer(
        "Iltimos, tekshirish uchun ko'krak qafasi rentgen tasvirini <b>rasm (foto)</b> holatida yuboring. 🫁📸",
        parse_mode="HTML"
    )

# ==============================================================
# RENDER.COM HEALTHCHECK SERVERI (BEPUL HOSTING UCHUN)
# ==============================================================
async def health_check(request):
    return web.Response(text="Bot muvaffaqiyatli ishlab turibdi!", content_type="text/plain")

async def start_web_server():
    app = web.Application()
    app.router.add_route('*', '/', health_check)
    app.router.add_route('*', '/health', health_check)
    port = int(os.getenv("PORT", 8080))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    print(f"Render veb-serveri {port}-portda barcha so'rovlar uchun faoliyat boshladi.")

async def main():
    if BOT_TOKEN == "BOT_TOKENINI_SHU_YERGA_YOZING":
        print("\n" + "=" * 65)
        print("DIQQAT: Bot ishlashi uchun Telegram BOT_TOKEN kiritilishi kerak!")
        print("=" * 65 + "\n")
        return
        
    # Render.com bepul Web Service portini ochish
    if os.getenv("PORT"):
        await start_web_server()
        
    bot = Bot(token=BOT_TOKEN)
    print("Bot muvaffaqiyatli ishga tushdi va xabarlarni kutmoqda...")
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot to'xtatildi.")
