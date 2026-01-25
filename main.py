import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import io
import cv2
import os
import json
import segmentation_models_pytorch as smp
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from datetime import datetime

# Firebase imports
import firebase_admin
from firebase_admin import credentials, storage, firestore

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

models = {}

# Firebase Storage 버킷명
FIREBASE_BUCKET = "knu-team-04.firebasestorage.app"

# 부품 목록 (24개)
PART_CLASSES = [
    "Front bumper", "Front fender(L)", "Front fender(R)", 
    "Head lights(L)", "Head lights(R)", "Rear bumper", 
    "Rear fender(L)", "Rear fender(R)", 
    "Front door(L)", "Front door(R)", "Rear door(L)", "Rear door(R)", 
    "Side mirror(L)", "Side mirror(R)", "Rocker panel(L)", "Rocker panel(R)", 
    "Front Wheel(L)", "Front Wheel(R)", "Rear Wheel(L)", "Rear Wheel(R)", 
    "Bonnet", "Windshield", "Trunk lid", "Rear windshield"
]

DAMAGE_CLASSES = ["Scratched", "Separated", "Crushed", "Breakage"]

# 손상 타입별 색상 (BGR)
DAMAGE_COLORS = {
    "Scratched": (0, 255, 255),    # 노란색
    "Separated": (255, 0, 255),    # 마젠타
    "Crushed": (0, 165, 255),      # 오렌지
    "Breakage": (0, 0, 255),       # 빨간색
}

# 차종 리스트 (전체 유지 - 간략화를 위해 주요 부분만 표시)
KNOWN_MODELS = ['레이', '레이(18)', '쏘나타(DN8)', '싼타페(15)', 'K5(18)', '그랜저IG(17)']  # 실제로는 전체 리스트 사용

DEFAULT_FALLBACK_CAR = "쏘나타(DN8)"

MINIMUM_COST_BY_PART = {
    "Front bumper": 200000, "Rear bumper": 200000,
    "Front door(L)": 300000, "Front door(R)": 300000,
    "Rear door(L)": 300000, "Rear door(R)": 300000,
    "Bonnet": 350000, "Trunk lid": 300000,
    "Front fender(L)": 250000, "Front fender(R)": 250000,
    "Rear fender(L)": 250000, "Rear fender(R)": 250000,
    "Windshield": 400000, "Rear windshield": 300000,
    "Head lights(L)": 200000, "Head lights(R)": 200000,
    "Side mirror(L)": 100000, "Side mirror(R)": 100000,
    "Rocker panel(L)": 200000, "Rocker panel(R)": 200000,
    "Front Wheel(L)": 150000, "Front Wheel(R)": 150000,
    "Rear Wheel(L)": 150000, "Rear Wheel(R)": 150000,
}

DAMAGE_MULTIPLIER = {
    "Scratched": 0.3, "Separated": 0.7, "Crushed": 0.9, "Breakage": 1.0,
}

# Firebase 함수
def initialize_firebase():
    if not firebase_admin._apps:
        cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service-account-key.json')
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
        else:
            cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_BUCKET})
        print(f"✅ Firebase 초기화: {FIREBASE_BUCKET}")

def upload_to_firebase_storage(file_bytes, folder, filename):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"{folder}/{filename}")
        content_type = 'application/json' if filename.endswith('.json') else 'image/jpeg'
        blob.upload_from_string(file_bytes, content_type=content_type)
        blob.make_public()
        print(f"✅ 업로드: {folder}/{filename}")
        return blob.public_url
    except Exception as e:
        print(f"❌ 업로드 실패: {e}")
        return None

def save_to_firestore(data):
    try:
        db = firestore.client()
        doc_ref = db.collection('damage_analyses').add({**data, 'timestamp': firestore.SERVER_TIMESTAMP})
        return doc_ref[1].id
    except:
        return None

def create_visualization(original_image_bytes, part_mask, damage_masks, detected_parts_info):
    original_img = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
    original_size = original_img.size
    img_resized = original_img.resize((512, 512))
    img_np = np.array(img_resized)
    overlay = img_np.copy()
    
    for info in detected_parts_info:
        try:
            part_id = PART_CLASSES.index(info["part"]) + 1
            part_area = (part_mask == part_id)
            damage_idx = DAMAGE_CLASSES.index(info["damage"])
            damage_area = (damage_masks[damage_idx] == 1)
            damaged_area = part_area & damage_area
            color = DAMAGE_COLORS.get(info["damage"], (255, 255, 255))
            overlay[damaged_area] = color
        except:
            continue
    
    blended = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
    blended_pil = Image.fromarray(blended.astype(np.uint8))
    blended_resized = blended_pil.resize(original_size, Image.LANCZOS)
    buffered = io.BytesIO()
    blended_resized.save(buffered, format="JPEG", quality=95)
    return buffered.getvalue()

def find_similar_model(user_input):
    if not user_input:
        return DEFAULT_FALLBACK_CAR
    if user_input in KNOWN_MODELS:
        return user_input
    user_clean = user_input.replace(" ", "").lower()
    for model in KNOWN_MODELS:
        if user_clean == model.replace(" ", "").lower():
            return model
    for model in KNOWN_MODELS:
        if user_clean in model.replace(" ", "").lower():
            return model
    return DEFAULT_FALLBACK_CAR

class DamageUnetWrapper(nn.Module):
    def __init__(self, num_classes, encoder, pre_weight):
        super().__init__()
        self.model = smp.Unet(classes=num_classes, encoder_name=encoder, 
                             encoder_weights=pre_weight, in_channels=3)
    def forward(self, x):
        return self.model(x)

def load_part_model(path):
    print(f"Loading Part Model from {path}...")
    model = smp.DeepLabV3(encoder_name="efficientnet-b2", encoder_weights=None, in_channels=3, classes=25)
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def load_damage_model(path):
    print(f"Loading Damage Model from {path}...")
    model_wrapper = DamageUnetWrapper(num_classes=2, encoder="resnet34", pre_weight=None)
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    try:
        model_wrapper.model.load_state_dict(new_state_dict)
    except RuntimeError:
        model_wrapper.load_state_dict(new_state_dict)
    model_wrapper.to(device)
    model_wrapper.eval()
    return model_wrapper

@app.on_event("startup")
async def startup_event():
    print("="*60)
    print("🚀 V3 모델 + Firebase 서버 시작")
    print("="*60)
    try:
        initialize_firebase()
    except Exception as e:
        print(f"⚠️ Firebase 초기화 실패: {e}")
    try:
        models['part'] = load_part_model("models/best_unet_size512_epoch25.pth")
        models['damage_0'] = load_damage_model("models/Unet_damage_label0.pt") 
        models['damage_1'] = load_damage_model("models/Unet_damage_label1.pt") 
        models['damage_2'] = load_damage_model("models/Unet_damage_label2.pt") 
        models['damage_3'] = load_damage_model("models/Unet_damage_label3.pt") 
        models['cost_predictor'] = joblib.load("models/cost_predictor_v3.pkl")
        print("✅ 모든 모델 로드 성공!")
        print("="*60)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        raise e

def preprocess_image(image_bytes, target_size=512):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((target_size, target_size))
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)
    return torch.from_numpy(img_np).unsqueeze(0).float().to(device)

def get_cost_prediction(car_model, part_name, damage_type):
    try:
        input_df = pd.DataFrame([{'Car_Model': car_model, 'Part': part_name, 'Damage_Type': damage_type}])
        pred_value = models['cost_predictor'].predict(input_df)[0]
        cost = int(max(pred_value, 0))
        min_cost = MINIMUM_COST_BY_PART.get(part_name, 100000)
        multiplier = DAMAGE_MULTIPLIER.get(damage_type, 1.0)
        guaranteed_min = int(min_cost * multiplier)
        if cost < guaranteed_min * 0.5:
            cost = guaranteed_min
        if cost > 5000000:
            cost = 5000000
        return cost, car_model
    except:
        fallback_df = pd.DataFrame([{'Car_Model': DEFAULT_FALLBACK_CAR, 'Part': part_name, 'Damage_Type': damage_type}])
        pred_value = models['cost_predictor'].predict(fallback_df)[0]
        cost = int(max(pred_value, 0))
        min_cost = MINIMUM_COST_BY_PART.get(part_name, 100000)
        multiplier = DAMAGE_MULTIPLIER.get(damage_type, 1.0)
        cost = max(cost, int(min_cost * multiplier))
        return cost, f"{car_model} (대체: {DEFAULT_FALLBACK_CAR})"

@app.post("/predict")
async def predict(car_model: str = Form(...), file: UploadFile = File(...), user_id: str = Form(default="anonymous")):
    try:
        matched_car_model = find_similar_model(car_model)
        print(f"📝 {car_model} -> {matched_car_model}")
        
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)
        
        with torch.no_grad():
            part_out = models['part'](input_tensor)
            part_mask = torch.argmax(part_out, dim=1).cpu().numpy()[0]
            damage_masks = {}
            for i in range(4):
                d_out = models[f'damage_{i}'](input_tensor)
                damage_masks[i] = torch.argmax(d_out, dim=1).cpu().numpy()[0]

        total_estimated_cost = 0
        detected_parts_info = []
        final_car_model_used = matched_car_model

        for part_id in np.unique(part_mask):
            if part_id == 0 or part_id - 1 >= len(PART_CLASSES):
                continue
            part_name = PART_CLASSES[part_id - 1] 
            current_part_mask = (part_mask == part_id)
            max_damage_pixels = 0
            detected_damage_type = "Scratched"
            found_damage = False
            
            for i, d_name in enumerate(DAMAGE_CLASSES):
                overlap = current_part_mask & (damage_masks[i] == 1)
                overlap_pixels = np.sum(overlap)
                if overlap_pixels > 100:
                    found_damage = True
                    if overlap_pixels > max_damage_pixels:
                        max_damage_pixels = overlap_pixels
                        detected_damage_type = d_name

            if found_damage:
                cost, used_model = get_cost_prediction(matched_car_model, part_name, detected_damage_type)
                if DEFAULT_FALLBACK_CAR in used_model:
                    final_car_model_used = used_model
                total_estimated_cost += cost
                detected_parts_info.append({"part": part_name, "damage": detected_damage_type, "cost": cost})

        visualization_bytes = create_visualization(image_bytes, part_mask, damage_masks, detected_parts_info)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        damage_image_url = upload_to_firebase_storage(visualization_bytes, "damage", f"{user_id}_{timestamp}_damage.jpg")
        estimate_data = {"userId": user_id, "carModel": car_model, "carModelApplied": final_car_model_used, 
                        "totalCost": total_estimated_cost, "details": detected_parts_info, 
                        "timestamp": timestamp, "analysisDate": datetime.now().isoformat()}
        estimate_url = upload_to_firebase_storage(json.dumps(estimate_data, ensure_ascii=False, indent=2).encode('utf-8'), 
                                                  "estimate", f"{user_id}_{timestamp}_estimate.json")
        
        doc_id = save_to_firestore({"userId": user_id, "carModel": car_model, "carModelApplied": final_car_model_used, 
                                    "totalCost": total_estimated_cost, "details": detected_parts_info, 
                                    "damageImageUrl": damage_image_url, "estimateUrl": estimate_url})

        print(f"💰 총 {total_estimated_cost:,}원")
        return {"status": "success", "analysisId": doc_id, "car_model_input": car_model, 
                "car_model_applied": final_car_model_used, "total_cost": total_estimated_cost, 
                "details": detected_parts_info, "damage_image_url": damage_image_url, "estimate_url": estimate_url}
    except Exception as e:
        print(f"❌ {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "v3-firebase", "models_loaded": len(models), "firebase_bucket": FIREBASE_BUCKET}