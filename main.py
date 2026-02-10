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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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

# [설정] 시각화 색상 (RGB 포맷)
# Part: Cyan (청록색)
VIS_PART_COLOR = (0, 255, 255)
# Damage: Orange (주황색) - 요청하신 색상 반영
VIS_DAMAGE_COLOR = (255, 165, 0)

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

# 차종 리스트 
KNOWN_MODELS = [
    '1SERIES 3door(12)-F21', '1SERIES 5door(12)-F20', '200c', '300C',
    '3SERIES COUPE(05)-E92', '3SERIES GT(13)-F34', '3SERIES TOURING(12)-F31',
    '3SERIES(12)-F30', '3SERIES(15)-F30', '3SERIES(19)-G20',
    '4SERIES GRANCOUPE(20)-G26', '500 C(13)', '500(13)', '5GT(10)-F07',
    '5SERIES TOURING(10)-F11', '5SERIES(03)-E60', '5SERIES(10)-F10',
    '5SERIES(13)-F10', '5SERIES(17)-G30', '6SERIES GT(17)-G32',
    '7SERIES(08)-F02, F04', '7SERIES(15)-G11', '7SERIES(15)-G12',
    'A CLASS SEDAN(20)-W177', 'A CLASS(13)-W176', 'A CLASS(16)-W176',
    'A CLASS(18)-W177', 'A3 3DOOR(12)-8V', 'A3 5DOOR(03)-8P, 8P3',
    'A3 5DOOR(12)-8V', 'A3(12)-8V', 'A4(07)-B8', 'A4(15)-B9',
    'A6(11)-C7', 'A6(15)-C7', 'A6(18)-C8', 'A6(97)-C5',
    'A7(10)-4G8', 'A8 LWB(09)-D4', 'A8 LWB(17)-D5', 'A8(02)-D3',
    'A8(10)-D4', 'ACCORD(15)-9TH', 'ACCORD(18)-10TH', 'ALTIMA(06)-L32',
    'ALTIMA(12)-L33', 'AMG GT 4DOOR(19)', 'ARTEON(19)', 'ATS',
    'ATS COUPE(15)', 'All New SM7', 'B CLASS(12)-W246', 'BMW 118D',
    'BMW 1시리즈', 'BMW 2시리즈', 'BMW 318i', 'BMW 320i', 'BMW 323i',
    'BMW 330is', 'BMW 3시리즈', 'BMW 4시리즈', 'BMW 520D', 'BMW 528i',
    'BMW 530', 'BMW 530is', 'BMW 535i', 'BMW 5시리즈', 'BMW 6시리즈',
    'BMW 7시리즈', 'BMW GT', 'BMW M3', 'BMW M5', 'BMW MINI',
    'BMW M시리즈', 'BMW X시리즈', 'BMW Z4', 'BMW Z시리즈', 'BMW 쿠퍼',
    'C CLASS CABRIOLET(17)', 'C CLASS COUPE(16)', 'C CLASS ESTATE(16)',
    'C CLASS(14)-W205', 'C30(07)', 'C4', 'C4 CACTUS(14)',
    'CAMRY HYBRID(15)-7.5TH ', 'CAMRY(12)-XV50', 'CAMRY(15)-XV50',
    'CAMRY(17)-XV70', 'CANYON(14)', 'CAPTUR(20)-HJB', 'CAYENNE(11)-958',
    'CAYENNE(17)-9Y0', 'CAYMAN(05)-987', 'CAYMAN(16)-982',
    'CHEROKEE(07)-KK', 'CHEROKEE(13)-KL', 'CK MINI VAN',
    'CLA CLASS(14)-C117', 'CLS CLASS SHOOTING BRAKE(13)',
    'CLS CLASS(11)-W218', 'CLS CLASS(18)-C257', 'COMPASS(16)-2ND',
    'COOPER 3DOOR(01)-1ST', 'COOPER 3DOOR(14)-3RD', 'COOPER 5DOOR(14)-3RD',
    'COOPER CONVERTIBLE(16)-3RD', 'COOPER COUPE(11)', 'COOPER D(14)-F56',
    'CORVETTE', 'COUNTRYMAN(11)-1ST', 'COUNTRYMAN(17)-2ND', 'CR-V',
    'CR-V(12)-4TH', 'CT(13)-ZWA10', 'CUBE(11)-Z12', 'Compass',
    'DBS', 'DBS Volante', 'DISCOVERY 5(17)-5TH', 'DISCOVERY SPORT(14)-1ST',
    'DISCOVERY SPORT(20)-2ND', 'DS4', 'Discovery (LJ)', 'Discovery II (LT)',
    'E CLASS CABRIOLET(13)', 'E CLASS CABRIOLET(16)', 'E CLASS COUPE(13)',
    'E CLASS COUPE(16)', 'E CLASS(13)-W212', 'E CLASS(16)-W213',
    'EQ900', 'ES(15)-XV60', 'ES(18)-XV70', 'ESCAPE(12)-3RD',
    'EVOQUE convertible(11)-1ST', 'EVOQUE(11)-L538(1ST)',
    'EXPLORER(10)-5TH', 'EXPLORER(15)-U502', 'EXPLORER(19)-6TH',
    'Escalade', 'F-PACE(16)', 'FLYINGSPUR SPEED(13)-2ND',
    'FOCUS 5DOOR(10)-3RD', 'Freelancer (LN)', 'G4 렉스턴',
    'G70(17)', 'G80(16)', 'G80(20)-RG3', 'GHIBLI(13)',
    'GLA CLASS(14)-X156', 'GLC CLASS COUPE(17)', 'GLC CLASS COUPE(20)',
    'GLC CLASS(16)-X253', 'GLE CLASS COUPE(16)', 'GLE CLASS(16)-W166',
    'GLE CLASS(20)-V167', 'GM', 'GM Jimmi 짚', 'GM 까메로',
    'GM 캐딜락', 'GOLF(05)-MK5', 'GOLF(13)-MK7', 'GV70(20)-JK1',
    'GV70(21)', 'GV80(20)', 'GV80(20)-JX1', 'HG그랜저', 'HG그랜져',
    'HUSTLER(15)', 'JETTA(11)-6TH', 'JUKE(14)', 'K3', 'K3(18)',
    'K3(4도어)(16)', 'K5', 'K5 하이브리드', 'K5(15)', 'K5(18)',
    'K5(19)-DL3', 'K7', 'K8(21)-GL3', 'K9', 'K9(15)', 'K9(18)',
    'KENBO600', 'LANDAU', 'LF 쏘나타', 'LF 쏘나타 뉴라이즈',
    'MACAN(13)-95B', 'MAXIMA(15)-A36', 'MAYBACH S CLASS(15)-X222',
    'MKS(12)-2ND', 'MKZ(12)-2ND', 'MKZ(12)-2ST', 'MONDEO(14)-4TH',
    'MUSTANG(14)-6TH', 'NEW BMW  520i', 'NEW BMW 525i', 'NEW BMW 730',
    'NEW BMW 740Li', 'NEW BMW 740iL', 'NF 쏘나타', 'NX(17)-AZ10',
    'New EF 쏘나타', 'New SM5 플래티넘', 'ODYSSEY(17)-5TH',
    'PANAMERA(09)-970', 'PASSAT GT(18)-B8', 'PASSAT(12)-B7',
    'PILOT(12)-2ND', 'PILOT(16)-3RD', 'POLO(13)-5TH', 'PRIUS C(18)-',
    'PRIUS(09)-XW30', 'PRIUS(16)-XW50', 'Q3(11)-8U', 'Q5(08)-8R',
    'Q5(12)-8R', 'Q5(17)', 'Q50(14)', 'Q7(16)-4M', 'QM3', 'QM5',
    'QM5(11)', 'QM5네오(14)', 'QM6', 'QX50(16)-J50', 'RAM SRT(18)-5TH',
    'RAV4(13)-XA40', 'RENEGADE(14)-1ST', 'RX(09)-AL20',
    'Range Rover (LH)', 'Range Rover (LP)', 'Rapide', 'Remegade',
    'S CLASS COUPE(18)', 'S CLASS LWB(18)-W222', 'S CLASS(14)-W222',
    'S3', 'S60(11)-2ND', 'S80(06)', 'S90(16)', 'SCIROCCO(12)-3RD',
    'SEBRING', 'SLK CLASS(11)-R172', 'SM 3', 'SM 7', 'SM3 (09)',
    'SM3 (2.0)', 'SM5 (2.5)', 'SM5 Nova', 'SM5 TCE(13)',
    'SM5 뉴임프레션', 'SM5(10)', 'SM6', 'SM7 New Art', 'SM7 Nova',
    'TG그랜져', 'THE BEETLE(12)-3RD', 'TIGUAN ALLSPACE(18)-2ND',
    'TIGUAN(11)-1ST', 'TIGUAN(11)-B7', 'TIGUAN(18)-2ND', 'UX(19)-1ST',
    'V40(13)-1ST', 'V8 Vantage', 'WRANGER SAHARA 4DOOR(07)-JK',
    'WRANGER SPORT 4DOOR(18)-JL', 'X-TRAIL(13)-T32', 'X1(16)-F48',
    'X3', 'X3(14)-F25', 'X3(17)-G01', 'X4(14)-F26', 'X5',
    'X5(13)-F15', 'X6', 'X6(14)-F16', 'XE(15)-X760', 'XF(11)-X250',
    'XF(16)-X260', 'XJ LWB(09)-X351', 'XJ(09)-X351', 'XJ(20)',
    'XM3(20)-LJL', 'YF 쏘나타', 'YF쏘나타 하이브리드', 'i30',
    'i30 (2012)', 'i30(2017)', 'i30cw', 'i40', 'i40 살룬',
    'i40 살룬(15)', 'i40(15)', '그랜드 체로키 짚', '그랜드스타렉스 3밴',
    '그랜드스타렉스 3밴(15)', '그랜드스타렉스 5밴', '그랜드스타렉스 5밴(15)',
    '그랜드스타렉스 웨건', '그랜드스타렉스 웨건(15)', '그랜드스타렉스3밴(18)',
    '그랜드스타렉스5밴(18)', '그랜드스타렉스웨건(18)', '그랜드카니발',
    '그랜저(19)-IG', '그랜저IG(17)', '그랜저IG(20)', '그랜져IG(17)',
    '그렌져XG', '기블리', '뉴 SM3', '뉴 SM5', '뉴 모닝',
    '뉴 베르나(05)', '뉴 체어맨', '뉴SM3(2012)', '뉴그랜져XG',
    '뉴렉스턴', '뉴벤츠 E200', '뉴벤츠 E230', '뉴벤츠 S430L',
    '뉴쏘렌토R(13)', '뉴아우디 A6', '뉴에쿠스', '뉴엑센트(4Dr)',
    '뉴엑센트(5Dr)', '뉴카니발9인승(06)', '뉴카렌스(06)', '뉴프라이드5DR(05)',
    '니로', '니로EV(19)', '닛산', '닛산 리프', '닛산 맥시마',
    '닛산 알티마', '닛산 인피니티', '닛산 쥬크', '닛산 캐시카이',
    '닛산 큐브', '닷지 다코타', '더넥스트스파크', '도요다',
    '도요다 86', '도요다 랙서스', '도요다 아벨론', '도요다 캠리',
    '도요다 툰두라', '도요타 라브4', '도요타 시에나', '라세티',
    '랜드로버 디스커버리', '랜드로버 렌지로버', '랜드로버 로버미니',
    '랭글러 짚', '레이', '레이(18)', '레인지로버 이보크',
    '렉스턴', '렉스턴 II', '로체', '로체 이노베이션',
    '마이바흐', '마쯔다 RX-5', '말리부', '맥스크루즈',
    '맥스크루즈(16)', '모닝', '모하비', '모하비(16)',
    '무쏘스포츠', '미찌비스', '베뉴(19)', '베라크루즈',
    '베르나(4Dr)', '벤츠', '벤츠 200', '벤츠 A클래스',
    '벤츠 B클래스', '벤츠 C200', '벤츠 C230', '벤츠 CLA클래스',
    '벤츠 CLS', '벤츠 CLS클래스', '벤츠 CL클래스', '벤츠 C클래스',
    '벤츠 E220', '벤츠 E300', '벤츠 E350', '벤츠 E420',
    '벤츠 E클래스', '벤츠 G350', '벤츠 GLE', '벤츠 GLK 220',
    '벤츠 GLK클래스', '벤츠 GLS', '벤츠 ML320', '벤츠 ML350',
    '벤츠 R클래스', '벤츠 S350L', '벤츠 S500', '벤츠 S500L',
    '벤츠 S63 AMG', '벤츠 SLK200', '벤츠 SLK230', '벤츠 SLK클래스',
    '벤츠 S클래스', '벤츠 컨버터블', '벤츠GLC', '벤츠스프린터',
    '벨로스터', '벨로스터(15)', '벨로스터(17)', '볼보',
    '볼보 S60', '볼보 S70', '볼보 S80', '볼보 S90',
    '볼보 V40', '볼보 V60', '볼보 V90', '볼보 XC60',
    '볼보 XC70', '볼보 XC90', '뷰티풀코란도', '세피아레오',
    '셀토스(19)', '스타리아 투어러(21)', '스타리아 투어러(21)-US4',
    '스토닉', '스팅어(17)', '스파크(마티즈크리에이티브)', '스포티지(16)',
    '스포티지R', '신형 스타렉스6밴', '신형 스타렉스점보', '싼타페',
    '싼타페(15)', '싼타페(2012)', '싼타페CM', '싼타페CM(10)',
    '싼타페DM', '싼타페TM(18)', '쏘나타 트랜스폼', '쏘나타(19)',
    '쏘렌토', '쏘렌토R', '쏘울', '쏘울(16)', '씨트로엥',
    '아반떼 하이브리드 LPi', '아반떼(20)-CN7', '아반떼AD(16)',
    '아반떼HD(06)', '아반떼MD(10)', '아반떼MD(13)', '아베오(세단)',
    '아베오(해치백)', '아슬란', '아우디 A1', '아우디 A3',
    '아우디 A4', '아우디 A5', '아우디 A6', '아우디 A7',
    '아우디 A8', '아우디 Q3', '아우디 Q5', '아우디 Q7',
    '아우디 R8', '아우디 S4', '아우디 S5', '아우디 S8',
    '아우디 SQ5', '아웃백', '아이오닉', '알토라팡',
    '알페온', '액티언스포츠', '에쿠스(09)', '엑센트(11)',
    '엑센트(15)', '오피러스(06)', '올 뉴 K7', '올 뉴 모닝(11)',
    '올 뉴 쏘울', '올뉴말리부', '올뉴말리부(2017)', '올뉴모닝(15)',
    '올뉴모닝(17)', '올뉴쏘렌토', '올뉴쏘렌토(18)', '올뉴쏘울(16)',
    '올뉴카니발(14)', '올뉴카렌스', '올뉴크루즈(17)', '올뉴투싼',
    '올뉴투싼(15)', '올뉴프라이드4DR(12)', '올뉴프라이드5DR(12)',
    '올란도', '윈스톰', '윈스톰맥스', '임팔라',
    '임팔라(15)-10TH', '재규어', '재규어 F-TYPE', '재규어 New XJ',
    '재규어 X-TYPE', '재규어 XE', '재규어 XF', '재규어 XJ',
    '제네시스', '제네시스 쿠페', '제네시스(14)', '제네시스쿠페(12)',
    '젠트라', '체로키 짚', '체어맨W', '카렌스(16)',
    '칼로스(4도어)', '캐딜락', '캐딜락 CTS', '컨티넨탈',
    '코나', '코나EV(19)', '코란도 C', '코란도 투리스모',
    '코란도스포츠', '콰트로포르테', '크라이슬러', '크루즈(라세티 프리미어)',
    '클리오', '투싼', '투싼 IX', '투싼(14)',
    '트라제XG', '트라제XG(디젤)', '트랙스', '트레일블레이저(20)-9BYC',
    '티볼리', '팰리세이드(19)', '포드', '포드 MKC',
    '포드 MKS', '포드 MKZ', '포드 MUSTANG', '포드 링컨',
    '포드 링컨 LS', '포드 몬데오', '포드 익스케이프', '포드 익스플로러',
    '포드 쿠커', '포드 토러스', '포드 포커스', '포드 퓨전',
    '포르쉐 911', '포르쉐 마칸', '포르쉐 박스타', '포르쉐 카이맨',
    '포르쉐 카이엔 터보', '포르쉐 파나메라', '포르테', '포르테 쿱',
    '폭스바겐 CC', '폭스바겐 골프', '폭스바겐 뉴비틀', '폭스바겐 비틀',
    '폭스바겐 시로코', '폭스바겐 아테온', '폭스바겐 제타', '폭스바겐 투아렉',
    '폭스바겐 티구안', '폭스바겐 파샤트', '폭스바겐 페이톤', '폭스바겐 폴로',
    '푸조', '푸조 2008', '푸조 3008', '푸조 508',
    '푸조2008', '푸조207', '푸조308', '푸조407',
    '프라이드4DR(15)', '프라이드5DR(15)', '프리우스', '피아트',
    '피아트 란시아카패', '허슬러', '혼다', '혼다 CIVIC',
    '혼다 S2000', '혼다 어코드', '혼다 오딧세이', '혼다 파일럿'
]

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

# [수정] 백그라운드 저장 프로세스: combined 이미지 인자 추가 및 저장 로직 추가
def background_save_process(visualization_bytes, part_visualization_bytes, combined_visualization_bytes, estimate_data, user_id, timestamp):
    try:
        print(f"🔄 [Background] 저장 작업 시작 (User: {user_id})")
        # 1. 파손 시각화 이미지 업로드 (damage 폴더)
        damage_image_url = upload_to_firebase_storage(visualization_bytes, "damage", f"{user_id}_{timestamp}_damage.jpg")
        
        # 2. 부품 시각화 이미지 업로드 (damage_part 폴더)
        part_image_url = upload_to_firebase_storage(part_visualization_bytes, "damage_part", f"{user_id}_{timestamp}_part.jpg")

        # 3. [추가] 통합 시각화 이미지 업로드 (analyzed_image 폴더)
        combined_image_url = upload_to_firebase_storage(combined_visualization_bytes, "analyzed image", f"{user_id}_{timestamp}_combined.jpg")
        
        # 4. 견적서 JSON 업로드
        estimate_data["damageImageUrl"] = damage_image_url
        estimate_data["partImageUrl"] = part_image_url
        estimate_data["combinedImageUrl"] = combined_image_url # URL 추가
        
        estimate_json = json.dumps(estimate_data, ensure_ascii=False, indent=2).encode('utf-8')
        estimate_url = upload_to_firebase_storage(estimate_json, "estimate", f"{user_id}_{timestamp}_estimate.json")
        
        # 5. Firestore 저장
        save_to_firestore({
            **estimate_data,
            "damageImageUrl": damage_image_url,
            "partImageUrl": part_image_url,
            "combinedImageUrl": combined_image_url, # URL 저장
            "estimateUrl": estimate_url
        })
        print(f"✅ [Background] 모든 저장 완료")
    except Exception as e:
        print(f"❌ [Background] 저장 실패: {e}")

# 부품 시각화 함수: Cyan 색상만 표시 (Damage 표시 안 함)
def create_part_visualization(original_image_bytes, part_mask):
    original_img = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
    original_size = original_img.size
    img_resized = original_img.resize((512, 512))
    img_np = np.array(img_resized)
    overlay = img_np.copy()
    
    for part_id in np.unique(part_mask):
        if part_id == 0: continue
        if part_id > len(PART_CLASSES): continue
        mask = (part_mask == part_id)
        overlay[mask] = VIS_PART_COLOR
        
    blended = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
    blended_pil = Image.fromarray(blended.astype(np.uint8))
    blended_resized = blended_pil.resize(original_size, Image.LANCZOS)
    
    buffered = io.BytesIO()
    blended_resized.save(buffered, format="JPEG", quality=95)
    return buffered.getvalue()

# 손상 시각화 함수: Orange 색상만 표시 (Part 표시 안 함)
# damage/ 폴더에 저장될 이미지
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
            
            # 부품 영역 내부의 손상만 표시
            final_damage_area = part_area & damage_area
            overlay[final_damage_area] = VIS_DAMAGE_COLOR # Orange
        except:
            continue
    
    blended = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
    blended_pil = Image.fromarray(blended.astype(np.uint8))
    blended_resized = blended_pil.resize(original_size, Image.LANCZOS)
    buffered = io.BytesIO()
    blended_resized.save(buffered, format="JPEG", quality=95)
    return buffered.getvalue()

# [추가] 통합 시각화 함수: Part(Cyan) 위에 Damage(Orange) 합치기
# analyzed_image/ 폴더에 저장될 이미지
def create_combined_visualization(original_image_bytes, part_mask, damage_masks, detected_parts_info):
    original_img = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
    original_size = original_img.size
    img_resized = original_img.resize((512, 512))
    img_np = np.array(img_resized)
    overlay = img_np.copy()

    # 1층: 부품 (Cyan) 그리기
    for part_id in np.unique(part_mask):
        if part_id == 0: continue
        if part_id > len(PART_CLASSES): continue
        mask = (part_mask == part_id)
        overlay[mask] = VIS_PART_COLOR

    # 2층: 손상 (Orange) 덧그리기
    for info in detected_parts_info:
        try:
            part_id = PART_CLASSES.index(info["part"]) + 1
            part_area = (part_mask == part_id)
            damage_idx = DAMAGE_CLASSES.index(info["damage"])
            damage_area = (damage_masks[damage_idx] == 1)
            # 부품 영역 내부의 손상만 표시
            final_damage_area = part_area & damage_area
            overlay[final_damage_area] = VIS_DAMAGE_COLOR # Orange
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
    print("🚀 V3 모델 + Firebase (3 Images: Damage/Part/Combined) 서버 시작")
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
        models['cost_predictor'] = joblib.load("models/cost_predictor_v4.pkl")
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

# 수리 방법 4가지 분류 로직 (Default: Painting)
def decide_repair_action(damage_type, pixel_count):
    if damage_type == "Breakage":
        return "Replace", "교환"
    if damage_type == "Separated":
        return "Detach", "탈착"
    if damage_type == "Crushed":
        if pixel_count > 5000: return "Replace", "교환"
        else: return "Sheet_Metal", "판금"
    return "Painting", "도장"

def get_cost_prediction(car_model, part_name, damage_type, repair_action_code):
    try:
        input_df = pd.DataFrame([{
            'Car_Model': car_model, 
            'Part': part_name, 
            'Damage_Type': damage_type,
            'Repair_Action': repair_action_code
        }])
        
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
        fallback_df = pd.DataFrame([{
            'Car_Model': DEFAULT_FALLBACK_CAR, 
            'Part': part_name, 
            'Damage_Type': damage_type,
            'Repair_Action': repair_action_code
        }])
        
        pred_value = models['cost_predictor'].predict(fallback_df)[0]
        cost = int(max(pred_value, 0))
        min_cost = MINIMUM_COST_BY_PART.get(part_name, 100000)
        multiplier = DAMAGE_MULTIPLIER.get(damage_type, 1.0)
        cost = max(cost, int(min_cost * multiplier))
        
        return cost, f"{car_model} (대체: {DEFAULT_FALLBACK_CAR})"

@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, car_model: str = Form(...), file: UploadFile = File(...), user_id: str = Form(default="anonymous")):
    try:
        matched_car_model = find_similar_model(car_model)
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)
        
        with torch.no_grad():
            part_out = models['part'](input_tensor)
            part_mask = torch.argmax(part_out, dim=1).cpu().numpy()[0]

        damage_masks = {}
        def run_damage_inference(i):
            with torch.no_grad():
                d_out = models[f'damage_{i}'](input_tensor)
                return i, torch.argmax(d_out, dim=1).cpu().numpy()[0]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(run_damage_inference, range(4)))
            for i, mask in results:
                damage_masks[i] = mask

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
                repair_code, repair_name = decide_repair_action(detected_damage_type, max_damage_pixels)
                cost, used_model = get_cost_prediction(matched_car_model, part_name, detected_damage_type, repair_code)
                if DEFAULT_FALLBACK_CAR in used_model:
                    final_car_model_used = used_model
                total_estimated_cost += cost
                
                detected_parts_info.append({
                    "part": part_name, 
                    "damage": detected_damage_type, 
                    "repair_method": repair_name,
                    "cost": cost
                })

        # [수정] 3가지 이미지 생성
        visualization_bytes = create_visualization(image_bytes, part_mask, damage_masks, detected_parts_info) # Orange Damage Only
        part_visualization_bytes = create_part_visualization(image_bytes, part_mask) # Cyan Part Only
        # [추가] 합친 이미지 생성
        combined_visualization_bytes = create_combined_visualization(image_bytes, part_mask, damage_masks, detected_parts_info)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        estimate_data = {
            "userId": user_id, 
            "carModel": car_model, 
            "carModelApplied": final_car_model_used, 
            "totalCost": total_estimated_cost, 
            "details": detected_parts_info, 
            "timestamp": timestamp, 
            "analysisDate": datetime.now().isoformat()
        }

        # [수정] 백그라운드 태스크에 combined 이미지 전달
        background_tasks.add_task(background_save_process, visualization_bytes, part_visualization_bytes, combined_visualization_bytes, estimate_data, user_id, timestamp)

        print(f"💰 [응답 반환] 총 {total_estimated_cost:,}원 (이미지 저장은 백그라운드 처리)")
        return {
            "status": "success", 
            "message": "Analysis complete. Data saving in background.",
            "car_model_input": car_model, 
            "car_model_applied": final_car_model_used, 
            "total_cost": total_estimated_cost, 
            "details": detected_parts_info
        }
        
    except Exception as e:
        print(f"❌ {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/car-models")
async def get_car_models():
    return {"models": KNOWN_MODELS}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "v3-3images-orange", "models_loaded": len(models), "firebase_bucket": FIREBASE_BUCKET}