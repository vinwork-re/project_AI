"""
app.py — Flask API Backend
Nhận dạng bình luận tiếng Việt quán ăn sử dụng SVM
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle, os, re, numpy as np

app = Flask(__name__, static_folder='.')
CORS(app)

ASPECTS   = ["AMBIENCE#GENERAL", "FOOD#PRICES", "FOOD#QUALITY", "SERVICE#GENERAL"]
ASPECT_VI = {
    "AMBIENCE#GENERAL": "Không gian / Môi trường",
    "FOOD#PRICES":      "Giá cả món ăn",
    "FOOD#QUALITY":     "Chất lượng món ăn",
    "SERVICE#GENERAL":  "Dịch vụ phục vụ",
}
LABEL_NAMES = {0: "None", 1: "Positive", 2: "Negative", 3: "Neutral"}
LABEL_VI    = {0: "Không đề cập", 1: "Tích cực", 2: "Tiêu cực", 3: "Trung lập"}
MODEL_DIR   = "models"
MODELS      = {}

# ── Import từ train_model (phải import SVMModel để pickle load được) ─────────
try:
    from train_model import preprocess as preprocess_fn, SVMModel
except ImportError:
    SVMModel = None
    def preprocess_fn(text):
        return text.lower().strip() if isinstance(text, str) else ""


def load_models():
    global MODELS
    for aspect in ASPECTS:
        path = os.path.join(MODEL_DIR, f"{aspect.replace('#','_')}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                MODELS[aspect] = pickle.load(f)
            print(f"✅ Loaded: {aspect}")
        else:
            print(f"⚠️  Not found: {path}")


# ── Spam detection ───────────────────────────────────────────────────────────
VIET_CHARS = re.compile(
    r'[àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ]'
)

# Từ khóa liên quan đến quán ăn — phải có ít nhất 1
FOOD_KEYWORDS = [
    'ăn','uống','món','quán','nhà hàng','đồ ăn','thức ăn','thực đơn',
    'ngon','dở','tệ','chán','giá','phục vụ','nhân viên','không gian',
    'view','vệ sinh','sạch','bẩn','nhanh','chậm','đắt','rẻ','hợp lý',
    'cay','mặn','nhạt','ngọt','chua','tươi','ôi','cứng','mềm','giòn',
    'thơm','hương','vị','nêm','nếm','bàn','ghế','chỗ','phòng','menu',
    'order','gọi','chờ','đợi','bill','tính tiền','ship','giao','takeaway',
    'buffet','lẩu','nướng','cơm','bún','phở','mì','bánh','chè','nước',
    'cafe','cà phê','trà','bia','rượu','hải sản','thịt','rau','salad',
    'recommend','quay lại','lần sau','hài lòng','thất vọng','tuyệt','xuất sắc',
]

def is_spam(text: str) -> bool:
    cleaned = text.lower().strip()
    tokens  = cleaned.split()

    # Quá ngắn
    if len(tokens) < 4:
        return True

    # Lặp từ quá nhiều (ok ok ok ok...)
    if len(set(tokens)) / len(tokens) < 0.4:
        return True

    # Không có chữ cái tiếng Việt có dấu VÀ không có từ khóa thực phẩm
    viet_count = len(VIET_CHARS.findall(text))
    has_food_kw = any(kw in cleaned for kw in FOOD_KEYWORDS)

    if viet_count == 0 and not has_food_kw:
        return True

    # Toàn số và ký tự không có nghĩa
    meaningful = re.sub(r'[0-9\s\W]', '', cleaned)
    if len(meaningful) < 4:
        return True

    return False


# ── Overall sentiment logic ──────────────────────────────────────────────────
def determine_overall(aspect_results: dict) -> dict:
    pos       = [k for k, v in aspect_results.items() if v['label'] == 1]
    neg       = [k for k, v in aspect_results.items() if v['label'] == 2]
    mentioned = [k for k, v in aspect_results.items() if v['label'] != 0]

    if not mentioned:
        return {
            'overall': 'None', 'overall_vi': 'Spam / Không liên quan',
            'overall_color': 'spam', 'breakdown': None,
        }

    score = len(pos) - len(neg)

    if len(pos) > 0 and len(neg) == 0:
        label, color, vi = 'Positive', 'positive', 'Tích cực'
    elif len(neg) > 0 and len(pos) == 0:
        label, color, vi = 'Negative', 'negative', 'Tiêu cực'
    elif score >= 2:
        label, color, vi = 'Positive', 'positive', 'Tích cực (thiên tốt)'
    elif score <= -2:
        label, color, vi = 'Negative', 'negative', 'Tiêu cực (thiên xấu)'
    elif len(pos) == 0 and len(neg) == 0:
        label, color, vi = 'Neutral', 'neutral', 'Trung lập'
    else:
        label, color, vi = 'Neutral', 'neutral', 'Trung lập — vừa tích cực vừa tiêu cực'

    # Breakdown chỉ khi Neutral có cả pos lẫn neg
    breakdown = None
    if label == 'Neutral' and pos and neg:
        breakdown = {
            'positive_aspects': [
                {'aspect': k, 'aspect_vi': ASPECT_VI[k],
                 'confidence': aspect_results[k]['confidence']}
                for k in pos
            ],
            'negative_aspects': [
                {'aspect': k, 'aspect_vi': ASPECT_VI[k],
                 'confidence': aspect_results[k]['confidence']}
                for k in neg
            ],
        }

    return {'overall': label, 'overall_vi': vi,
            'overall_color': color, 'breakdown': breakdown}


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json(force=True)
    review = data.get('review', '').strip()
    if not review:
        return jsonify({'error': 'Vui lòng nhập nội dung bình luận.'}), 400

    if is_spam(review):
        return jsonify({
            'review': review,
            'overall': 'None', 'overall_vi': 'Spam / Không liên quan',
            'overall_color': 'spam', 'aspects': {}, 'breakdown': None,
        })

    cleaned = preprocess_fn(review)
    aspect_results = {}

    for aspect in ASPECTS:
        if aspect not in MODELS:
            aspect_results[aspect] = {
                'label': 0, 'label_en': 'None',
                'label_vi': 'Mô hình chưa tải', 'confidence': 0.0,
            }
            continue
        try:
            model  = MODELS[aspect]
            proba  = model.predict_proba([cleaned])[0]
            classes = model.classes_
            idx    = int(np.argmax(proba))
            pred   = int(classes[idx])
            conf   = float(proba[idx])
            prob_dict = {int(c): float(p) for c, p in zip(classes, proba)}

            aspect_results[aspect] = {
                'label':    pred,
                'label_en': LABEL_NAMES[pred],
                'label_vi': LABEL_VI[pred],
                'confidence': round(conf * 100, 1),
                'probabilities': {
                    LABEL_VI[k]: round(v * 100, 1)
                    for k, v in sorted(prob_dict.items())
                },
            }
        except Exception as e:
            aspect_results[aspect] = {
                'label': 0, 'label_en': 'None',
                'label_vi': f'Lỗi: {e}', 'confidence': 0.0,
            }

    overall = determine_overall(aspect_results)
    return jsonify({
        'review': review,
        **overall,
        'aspects': {
            k: {**v, 'aspect_vi': ASPECT_VI[k]}
            for k, v in aspect_results.items()
        },
    })


@app.route('/train', methods=['POST'])
def train():
    try:
        from train_model import train_and_save
        train_and_save()
        load_models()
        return jsonify({'status': 'success',
                        'message': f'Đã huấn luyện xong {len(MODELS)} mô hình SVM!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Chỉ chấp nhận file CSV'}), 400
    csv_path = 'uploaded_data.csv'
    file.save(csv_path)
    try:
        from train_model import train_and_save
        train_and_save(csv_path)
        load_models()
        return jsonify({'status': 'success',
                        'message': f'Đã upload & huấn luyện từ {file.filename}!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/status')
def status():
    return jsonify({
        'models_loaded': list(MODELS.keys()),
        'ready': len(MODELS) == len(ASPECTS),
    })


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    load_models()
    if not MODELS:
        print('\n⚡ Không có model, tự động huấn luyện bằng dữ liệu mẫu...')
        from train_model import train_and_save
        train_and_save()
        load_models()
    print('\n🚀 Server: http://localhost:5000')
    app.run(debug=True, port=5000)
