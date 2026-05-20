"""
train_model.py  —  SVM + SMOTE cho ABSA tiếng Việt quán ăn
CSV: Review | AMBIENCE#GENERAL | FOOD#PRICES | FOOD#QUALITY | SERVICE#GENERAL
Nhãn: 0=None  1=Positive  2=Negative  3=Neutral

Dùng LinearSVC (tốt hơn RBF cho text TF-IDF nhiều chiều)
+ SMOTE cân bằng dữ liệu
+ CalibratedClassifierCV để có predict_proba
"""

import pandas as pd
import numpy as np
import pickle, os, re
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
    print("✅ SMOTE sẵn sàng")
except ImportError:
    HAS_SMOTE = False
    print("⚠️  Không có imbalanced-learn → pip install imbalanced-learn")

# ── Config ────────────────────────────────────────────────────────────────────
ASPECTS   = ["AMBIENCE#GENERAL", "FOOD#PRICES", "FOOD#QUALITY", "SERVICE#GENERAL"]
ASPECT_VI = {
    "AMBIENCE#GENERAL": "Không gian / Môi trường",
    "FOOD#PRICES":      "Giá cả món ăn",
    "FOOD#QUALITY":     "Chất lượng món ăn",
    "SERVICE#GENERAL":  "Dịch vụ phục vụ",
}
LABEL_NAMES = {0:"None", 1:"Positive", 2:"Negative", 3:"Neutral"}
LABEL_VI    = {0:"Không đề cập", 1:"Tích cực", 2:"Tiêu cực", 3:"Trung lập"}
MODEL_DIR   = "models"

# ── Tiền xử lý ────────────────────────────────────────────────────────────────
ABBREVIATIONS = {
    r'\bko\b':'không', r'\bk\b':'không', r'\bkh\b':'không',
    r'\bbt\b':'bình thường', r'\bđc\b':'được', r'\bdc\b':'được',
    r'\bvs\b':'với', r'\bnhìu\b':'nhiều', r'\bthik\b':'thích',
    r'\bok\b':'ổn', r'\bpv\b':'phục vụ', r'\bnv\b':'nhân viên',
    r'\bhok\b':'không', r'\bhk\b':'không', r'\bqá\b':'quá',
    r'\bcl\b':'chất lượng', r'\bgc\b':'giá cả', r'\bj\b':'gì',
}
POS_WORDS = ['ngon','tuyệt','xuất sắc','hài lòng','thích','đỉnh','hoàn hảo',
             'sạch','đẹp','rẻ','hợp lý','nhanh','nhiệt tình','chu đáo',
             'thân thiện','tốt','đậm đà','đáng tiền','quay lại','thơm','tươi']
NEG_WORDS = ['dở','tệ','chán','thất vọng','chờ lâu','chậm','đắt','cắt cổ',
             'bẩn','ôi','nhạt','mặn','cứng','nguội','thô lỗ','không ngon',
             'kém','tồi','lạnh','khó chịu','không quay lại','đắt quá']
NEU_WORDS = ['bình thường','tạm','được','ổn','không có gì','trung bình',
             'tàm tạm','cũng được','không đặc sắc','vừa phải']

EMOJI_RE = re.compile(
    "[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
    u"\U00002700-\U000027BF" u"\U0001F900-\U0001F9FF"
    u"\U00002600-\U000026FF" "]+", flags=re.UNICODE)

def preprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = EMOJI_RE.sub(' ', text).lower().strip()
    for pat, rep in ABBREVIATIONS.items():
        text = re.sub(pat, rep, text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text)
    text = re.sub(
        r'[^\w\sàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ]',
        ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()

    # ── Xử lý phủ định: "không ngon" → "không_ngon" ──────────────────────────
    neg_set = {'không', 'chẳng', 'chả', 'chưa', 'ko', 'hok'}
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        if words[i] in neg_set and i + 1 < len(words):
            result.append(words[i] + '_' + words[i+1])
            i += 2
        else:
            result.append(words[i])
            i += 1
    text = ' '.join(result)

    # ── Gắn tag tình cảm ──────────────────────────────────────────────────────
    has_negation = any(('không_' + w) in text or ('chưa_' + w) in text
                       for w in POS_WORDS)
    tags = []
    if any(w in text for w in POS_WORDS) and not has_negation:
        tags.append('__POS__')
    if any(w in text for w in NEG_WORDS) or has_negation:
        tags.append('__NEG__')
    if any(w in text for w in NEU_WORDS):
        tags.append('__NEU__')
    return text + (' ' + ' '.join(tags) if tags else '')


def make_vectorizer():
    """TF-IDF word + char ngram — tối ưu cho text tiếng Việt."""
    return FeatureUnion([
        ('word', TfidfVectorizer(
            analyzer='word', ngram_range=(1,3),
            max_features=50000, sublinear_tf=True, min_df=1)),
        ('char', TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2,5),
            max_features=50000, sublinear_tf=True, min_df=1)),
    ])


def make_svm(class_weight=None):
    """
    LinearSVC + CalibratedClassifierCV:
    - LinearSVC tốt hơn RBF cho dữ liệu text TF-IDF nhiều chiều (60k+ features)
    - CalibratedClassifierCV để có predict_proba (cần cho web)
    """
    base = LinearSVC(
        C=0.5,                    # nhỏ hơn để tránh overfit nhãn đa số
        max_iter=2000,
        class_weight=class_weight,
        random_state=42,
    )
    return CalibratedClassifierCV(base, cv=3, method='sigmoid')


class SVMModel:
    """Gói vectorizer + SVM thành 1 object để pickle."""
    def __init__(self, vec, svm):
        self.vec = vec
        self.svm = svm
        self.classes_ = np.array(svm.classes_)

    def predict(self, X):
        return self.svm.predict(self.vec.transform(X))

    def predict_proba(self, X):
        return self.svm.predict_proba(self.vec.transform(X))


def load_data(csv_path=None):
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy: {csv_path}")

    for enc in ['utf-8-sig', 'utf-8', 'latin-1']:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue

    df.columns = df.columns.str.strip()
    print(f"✅ Load {len(df)} dòng từ: {csv_path}")

    missing = [c for c in ['Review']+ASPECTS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}")

    df['Review'] = df['Review'].fillna('').apply(preprocess)
    for asp in ASPECTS:
        df[asp] = pd.to_numeric(df[asp], errors='coerce').fillna(0).astype(int).clip(0,3)
    df = df[df['Review'].str.len() > 3].reset_index(drop=True)
    print(f"   Sau tiền xử lý: {len(df)} dòng hợp lệ\n")
    return df


def train_one_aspect(X_text, y, aspect_name):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    print(f"   Phân phối nhãn:")
    for u, c in zip(unique, counts):
        bar = '█' * max(1, int(c/total*30))
        print(f"     {LABEL_NAMES[u]:10s}: {c:5d}  {bar}  ({c/total*100:.1f}%)")

    # ── Vectorize toàn bộ ────────────────────────────────────────────────────
    vec = make_vectorizer()
    X_vec = vec.fit_transform(X_text)

    # ── Tách train / test 80/20 ──────────────────────────────────────────────
    min_count = int(min(counts))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_vec, y, test_size=0.2, random_state=42,
        stratify=y if min_count >= 2 else None
    )

    # ── SMOTE trên tập TRAIN ─────────────────────────────────────────────────
    if HAS_SMOTE and min_count >= 2:
        tr_unique, tr_counts = np.unique(y_tr, return_counts=True)
        k = max(1, min(5, int(min(tr_counts)) - 1))
        try:
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_tr_bal, y_tr_bal = sm.fit_resample(X_tr, y_tr)
            res_u, res_c = np.unique(y_tr_bal, return_counts=True)
            print(f"\n   ⚖️  Sau SMOTE (tập train):")
            for u, c in zip(res_u, res_c):
                print(f"     {LABEL_NAMES[u]:10s}: {c}")
        except Exception as e:
            print(f"   ⚠️  SMOTE lỗi ({e}) → dùng class_weight")
            X_tr_bal, y_tr_bal = X_tr, y_tr
    else:
        X_tr_bal, y_tr_bal = X_tr, y_tr

    # class_weight trên tập đã SMOTE
    cw = compute_class_weight('balanced', classes=np.unique(y_tr_bal), y=y_tr_bal)
    cw_dict = {int(k):float(v) for k,v in zip(np.unique(y_tr_bal), cw)}

    # ── Train + Evaluate trên TEST ───────────────────────────────────────────
    svm_eval = make_svm(class_weight=cw_dict)
    svm_eval.fit(X_tr_bal, y_tr_bal)

    y_pred = svm_eval.predict(X_te)
    test_labels = sorted(np.unique(np.concatenate([y_te, y_pred])))
    print(f"\n   📊 Kết quả trên tập TEST ({len(y_te)} mẫu — chưa thấy khi train):")
    print(classification_report(
        y_te, y_pred,
        labels=test_labels,
        target_names=[LABEL_NAMES[i] for i in test_labels],
        zero_division=0
    ))
    macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    print(f"   🎯 Macro F1 = {macro_f1:.3f}  ({'✅ Tốt' if macro_f1>=0.6 else '⚠️ Trung bình' if macro_f1>=0.4 else '❌ Cần thêm dữ liệu'})\n")

    # ── Train lại toàn bộ data để lưu ───────────────────────────────────────
    print(f"   🔄 Train lại toàn bộ {total} mẫu để lưu model...")
    if HAS_SMOTE and min_count >= 2:
        k_full = max(1, min(5, int(min(counts)) - 1))
        try:
            X_full_bal, y_full_bal = SMOTE(random_state=42, k_neighbors=k_full).fit_resample(X_vec, y)
        except Exception:
            X_full_bal, y_full_bal = X_vec, y
    else:
        X_full_bal, y_full_bal = X_vec, y

    cw_full = compute_class_weight('balanced', classes=np.unique(y_full_bal), y=y_full_bal)
    cw_full_dict = {int(k):float(v) for k,v in zip(np.unique(y_full_bal), cw_full)}

    svm_final = make_svm(class_weight=cw_full_dict)
    svm_final.fit(X_full_bal, y_full_bal)

    return SVMModel(vec, svm_final)


def train_and_save(csv_path=None, model_dir=MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    df = load_data(csv_path)

    for aspect in ASPECTS:
        print(f"\n{'═'*55}")
        print(f"  📌  {aspect}  —  {ASPECT_VI[aspect]}")
        print(f"{'═'*55}")

        model = train_one_aspect(
            df['Review'].values,
            df[aspect].astype(int).values,
            aspect
        )

        path = os.path.join(model_dir, f"{aspect.replace('#','_')}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   💾 Đã lưu: {path}")

    meta = {'aspects':ASPECTS, 'aspect_vi':ASPECT_VI,
            'label_names':LABEL_NAMES, 'label_vi':LABEL_VI,
            'n_samples':len(df)}
    with open(os.path.join(model_dir,'meta.pkl'),'wb') as f:
        pickle.dump(meta, f)

    print(f"\n{'═'*55}")
    print(f"  ✅ Hoàn tất! 4 mô hình SVM lưu tại: {model_dir}/")
    print(f"{'═'*55}\n")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("❌ Cách dùng: python train_model.py \"đường_dẫn_file.csv\"")
        sys.exit(1)
    train_and_save(sys.argv[1])
