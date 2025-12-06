from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import re

app = Flask(__name__)

# ==========================================
# 1. DỮ LIỆU HUẤN LUYỆN (DATASET)
# ==========================================
# Đây là "vốn hiểu biết" của AI. Bạn càng thêm nhiều câu, AI càng khôn.
# Cấu trúc: (Nội dung mẫu, Nhãn phân loại)

training_data = [
    # --- NHÃN: TAI_NAN ---
    ("Có tai nạn giao thông nghiêm trọng ở ngã tư", "Tai nạn giao thông"),
    ("Hai xe máy va chạm mạnh, người bị thương", "Tai nạn giao thông"),
    ("Xe tải lật chắn ngang đường, kẹt xe", "Tai nạn giao thông"),
    ("Tông xe liên hoàn, cần cứu thương gấp", "Tai nạn giao thông"),
    ("Người đi bộ bị xe tông trúng", "Tai nạn giao thông"),
    ("Va quẹt xe nhẹ, đang cãi nhau to", "Tai nạn giao thông"),

    # --- NHÃN: CUOP_GIAT ---
    ("Bị giật điện thoại khi đang nghe máy", "Cướp giật"),
    ("Có kẻ móc túi ở chợ, cẩn thận", "Cướp giật"),
    ("Cướp giật dây chuyền rồi bỏ chạy xe máy", "Cướp giật"),
    ("Mất xe máy, trộm bẻ khóa vào nhà", "Cướp giật"),
    ("Thấy đối tượng khả nghi rình mò bẻ khóa", "Cướp giật"),
    ("Bị trấn lột tiền trong hẻm vắng", "Cướp giật"),

    # --- NHÃN: DUA_XE ---
    ("Đám thanh niên tụ tập nẹt pô ầm ĩ", "Tụ tập đua xe"),
    ("Đua xe trái phép gây rối trật tự", "Tụ tập đua xe"),
    ("Lạng lách đánh võng, bốc đầu xe", "Tụ tập đua xe"),
    ("Nhóm quái xế tụ tập chuẩn bị đua", "Tụ tập đua xe"),

    # --- NHÃN: HOA_HOAN (CHÁY) ---
    ("Cháy nhà dân, khói bốc lên nghi ngút", "Hỏa hoạn"),
    ("Có mùi khét lẹt, nghi chập điện cháy", "Hỏa hoạn"),
    ("Lửa bùng lên dữ dội tại kho hàng", "Hỏa hoạn"),
    ("Cần cứu hỏa gấp, cháy lớn quá", "Hỏa hoạn"),

    # --- NHÃN: NGAP_LUT / HA_TANG ---
    ("Đường ngập nước sâu không đi được", "Hư hỏng hạ tầng"),
    ("Hố ga mất nắp rất nguy hiểm", "Hư hỏng hạ tầng"),
    ("Cây xanh gãy đổ chắn ngang đường", "Hư hỏng hạ tầng"),
    ("Dây điện bị đứt sà xuống đất", "Hư hỏng hạ tầng"),

    # --- NHÃN: SPAM (RÁC) ---
    ("Alo alo 123 test", "Spam"),
    ("Thử nghiệm tính năng", "Spam"),
    ("Test báo cáo abc xyz", "Spam"),
    ("Bán sim số đẹp giá rẻ", "Spam"),
    ("Tuyển dụng việc làm lương cao", "Spam"),
    ("Vay tiền nhanh không thế chấp", "Spam"),
    ("dfhjsdfhksdf sdjkfhsdkjf", "Spam"), # Ký tự loạn
    ("aaaaaaaaaaaaaaaaa", "Spam"),
]

# Tách dữ liệu ra 2 mảng để huấn luyện
train_texts = [item[0] for item in training_data]
train_labels = [item[1] for item in training_data]

# ==========================================
# 2. HUẤN LUYỆN MODEL (TRAINING)
# ==========================================
print("Dang huan luyen AI... Vui long cho...")

# Tạo một Pipeline đơn giản:
# 1. CountVectorizer: Biến đổi chữ thành số (đếm từ)
# 2. MultinomialNB: Thuật toán Naive Bayes (Cực nhanh và tốt cho văn bản)
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Bắt đầu học
model.fit(train_texts, train_labels)

print("AI da hoc xong! San sang phuc vu.")


# ==========================================
# 3. API XỬ LÝ
# ==========================================

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    # Lấy dữ liệu
    title = data.get('title', '')
    desc = data.get('text', '')
    
    # Gộp tiêu đề và mô tả để AI có nhiều dữ liệu phân tích hơn
    full_text = f"{title} {desc}"

    # --- BƯỚC 1: Kiểm tra quy tắc cứng (Hard Rules) trước ---
    # Nếu chuỗi quá ngắn hoặc lặp ký tự -> Phán Spam luôn, không cần AI đoán
    if len(desc) < 10 and len(title) < 10:
         return jsonify({
            "suggestion": "Nghi ngờ Spam (Quá ngắn)",
            "confidence": 0.9,
            "source": "rule-based"
        })
    
    # --- BƯỚC 2: Dùng AI dự đoán (Machine Learning) ---
    # Dự đoán nhãn
    prediction = model.predict([full_text])[0]
    
    # Lấy độ tin cậy (Probability) của dự đoán đó
    # model.predict_proba trả về mảng xác suất cho từng nhãn
    probs = model.predict_proba([full_text])[0]
    confidence = max(probs) # Lấy xác suất cao nhất

    # Làm đẹp kết quả trả về
    suggestion_text = ""
    if prediction == "Spam":
        suggestion_text = "Nghi ngờ Spam (Nội dung rác)"
    else:
        suggestion_text = f"Phân loại: {prediction}"

    return jsonify({
        "suggestion": suggestion_text,
        "confidence": float(round(confidence, 2)), # Làm tròn 2 số lẻ
        "source": "ai-machine-learning"
    })

if __name__ == '__main__':
    app.run(debug=True)