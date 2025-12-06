from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import re

app = Flask(__name__)

# ==========================================
# 1. DỮ LIỆU HUẤN LUYỆN (DATASET MỞ RỘNG)
# ==========================================
# Cấu trúc: ("Nội dung mẫu", "Nhãn phân loại")

training_data = [
    # --- 1. TAI NẠN GIAO THÔNG ---
    ("Có tai nạn giao thông nghiêm trọng ở ngã tư", "Tai nạn giao thông"),
    ("Hai xe máy va chạm mạnh, người bị thương nặng", "Tai nạn giao thông"),
    ("Xe tải lật chắn ngang đường, kẹt xe kinh khủng", "Tai nạn giao thông"),
    ("Tông xe liên hoàn, cần cứu thương gấp", "Tai nạn giao thông"),
    ("Người đi bộ bị xe tông trúng nằm bất động", "Tai nạn giao thông"),
    ("Va quẹt xe nhẹ, đang cãi nhau to gây ùn tắc", "Tai nạn giao thông"),
    ("Xe buýt cán qua xe máy, hiện trường rất thảm", "Tai nạn giao thông"),
    ("Tai nạn chết người ngay khúc cua", "Tai nạn giao thông"),
    ("Xe container mất lái lao vào nhà dân", "Tai nạn giao thông"),
    ("Ngã xe do đường trơn trượt", "Tai nạn giao thông"),
    ("Tông xe rồi bỏ chạy", "Tai nạn giao thông"),
    ("tai nan xe may", "Tai nạn giao thông"), # Không dấu
    ("dung xe gay tai nan", "Tai nạn giao thông"),

    # --- 2. CƯỚP GIẬT / TRỘM CẮP ---
    ("Bị giật điện thoại khi đang nghe máy", "Cướp giật"),
    ("Có kẻ móc túi ở chợ, cẩn thận ví tiền", "Cướp giật"),
    ("Cướp giật dây chuyền rồi bỏ chạy xe Exciter", "Cướp giật"),
    ("Mất xe máy, trộm bẻ khóa vào nhà", "Cướp giật"),
    ("Thấy đối tượng khả nghi rình mò bẻ khóa cổng", "Cướp giật"),
    ("Bị trấn lột tiền trong hẻm vắng", "Cướp giật"),
    ("Đua nóng xe SH, trộm chạy về hướng cầu", "Cướp giật"),
    ("Cướp ngân hàng, có vũ khí", "Cướp giật"),
    ("Giật túi xách khiến nạn nhân té ngã", "Cướp giật"),
    ("trom cho", "Cướp giật"),
    ("bi moc tui tren xe bus", "Cướp giật"),
    ("thay an trom leo rao", "Cướp giật"),

    # --- 3. TỤ TẬP ĐUA XE / GÂY RỐI ---
    ("Đám thanh niên tụ tập nẹt pô ầm ĩ", "Tụ tập đua xe"),
    ("Đua xe trái phép gây rối trật tự cả khu phố", "Tụ tập đua xe"),
    ("Lạng lách đánh võng, bốc đầu xe nguy hiểm", "Tụ tập đua xe"),
    ("Nhóm quái xế tụ tập chuẩn bị đua xe", "Tụ tập đua xe"),
    ("Thanh niên chạy xe rú ga, nẹt pô inh ỏi", "Tụ tập đua xe"),
    ("Bão đêm, đua xe đông quá", "Tụ tập đua xe"),
    ("Tụ tập đánh nhau, gây gổ", "Tụ tập đua xe"), # Gộp chung vào gây rối
    ("Dàn hàng ngang chặn đầu xe", "Tụ tập đua xe"),
    ("dua xe trai phep", "Tụ tập đua xe"),

    # --- 4. KẺ GIAN / BIẾN THÁI (AN NINH) ---
    ("Có kẻ biến thái hay đi theo nữ sinh", "Kẻ gian đáng ngờ"),
    ("Người lạ mặt lảng vảng trước cổng trường", "Kẻ gian đáng ngờ"),
    ("Đối tượng nghiện hút tụ tập chích thuốc", "Kẻ gian đáng ngờ"),
    ("Kẻ gian rình mò nhà dân ban đêm", "Kẻ gian đáng ngờ"),
    ("Có người lạ đeo khẩu trang nhìn rất khả nghi", "Kẻ gian đáng ngờ"),
    ("Giả danh nhân viên thu tiền điện lừa đảo", "Kẻ gian đáng ngờ"),
    ("Ke bien thai lo hang", "Kẻ gian đáng ngờ"),

    # --- 5. HỎA HOẠN (CHÁY NỔ) ---
    ("Cháy nhà dân, khói bốc lên nghi ngút", "Hỏa hoạn"),
    ("Có mùi khét lẹt, nghi chập điện cháy", "Hỏa hoạn"),
    ("Lửa bùng lên dữ dội tại kho hàng", "Hỏa hoạn"),
    ("Cần cứu hỏa gấp, cháy lớn quá", "Hỏa hoạn"),
    ("Nổ bình gas gây cháy nhà", "Hỏa hoạn"),
    ("Cháy rừng, cháy cỏ khô lan rộng", "Hỏa hoạn"),
    ("chay nha roi", "Hỏa hoạn"),

    # --- 6. HẠ TẦNG / NGẬP LỤT / KHÁC ---
    ("Đường ngập nước sâu không đi được", "Nguy hiểm khác"),
    ("Hố ga mất nắp rất nguy hiểm cho người đi đường", "Nguy hiểm khác"),
    ("Cây xanh gãy đổ chắn ngang đường", "Nguy hiểm khác"),
    ("Dây điện bị đứt sà xuống đất, coi chừng giật", "Nguy hiểm khác"),
    ("Sạt lở đất bờ sông", "Nguy hiểm khác"),
    ("Cống trào nước hôi thối", "Nguy hiểm khác"),
    ("Đèn đường hỏng tối om", "Nguy hiểm khác"),
    ("ngap nuoc", "Nguy hiểm khác"),

    # --- 7. SPAM (RÁC) ---
    ("Alo alo 123 test", "Spam"),
    ("Thử nghiệm tính năng báo cáo", "Spam"),
    ("Test báo cáo abc xyz", "Spam"),
    ("Bán sim số đẹp giá rẻ 0909xxx", "Spam"),
    ("Tuyển dụng việc làm lương cao, việc nhẹ lương cao", "Spam"),
    ("Vay tiền nhanh không thế chấp", "Spam"),
    ("dfhjsdfhksdf sdjkfhsdkjf", "Spam"),
    ("aaaaaaaaaaaaaaaaa", "Spam"),
    ("Xin chào, hôm nay trời đẹp quá", "Spam"),
    ("Bán đất nền giá rẻ", "Spam"),
    ("Game bài đổi thưởng uy tín", "Spam"),
    ("Click vào link để nhận quà", "Spam"),
    ("test", "Spam"),
    ("demo", "Spam"),
]

# Tách dữ liệu
train_texts = [item[0] for item in training_data]
train_labels = [item[1] for item in training_data]

# ==========================================
# 2. HUẤN LUYỆN MODEL
# ==========================================
print("[AI] Dang huan luyen (Scikit-learn)...")

# Pipeline: Biến đổi văn bản -> Đếm từ -> Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(train_texts, train_labels)

print("[AI] Da hoc xong! San sang phuc vu.")

# ==========================================
# 3. API XỬ LÝ
# ==========================================
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    title = data.get('title', '')
    desc = data.get('text', '')
    full_text = f"{title} {desc}"

    # 1. Kiểm tra quy tắc cứng (Hard Rules) trước
    if len(desc) < 10 and len(title) < 10:
         return jsonify({
            "suggestion": "Nghi ngờ Spam (Nội dung quá ngắn)",
            "confidence": 0.95,
            "source": "rule-based"
        })
    
    # 2. Dùng Model AI dự đoán
    prediction = model.predict([full_text])[0]
    probs = model.predict_proba([full_text])[0]
    confidence = max(probs)

    # Logic hiển thị
    if prediction == "Spam":
        suggestion_text = "Nghi ngờ Spam (Nội dung rác/Quảng cáo)"
    else:
        suggestion_text = f"Phân loại: {prediction}"

    # Nếu độ tin cậy quá thấp (< 50%), AI sẽ lưỡng lự
    if confidence < 0.5:
        suggestion_text = f"Chưa rõ ràng (Nghiêng về: {prediction})"

    return jsonify({
        "suggestion": suggestion_text,
        "confidence": float(round(confidence, 2)),
        "source": "ai-machine-learning"
    })

if __name__ == '__main__':
    app.run(debug=True)