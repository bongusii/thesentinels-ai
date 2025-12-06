from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# --- 1. KHỞI TẠO MODEL AI (Load 1 lần duy nhất khi chạy server) ---
print("Dang khoi dong AI Model (MobileNetV2)... Vui long cho...")
# MobileNetV2 là model nhẹ, được Google huấn luyện trên 1.4 triệu bức ảnh
model = MobileNetV2(weights='imagenet')
print("AI Model da san sang!")

# --- 2. CÁC HÀM XỬ LÝ ---

def analyze_text_logic(title, description):
    """Logic phân tích văn bản (như cũ)"""
    title = title.lower() if title else ""
    desc = description.lower() if description else ""
    
    # Check Spam cơ bản
    if len(desc) < 20:
        return {"suggestion": "Nghi ngờ Spam (Mô tả quá ngắn)", "confidence": 0.60}
    
    # Check từ khóa nhạy cảm
    spam_keywords = ['test', 'thử', 'abc', '123', 'quảng cáo', 'bán']
    if any(word in title for word in spam_keywords) or any(word in desc for word in spam_keywords):
        return {"suggestion": "Nghi ngờ Spam (Nội dung rác)", "confidence": 0.80}

    return {"suggestion": "Văn bản hợp lệ", "confidence": 0.20} # Confidence thấp vì chưa check ảnh

def analyze_image_logic(image_bytes):
    """Logic nhận diện hình ảnh bằng Deep Learning"""
    try:
        # 1. Chuẩn bị ảnh
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((224, 224)) # MobileNet yêu cầu kích thước 224x224
        
        # 2. Chuyển sang mảng số (Array)
        x = img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # 3. Dự đoán (Predict)
        preds = model.predict(x)
        
        # 4. Giải mã kết quả (Lấy top 3 khả năng cao nhất)
        # Kết quả trả về dạng: [('id', 'label', probability), ...]
        decoded = decode_predictions(preds, top=3)[0]
        
        # Chuyển thành list đơn giản để trả về JSON
        results = []
        for item in decoded:
            results.append({
                "label": item[1],   # Tên vật thể (Tiếng Anh)
                "score": float(item[2]) # Độ tin cậy
            })
            
        return results

    except Exception as e:
        print("Loi xu ly anh:", str(e))
        return []

# --- 3. API ENDPOINT ---

@app.route('/analyze', methods=['POST'])
def analyze():
    # Lấy dữ liệu văn bản
    title = request.form.get('title', '')
    description = request.form.get('description', '')
    
    # 1. Phân tích văn bản trước
    text_result = analyze_text_logic(title, description)
    final_suggestion = text_result['suggestion']
    final_confidence = text_result['confidence']
    ai_source = "python-nlp"

    # 2. Phân tích ảnh (Nếu có upload)
    image_labels = []
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            image_bytes = file.read()
            image_labels = analyze_image_logic(image_bytes)
            
            if image_labels:
                top_label = image_labels[0]['label']
                top_score = image_labels[0]['score']
                
                # Logic kết hợp: Nếu AI nhận diện được vật thể nguy hiểm
                # (Đây là danh sách ví dụ, bạn có thể mở rộng)
                danger_objects = ['crash', 'car_wreck', 'fire', 'flame', 'police', 'ambulance', 'weapon', 'gun', 'knife']
                
                is_dangerous = any(obj in top_label.lower() for obj in danger_objects)
                
                if is_dangerous:
                    final_suggestion = f"Nguy hiểm cao! AI phát hiện: {top_label}"
                    final_confidence = 0.95 # Tăng độ tin cậy lên rất cao
                    ai_source = "python-vision (High Risk)"
                else:
                    final_suggestion = f"Ảnh chứa: {top_label}"
                    # Nếu ảnh rõ ràng nhưng không phải nguy hiểm, confidence giữ nguyên hoặc tăng nhẹ
                    final_confidence = max(final_confidence, 0.7)
                    ai_source = "python-vision"

    # Trả về kết quả tổng hợp
    return jsonify({
        "suggestion": final_suggestion,
        "confidence": final_confidence,
        "source": ai_source,
        "image_analysis": image_labels # Trả về chi tiết để debug nếu cần
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)