from flask import Flask, request, jsonify
import re

app = Flask(__name__)

# --- LOGIC AI (VĂN BẢN) ---

def is_repetitive(text):
    """Kiểm tra spam ký tự lặp (vd: aaaaa, abcabcabc)"""
    if not text or len(text) < 5:
        return False
        
    # 1. Ký tự giống hệt nhau quá nhiều (aaaaaaaa)
    char = text[0]
    if text.count(char) / len(text) > 0.8:
        return True
        
    # 2. Chuỗi lặp lại (abcabcabc) - Set unique char ít
    if len(set(text)) < 4 and len(text) > 10:
        return True
        
    return False

def analyze_text_logic(title, description):
    """Phân tích mức độ tin cậy của báo cáo"""
    title = title.lower() if title else ""
    desc = description.lower() if description else ""
    
    # 1. Kiểm tra độ dài
    if len(desc) < 15:
        return {"suggestion": "Nghi ngờ Spam (Mô tả quá ngắn)", "confidence": 0.80, "source": "python-nlp"}
    
    # 2. Kiểm tra lặp lại (Spam)
    if is_repetitive(desc) or is_repetitive(title):
        return {"suggestion": "Nghi ngờ Spam (Nội dung vô nghĩa)", "confidence": 0.90, "source": "python-nlp"}

    # 3. Kiểm tra từ khóa Test/Spam
    spam_keywords = ['test', 'thử', 'abc', '123', 'demo', 'alo']
    if any(w in title for w in spam_keywords) or any(w in desc for w in spam_keywords):
        return {"suggestion": "Báo cáo thử nghiệm/Spam", "confidence": 0.85, "source": "python-nlp"}

    # 4. Kiểm tra từ khóa nguy hiểm (Tăng độ tin cậy)
    # Nếu có các từ này thì khả năng là báo cáo thật
    danger_keywords = ['tai nạn', 'cướp', 'cháy', 'đua xe', 'đánh nhau', 'ngập', 'kẹt xe', 'hố ga']
    if any(w in title for w in danger_keywords) or any(w in desc for w in danger_keywords):
        return {"suggestion": "Báo cáo hợp lệ (Có từ khóa nguy hiểm)", "confidence": 0.95, "source": "python-nlp"}

    # Mặc định
    return {"suggestion": "Báo cáo cần xem xét", "confidence": 0.50, "source": "python-nlp"}


# --- API ENDPOINT ---

@app.route('/analyze', methods=['POST'])
def analyze():
    # Nhận JSON thay vì Form Data
    data = request.json
    
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    title = data.get('title', '')
    description = data.get('text', '') # Laravel gửi key là 'text'
    
    result = analyze_text_logic(title, description)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)