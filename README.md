# 🛡️ Layered WAF with Deep Learning (Bi-LSTM)
Đồ án môn học: Bảo mật Web và Ứng dụng

Giảng viên hướng dẫn: Thầy Ngô Khánh Khoa

Thực hiện: Nhóm 18

# 📖 Giới thiệu (Introduction)
Dự án này xây dựng một hệ thống Tường lửa ứng dụng Web (WAF) thế hệ mới, áp dụng kiến trúc phân lớp (Layered Architecture) để bảo vệ ứng dụng khỏi các cuộc tấn công mạng phổ biến.

Hệ thống kết hợp giữa tốc độ của Rule-based và độ chính xác của Deep Learning:

Layer 1 (Proxy): Chặn tấn công từ chối dịch vụ (DDoS) dựa trên tần suất (Rate Limiting).

Layer 2 (AI Model): Sử dụng mô hình Bi-Directional LSTM xử lý cấp độ ký tự (Character-level) để phát hiện SQL Injection và XSS.

# 🏗️ Kiến trúc Hệ thống (Architecture)
Hệ thống hoạt động trên nền tảng Docker, bao gồm 4 dịch vụ chính:

Reverse Proxy (FastAPI): Cổng vào, kiểm tra DDoS (Layer 1).

Model API (PyTorch): Bộ não AI, kiểm tra nội dung độc hại (Layer 2).

WebGoat (Target): Ứng dụng web chứa lỗ hổng để kiểm thử.

Dashboard (Streamlit): Giao diện giám sát tấn công theo thời gian thực.

# 🚀 Cài đặt & Chạy (Installation & Usage)
1. Yêu cầu (Prerequisites)
Docker Desktop (đã cài đặt và đang chạy).

Git (tùy chọn).

2. Khởi chạy hệ thống
Mở Terminal (CMD/PowerShell) tại thư mục gốc của dự án và chạy lệnh:



-Build và chạy toàn bộ hệ thống
docker-compose up --build
Lần đầu chạy sẽ mất vài phút để tải image và cài đặt thư viện.

3. Truy cập
Sau khi chạy thành công, bạn có thể truy cập:

📊 Dashboard giám sát: http://localhost:8501

🎯 WebGoat (Target App): http://localhost:8010/WebGoat/login

🧪 Hướng dẫn Demo (Testing Scenarios)
Sử dụng curl hoặc Postman để gửi các request kiểm thử.

Kịch bản 1: Traffic Sạch (Normal Traffic)
Hệ thống cho phép đi qua (HTTP 200/302).



curl -X POST -d "username=admin&password=123" http://localhost:8010/WebGoat/login -v
Kịch bản 2: Tấn công SQL Injection (Layer 2 Block)
Hệ thống AI phát hiện và chặn (HTTP 403).



curl -X POST -d "username=' OR '1'='1'--&password=123" http://localhost:8010/WebGoat/login -v
Kịch bản 3: Tấn công XSS (Layer 2 Block)
Hệ thống AI phát hiện mã Script và chặn (HTTP 403).



curl -X POST -d "comment=<script>alert(1)</script>" http://localhost:8010/WebGoat/somepage -v
Kịch bản 4: Tấn công DDoS (Layer 1 Block)
Gửi liên tục 60 requests. Các request đầu đi qua, các request sau bị chặn do vượt ngưỡng 50 req/10s (HTTP 429).


FOR /L %i IN (1,1,60) DO curl -s -o NUL -w "%{http_code} " http://localhost:8010/WebGoat/
🧠 Huấn luyện lại Mô hình (Retraining Model)
Nếu bạn muốn cập nhật dataset để mô hình thông minh hơn:

Cập nhật dữ liệu: Thêm mẫu tấn công mới vào file data/labeled_requests.csv.

Chạy script train:



python notebooks/train_simple.py
Cập nhật vào Docker:



-Copy model mới vào container
docker cp notebooks/model.pt model_api:/app/model.pt

-Khởi động lại service AI
docker-compose restart model_api

🛠️ Khắc phục sự cố (Troubleshooting)
Lỗi "Port already in use": Tắt các ứng dụng đang chiếm dụng port 8010 hoặc 8501, hoặc sửa trong docker-compose.yml.

Log Dashboard không chạy: Bấm nút Refresh Data Now hoặc nút DELETE ALL LOGS trên giao diện Dashboard để reset.

Model không chặn được tấn công: Hãy thực hiện bước "Huấn luyện lại Mô hình" và đảm bảo đã copy file model.pt mới vào container.

© 2025 - Nhóm 18 - UIT