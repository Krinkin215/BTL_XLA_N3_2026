# Hướng dẫn sử dụng StarDist pretrained

Thí nghiệm sử dụng mô hình pretrained với trọng số có sẵn, không huấn luyện hoặc fine-tune. Mô hình đề xuất là **2D_versatile_fluo**, dành cho ảnh nhân huỳnh quang đơn kênh theo tài liệu StarDist [1]. Cần kiểm tra trực quan để xác nhận miền ảnh; nhóm xám nền tối được phân loại theo diện mạo pixel. Định dạng RGB không đủ để xác định ảnh H&E.

## Tập dữ liệu thực nghiệm

- Manifest: `experiments/pretrained_fluo/manifest.csv`.
- Validation: **20 ảnh** để kiểm tra đầu vào, chọn tiền xử lý/ngưỡng StarDist, Otsu và Watershed.
- Test: **20 ảnh** để so sánh cuối cùng sau khi chốt cấu hình trên validation.
- Không cần tập train để chạy trọng số có sẵn. 60 ảnh train của tập grayscale100 và phần train đầy đủ vẫn được giữ dự phòng, chưa được sử dụng trong protocol này.
- Giữ nguyên ID và split; không chuyển ảnh train sang test hay chia lại theo kết quả dự đoán. Các thuật toán sử dụng cùng danh sách ID ở thư mục trên.

Tập thực nghiệm ban đầu gồm 40 ảnh. Khi mở rộng, lấy thêm ảnh từ cùng split gốc và chốt danh sách trước khi xem điểm test.

## Đầu vào và cách tải mô hình

`normalized_path` trỏ tới NPY float32 H×W đã chuẩn hóa P1/P99.8 và clip [0,1]. Mỗi ảnh dùng percentile riêng; không chuẩn hóa thêm lần nữa. Ảnh gốc và TIFF ground truth vẫn có trong manifest. Pipeline chuẩn hóa này được chọn minh bạch cho thí nghiệm; không mặc định giống mọi ví dụ StarDist.

Ví dụ chạy dự đoán trên Colab (mô hình chưa được thực thi trong gói dữ liệu này):

```python
# Ô Colab riêng: %pip install tensorflow stardist
from pathlib import Path
import pandas as pd
import numpy as np
from stardist.models import StarDist2D

project = Path('/content/StarDist_Dataset')
df = pd.read_csv(project / 'experiments/pretrained_fluo/manifest.csv')
row = df[df['split'] == 'val'].iloc[0]  # bắt đầu trên validation
image = np.load(project / row['normalized_path'])
model = StarDist2D.from_pretrained('2D_versatile_fluo')
predicted_instances, details = model.predict_instances(image)
print(predicted_instances.shape, predicted_instances.max())
```

Khởi đầu với ngưỡng đi kèm mô hình. Nếu thay đổi ngưỡng xác suất/NMS, chọn trên validation và lưu giá trị trước khi chạy test. Khi so sánh instance, sử dụng cùng ngưỡng IoU matching, chẳng hạn 0.5, cùng precision/recall/F1 và TP/FP/FN; không gọi trung bình IoU pixel hoặc accuracy của matching là COCO AP. Gói dữ liệu chưa chứa kết quả so sánh các thuật toán.

## Giới hạn đánh giá

Theo tài liệu chính thức, **cả 2D_versatile_fluo và 2D_paper_dsb2018 đều đã được huấn luyện trên một phần DSB2018** [1]. Bản chia val/test cục bộ tách dữ liệu chỉnh tham số khỏi dữ liệu đánh giá cuối cùng; chưa xác định ảnh test có xuất hiện trong dữ liệu huấn luyện mô hình pretrained hay không.

Thí nghiệm sử dụng StarDist pretrained với trọng số cố định trên tập ảnh DSB2018 được chọn. Do chưa đối chiếu chính xác danh sách huấn luyện gốc, kết quả chỉ phục vụ minh họa và so sánh trong phạm vi dữ liệu này, chưa phải ước lượng khả năng tổng quát hóa độc lập.

[1] https://github.com/stardist/stardist#pretrained-models-for-2d
