# Kết quả chạy thực nghiệm

Số liệu được sinh trực tiếp bởi `run_experiment.py` trong lần chạy này.

| Phương pháp | Ảnh test | TP | FP | FN | Precision | Recall | F1 | Matching accuracy | Giây/ảnh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stardist | 20 | 755 | 83 | 229 | 0.9010 | 0.7673 | 0.8288 | 0.7076 | 0.0936 |
| otsu | 20 | 420 | 57 | 564 | 0.8805 | 0.4268 | 0.5749 | 0.4035 | 0.0048 |
| watershed | 20 | 470 | 98 | 514 | 0.8275 | 0.4776 | 0.6057 | 0.4344 | 0.0157 |

IoU matching = 0.5; cộng TP/FP/FN trên toàn bộ tập trước khi tính các chỉ số (micro).
Matching accuracy = TP/(TP+FP+FN); đây không phải COCO AP hoặc diện tích dưới đường precision–recall.

StarDist đạt F1 = 0.8288, precision = 0.9010, recall = 0.7673.
Chênh lệch F1 với baseline cao nhất (watershed): +0.2231.

Cấu hình đã chốt trên validation trước khi nạp nhãn test:

```json
{
  "stardist": {
    "prob_thresh": 0.47576062024687477,
    "nms_thresh": 0.3
  },
  "otsu": {
    "sigma": 0.8,
    "min_size": 20
  },
  "watershed": {
    "sigma": 1.0,
    "min_size": 30,
    "min_distance": 9
  }
}
```

Thời gian chỉ gồm suy luận và hậu xử lý; đã warm-up trên một ảnh validation. Không gồm tải mô hình, tuning, đọc/ghi tệp, matching hoặc vẽ hình. Phụ thuộc phần cứng và kích thước ảnh.

Đây là so sánh trong tập DSB2018 đã chọn. Mô hình pretrained đã học một phần DSB2018; chưa xác minh giao nhau với tập test, nên chưa thể kết luận khả năng tổng quát hóa độc lập.

Chi tiết: `metrics_summary.csv`, `metrics_per_image.csv`, `predictions_manifest.csv`, `selected_config.json`, `run_metadata.json`, `figures/`.
