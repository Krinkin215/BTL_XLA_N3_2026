# Chạy StarDist và bàn giao kết quả

Đức phụ trách mã thực nghiệm, điều chỉnh ngưỡng trên validation, phân đoạn dữ liệu,
tích hợp hai baseline và phần phương pháp/kết luận kỹ thuật. Điểm vào chính là
`run_experiment.py`; notebook `compare stardist - otsu - watershred.ipynb` gọi cùng mã này.

## Cài đặt trên máy cá nhân

Cấu hình kiểm chứng: Python 3.11, TensorFlow 2.16.2, tf-keras 2.16.0,
StarDist 0.9.2, CSBDeep 0.8.2, NumPy 1.26.4. Các phiên bản chính được cố định trong
`requirements-stardist.txt`. Bộ phụ thuộc nhẹ `requirements.txt` vẫn dành cho chuẩn bị dữ liệu.
`requirements-stardist-lock.txt` ghi đầy đủ phiên bản thư viện của môi trường chạy đã
kiểm chứng trên macOS ARM64/Python 3.11; dùng file này khi cần tái tạo chính xác môi trường đó.

macOS/Linux, chạy từ thư mục gốc repo:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-stardist.txt
python -m unittest test_dataset_integrity test_experiment -v
python run_experiment.py
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-stardist.txt
.\.venv\Scripts\python.exe -m unittest test_dataset_integrity test_experiment -v
.\.venv\Scripts\python.exe run_experiment.py
```

Lần đầu cần Internet để cài thư viện và tải trọng số chính thức (~5 MB).
Trọng số được kiểm tra bằng hash trong registry StarDist và lưu ở `.cache/keras/`.
Sau khi đã cài và cache trọng số, chạy lại không cần tải mô hình.
Không cần GPU cho tập 40 ảnh này. Script đặt `TF_USE_LEGACY_KERAS=1` trước khi import
TensorFlow để dùng môi trường đã kiểm chứng; nếu dùng notebook, khởi động lại kernel
sau khi cài/chuyển phiên bản TensorFlow.

## Dữ liệu và lệnh chạy

Mặc định đọc `StarDist_Dataset/experiments/pretrained_fluo/manifest.csv`:
20 ảnh validation (862 nhân) và 20 ảnh test (984 nhân).
Ảnh đầu vào là NPY float32 H×W đã chuẩn hóa P1/P99.8, clip [0,1]; không chuẩn hóa lần hai.
TIFF ground truth chứa ID instance và luôn được giữ nguyên.

```bash
python run_experiment.py --output results/stardist_pretrained
```

Mỗi lần chạy dùng **thư mục đầu ra mới hoặc rỗng**; script từ chối ghi đè một lần chạy
đã có tệp. Ví dụ chạy đối chứng chỉ StarDist, giữ ngưỡng pretrained:

```bash
python run_experiment.py --stardist-only --default-thresholds --output results/stardist_defaults
```

Khi dữ liệu nằm nơi khác:

```bash
python run_experiment.py --dataset /path/to/StarDist_Dataset --output results/run_02
```

`--manifest` nhận đường dẫn tuyệt đối hoặc đường dẫn tính từ `--dataset`.
Manifest thực nghiệm phải chỉ chứa hai tập val/test, ID duy nhất, có cả hai tập,
và metadata/đường dẫn/split phải khớp `dataset_manifest.csv` của gói dữ liệu.
Protocol này dành cho nhóm ảnh xám nền tối với NPY đơn kênh; không tự chuyển ảnh RGB
hoặc ảnh nền sáng sang miền huỳnh quang.

## Quy trình chọn tham số

1. Kiểm tra manifest và nạp **validation**. Kiểm tra kích thước, dtype, khoảng giá trị
   và số instance trong nhãn.
2. Tải `2D_versatile_fluo`, giữ nguyên trọng số; không train/fine-tune.
3. Dùng `optimize_thresholds` của StarDist để chọn xác suất/NMS trên validation,
   với NMS ∈ {0.3, 0.4, 0.5}, mục tiêu matching accuracy tại IoU = 0.5.
4. Chọn Otsu trên 9 cấu hình sigma/min_size; Watershed trên 36 cấu hình
   sigma/min_size/min_distance. Hai baseline dùng cùng mục tiêu và validation.
5. Ghi `selected_config.json` trước khi nạp ảnh/nhãn test. Các ngưỡng StarDist được
   truyền tường minh khi dự đoán. Không lưu ngưỡng mới vào cache mô hình.
6. Warm-up mỗi phương pháp trên một ảnh validation; xuất dự đoán val/test,
   điểm đánh giá và các hình minh họa. Thời gian không gồm đọc/ghi hoặc matching.

Không chọn lại ngưỡng sau khi xem test. Nếu mở rộng tập đánh giá, chốt manifest
trước khi chạy và giữ nguyên split gốc. Kết quả validation là điểm sau tuning.

## Tệp bàn giao

Kết quả nằm ngoài gói `StarDist_Dataset/` để giữ nguyên checksum gói dữ liệu:

```text
results/stardist_pretrained/
  KET_QUA.md                         # Bảng test và kết luận sinh từ lần chạy
  selected_config.json               # Ngưỡng, tham số và ID validation
  run_metadata.json                  # Phiên bản, thiết bị, hash mã/đầu vào/mô hình
  experiment_manifest.csv            # Danh sách ảnh của lần chạy
  validation_search_otsu.csv
  validation_search_watershed.csv
  metrics_summary.csv                # Micro metrics cho val/test, IoU 0.50..0.95
  metrics_per_image.csv              # TP/FP/FN và thời gian từng ảnh tại IoU 0.5
  predictions_manifest.csv           # Truy từ source_id tới TIFF và checksum
  predictions/{stardist,otsu,watershed}/{val,test}/*.tif
  figures/
    segmentation_comparison.png
    matching_accuracy_iou.png
    stardist_probability.png
    stardist_probability.npy
    watershed_steps.png
    sample_ids.json
  checksums.sha256
```

TIFF dự đoán dùng uint16, tự chuyển uint32 nếu ID vượt 65.535; nền 0, mỗi nhân một ID.
Không dùng ảnh overlay PNG làm nhãn. Ví dụ đọc kết quả:

```python
from pathlib import Path
import pandas as pd
import tifffile

run = Path('results/stardist_pretrained')
files = pd.read_csv(run / 'predictions_manifest.csv')
row = files.query("method == 'stardist' and split == 'test'").iloc[0]
labels = tifffile.imread(run / row.prediction_path)
print(row.source_id, labels.shape, labels.dtype)
```

`run_metadata.json` chỉ được ghi với `status=complete` khi bước suy luận/đánh giá/vẽ hình
đã xong. Nếu chạy lỗi giữa chừng, chọn thư mục đầu ra mới sau khi xử lý lỗi.

## Notebook và Colab/Kaggle

Mở notebook ở thư mục gốc repo, chạy lần lượt các ô. Ô đầu tìm `run_experiment.py`;
nếu repo nằm nơi khác, sửa `REPO`. Trên Colab/Kaggle cần đưa **cả repo** vào máy chạy,
gồm mã, `requirements-stardist.txt` và `StarDist_Dataset/`, rồi đặt `REPO` đúng vị trí.
`OUTPUT` phải nằm trong vùng có quyền ghi; ví dụ `/content/results/run_01` hoặc
`/kaggle/working/results/run_01`. Giữ kernel Python tương thích với bộ phụ thuộc.

Notebook cài thư viện vào kernel đang chạy và gọi script bằng subprocess mới,
giúp thiết lập Keras trước khi TensorFlow được import. Các ô sau đọc bảng và hình
từ thư mục kết quả. Dùng `RUN_EXPERIMENT=False` khi chỉ muốn xem lại lần chạy đã có.

Định nghĩa chỉ số, phương pháp và giới hạn diễn giải nằm trong
[BAO_CAO_KY_THUAT.md](BAO_CAO_KY_THUAT.md).
