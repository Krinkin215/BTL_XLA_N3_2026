# StarDist — Phân đoạn nhân tế bào

Dự án chuẩn bị dữ liệu DSB2018/BBBC038 và chạy phân đoạn từng đối tượng bằng StarDist pretrained, so sánh với Otsu và Watershed.

## Chạy thực nghiệm — phần Đức

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-stardist.txt
python -m unittest test_dataset_integrity test_experiment -v
python run_experiment.py
```

Quy trình chọn ngưỡng trên 20 ảnh validation, chốt cấu hình rồi đánh giá 20 ảnh test.
Mỗi lần chạy xuất mask TIFF giữ ID instance, cấu hình, TP/FP/FN, precision/recall/F1,
matching accuracy, thời gian từng ảnh và hình minh họa. Không huấn luyện/fine-tune.

- [Hướng dẫn chạy, Windows và Colab/Kaggle](HUONG_DAN_CHAY_STARDIST.md).
- [Notebook thực nghiệm](compare%20stardist%20-%20otsu%20-%20watershred.ipynb), dùng chung `run_experiment.py`.
- [Phương pháp và kết luận kỹ thuật](BAO_CAO_KY_THUAT.md).
- [Kết quả chạy và bảng so sánh](results/stardist_pretrained/KET_QUA.md).

Đầu ra mặc định ở `results/stardist_pretrained/`, nằm ngoài gói dữ liệu để giữ nguyên
checksum bàn giao. Chọn `--output results/run_02` khi chạy lại; không ghi đè lần chạy trước.

## Dữ liệu

Đã chuẩn bị **670 ảnh và 29.461 nhãn nhân tế bào**. Ảnh PNG giữ nguyên nội dung nguồn; các mask nhị phân được gộp thành TIFF `uint16`, nền 0 và mỗi nhân một ID riêng.

| Tập | Số ảnh | Số nhân |
|---|---:|---:|
| Train | 398 | 17.350 |
| Validation | 130 | 5.707 |
| Test | 142 | 6.404 |

Thí nghiệm pretrained ban đầu sử dụng **20 ảnh validation và 20 ảnh test**, được liệt kê trong [manifest thực nghiệm](StarDist_Dataset/experiments/pretrained_fluo/manifest.csv). Phần train được giữ dự phòng; không huấn luyện hoặc fine-tune trong cấu hình này.

## Cấu trúc

```text
StarDist_Dataset/
  processed_data/{train,val,test}/{images,masks,normalized}/
  experiments/pretrained_fluo/
  experiments/grayscale100/
  splits/
  results/                         # QC dữ liệu
  Chuan_bi_du_lieu.ipynb
  dataset_manifest.csv
  instance_mapping.csv
  quality_report.json
prepare_dataset.py
test_dataset_integrity.py
finalize_delivery.py
requirements.txt
requirements-stardist.txt
run_experiment.py
test_experiment.py
results/stardist_pretrained/        # Mask, bảng điểm và hình thực nghiệm
HUONG_DAN_CHAY_STARDIST.md
BAO_CAO_KY_THUAT.md
HUONG_DAN_BAT_DAU.md
```

Ảnh, TIFF nhãn và NPY chuẩn hóa có trong repo để có thể đọc dữ liệu sau khi clone. ZIP đóng gói và dữ liệu nguồn `stage1_train.zip` không được lưu trong Git; ZIP có thể tạo từ thư mục dữ liệu hiện có.

## Bắt đầu

```bash
git clone git@github.com:Krinkin215/BTL_XLA_N3_2026.git
cd BTL_XLA_N3_2026
python -m pip install -r requirements.txt
python -m unittest test_dataset_integrity -v
```

- Hướng dẫn sử dụng: [HUONG_DAN_BAT_DAU.md](HUONG_DAN_BAT_DAU.md).
- Notebook máy cá nhân/Colab: [Chuan_bi_du_lieu.ipynb](StarDist_Dataset/Chuan_bi_du_lieu.ipynb). Đặt `PROJECT` tới thư mục `StarDist_Dataset`.
- Dữ liệu và tiền xử lý: [README bộ dữ liệu](StarDist_Dataset/README.md), [báo cáo](StarDist_Dataset/BAO_CAO_DU_LIEU.md).
- Cấu hình pretrained và ví dụ dự đoán: [HUONG_DAN_PRETRAINED.md](StarDist_Dataset/HUONG_DAN_PRETRAINED.md).

Để tái tạo dữ liệu từ nguồn tải riêng, chọn thư mục đầu ra mới:

```bash
python prepare_dataset.py --raw "PATH_TO/stage1_train.zip" --out "StarDist_Dataset_rebuilt" --zip
```

`finalize_delivery.py` dùng để cập nhật tài liệu và kiểm tra ZIP của bản dữ liệu hiện có; đường dẫn nguồn cục bộ cần phù hợp môi trường thực thi. Phần chuẩn bị dữ liệu chỉ cần NumPy và Pillow. Notebook cài thêm pandas; dự đoán pretrained cần TensorFlow và StarDist theo hướng dẫn riêng.

## Nhánh Git

Nhánh `main` hiện có dữ liệu đã chuẩn bị và notebook so sánh. Nhóm có thể tiếp tục
dùng nhánh theo thành viên, tích hợp qua pull request. Commit dùng định dạng
Conventional Commits, ví dụ `feat(exp): ...`, `fix(data): ...`, `docs: ...`.

## Giới hạn và nguồn

Mô hình `2D_versatile_fluo` dành cho ảnh nhân huỳnh quang đơn kênh; phân loại ảnh xám nền tối trong dữ liệu mới dựa trên diện mạo. Cả `2D_versatile_fluo` và `2D_paper_dsb2018` đã học một phần DSB2018, nên bản test cục bộ chưa chứng minh đánh giá tổng quát hóa độc lập. Kết quả thực nghiệm nằm trong `results/stardist_pretrained/`; `StarDist_Dataset/results/` chỉ chứa hình kiểm tra dữ liệu. Matching accuracy = TP/(TP+FP+FN), không phải COCO AP.

- [BBBC038v1](https://bbbc.broadinstitute.org/BBBC038): nguồn dữ liệu, CC0.
- [Caicedo et al. (2019)](https://doi.org/10.1038/s41592-019-0612-7): bài báo DSB2018.
- [StarDist](https://github.com/stardist/stardist#pretrained-models-for-2d): mô hình pretrained và hướng dẫn sử dụng.
