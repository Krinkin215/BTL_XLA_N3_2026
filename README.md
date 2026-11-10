# StarDist — Phân đoạn nhân tế bào

Dự án chuẩn bị dữ liệu DSB2018/BBBC038 cho thí nghiệm phân đoạn từng đối tượng bằng StarDist pretrained và so sánh với Otsu, Watershed.

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
  results/
  Chuan_bi_du_lieu.ipynb
  dataset_manifest.csv
  instance_mapping.csv
  quality_report.json
prepare_dataset.py
test_dataset_integrity.py
finalize_delivery.py
requirements.txt
HUONG_DAN_BAT_DAU.md
```

Ảnh, TIFF nhãn và NPY chuẩn hóa có trong repo để có thể đọc dữ liệu sau khi clone. ZIP đóng gói và dữ liệu nguồn `stage1_train.zip` không được lưu trong Git; ZIP có thể tạo từ thư mục dữ liệu hiện có.

## Bắt đầu

```bash
git clone --branch kien git@github.com:Krinkin215/BTL_XLA_N3_2026.git
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

- `main`: chỉ có commit khởi tạo rỗng, chưa chứa dự án.
- `develop`: nhánh tích hợp, bắt đầu từ cùng commit khởi tạo.
- `kien`: nhánh phát triển dự án, tách từ `develop` theo tên nhánh được sử dụng trong repo.

Luồng tích hợp: `kien` → pull request vào `develop` → phát hành vào `main`. Hiện chưa merge dự án vào `develop` hoặc `main`. Commit dùng định dạng Conventional Commits, ví dụ `feat(data): ...`, `fix(data): ...`, `docs: ...`.

## Giới hạn và nguồn

Mô hình đề xuất `2D_versatile_fluo` cần ảnh nhân huỳnh quang đơn kênh; phân loại ảnh xám nền tối trong dữ liệu mới dựa trên diện mạo. Cả `2D_versatile_fluo` và `2D_paper_dsb2018` đã học một phần DSB2018, nên bản test cục bộ chưa chứng minh đánh giá tổng quát hóa độc lập. Gói hiện tại chưa chứa kết quả chạy các thuật toán.

- [BBBC038v1](https://bbbc.broadinstitute.org/BBBC038): nguồn dữ liệu, CC0.
- [Caicedo et al. (2019)](https://doi.org/10.1038/s41592-019-0612-7): bài báo DSB2018.
- [StarDist](https://github.com/stardist/stardist#pretrained-models-for-2d): mô hình pretrained và hướng dẫn sử dụng.
