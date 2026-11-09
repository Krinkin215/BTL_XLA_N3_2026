# Bộ dữ liệu StarDist

Đã xử lý thực tế 670 ảnh, 29,461 nhân; 0 mẫu bị loại do kiểm tra dữ liệu. Nguồn gốc vẫn ở Downloads và không bị sửa. Chuẩn bị ngày 09/11/2026 (Asia/Saigon).

## Mở và sử dụng

**Chế độ pretrained:** không train/fine-tune. Tập thực nghiệm ban đầu là `experiments/pretrained_fluo/manifest.csv` (20 validation, 20 test, giữ nguyên split); mô hình đề xuất `2D_versatile_fluo`. Phần train được giữ dự phòng. Xem `HUONG_DAN_PRETRAINED.md` và `experiments/pretrained_fluo/config.json`. Cả `2D_versatile_fluo` lẫn `2D_paper_dsb2018` đã học một phần DSB2018 theo tài liệu StarDist, nên bản test này chưa chứng minh đánh giá tổng quát hóa độc lập.

1. Giải nén ZIP vào một thư mục; mở `Chuan_bi_du_lieu.ipynb` trên Jupyter/Colab, sửa `PROJECT` tới thư mục này.
2. Các thuật toán sử dụng cùng manifest. Dùng cột `image_path`, `mask_path`; đường dẫn tương đối với thư mục này.
3. TIFF mask là **uint16**, nền = 0, mỗi nhân = một ID 1..N. Đọc bằng `np.asarray(Image.open(path))` hoặc `tifffile.imread(path)`; không convert mask sang uint8 hoặc nhị phân khi tính matching instance.
4. `experiments/grayscale100/manifest.csv` có 100 ảnh khởi đầu; `train_ids.txt`, `val_ids.txt`, `test_ids.txt` là các source ID dùng chung. Đó là các dòng tham chiếu dữ liệu, không nhân bản tệp.
5. Ảnh PNG gốc giữ nguyên byte và kích thước. Cột `normalized_path` trỏ đến NPY float32 đơn kênh cho nhóm ảnh xám nền tối; cột trống nghĩa là chưa áp dụng chuẩn hóa này.

## Phân chia toàn bộ dữ liệu

| Tập | Ảnh | Nhân | Ảnh trong tập khởi đầu |
|---|---:|---:|---:|
| train | 398 | 17350 | 60 |
| val | 130 | 5707 | 20 |
| test | 142 | 6404 | 20 |

Seed = 42; phân tầng theo diện mạo và số nhân (<20, 20–59, >=60). Giữ ảnh có cùng RGB pixel hash trong một tập. Có 0 nhóm ảnh trùng pixel. Tỉ lệ mục tiêu 60/20/20, làm tròn theo từng tầng/nhóm. Chia ở cấp ảnh, trước khi tạo patch/augmentation. Không dùng test để chọn tham số. Nếu chỉ dùng pretrained, phần train có thể chưa dùng.

## Nhóm diện mạo

- `color`: 108 ảnh.
- `grayscale_dark_background`: 546 ảnh.
- `grayscale_light_background`: 16 ảnh.

`grayscale_*`: ba kênh RGB bằng nhau ở mọi pixel (bao gồm ảnh xám lưu dưới dạng RGB/RGBA). Nền tối/sáng được gợi ý bằng median dải biên 5%, ngưỡng 127.5 trên thang 0..255. Đây là **quy tắc diện mạo**, không xác nhận huỳnh quang hoặc H&E. Không coi mọi ảnh màu là H&E. Alpha gốc được giữ trong PNG; khi phân loại/hash dùng RGB, không dùng alpha.

Tập khởi đầu lấy ngẫu nhiên tối đa 60/20/20 ảnh từ nhóm xám nền tối của từng split, không chọn bằng chất lượng dự đoán. Cần xác nhận mô hình StarDist và kiểm tra trực quan ảnh trước khi chốt thí nghiệm. Các nhóm diện mạo khác được giữ trong manifest đầy đủ.

## Tiền xử lý

Không resize/crop, đổi hướng, Gaussian, CLAHE hoặc chỉnh sửa nhãn. Ảnh gốc được giữ để thử tiền xử lý riêng cho Otsu/Watershed trên validation.

Với nhóm xám nền tối: lấy kênh R (R=G=B), tính P1 và P99.8 trên từng ảnh, `clip((I-P1)/(P99.8-P1),0,1)` và lưu NPY float32. Tham số nằm trong manifest; không tính percentile chung trên test/train. Đây là lựa chọn được ghi rõ của pipeline này; không mặc định trùng mọi cấu hình chuẩn hóa StarDist. Không chuẩn hóa hai lần. Các nhóm khác cần pipeline riêng phù hợp mô hình.

## Kiểm tra và truy xuất

- `quality_report.json`: kết quả kiểm tra số lượng, kích thước, giá trị mask, chồng lấn, TIFF đọc lại, hash ảnh và chuẩn hóa.
- `instance_mapping.csv`: truy từ mỗi instance ID về tên mask nguồn và diện tích.
- `source_inventory.csv`: thống kê và lý do loại, gồm tất cả thư mục nguồn được phát hiện.
- `results/quality_control.png`, `results/normalization_example.png`: hình minh họa; `preview_samples.csv` ghi đầy đủ ID các ảnh xem.
- `checksums.sha256`: SHA256 các tệp của gói (trừ chính tệp checksum).
- `prepare_dataset.py`: mã tái tạo; chỉ cần numpy và Pillow. Script từ chối ghi đè thư mục đầu ra đã có nội dung.

Chạy lại ở Windows/Colab:
```bash
python -m pip install -r requirements.txt
python prepare_dataset.py --raw "PATH_TO/stage1_train" --out "PATH_TO/new_output"
```
`--raw` nhận thư mục chứa các thư mục image ID hoặc file `stage1_train.zip`. ZIP bàn giao chỉ chứa ảnh gốc được sao chép, nhãn gộp và tài liệu; không kèm lại hàng nghìn mask nguồn. Nguồn chạy thực tế: `C:\Users\ADMIN\Downloads\stage1_train.zip`.

## Giới hạn cần ghi trong báo cáo

Không có metadata nguồn thí nghiệm được tích hợp vào bản chia này, nên chưa kiểm soát nhóm bệnh nhân/đĩa/nguồn thu nhận hoặc ảnh gần trùng. Hash chỉ loại rò rỉ ảnh trùng pixel; chia ngẫu nhiên không chứng minh tính độc lập sinh học. Một số pretrained StarDist có liên quan DSB2018: đánh giá trên đây không tự động là đánh giá tổng quát hóa độc lập. Nhãn gốc có thể có thiếu sót chú giải; chưa sửa nhãn bằng mô hình. Gói dữ liệu chưa chứa kết quả dự đoán hoặc điểm đánh giá phân đoạn.

## Thành phần sử dụng

- **StarDist:** ảnh, TIFF instance và danh sách ID; chọn mô hình và pipeline phù hợp miền ảnh.
- **Otsu/Watershed:** cùng danh sách val/test; chỉnh tham số trên validation và lưu kết quả ở cấp instance để so sánh.
- **Báo cáo:** `BAO_CAO_DU_LIEU.md`, bảng thống kê và hình minh họa.

## Nguồn

- BBBC038v1: https://bbbc.broadinstitute.org/BBBC038 (CC0; mô tả cấu trúc ảnh và mask).
- Caicedo et al. (2019), *Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl*, https://doi.org/10.1038/s41592-019-0612-7.
- StarDist: https://github.com/stardist/stardist ; danh sách mô hình https://github.com/stardist/stardist/blob/main/stardist/models/__init__.py.
