# Dữ liệu và tiền xử lý

## 3.1. Giới thiệu bộ dữ liệu

Thí nghiệm sử dụng gói stage1_train của Data Science Bowl 2018, được phân phối qua BBBC038v1 [1]. Bộ dữ liệu gồm ảnh nhân tế bào từ nhiều điều kiện thu nhận; mỗi nhân được chú giải bằng một mask nhị phân riêng. Pipeline phát hiện 670 thư mục mẫu, giữ 670 ảnh hợp lệ, tổng cộng 29,461 nhân; 0 mẫu bị loại. Số liệu được tính trực tiếp từ tệp cục bộ.

## 3.2. Cấu trúc và nhãn

Ảnh PNG được sao chép nguyên byte, giữ kích thước gốc, từ 256–1388 pixel chiều rộng và 256–1040 pixel chiều cao. Các mask nguồn được sắp xếp theo tên, kiểm tra rồi gộp thành một label map TIFF uint16: nền 0 và ID 1..N cho từng nhân. Không gộp các nhân thành nhãn nhị phân. Mỗi nhãn có bản ghi truy xuất về mask PNG nguồn trong instance_mapping.csv. Pipeline phát hiện mask sai kích thước, rỗng, không nhị phân hoặc chồng lấn và loại cả mẫu đó kèm lý do thay vì âm thầm ghi đè.

## 3.3. Phân chia dữ liệu

| Tập | Số ảnh | Số nhân | Tập khởi đầu |
|---|---:|---:|---:|
| train | 398 | 17350 | 60 |
| val | 130 | 5707 | 20 |
| test | 142 | 6404 | 20 |

Chia train/validation/test theo mục tiêu 60/20/20, seed 42, phân tầng theo nhóm diện mạo và mức số lượng nhân. Ảnh trùng chính xác RGB pixel được gom vào cùng một tập (0 nhóm trùng). Tỉ lệ thực tế chịu ảnh hưởng làm tròn và kích thước nhóm. Validation phục vụ chỉnh tham số; test dùng cho đánh giá cuối cùng. Các thuật toán dùng cùng source ID để phép so sánh nhất quán. Nếu tạo patch/augmentation, phải thực hiện sau khi chia tập và chỉ augment train.

Tập khởi đầu gồm 100 ảnh xám nền tối trong experiments/grayscale100/manifest.csv (mục tiêu 60 train, 20 val, 20 test). Các ảnh được lấy ngẫu nhiên trong từng split, chưa xác nhận miền nhuộm và chưa dựa trên điểm dự đoán. Quy mô thực nghiệm được xác định theo manifest sử dụng.

**Cấu hình thực nghiệm: pretrained với trọng số cố định, không huấn luyện/fine-tune.** Thí nghiệm khởi đầu dùng 20 ảnh validation và 20 ảnh test trong experiments/pretrained_fluo/manifest.csv, lấy nguyên các split của tập grayscale100; 60 ảnh train được giữ dự phòng. Mô hình đề xuất là 2D_versatile_fluo sau khi xác nhận miền ảnh. Cả 2D_versatile_fluo và 2D_paper_dsb2018 được StarDist mô tả là đã huấn luyện trên một phần DSB2018 [2]; do chưa đối chiếu danh sách huấn luyện gốc, không diễn giải kết quả là đánh giá tổng quát hóa độc lập.

## 3.4. Tiền xử lý

Các nhóm diện mạo được thống kê:

- `color`: 108 ảnh.
- `grayscale_dark_background`: 546 ảnh.
- `grayscale_light_background`: 16 ảnh.

Quy tắc ảnh xám yêu cầu R=G=B tại mọi pixel; nền được gợi ý từ median dải biên 5% với ngưỡng 127.5. Quy tắc này không thay thế thông tin loại kính hiển vi hoặc nhuộm. Với ảnh xám nền tối, xuất thêm ảnh đơn kênh float32 theo I_norm = clip((I−P1)/(P99.8−P1),0,1); percentile được tính riêng từng ảnh và ghi trong manifest. Ảnh PNG gốc luôn giữ nguyên. Không resize, làm mượt hoặc áp biến đổi cường độ lên nhãn. Otsu/Watershed có thể cần tiền xử lý riêng, được chọn trên validation và ghi rõ trong phần thực nghiệm.

## 3.5. Kiểm tra chất lượng và giới hạn

Kiểm tra tự động bao gồm cấu trúc ảnh/mask, kích thước, mask nhị phân không rỗng, chồng lấn, đọc lại TIFF, ID liên tục và diện tích khớp từng mask nguồn, SHA256 bản sao PNG, phạm vi NPY chuẩn hóa và ảnh trùng qua các split. Minh họa original/instance label/biên nhãn nằm trong results/quality_control.png; thông tin mẫu trong results/preview_samples.csv. Hình chuẩn hóa nằm trong results/normalization_example.png. Kết quả chi tiết trong quality_report.json.

Bản chia ở cấp ảnh chưa sử dụng nhóm nguồn thí nghiệm nên không loại trừ tương quan giữa các ảnh hoặc ảnh gần trùng. Việc kiểm tra cơ học không xác nhận tính đúng đắn sinh học của mọi chú giải. Không dùng dự đoán làm ground truth. Với mô hình pretrained, cần kiểm tra sự giao nhau với dữ liệu huấn luyện trước khi tuyên bố đánh giá độc lập; StarDist có mô hình mang tên 2D_paper_dsb2018 [2]. Phần này mới mô tả chuẩn bị dữ liệu, chưa có kết quả so sánh StarDist/Otsu/Watershed.

## Tài liệu tham khảo

[1] BBBC038v1, Broad Bioimage Benchmark Collection. https://bbbc.broadinstitute.org/BBBC038 ; Caicedo et al. (2019), Nature Methods. https://doi.org/10.1038/s41592-019-0612-7.

[2] StarDist, mã nguồn và danh sách mô hình pretrained. https://github.com/stardist/stardist ; https://github.com/stardist/stardist/blob/main/stardist/models/__init__.py.
