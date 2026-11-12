# Phương pháp và kết luận kỹ thuật — phần Đức

## Phạm vi thực nghiệm

Thí nghiệm phân đoạn từng nhân tế bào dùng StarDist pretrained `2D_versatile_fluo`,
so sánh với Otsu và Otsu kết hợp Watershed. Không huấn luyện/fine-tune mạng và không
sử dụng phần train trong quy trình này. Manifest cố định gồm 20 ảnh validation
(862 nhân) và 20 ảnh test (984 nhân), từ nhóm xám nền tối của dữ liệu DSB2018 đã chuẩn bị.

Ảnh xám nền tối là nhóm diện mạo pixel, chưa xác nhận phương pháp nhuộm cho từng ảnh.
Mô hình chính thức dành cho ảnh nhân huỳnh quang đơn kênh [1]; cần cân nhắc sự phù hợp
miền ảnh khi diễn giải kết quả.

## StarDist

StarDist biểu diễn đối tượng bằng đa giác hình sao: mạng dự đoán xác suất đối tượng
và các khoảng cách theo những tia xuyên tâm, sau đó dùng non-maximum suppression
(NMS) để loại các đề xuất chồng lấn [1, 2]. Dự đoán cuối cùng là label map, trong đó
0 là nền và mỗi nhân có ID riêng. Mã dùng trọng số chính thức, có kiểm tra hash tệp tải.

Đầu vào là ảnh H×W float32 đã chuẩn hóa riêng từng ảnh:

`I_norm = clip((I − P1)/(P99.8 − P1), 0, 1)`.

Không chuẩn hóa lại NPY đã chuẩn bị, không resize ảnh và không làm mượt, nhị phân hóa
hoặc sửa ground truth. Mạng có thể pad/crop nội bộ khi cần để tương thích kích thước,
nhưng label map xuất ra phải cùng kích thước ảnh ban đầu.

Hai tham số được chọn bằng `StarDist2D.optimize_thresholds` trên validation:
ngưỡng xác suất (`prob_thresh`) và ngưỡng NMS (`nms_thresh`). NMS thử {0.3, 0.4, 0.5};
ở mỗi giá trị, thư viện tối ưu ngưỡng xác suất bằng tìm kiếm golden-section với
thiết lập mặc định của StarDist 0.9.2. Mục tiêu là matching accuracy tại IoU = 0.5 [3, 4].
Đây là tìm kiếm số trên validation, không bảo đảm tối ưu toàn cục mọi ngưỡng có thể có.
Giá trị cuối cùng được lưu trong `selected_config.json` trước khi nạp test.

## Baseline và tích hợp kết quả

Otsu làm mượt Gaussian, tìm ngưỡng toàn cục, opening với đĩa bán kính 2,
loại vùng nhỏ rồi gán ID cho các thành phần liên thông. Tìm tham số trên validation:
sigma ∈ {0.8, 1.0, 1.5}, diện tích tối thiểu ∈ {20, 30, 50} pixel.

Watershed dùng cùng bước tạo foreground, tính biến đổi khoảng cách Euclid,
tìm cực đại cục bộ làm marker và watershed trên âm của khoảng cách trong foreground.
Thử thêm khoảng cách cực đại ∈ {3, 5, 7, 9} pixel. Cho phép marker sát biên và bổ sung
một marker tại khoảng cách lớn nhất nếu một thành phần chưa có marker; nhờ đó tránh
bỏ mất cả vùng chỉ vì không có cực đại thỏa điều kiện. Đây là thay đổi có chủ đích
so với notebook cũ; điểm và thời gian có thể khác các output cũ đã lưu.

Cả ba phương pháp dùng cùng NPY đầu vào, source ID, split và ground truth.
Otsu thử 9 cấu hình; Watershed thử 36 cấu hình độc lập. Khi bằng điểm, lấy cấu hình
xuất hiện trước trong lưới cố định. Các bảng tìm kiếm được lưu để truy xuất lựa chọn.
Phạm vi baseline chỉ bao gồm những lưới này, không khẳng định là cấu hình cổ điển tốt nhất.

## Đánh giá ở cấp instance

Hai đối tượng được ghép một-một nếu giao trên hợp (IoU) đạt ngưỡng τ.
Sử dụng `stardist.matching` để tìm matching và đếm TP/FP/FN [3]. Tổng hợp **micro**:
cộng số đếm của toàn tập rồi tính:

- Precision = TP/(TP+FP).
- Recall = TP/(TP+FN).
- F1 = 2TP/(2TP+FP+FN).
- Matching accuracy = TP/(TP+FP+FN).

Thư viện trả về 0 khi mẫu số bằng 0. Các cột `n_true`, `n_pred`, mean matched IoU
và panoptic quality được xuất kèm trong bảng tổng hợp. Bảng chính dùng τ = 0.5;
đường cong đánh giá dùng τ từ 0.50 đến 0.95 với bước 0.05.

Notebook cũ gọi `matching.accuracy` là “AP”. Báo cáo này dùng tên **matching accuracy**
và ghi rõ công thức; chỉ số không phải COCO AP hoặc diện tích dưới đường precision–recall.
Không gọi trung bình các điểm micro trên dải IoU là điểm leaderboard DSB2018, vì giao thức
tổng hợp có thể khác. Điểm từng ảnh tại IoU 0.5 được lưu riêng để kiểm tra lỗi và phân bố.

Thời gian dùng `time.perf_counter`, sau một lần warm-up mỗi phương pháp trên validation.
Phép đo gồm suy luận và hậu xử lý; không gồm tải trọng số, chọn tham số, đọc/ghi,
matching và tạo hình. Đây là thời gian một lần chạy trên phần cứng trong metadata,
không phải benchmark hiệu năng đa thiết bị.

## Kết quả và kết luận kỹ thuật

Bảng số liệu và kết luận tính trực tiếp từ lần chạy đã kiểm chứng nằm ở
[results/stardist_pretrained/KET_QUA.md](results/stardist_pretrained/KET_QUA.md).
Lần chạy ngày 12/11/2026 trên CPU macOS ARM64 thu được kết quả test tại IoU = 0.5:

| Phương pháp | TP | FP | FN | Precision | Recall | F1 | Matching accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| StarDist pretrained | 755 | 83 | 229 | 0.9010 | 0.7673 | 0.8288 | 0.7076 |
| Otsu | 420 | 57 | 564 | 0.8805 | 0.4268 | 0.5749 | 0.4035 |
| Otsu + Watershed | 470 | 98 | 514 | 0.8275 | 0.4776 | 0.6057 | 0.4344 |

Ngưỡng StarDist được chốt là `prob_thresh=0.47576062024687477`, `nms_thresh=0.3`.
StarDist có F1 cao nhất trong ba cấu hình đã chọn, hơn Watershed 0.2231 và có ít
đối tượng bị bỏ sót hơn (FN 229 so với 514). Đổi lại, thời gian trung bình mỗi ảnh
của StarDist là 0.0931 giây, so với 0.0049 giây cho Otsu và 0.0155 giây cho Watershed
trong lần đo này. Thời gian không được dùng làm thuộc tính cố định của thuật toán.

Khi nâng IoU lên 0.9, matching accuracy của StarDist giảm còn 0.1877; lợi thế so với
Watershed (0.1766) hẹp hơn rõ rệt. Điều này cho thấy lợi ích ở mức ghép đối tượng
IoU 0.5 chưa chuyển thành lợi thế tương đương khi yêu cầu vùng dự đoán khớp rất sát
ground truth. Hình mật độ cao cho thấy hai baseline còn gộp nhiều nhân, trong khi
StarDist tách được nhiều đối tượng hơn nhưng vẫn có nhân bị bỏ sót.
Tại IoU = 0.95, matching accuracy của StarDist là 0.0532, thấp hơn Otsu (0.0695)
và Watershed (0.0630); kết quả không cho thấy StarDist vượt baseline ở mọi yêu cầu IoU.

Các mask, cấu hình và checksum đi kèm cho phép kiểm tra lại kết quả ở cấp ảnh.
Hình so sánh chọn ba ảnh có số nhân thấp nhất, trung vị và cao nhất; không chọn theo
chất lượng dự đoán. Với số ảnh nhỏ, ID trùng trong lựa chọn minh họa được bỏ lặp.

Đánh giá lợi ích của StarDist dựa trên F1/precision/recall, TP/FP/FN và mức giảm điểm
khi tăng ngưỡng IoU. F1 cao ở IoU 0.5 nhưng giảm tại IoU cao cho thấy nhận diện đối tượng
tốt hơn độ khớp biên chính xác; cần xem hình và matching để xác định loại lỗi cụ thể.
Không suy diễn chất lượng biên chỉ từ số lượng đối tượng dự đoán.

Giới hạn chính: dữ liệu test chưa được đối chiếu với danh sách huấn luyện pretrained.
StarDist xác nhận `2D_versatile_fluo` và `2D_paper_dsb2018` đều học một phần DSB2018 [1].
Vì vậy kết quả thể hiện so sánh trong tập ảnh đã chọn, chưa chứng minh tổng quát hóa
độc lập. Chia dữ liệu chưa kiểm soát nguồn thu nhận/nhóm bệnh nhân và ảnh gần trùng;
20 ảnh test cũng chưa đủ đại diện toàn bộ các miền ảnh của bộ dữ liệu.

Hướng mở rộng gồm xác nhận miền ảnh, dùng dữ liệu kiểm thử độc lập với pretrained,
đánh giá thêm lỗi tách/gộp trên ảnh dày đặc và mở rộng tham số trên validation trước
khi chốt thí nghiệm mới. Không chọn tham số dựa trên điểm test hiện tại.

## Tài liệu tham khảo

[1] StarDist — pretrained models và cách sử dụng: https://github.com/stardist/stardist#pretrained-models-for-2d.

[2] Schmidt et al., *Cell Detection with Star-convex Polygons*, MICCAI 2018.
https://arxiv.org/abs/1806.03535.

[3] Mã nguồn matching và định nghĩa các chỉ số:
https://github.com/stardist/stardist/blob/0.9.2/stardist/matching.py.

[4] Chọn ngưỡng và hàm mục tiêu:
https://github.com/stardist/stardist/blob/0.9.2/stardist/models/base.py và
https://github.com/stardist/stardist/blob/0.9.2/stardist/utils.py.
