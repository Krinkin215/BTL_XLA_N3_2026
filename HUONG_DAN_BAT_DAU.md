# Hướng dẫn sử dụng dự án StarDist

Các sản phẩm chính nằm trong **StarDist_Dataset/**. Dữ liệu gốc trong Downloads được giữ nguyên. Lần chạy hoàn chỉnh đọc từ `C:\Users\ADMIN\Downloads\stage1_train.zip` để giảm thời gian mở hàng chục nghìn tệp nhỏ.

## 1. Tổng quan quy trình

Mở `StarDist_Dataset/README.md` để xem thống kê và cách sử dụng dữ liệu. Quy trình gồm:

1. Sắp xếp ảnh vào `processed_data/train`, `val`, `test`, giữ nguyên tên image ID để truy xuất về nguồn.
2. Gộp các mask nhị phân của mỗi ảnh thành một TIFF **uint16**: nền 0; mỗi nhân một số 1, 2, 3… Nhờ vậy hai nhân sát nhau vẫn là hai đối tượng khác nhau.
3. Chia dữ liệu với seed 42, phân tầng theo diện mạo và số nhân; ảnh trùng pixel được giữ cùng tập. Train dành cho huấn luyện nếu cần, val để chỉnh tham số, test để đánh giá cuối cùng.
4. Kiểm tra dữ liệu, lưu ảnh minh họa. Giữ ảnh gốc; xuất thêm ảnh chuẩn hóa đơn kênh cho nhóm xám nền tối. Nhãn không bị chuẩn hóa hay làm mượt.
5. Tạo notebook Python/Colab, manifest, danh sách ảnh dùng chung, thống kê, phần viết báo cáo và ZIP bàn giao.

## 2. Kiểm tra trực quan ảnh và nhãn

Mở `StarDist_Dataset/results/quality_control.png`. Mỗi hàng gồm ảnh gốc, nhãn tô màu cho từng nhân, và đường biên nhãn đặt lên ảnh. Màu chỉ phục vụ hiển thị; dữ liệu TIFF thực tế là số nguyên.

Mở `normalization_example.png` trong cùng thư mục để so sánh trước/sau chuẩn hóa. Không dùng ảnh minh họa PNG làm nhãn cho thuật toán.

## 3. Dùng notebook trên Colab

1. Đưa `StarDist_Dataset_BanGiao.zip` lên Google Drive hoặc vùng Files của Colab; giải nén để có thư mục `StarDist_Dataset`.
2. Chọn **File → Upload notebook**, mở `StarDist_Dataset/Chuan_bi_du_lieu.ipynb`.
3. Sửa dòng cấu hình thành `PROJECT = Path('/content/StarDist_Dataset')` nếu giải nén vào `/content`.
4. Chạy lần lượt các ô. Giữ `RUN_PREPARATION = False` để dùng dữ liệu đã chuẩn bị. Phần này không cần GPU.
5. Chỉ bật tái tạo khi muốn chạy lại từ dữ liệu gốc; dùng thư mục đầu ra mới. `--raw` nhận thư mục đã giải nén hoặc `stage1_train.zip`.

Ví dụ giải nén ZIP đã tải lên `/content` trong một ô Colab:

```python
from zipfile import ZipFile
with ZipFile('/content/StarDist_Dataset_BanGiao.zip') as z:
    z.extractall('/content')
```

## 4. Sử dụng dữ liệu và tài liệu

- **Thực nghiệm pretrained:** dùng `experiments/pretrained_fluo/manifest.csv`: 20 ảnh validation để chọn cấu hình và 20 ảnh test để so sánh cuối cùng; đọc `HUONG_DAN_PRETRAINED.md`. Giữ nguyên split, phần train lưu dự phòng. Các đường dẫn trong manifest tính từ thư mục `StarDist_Dataset`, không phải từ thư mục chứa manifest con.
- **Báo cáo và hình minh họa:** `BAO_CAO_DU_LIEU.md`, `dataset_summary.csv`, các hình trong `results/`.
- **Kiểm tra chất lượng:** `quality_report.json` ghi kết quả kiểm tra nhãn, ảnh, chuẩn hóa và phân chia dữ liệu.

## 5. Cấu hình mô hình và giới hạn đánh giá

Thí nghiệm sử dụng pretrained, không huấn luyện hoặc fine-tune. Mô hình đề xuất là `2D_versatile_fluo`. Nhóm ảnh xám nền tối được phân loại theo pixel, chưa xác nhận huỳnh quang; cần kiểm tra trực quan để xác nhận miền phù hợp. Ảnh màu chưa được tự động gán là H&E. Các nhóm ảnh được giữ trong bộ dữ liệu đầy đủ.

Không chọn tham số từ test. Theo tài liệu StarDist, cả `2D_versatile_fluo` và `2D_paper_dsb2018` đã học một phần DSB2018; không gọi kết quả trên bộ này là đánh giá tổng quát hóa độc lập khi chưa đối chiếu dữ liệu huấn luyện gốc. Bản chia hiện tại chưa sử dụng metadata thí nghiệm và chưa kiểm tra ảnh gần trùng; các giới hạn đã được ghi trong báo cáo. Nguồn: https://github.com/stardist/stardist#pretrained-models-for-2d.

Mã chuẩn bị dữ liệu nằm ở `prepare_dataset.py`; các kiểm tra bảo toàn nhãn nằm ở `test_dataset_integrity.py`. Gói hiện tại gồm dữ liệu đã xử lý, tài liệu và ví dụ sử dụng pretrained; chưa chứa kết quả chạy StarDist hoặc Otsu/Watershed.
