# Nhận diện biển báo giao thông & phát hiện lệch làn trên Jetson Nano

Hệ thống nhận diện biển báo giao thông và cảnh báo lệch làn **thời gian thực**, xây dựng bằng **YOLOv8** và triển khai trên **NVIDIA Jetson Nano** để suy luận trực tiếp trên thiết bị nhúng.

> Đồ án tốt nghiệp — ngành Công nghệ Kỹ thuật Máy tính, Trường Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE), 06/2025.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8)
![Jetson Nano](https://img.shields.io/badge/NVIDIA-Jetson%20Nano-76B900)

---

## Demo

<!-- TODO: thay bằng GIF thật hoặc link YouTube (unlisted).
     Mẹo: đừng commit file video vào git. Upload clip lên YouTube (unlisted) rồi
     dán ảnh thumbnail có link, hoặc cắt một đoạn ngắn thành GIF < 10 MB. -->

<!-- Ví dụ khi đã có GIF trong docs/:  ![demo](docs/demo.gif)  -->

_Demo GIF / video sẽ được cập nhật._

---

## Điểm nổi bật

- Nhận diện và phân loại **33 lớp biển báo + 1 lớp vạch kẻ đường** (tổng 34 lớp).
- **Suy luận thời gian thực ngay trên thiết bị, ~13.57 FPS** trên Jetson Nano (YOLOv8n) — đủ nhanh để chạy trực tiếp.
- **mAP@0.5 ≈ 0.94** trên bài toán nhận diện biển báo.
- **Cảnh báo lệch làn** theo luật (rule-based), hiển thị trực tiếp trên video ("Departure" / "No Departure").
- Huấn luyện trên **bộ dữ liệu tùy chỉnh gồm 7.803 ảnh**, gắn nhãn và tăng cường dữ liệu bằng Roboflow.
- So sánh các phiên bản YOLOv8 (n / s / m …) theo **mAP, F1-score và FPS**, đối chiếu Jetson Nano với Google Colab.

---

## Tính năng

- **Nhận diện & phân loại biển báo** thuộc bốn nhóm: biển cấm, biển hiệu lệnh, biển nguy hiểm và biển chỉ dẫn.
- **Nhận diện vạch kẻ đường.** Lớp `LANE` được phát hiện dưới dạng bounding box; chỉ những box có cạnh dưới nằm ở phần dưới khung hình mới được coi là làn đang đi.
- **Logic phát hiện lệch làn.** Giả định xe nằm chính giữa khung hình; nếu một box vạch kẻ đường lệch vào trong ngưỡng pixel so với đường tâm, khung hình sẽ bị đánh dấu `Departure` (box chuyển sang màu đỏ).
- **Xuất video đã gán nhãn.** Mỗi khung hình được vẽ box biển báo + độ tin cậy, box làn đường, và trạng thái lệch làn, rồi ghi ra file video.

---

## Kiến trúc hệ thống

```
Ảnh / video đầu vào
        │
        ▼
   Tiền xử lý
        │
        ▼
  Mô hình YOLOv8  ──►  Trích xuất đặc trưng  ──►  Phân lớp
        │
        ▼
  Box phát hiện + độ tin cậy
        │
        ├──► Biển báo (cấm / hiệu lệnh / nguy hiểm / chỉ dẫn)
        │
        └──► Vạch kẻ đường ──► Heuristic lệch làn ──► "Departure" / "No Departure"
        │
        ▼
  Khung hình / video đầu ra đã gán nhãn
```

---

## Bộ dữ liệu

| Thuộc tính            | Giá trị                                             |
|-----------------------|-----------------------------------------------------|
| Tổng số ảnh           | 7.803                                               |
| Số lớp                | 34 (33 biển báo + 1 vạch kẻ đường)                  |
| Gắn nhãn & tăng cường | Roboflow                                            |
| Nhóm biển báo         | Cấm, Hiệu lệnh, Nguy hiểm, Chỉ dẫn                  |

<!-- TODO: Bộ dữ liệu quá lớn để đưa vào git. Hãy đăng lên Roboflow / Kaggle /
     Hugging Face Datasets rồi để link ở đây, ví dụ:
     Dataset: https://universe.roboflow.com/your-workspace/your-dataset -->

---

## Huấn luyện

Huấn luyện bằng framework Ultralytics YOLOv8 trên Google Colab (bộ dữ liệu tùy chỉnh).

| Siêu tham số     | Giá trị     |
|------------------|-------------|
| Optimizer        | AdamW       |
| Learning rate    | 0.01        |
| Batch size       | 4           |
| Kích thước ảnh   | 352         |
| Số epoch         | <!-- TODO --> |

---

## Kết quả

**Độ chính xác & hiệu năng thời gian thực**

| Chỉ số                          | Giá trị      |
|---------------------------------|--------------|
| mAP@0.5 (biển báo)              | ≈ 0.94       |
| FPS trung bình trên Jetson Nano (n) | 13.57    |

**So sánh các phiên bản YOLOv8**

<!-- TODO: điền số liệu từ slide báo cáo (slide 15) -->

| Mô hình   | mAP@0.5 | F1-score | FPS (Jetson Nano) | FPS (Colab) |
|-----------|---------|----------|-------------------|-------------|
| YOLOv8n   | 0.94    |          | 13.57             |             |
| YOLOv8s   |         |          |                   |             |
| YOLOv8m   |         |          |                   |             |

Một số nhận xét từ nghiên cứu:
- Jetson Nano và Colab đạt **độ chính xác tương đương**; GPU là yếu tố quyết định FPS.
- Việc tinh chỉnh **learning rate và batch size** ảnh hưởng rõ rệt đến hiệu suất cuối của mô hình.

<!-- TODO: thêm biểu đồ huấn luyện (Loss / mAP) và biểu đồ mAP@0.5 theo từng lớp vào docs/ rồi nhúng vào đây. -->

---

## Cài đặt

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install ultralytics opencv-python
```

Trọng số đã huấn luyện (`best.pt`) được đăng trong mục
[**Releases**](../../releases) <!-- TODO: upload best.pt vào release (hoặc dùng Git LFS) thay vì commit thẳng vào repo -->.
Tải về và đặt vào thư mục gốc dự án trước khi chạy.

---

## Sử dụng

Chạy suy luận trên một video:

```bash
python datn.py --model best.pt --source 3.mp4 --output out.avi
```

Script `datn.py` nhận các tham số dòng lệnh:

| Tham số                  | Mặc định                        | Mô tả                                                        |
|--------------------------|---------------------------------|-------------------------------------------------------------|
| `--model`                | `best.pt`                       | Đường dẫn tới trọng số YOLOv8.                              |
| `--source`               | `3.mp4`                         | File video, hoặc chỉ số camera như `0`.                     |
| `--output`               | `output_lane_detection.avi`     | Đường dẫn video đầu ra đã gán nhãn.                         |
| `--device`               | `cpu`                           | Thiết bị suy luận: `cpu` hoặc `cuda`.                       |
| `--conf`                 | `0.25`                          | Ngưỡng độ tin cậy.                                          |
| `--lane-region`          | `0.55`                          | Chỉ giữ box làn đường có cạnh dưới nằm dưới tỉ lệ này của chiều cao khung hình. |
| `--departure-threshold`  | `125`                           | Khoảng cách pixel so với tâm khung hình để kích hoạt cảnh báo lệch làn. |
| `--no-display`           | –                               | Tắt cửa sổ preview (hữu ích trên thiết bị không màn hình).  |

Trong khi chạy, cửa sổ preview sẽ hiển thị; nhấn **`q`** để thoát.

Ví dụ chạy webcam trên GPU:

```bash
python datn.py --model best.pt --source 0 --device cuda
```

> Lưu ý: mặc định script chạy trên **CPU**. Trên Jetson Nano, mô hình chạy trên
> GPU để đạt con số ~13.57 FPS thời gian thực nêu trên.

---

## Cấu trúc dự án

```
.
├── datn.py                 # Suy luận: nhận diện biển báo + phát hiện lệch làn trên video
├── best.pt                 # Trọng số YOLOv8 đã huấn luyện (qua Releases / Git LFS)
├── docs/                   # GIF demo, biểu đồ kết quả, ảnh chụp màn hình
├── requirements.txt
└── README.md
```

<!-- TODO: chỉnh lại cho khớp repo thực tế. Nếu còn giữ notebook/config huấn luyện,
     hãy thêm vào (ví dụ train.ipynb hoặc data.yaml) — nhà tuyển dụng đánh giá cao
     việc thấy code huấn luyện, không chỉ mỗi file trọng số. -->

---

## Hạn chế

- Đôi lúc **nhận diện nhầm hoặc bỏ sót** biển báo.
- **Nhận diện vạch kẻ đường** kém ổn định hơn so với biển báo, đặc biệt trong điều kiện thiếu sáng.
- Bị giới hạn bởi **năng lực tính toán của Jetson Nano** và thời gian huấn luyện.

## Hướng phát triển

- Cải thiện và **cân bằng lớp** cho bộ dữ liệu.
- Bổ sung **camera hồng ngoại / cảm biến ánh sáng** để cải thiện nhận diện trong điều kiện thiếu sáng.
- Nâng cấp nền tảng phần cứng để tăng thông lượng.
- Xây dựng **giao diện người dùng**.
- Tiến tới **triển khai thực tế**.

---

## Tác giả

- **Lâm Hoàng Phúc** — 21119350
- **Nguyễn Văn Thảo** — 21119129

Thực hiện tại Khoa Điện – Điện tử, Bộ môn Kỹ thuật Máy tính – Viễn thông, HCMUTE.

---

## Lời cảm ơn

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/) cho việc gắn nhãn & tăng cường dữ liệu
- NVIDIA Jetson Nano
