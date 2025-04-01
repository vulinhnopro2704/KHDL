# 📱 Phân Tích Ảnh Hưởng của Phần Cứng và Thương Hiệu đến Giá Điện Thoại Di Động

-   Nhóm thực hiện: **Trương Vũ Linh (Crawl data from Web), Hoàng Văn Đạt(Data Cleaning), Đỗ Văn Tuấn (Data Analysis)**

## 📊 1. Tổng Quan

Phân tích mức độ ảnh hưởng của các thông số phần cứng và thương hiệu đến giá bán của điện thoại di động. Dự án này tập trung vào thu thập dữ liệu, xem sét sự ảnh hưởng của phần cứng và thương hiệu đến mức giá của điện thoại di động, tiền xử lý và phân tích để chuẩn bị cho việc xây dựng mô hình dự đoán giá.

## 📚 2. Nguồn Dữ Liệu

Dữ liệu được thu thập từ hai trang thương mại điện tử uy tín:

-   🛒 [MobileCity](https://mobilecity.vn/dien-thoai)
-   🛍️ [CellphoneS](https://cellphones.com.vn/mobile.html)

## 🎯 3. Mục Tiêu Phân Tích

Khảo sát tính khả thi cho việc xây dựng mô hình dự đoán biến mục tiêu "Giá - Price" dựa trên các đặc trưng về phần cứng và thương hiệu. Với biến mục tiêu là biến liên tục, bài toán được định hướng theo mô hình hồi quy (regression).

## 🔍 4. Phương Pháp Tiếp Cận

### 4.1. Quy Trình Thực Hiện

1. **🔎 Khai thác dữ liệu** (Data Exploration)
2. **🧹 Tiền xử lý dữ liệu** (Data Preprocessing - Data Cleaning)
3. **📊 Phân tích dữ liệu** (Data Analysis)
4. **📈 Xây dựng mô hình hồi quy** (Regression Model Building) - _sẽ được thực hiện ở bài tập sau_

### 4.2. Thứ Tự Thực Thi File

1. `craw_data_cellphones.ipynb` - Thu thập dữ liệu từ CellphoneS
2. `craw_data_mobilecity.ipynb` - Thu thập dữ liệu từ MobileCity
3. `cleaning_data.ipynb` - Tiền xử lý dữ liệu
4. `data_analysis.ipynb` - Phân tích dữ liệu

## 📁 5. Cấu Trúc Dữ Liệu

-   Dữ liệu chi tiết của từng sản phẩm được lưu trong thư mục `data/json`
-   Dữ liệu thô đã thu thập được lưu trong thư mục `data/raw`
-   Dữ liệu tổng hợp xử lí, được làm sạch và chọn lọc cuối cùng từ dữ liệu thô được lưu trong file `data/clean/feature_engineering.csv`. Đây chính là file dữ liệu chính nhóm dùng để phân tích.

## 📂 6. Cấu Trúc Thư Mục

```
phone-price-analysis/
│
├── 📓 notebooks/
│   ├── craw_data_cellphones.ipynb      # Thu thập dữ liệu từ CellphoneS
│   ├── craw_data_mobilecity.ipynb      # Thu thập dữ liệu từ MobileCity
│   ├── cleaning_data.ipynb             # Tiền xử lý - làm sách dữ liệu dữ liệu
│   └── data_analysis.ipynb        # Phân tích dữ liệu và trực quan hóa
│
├── link/
|   ├── product_links_cellphones.txt             # Link sản phẩm từ CellphoneS
|   └── product_links_mobilecity.txt             # Link sản phẩm từ MobileCity
├── data/                               # Thư mục chứa dữ liệu
│   ├── raw/                     # Dữ liệu thô đã thu thập được
│   ├── json/                    # Dữ liệu chi tiết từng sản phẩm đã thu thập
│   ├── processed/               # Dữ liệu sau xử lý
│   └── clean/                   # Dữ liệu đã tổng hợp và làm sạch (`feature_engineering.csv` là file chính)
│
│
└── 📝 README.md                 # Tài liệu dự án
```

## 🚀 7. Hướng Dẫn Thực Thi

1. **Thu thập dữ liệu**

-   Xem và chạy các notebook sau để thu thập dữ liệu từ CellphoneS và MobileCity:
    ```bash
    notebooks/craw_data_cellphones.ipynb
    notebooks/craw_data_mobilecity.ipynb
    ```

2. **Tiền xử lý và phân tích**

-   Sau khi thu thập dữ liệu, bạn có thể chạy các notebook sau để tiền xử lý và phân tích dữ liệu:
    ```bash
    notebooks/cleaning_data.ipynb
    notebooks/data_analysis.ipynb
    ```

## 🔧 8. Công Nghệ Sử Dụng

-   **Python**: Ngôn ngữ lập trình chính
-   **Pandas & NumPy**: Xử lý và phân tích dữ liệu
-   **Matplotlib & Seaborn**: Trực quan hóa dữ liệu
-   **Selenium**: Thu thập dữ liệu web

## 📝 9. Tài Liệu Tham Khảo

-   [Pandas Documentation](https://pandas.pydata.org/docs/)
-   [NumPy Documentation](https://numpy.org/doc/stable/)
-   [Seaborn Documentation](https://seaborn.pydata.org/)
-   [Selenium Documentation](https://www.selenium.dev/documentation/en/)
