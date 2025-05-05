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
4. **🔄 Mã hóa dữ liệu** (Data Encoding)
5. **🔪 Phân chia tập dữ liệu** (Dataset Splitting)
6. **📈 Xây dựng mô hình hồi quy** (Regression Model Building) - _sẽ được thực hiện ở bài tập sau_

### 4.2. Thứ Tự Thực Thi File

1. `notebooks/craw_data_cellphones.ipynb` - Thu thập dữ liệu từ CellphoneS
2. `notebooks/craw_data_mobilecity.ipynb` - Thu thập dữ liệu từ MobileCity
3. `notebooks/cleaning_data.ipynb` - Tiền xử lý dữ liệu
4. `notebooks/data_analysis.ipynb` - Phân tích dữ liệu
5. `code/target_encoding.py` hoặc `code/advanced_target_encoding.py` - Mã hóa biến phân loại
6. `code/improved_visualization.py` - Trực quan hóa kết quả mã hóa
7. `code/split_dataset.py` - Phân chia dữ liệu theo nhãn
8. `code/prepare_ml_datasets.py` - Chuẩn bị dữ liệu cho mô hình học máy

## 📁 5. Cấu Trúc Dữ Liệu

-   Dữ liệu chi tiết của từng sản phẩm được lưu trong thư mục `data/json`
-   Dữ liệu thô đã thu thập được lưu trong thư mục `data/raw`
-   Dữ liệu tổng hợp xử lí, được làm sạch và chọn lọc cuối cùng từ dữ liệu thô được lưu trong file `data/clean/feature_engineering.csv`. Đây chính là file dữ liệu chính nhóm dùng để phân tích.
-   Dữ liệu đã mã hóa được lưu trong `data/clean/feature_engineering_encoding.csv`
-   Dữ liệu phân chia theo nhãn được lưu trong các file `data/clean/category_label_dataset.csv` và `data/clean/price_label_dataset.csv`
-   Dữ liệu đã chuẩn bị cho mô hình học máy được lưu trong thư mục `data/ml_ready/`

## 📂 6. Cấu Trúc Thư Mục

```
phone-price-analysis/
│
├── 📓 notebooks/
│   ├── craw_data_cellphones.ipynb      # Thu thập dữ liệu từ CellphoneS
│   ├── craw_data_mobilecity.ipynb      # Thu thập dữ liệu từ MobileCity
│   ├── cleaning_data.ipynb             # Tiền xử lý - làm sạch dữ liệu
│   └── data_analysis.ipynb             # Phân tích dữ liệu và trực quan hóa
│
├── 📜 code/                            # Thư mục chứa các script Python
│   ├── target_encoding.py              # Mã hóa đơn giản các biến phân loại
│   ├── advanced_target_encoding.py     # Mã hóa nâng cao cho các biến phân loại
│   ├── improved_visualization.py       # Trực quan hóa nâng cao cho dữ liệu đã mã hóa
│   ├── split_dataset.py                # Phân chia dữ liệu thành hai tập với nhãn khác nhau
│   └── prepare_ml_datasets.py          # Chuẩn bị tập dữ liệu để huấn luyện mô hình học máy
│
├── 🔗 link/
|   ├── product_links_cellphones.txt    # Link sản phẩm từ CellphoneS
|   └── product_links_mobilecity.txt    # Link sản phẩm từ MobileCity
│
├── 📊 data/                            # Thư mục chứa dữ liệu
│   ├── raw/                            # Dữ liệu thô đã thu thập được
│   ├── json/                           # Dữ liệu chi tiết từng sản phẩm đã thu thập
│   ├── processed/                      # Dữ liệu sau xử lý
│   ├── clean/                          # Dữ liệu đã tổng hợp và làm sạch
│   │   ├── feature_engineering.csv     # Dữ liệu gốc đã làm sạch
│   │   ├── feature_engineering_encoding.csv  # Dữ liệu đã mã hóa biến phân loại
│   │   ├── category_label_dataset.csv  # Dữ liệu với nhãn price_category
│   │   └── price_label_dataset.csv     # Dữ liệu với nhãn price
│   │
│   ├── ml_ready/                       # Dữ liệu đã chia tập train/test cho học máy
│   │   ├── X_train_category.csv        # Dữ liệu huấn luyện với đặc trưng cho mô hình phân loại
│   │   ├── X_test_category.csv         # Dữ liệu kiểm tra với đặc trưng cho mô hình phân loại
│   │   ├── y_train_category.csv        # Nhãn huấn luyện cho mô hình phân loại
│   │   ├── y_test_category.csv         # Nhãn kiểm tra cho mô hình phân loại
│   │   ├── X_train_price.csv           # Dữ liệu huấn luyện với đặc trưng cho mô hình hồi quy giá
│   │   ├── X_test_price.csv            # Dữ liệu kiểm tra với đặc trưng cho mô hình hồi quy giá
│   │   ├── y_train_price.csv           # Nhãn huấn luyện cho mô hình hồi quy giá
│   │   └── y_test_price.csv            # Nhãn kiểm tra cho mô hình hồi quy giá
│   │
│   └── datasets/                       # Dữ liệu bổ sung
│
├── 📈 visualizations/                  # Thư mục chứa hình ảnh trực quan hóa
│   ├── brand_encoding_impact.png       # Biểu đồ ảnh hưởng của mã hóa thương hiệu
│   ├── chip_model_encoding_impact.png  # Biểu đồ ảnh hưởng của mã hóa chip
│   ├── screen_resolution_k_encoding_impact.png  # Biểu đồ ảnh hưởng của mã hóa độ phân giải
│   ├── screen_tech_encoding_impact.png # Biểu đồ ảnh hưởng của mã hóa công nghệ màn hình
│   └── improved/                       # Trực quan hóa nâng cao
│       ├── brand_bubble_plot.png       # Biểu đồ bong bóng cho thương hiệu
│       ├── chip_model_bubble_plot.png  # Biểu đồ bong bóng cho model chip
│       ├── encoding_correlation_heatmap.png  # Biểu đồ nhiệt tương quan
│       └── ...                         # Các biểu đồ khác
│
├── 🧠 models/                          # Thư mục sẽ chứa các mô hình học máy (trong bài sau)
│
└── 📝 README.md                        # Tài liệu dự án
```

## 🚀 7. Hướng Dẫn Thực Thi

### 7.1. Thu thập và phân tích dữ liệu

-   Xem và chạy các notebook sau để thu thập dữ liệu từ CellphoneS và MobileCity:
    ```bash
    notebooks/craw_data_cellphones.ipynb
    notebooks/craw_data_mobilecity.ipynb
    notebooks/cleaning_data.ipynb
    notebooks/data_analysis.ipynb
    ```

### 7.2. Mã hóa và chuẩn bị dữ liệu

#### Mã hóa biến phân loại (Target Encoding)

Mã hóa các biến phân loại như chip_model, brand, screen_tech và screen_resolution_k để giữ mối quan hệ với giá:

```bash
# Chạy phiên bản cơ bản
python code/target_encoding.py

# Hoặc chạy phiên bản nâng cao với nhiều kỹ thuật mã hóa hơn
python code/advanced_target_encoding.py
```

#### Trực quan hóa nâng cao cho dữ liệu đã mã hóa

Tạo các biểu đồ trực quan nâng cao với nhiều phương pháp hiển thị khác nhau:

```bash
python code/improved_visualization.py
```

#### Tách dữ liệu thành hai phiên bản với nhãn khác nhau

Tạo hai phiên bản của tập dữ liệu: một với nhãn price_category (phân loại) và một với nhãn price (hồi quy):

```bash
python code/split_dataset.py
```

#### Chuẩn bị dữ liệu cho mô hình học máy

Chuẩn bị dữ liệu bằng cách chia thành tập huấn luyện và kiểm tra:

```bash
python code/prepare_ml_datasets.py
```

## 🔧 8. Công Nghệ Sử Dụng

-   **Python**: Ngôn ngữ lập trình chính
-   **Pandas & NumPy**: Xử lý và phân tích dữ liệu
-   **Matplotlib & Seaborn**: Trực quan hóa dữ liệu
-   **Selenium**: Thu thập dữ liệu web
-   **Scikit-learn**: Mã hóa dữ liệu và phân chia tập dữ liệu

## 🧩 9. Thông Tin Chi Tiết Về Các Script Mới

### 9.1. Target Encoding (Mã hóa Biến Mục Tiêu)

Mã hóa các biến phân loại bằng cách thay thế giá trị bằng giá trị trung bình của biến mục tiêu (price) cho mỗi danh mục.

#### `target_encoding.py`

-   **Mục đích**: Thực hiện mã hóa cơ bản cho các biến phân loại
-   **Biến đầu vào**: `chip_model`, `brand`, `screen_tech`, `screen_resolution_k`
-   **Biến đầu ra**: Các biến đã mã hóa (`*_encoded`)
-   **Phương pháp**: K-fold target encoding với cơ chế làm mịn (smoothing)

#### `advanced_target_encoding.py`

-   **Mục đích**: Mã hóa nâng cao với xử lý dữ liệu huấn luyện/kiểm tra riêng biệt
-   **Tính năng bổ sung**:
    -   Điều chỉnh tham số alpha cho từng biến dựa vào số lượng giá trị phân biệt
    -   Tạo biểu đồ trực quan hóa tác động của mã hóa
    -   Phân tích tương quan giữa các biến đã mã hóa và giá

### 9.2. Trực Quan Hóa Nâng Cao

#### `improved_visualization.py`

-   **Mục đích**: Cung cấp nhiều phương pháp trực quan hóa khác nhau cho dữ liệu đã mã hóa
-   **Các phương pháp trực quan hóa**:
    -   Biểu đồ bong bóng (Bubble plot) với nhóm "Others" cho các giá trị ít phổ biến
    -   Nhiều biểu đồ nhỏ (Multiple small plots) hiển thị tất cả các giá trị
    -   Biểu đồ phân cụm phân cấp (Hierarchical clustering heatmap)
    -   Biểu đồ phân bố giá trị (Value distribution)
    -   Ma trận tương quan (Correlation heatmap)

### 9.3. Phân Chia Dữ Liệu

#### `split_dataset.py`

-   **Mục đích**: Tạo hai phiên bản của tập dữ liệu với nhãn khác nhau
-   **Đầu ra**:
    -   `category_label_dataset.csv`: Sử dụng `price_category` làm nhãn (cho bài toán phân loại)
    -   `price_label_dataset.csv`: Sử dụng `price` làm nhãn (cho bài toán hồi quy)

#### `prepare_ml_datasets.py`

-   **Mục đích**: Chuẩn bị dữ liệu cho việc huấn luyện mô hình học máy
-   **Tính năng**:
    -   Phân chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)
    -   Xử lý các biến phân loại
    -   Lưu các tập X_train, X_test, y_train, y_test riêng biệt

## 📝 10. Tài Liệu Tham Khảo

-   [Pandas Documentation](https://pandas.pydata.org/docs/)
-   [NumPy Documentation](https://numpy.org/doc/stable/)
-   [Seaborn Documentation](https://seaborn.pydata.org/)
-   [Selenium Documentation](https://www.selenium.dev/documentation/en/)
-   [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
