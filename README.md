# Đề tài: Hệ thống AI giám sát tòa nhà thông minh dựa trên dữ liệu camera

## 1. Bối cảnh và lý do chọn đề tài

Trong các tòa nhà hiện đại (chung cư, văn phòng, trung tâm thương mại), hệ thống camera giám sát (CCTV) đã trở nên rất phổ biến. Tuy nhiên, phần lớn hệ thống hiện tại chỉ dừng lại ở mức **ghi hình và xem lại thủ công**, phụ thuộc nhiều vào nhân viên trực phòng camera. Điều này dẫn tới:

* Khó phát hiện kịp thời các **tình huống khẩn cấp**: té ngã, gây gổ, trộm cắp, cháy nổ…
* Khó giám sát hiệu quả các khu vực **ít người qua lại** hoặc camera nhiều, nhân sự theo dõi ít.
* Chưa tận dụng được sức mạnh của **AI và học sâu** để tự động hóa giám sát và cảnh báo.

Vì vậy, đề tài hướng tới xây dựng một **hệ thống AI giám sát tòa nhà thông minh**, có khả năng **phân tích video camera theo thời gian thực**, hỗ trợ các chức năng:

* Đếm số người trong khu vực.
* Phát hiện người lạ.
* Phát hiện hành vi bất thường, lảng vảng (loitering).
* Phát hiện té ngã.

Từ đó giúp nâng cao **an ninh – an toàn – hiệu quả vận hành** của tòa nhà.

---

## 2. Mục tiêu nghiên cứu

### 2.1. Mục tiêu tổng quát

Xây dựng và đánh giá một hệ thống AI giám sát tòa nhà dựa trên dữ liệu camera, tích hợp nhiều chức năng phân tích hành vi và trạng thái con người (đếm người, phát hiện bất thường, phát hiện té ngã, phát hiện người lạ), có khả năng hoạt động gần thời gian thực.

### 2.2. Mục tiêu cụ thể

1. **Xây dựng module phát hiện và bám đuổi người (person detection & tracking)** làm nền tảng chung cho toàn hệ thống.
2. **Xây dựng module đếm người** trong một khu vực (sảnh, hành lang…) dựa trên kết quả detection/tracking.
3. **Xây dựng module phát hiện bất thường (anomaly detection)** trong video camera:

   * Ví dụ: chạy bất thường, tụ tập, lảng vảng lâu trong khu vực, hành vi bất thường theo ngữ cảnh.
4. **Xây dựng module phát hiện té ngã (fall detection)** dựa trên chuỗi hành vi/pose của từng người.
5. **Xây dựng module phát hiện người lạ (stranger detection)** dựa trên mô hình Person Re-Identification (ReID) và tập người đã đăng ký (whitelist).
6. Đề xuất **kiến trúc tổng thể hệ thống** (pipeline từ camera → AI → giao diện giám sát/cảnh báo) và **đánh giá** hệ thống trên các bộ dữ liệu chuẩn.

---

## 3. Phạm vi và giới hạn đề tài

1. Hệ thống tập trung vào **giám sát trong phạm vi một tòa nhà** (hoặc mô phỏng tòa nhà) với camera cố định, không xử lý môi trường đường phố phức tạp như giao thông đô thị.
2. Dữ liệu dùng trong nghiên cứu chủ yếu là **các bộ dữ liệu công khai** (public datasets) và **một tập dữ liệu nhỏ tự xây dựng** (nếu có) cho phần người lạ.
3. Đề tài tập trung vào **phân tích video từ camera cố định**; không xem xét sâu các vấn đề:

   * Hạ tầng mạng, lưu trữ lớn, triển khai trên nhiều server.
   * Quy định pháp lý chi tiết, bảo mật hệ thống ở mức triển khai thương mại.
4. Hệ thống được xây dựng ở mức **mô hình thử nghiệm (prototype)**: chạy trên một máy tính/GPU, có thể xử lý gần thời gian thực tùy cấu hình.
5. Phần nhận dạng danh tính **không sử dụng Face Recognition trực tiếp** mà dựa trên Person ReID + khái niệm “người quen/người lạ” trong phạm vi kỹ thuật, tránh vấn đề nhạy cảm về nhận diện khuôn mặt.

---

## 4. Dữ liệu nghiên cứu

Dữ liệu được chia theo từng chức năng của hệ thống.

### 4.1. Dữ liệu cho module đếm người

* **Mall Dataset**:

  * Video từ camera giám sát trong trung tâm thương mại (indoor), camera cố định trên cao.
  * Có anot số lượng người trong mỗi frame → phục vụ bài toán **crowd counting / people counting**.
  * Phù hợp với bối cảnh sảnh tòa nhà, hành lang đông người.

(Có thể tham khảo thêm UCSD Pedestrian Dataset hoặc ShanghaiTech Crowd cho các thí nghiệm mở rộng nếu cần.)

### 4.2. Dữ liệu cho module bất thường / lảng vảng

* **IITB-Corridor Dataset**:

  * Cảnh: hành lang (corridor) trong khuôn viên trường, giống hành lang tòa nhà.
  * Có nhiều loại hành vi bình thường và bất thường: đi lại, chạy, tụ tập, lảng vảng, bỏ lại đồ, v.v.
  * Rất phù hợp cho bài toán **video anomaly detection** và **loitering detection**.

* (Tùy phạm vi có thể bổ sung thêm **ShanghaiTech Campus** hoặc **NWPU-Campus** để tăng độ đa dạng bối cảnh.)

### 4.3. Dữ liệu cho module phát hiện té ngã

* **Le2i Fall Detection Dataset**:

  * Video té ngã trong nhiều bối cảnh indoor (phòng khách, văn phòng, phòng họp…).
  * Gồm cả video té ngã và hoạt động thường ngày (ADL) để phân biệt.

* **UR Fall Detection Dataset (URFD)**:

  * Các đoạn video té ngã và không té ngã quay bằng camera cố định.
  * Dùng để huấn luyện và đánh giá mô hình phân loại “fall” vs “no fall”.

* Có thể bổ sung các bộ như **CAUCAFall, Multicam Fall, GMDCSA-24** nếu cần tăng độ phức tạp và tính thực tế của môi trường.

### 4.4. Dữ liệu cho module phát hiện người lạ (ReID)

* **Market-1501** (và/hoặc một số dataset ReID khác):

  * Dùng để tiền huấn luyện (pre-train) backbone ReID, học đặc trưng người qua nhiều camera.

* **Tập dữ liệu nội bộ (mô phỏng)**:

  * Một tập hình/video nhỏ gồm các cư dân/nhân viên “đã đăng ký” trong tòa nhà (whitelist).
  * Dùng để xây dựng **gallery người quen** và đánh giá khả năng phân biệt “người quen” vs “người lạ” trong điều kiện mô phỏng.

---

## 5. Phương pháp nghiên cứu

### 5.1. Kiến trúc tổng thể hệ thống

Hệ thống được thiết kế theo pipeline:

1. **Camera cố định** trong tòa nhà (hành lang, sảnh, bãi gửi xe, khu vực quan trọng…).
2. **Module phát hiện & bám đuổi người (Person Detection & Tracking)**.
3. Từ kết quả tracking, tách ra các luồng thông tin để đưa vào:

   * Module **đếm người**.
   * Module **phát hiện bất thường / lảng vảng**.
   * Module **phát hiện té ngã**.
   * Module **phát hiện người lạ (ReID)**.
4. **Module tổng hợp & cảnh báo**:

   * Hiển thị thông tin lên giao diện (dashboard).
   * Gửi sự kiện cảnh báo (ví dụ: té ngã, loitering lâu, người lạ vào khu vực cấm…).

### 5.2. Module phát hiện & bám đuổi người

* Sử dụng mô hình **YOLOv8/YOLOv10** (hoặc tương đương) cho **person detection**.
* Kết hợp với thuật toán **tracking** (ví dụ: BYTETrack, OC-SORT hoặc DeepSORT) để gán ID cho từng người và theo dõi quỹ đạo di chuyển theo thời gian.
* Đánh giá bằng các chỉ số:

  * Phát hiện: mAP, Precision, Recall.
  * Tracking: IDF1, MOTA/MOTP (nếu có điều kiện).

### 5.3. Module đếm người

* Dựa trên kết quả detection/tracking:

  * Đếm số lượng người xuất hiện trong mỗi frame hoặc trong từng vùng quan tâm (ROI).
  * Có thể sử dụng logic: **count theo track ID** để tránh đếm trùng.
* So sánh/đánh giá kết quả trên Mall Dataset bằng các chỉ số:

  * MAE, MSE giữa số người dự đoán và ground truth.

### 5.4. Module phát hiện bất thường / lảng vảng

Có hai hướng tiếp cận chính (có thể chọn một hoặc kết hợp):

1. **Học theo video:**

   * Sử dụng mô hình 3D CNN, ConvLSTM hoặc Transformer để học biểu diễn động theo thời gian trên các đoạn video từ IITB-Corridor, ShanghaiTech Campus, NWPU-Campus…
   * Huấn luyện theo hướng:

     * Học trên dữ liệu “bình thường” (unsupervised/semi-supervised), bất thường được phát hiện qua **độ lệch/reconstruction error**.
     * Hoặc supervised nếu dataset có nhãn chi tiết.

2. **Học theo quỹ đạo (trajectory-based):**

   * Dùng thông tin vị trí/ID người theo thời gian từ tracker để phân tích:

     * Người lảng vảng ở một vùng quá lâu.
     * Người di chuyển bất thường (chạy, quay lại nhiều lần, tụ tập…).
   * Xây dựng các rule hoặc mô hình học máy trên đặc trưng quỹ đạo.

* Đánh giá bằng các chỉ số như AUC, EER, F1-score trên bộ dữ liệu anomaly.

### 5.5. Module phát hiện té ngã

* Từ kết quả tracking, cắt các **clip video ngắn** (ví dụ 1–2 giây) quanh thời điểm nghi ngờ té ngã của một người.
* Hai hướng tiếp cận:

1. **Video-based (pixel-level):**

   * Dùng 3D CNN, CNN+LSTM, hoặc Attention-based network để phân loại clip là “fall” hay “no fall”.
   * Train/fine-tune trên Le2i, URFD, CAUCAFall…

2. **Pose-based (skeleton):**

   * Sử dụng mô hình pose estimation (ví dụ: YOLOv8-pose hoặc MediaPipe) để trích xuất keypoints skeleton.
   * Đưa chuỗi skeleton qua LSTM/Transformer để phân loại té ngã.

* Đánh giá bằng các chỉ số: Accuracy, Precision, Recall, F1-score trên các bộ dữ liệu fall detection.

### 5.6. Module phát hiện người lạ (Person Re-Identification)

* **Bước 1 – Pre-train backbone ReID:**

  * Sử dụng các bộ dữ liệu ReID như Market-1501 để huấn luyện mô hình trích xuất đặc trưng người qua nhiều camera.

* **Bước 2 – Xây dựng gallery người quen:**

  * Thu và lưu trữ embedding của các cư dân/nhân viên “đã đăng ký” trong tòa nhà.

* **Bước 3 – Nhận diện người lạ:**

  * Với mỗi track người mới, trích embedding và so sánh với gallery.
  * Nếu khoảng cách tới tất cả vector trong gallery đều lớn hơn ngưỡng cho trước → gán nhãn **stranger**.

* Đánh giá:

  * Trên dataset ReID: Rank-1 accuracy, mAP.
  * Trên tập dữ liệu mô phỏng tòa nhà: tỉ lệ phát hiện đúng người quen/người lạ.

---

## 6. Kết quả mong đợi

1. Xây dựng được **một prototype hệ thống AI giám sát tòa nhà** chạy được trên dữ liệu video (offline hoặc gần thời gian thực), tích hợp:

   * Đếm người.
   * Phát hiện bất thường / lảng vảng.
   * Phát hiện té ngã.
   * Phát hiện người lạ.

2. Có **bộ kết quả thực nghiệm** trên các dataset chuẩn (Mall, IITB-Corridor, Le2i, URFD, Market-1501…), so sánh được hiệu quả của từng module qua các chỉ số đánh giá phù hợp.

3. Đề xuất được một **khung kiến trúc tổng quát** có thể áp dụng và mở rộng cho các bài toán giám sát tòa nhà thông minh trong thực tế (mở rộng thêm phát hiện cháy, xâm nhập khu vực cấm, thống kê mật độ người theo thời gian…).

4. Làm cơ sở để phát triển tiếp thành một **hệ thống demo** hoặc sản phẩm ứng dụng trong các tòa nhà, chung cư, khuôn viên trường học, bệnh viện, nhà máy… trong tương lai.
