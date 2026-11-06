# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** <06/11/2026>

**Runtime Colab:** GPU (cuda)

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Trả lời: Record mô tả **cả bức ảnh** vì mô hình Classification phân tích nội dung tổng thể và đưa ra các nhãn dự đoán cho cả ảnh. Kết quả không có bounding box để xác định vị trí của từng vật thể riêng lẻ.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Trả lời: Danh sách tên lớp được định nghĩa sẵn trong taxonomy đi kèm với checkpoint, không phải do model tự nghĩ ra. Trong bài này, model sử dụng taxonomy ImageNet-1K.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Trả lời: Cần ghi cả ba thông tin để xác định chính xác lớp dự đoán. `class_id` chỉ có nghĩa trong một taxonomy: trong ImageNet-1K thì `468 = cab`, còn trong COCO-80 thì `2 = car`. Nếu không ghi `taxonomy_name`, cùng một số ID có thể bị hiểu thành hai lớp khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Trả lời: Cần quy định cách chọn nhãn chính dựa trên nội dung hoặc chủ thể nổi bật nhất trong ảnh, để các annotator gán nhãn thống nhất và tránh kết quả khác nhau.
6. Vì sao không được lấy `score` của model làm ground truth?
- Trả lời: Không thể xem `score` là ground truth vì đây chỉ là mức độ tin cậy của model đối với dự đoán, và `rank` chỉ là thứ tự sắp theo score (rank 1 là score cao nhất, không phải đáp án đúng). Ví dụ ảnh `traffic` chủ yếu là xe buýt, nhưng model xếp `cab` (0.51) hạng 1 và `minibus` (0.16) hạng 2. Ground truth cần được xác định thông qua quá trình gán nhãn và kiểm tra bởi con người.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `sample_id`=kitchen ; `class_name`=bowl ; `score`=0.719063 ; `bbox_xyxy`=[32.65, 342.12, 100.16, 384.93] ; `bbox_width`=67.51 ; `bbox_height`=42.81
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Trả lời: 
    + Góc trên-trái cách mép trái ảnh 32.65 pixel, mép trên 342.12 pixel.
    + Góc dưới-phải: cách mép trái ảnh 100.16 pixel, mép trên 384.93 pixel.
    + Chiều rộng: 67.51 pixel.
    + Chiều cao: 42.81 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Trả lời:
    - Ngưỡng 0.20: có 17 predictions.
    - Ngưỡng 0.60: có 6 predictions.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Trả lời: người **reviewer** phải kiểm tra thêm các box mới được giữ lại, xác định box nào đúng, box nào sai hoặc trùng lặp để quyết định giữ hay bỏ.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Trả lời: 
    - Vẽ bounding box bao sát toàn bộ phần nhìn thấy của đối tượng.
    - Không lấy quá nhiều phần nền xung quanh.
    - Nếu đối tượng bị che khuất hoặc cắt khỏi mép ảnh, chỉ vẽ box quanh phần nhìn thấy, không tự suy đoán phần bị khuất.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Trả lời:
    - Vật bị cắt mép: `person` 0.61, `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78], nằm sát mép trái ảnh. Chỉ thấy cánh tay/bàn tay, phần còn lại của người nằm ngoài khung ảnh.
    - Cắt mép là trường hợp hay gặp nên đưa vào guideline. Quy tắc: khi đối tượng bị cắt bởi mép ảnh, chỉ vẽ bounding box quanh phần còn nhìn thấy, không tự suy đoán phần nằm ngoài ảnh.
    - Điểm mơ hồ: nếu chỉ thấy một cánh tay thì có gán nhãn `person` hay không? Guideline hiện chưa nói rõ, nên mình sẽ hỏi Lab Coach (escalation) thay vì tự quyết.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-004 ; `class_name`=potted plant ; `score`=0.632235 ; số điểm polygon=`polygon_point_count`=100 ; 3–5 điểm đầu của `polygon_xy`=[1.0, 2.0], [1.0, 148.0], [3.0, 148.0], [3.0, 147.0], [4.0, 146.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Trả lời: Polygon mô tả đường biên và hình dạng thực tế của vật thể chi tiết hơn. Trong khi đó, box chỉ là khung chữ nhật bao quanh vật thể.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- Trả lời: `instance_id` dùng để phân biệt từng thực thể riêng biệt trong ảnh. Nó không phải class ID dùng để xác định loại vật thể, cũng không phải tracking ID dùng để theo dõi vật thể qua nhiều khung hình hoặc video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Trả lời: Người gán nhãn cần vẽ mask sát theo đường viền của vật thể, chỉ bao gồm phần nhìn thấy và không lấy phần nền xung quanh.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Trả lời:
    - Chọn `kitchen-004` (`potted plant`, 0.632235) ở sát mép trái ảnh. Polygon bắt đầu `[1.0, 2.0] → [1.0, 148.0]`, chạy thẳng dọc mép trái, cho thấy vật thể bị cắt bởi khung ảnh.
    - Guideline: vật bị cắt mép là trường hợp có thể gặp ở nhiều ảnh. Quy tắc: chỉ vẽ mask theo phần vật thể còn nhìn thấy, không tự đoán phần nằm ngoài ảnh.
    - Escalation: vật này trông giống một bó lá/thảo mộc khô treo trên giá, không thấy chậu. Mình không tự quyết có gán `potted plant` hay không mà hỏi Lab Coach.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn mô tả nội dung chính của toàn bộ ảnh. | Ảnh `traffic` chủ yếu là xe buýt và ô tô nhưng top-1 là `cab` (0.51), top-2 là `minibus` (0.16); taxonomy ImageNet không có lớp khớp hẳn nên khó chọn một nhãn duy nhất. | Chọn một nhãn theo chủ thể chính và guideline. | Kiểm tra nhãn có phù hợp nội dung ảnh và có theo đúng quy tắc chọn chủ thể chính không. |
| Phát hiện vật thể | Mỗi vật thể có một nhãn lớp và một bounding box. | Trên `kitchen`, các box chồng lấn: `oven` 0.69 chồng lên `person` 0.61 và `bowl` 0.50; vài box `bowl` chưa bao sát vật thể ở khu vực bàn. | Gán đúng loại vật thể và vẽ box bao sát từng vật thể. | Kiểm tra nhãn, vị trí box, box trùng lặp và vật thể bị bỏ sót. |
| Instance segmentation | Mỗi vật thể có một nhãn lớp và một mask theo đường biên. | Mask `dining table` (`kitchen-005`, 0.578416, 598 điểm) phủ lên các vật đặt trên bàn; `potted plant` bị cắt ở mép trái ảnh. | Vẽ mask sát đường biên phần nhìn thấy, tách riêng từng vật thể. | Kiểm tra mask có đúng hình dạng, không lấn sang vật thể khác và không bỏ sót vùng cần gán nhãn. |

Nối vòng đời: guideline quy định cách gán nhãn, annotator gán theo guideline để tạo ground truth, model huấn luyện trên ground truth rồi cho prediction. Reviewer/QC so prediction và nhãn với guideline. Nếu phát hiện lỗi thì trả về annotator làm lại, và nếu lỗi lặp lại do guideline mơ hồ thì sửa guideline.

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Trả lời: Chỉ sử dụng ảnh và dữ liệu công khai được cung cấp trong bài lab, không tải lên hoặc chia sẻ dữ liệu cá nhân, dữ liệu khách hàng hay dữ liệu nhạy cảm.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Trả lời: Dừng xử lý và báo cáo với Lab Coach để được hướng dẫn.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x ] Đã nộp **link repository** trên LMS
