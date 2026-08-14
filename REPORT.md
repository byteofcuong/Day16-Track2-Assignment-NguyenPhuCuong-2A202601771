# Lab 16 — Cloud AI Environment Setup (AWS)

**Học viên:** Nguyễn Phú Cường — 2A202601771
**Ngày thực hiện:** 14/08/2026
**Cloud:** AWS (us-east-1) — luồng chính CPU + LightGBM

---

## 1. Hạ tầng đã triển khai

Triển khai bằng Terraform (`terraform/`), 27 tài nguyên:

| Thành phần | Cấu hình |
|---|---|
| VPC | `10.0.0.0/16`, 2 public + 2 private subnet trên 2 AZ |
| Bastion Host | `t3.micro`, public subnet, SSH chỉ từ dải IP của học viên |
| Compute Node | `c7i-flex.large` (2 vCPU / 3.73 GB), **private subnet, không có IP public** |
| NAT Gateway | Cho private subnet tải package/dataset |
| ALB | Port 80 → 8000 (health check `unhealthy` là đúng ở luồng CPU) |
| IAM Role + Instance Profile | Gắn vào Compute Node |

**Môi trường:** Ubuntu 22.04, Python 3.10.12, LightGBM 4.7.0, scikit-learn 1.7.2, pandas 2.3.3 — cài tự động qua `user_data`.

**CPU thực tế:** Intel Xeon Platinum 8488C (Sapphire Rapids).

**Thời gian từ `terraform apply` đến benchmark chạy thành công:** 11:56:43 → 12:39:14 = **42 phút 31 giây** (bao gồm thời gian xử lý 3 sự cố ở mục 4).

---

## 2. Bảng kết quả benchmark

Dataset: **Credit Card Fraud Detection** — 284,807 giao dịch, 492 gian lận (0.1727%).
Chia: 182,276 train / 45,569 validation / 56,962 test (stratified).

| Metric | Kết quả |
|---|---|
| Thời gian load data | **0.9580 s** |
| Thời gian training | **2.8045 s** |
| Best iteration | **138** / 1000 |
| AUC-ROC | **0.973571** |
| Accuracy | **0.999421** |
| F1-Score | **0.807018** |
| Precision | **0.945205** |
| Recall | **0.704082** |
| Inference latency (1 row) | **0.7717 ms** (p50 0.7659 / p95 0.8339 / p99 0.8659) |
| Inference throughput (1000 rows) | **2.1971 ms** → **455,145 dòng/giây** |

**Confusion matrix (ngưỡng 0.5):** TN=56,860 · FP=4 · FN=29 · TP=69
**Average Precision:** 0.842763 · **Ngưỡng F1 tối ưu:** 0.35 (F1 = 0.828729)

---

## 3. Nhận xét (báo cáo ngắn)

Training chỉ mất **2.8 giây** cho 138 cây trên 182k dòng với 2 vCPU — LightGBM dùng thuật toán histogram nên chi phí tính toán tỉ lệ theo số bin chứ không theo số mẫu, vì vậy bài toán quy mô này hoàn toàn không cần GPU; thời gian load CSV (0.96s) chiếm tới một phần ba tổng thời gian huấn luyện.

**AUC-ROC 0.9736** là mức tốt, nhưng với dữ liệu lệch 0.17% thì AUC-ROC dễ gây ảo tưởng — **Accuracy 0.9994 gần như vô nghĩa** vì đoán bừa "không gian lận" cho mọi giao dịch đã đạt 99.83%. Chỉ số đáng tin hơn là **Average Precision 0.8428**. Model đạt **Precision 0.945 nhưng Recall chỉ 0.704**: bắt được 69/98 giao dịch gian lận và chỉ báo nhầm 4 giao dịch sạch. Trong nghiệp vụ chống gian lận, bỏ lọt 29 giao dịch (FN) thường tốn kém hơn nhiều so với 4 lần báo động nhầm, nên nên hạ ngưỡng xuống **0.35** (F1 tăng lên 0.8287) hoặc thấp hơn nữa để đổi precision lấy recall.

**Inference trên CPU nhanh hơn mức cần thiết cho hầu hết use case thực tế:** 0.77 ms cho 1 dòng, đủ để chấm điểm gian lận đồng bộ ngay trong luồng thanh toán. Đáng chú ý là **xử lý theo lô hiệu quả gấp ~350 lần**: 1000 dòng chỉ mất 2.20 ms (0.0022 ms/dòng) so với 0.77 ms/dòng khi gọi lẻ — gần như toàn bộ 0.77 ms đó là chi phí cố định của mỗi lời gọi Python/numpy chứ không phải chi phí duyệt cây. Bài học rút ra: nếu hệ thống có thể gom batch thì throughput tăng hàng trăm lần mà không cần thêm một đồng phần cứng nào.

---

## 4. Sự cố gặp phải và cách xử lý

**(1) `t3.medium` bị AWS từ chối.** Tài khoản đang ở **AWS Free Plan**, chỉ cho phép khởi tạo instance type nằm trong danh sách free-tier-eligible; `t3.medium` mà lab chỉ định không nằm trong đó (`InvalidParameterCombination`). Đổi sang **`c7i-flex.large`** — cùng 2 vCPU / 4 GB, có trong danh sách, và là x86_64 nên dùng chung được Ubuntu AMI amd64 (các lựa chọn `t4g.*` là ARM nên không dùng được).

**(2) Siết SSH về một `/32` làm mất kết nối.** Ban đầu giới hạn Security Group về đúng IP public (`203.171.27.42/32`) theo nguyên tắc least-privilege, nhưng ISP định tuyến ra internet qua nhiều IP khác nhau (đo được nhảy sang `14.238.145.226`), khiến kết nối bị rớt ngẫu nhiên — chỉ 1/5 lần thành công. Phải nới thành hai dải `/24`. Đây là đánh đổi thực tế giữa least-privilege và tính khả dụng.

**(3) `ssh -J` không truyền khóa sang chặng nhảy.** Lệnh `ssh -i lab-key -J ubuntu@<bastion> ubuntu@<node>` báo `Permission denied (publickey)` vì `-J` không áp `-i` cho kết nối tới Bastion. Thay bằng `-o "ProxyCommand=ssh -i lab-key -W %h:%p ubuntu@<bastion>"`. (Hướng dẫn trong README — SSH vào Bastion rồi `ssh` tiếp sang node — cũng không chạy được vì private key không nằm trên Bastion.)

**(4) LightGBM chỉ dựng 1 cây, AUC 0.9347.** Đây là sự cố đáng chú ý nhất về mặt kỹ thuật. Với `learning_rate=0.05` và `num_leaves=31`, AUC trên validation **đạt đỉnh ngay ở cây đầu tiên (0.9124) rồi tụt dần** (0.757 ở cây 2, 0.752 ở cây 300), nên early stopping cắt ngay tại iteration 1. Nguyên nhân: chỉ có 315 giao dịch gian lận trong 182k dòng train, mỗi cây 31 lá overfit dữ dội vào nhúm positive đó khiến boosting đi sai hướng. Quét thực nghiệm 6 cấu hình cho thấy phải regularize mạnh hơn — `learning_rate=0.01`, `num_leaves=15`, `min_child_samples=50` — thì mô hình mới hội tụ bình thường (138 cây, AUC 0.9736, Precision tăng từ 0.677 lên 0.945, số false positive giảm từ 41 xuống 4).

---

## 5. Chi phí

| Dịch vụ | Đơn giá |
|---|---|
| NAT Gateway | ~$0.045/giờ + data |
| Application Load Balancer | ~$0.023/giờ |
| EC2 `c7i-flex.large` | ~$0.084/giờ |
| EC2 `t3.micro` (Bastion) | ~$0.010/giờ |
| **Tổng** | **~$0.16/giờ** |

Hạ tầng chạy khoảng 1 giờ (11:56 → ~12:56), tương ứng **ước tính ~$0.16**.

**Ghi chú về ảnh chụp Billing:** tại thời điểm chụp (12:56 ngày 14/08/2026), AWS Billing Dashboard hiển thị **Estimated grand total: USD 0.00** và **"No data to display"**. Đây không phải sai sót khi thu thập. Chính trang Bills hiển thị banner giải thích: *"Your free plan account does not get charged. Credits cover your free plan costs."* — tài khoản đang ở **AWS Free Plan**, toàn bộ chi phí được credit miễn phí bù trừ nên không phát sinh khoản phải trả. Cùng ràng buộc Free Plan này cũng là nguyên nhân chặn `t3.medium` ở mục 4(1). Ngoài ra AWS còn cập nhật dữ liệu billing trễ 6-24 giờ trong khi hạ tầng chỉ chạy 1 giờ; Cost Explorer API trả về `DataUnavailableException` vì lý do đó. Bảng đơn giá phía trên lấy từ bảng giá công bố của AWS cho region `us-east-1` để ước tính chi phí thực tế nếu tài khoản không có credit.

Một quan sát thêm về phân quyền: IAM user `track2-admin` tuy có `AdministratorAccess` nhưng **vẫn bị chặn xem Billing**, cho tới khi tài khoản root bật công tắc *IAM user and role access to Billing information* trong Account settings. Đây là cơ chế bảo vệ riêng của AWS cho dữ liệu thanh toán, nằm ngoài hệ thống IAM policy thông thường.

Toàn bộ tài nguyên đã được xóa bằng `terraform destroy` ngay sau khi thu thập đủ kết quả.

---

## 6. Danh mục nộp bài

| # | Deliverable | File |
|---|---|---|
| 1 | Screenshot chạy `benchmark.py` | `01-benchmark-output.png` |
| 2 | Metrics đầy đủ | `benchmark_result.json` |
| 3 | Screenshot CPU/RAM/Network | `02-resource-usage.png` |
| 4 | Screenshot AWS Billing | `03-aws-billing.png` |
| 5 | Mã nguồn Terraform | `terraform-source.zip` |
| 6 | Báo cáo | `REPORT.md` (file này) |
| 7 | Script benchmark | `benchmark.py` |
