# Lab 16 — GCP CPU LightGBM Benchmark

1. Tôi dùng Google Cloud, region `us-central1`, zone `us-central1-a`, VM `e2-medium` (2 vCPU/4 GB RAM), source commit `55539f67d7c78b43afe334a2ec3271c4bfdbbe2d`.
2. Dataset Credit Card Fraud Detection có 284,807 dòng, 31 cột, 492 fraud records; split stratified với seed 16: 170,883 train / 56,962 validation / 56,962 test.
3. Data load mất 2.702 giây; training mất 4.148 giây; LightGBM early stopping chọn best iteration 68.
4. Trên test set: AUC-ROC 0.976848, Accuracy 0.999508, F1 0.847826, Precision 0.906977, Recall 0.795918.
5. Latency một dòng là 1.483 ms; throughput batch 1,000 dòng là 248,796.94 dòng/giây. Tôi đo median, loại warm-up khỏi phép đo, dùng predict_proba trên pandas input.
6. CPU/RAM/Network được quan sát bằng top, free -h và ip -s link; ảnh tài nguyên nằm trong submission/screenshots/. Nếu ảnh được chụp sau benchmark, đó là trạng thái sau chạy.
7. Billing Reports được lọc theo Project và ngày làm lab; Billing có thể chưa cập nhật tại thời điểm quan sát do độ trễ báo cáo. Ảnh Billing nằm trong submission/screenshots/.
8. Tôi đã tải benchmark.py và benchmark_result.json về laptop trước khi dùng Terraform destroy để xóa VM, Cloud NAT, Load Balancer, disk và mạng của lab.
