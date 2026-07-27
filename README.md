# Adult Income — Dự đoán thu nhập & Phân tích/Giảm thiểu thiên vị thuật toán

Dự đoán một cá nhân có thu nhập hằng năm **> 50.000 USD** hay không, dựa trên bộ dữ liệu census
**Adult Income** (UCI, ~32.5k dòng). Bên cạnh việc tối ưu độ chính xác, dự án tập trung phân tích
sâu về **fairness**: đo lường và giảm thiểu thiên vị thuật toán (algorithmic bias) theo hai thuộc
tính nhạy cảm `sex` và `race`, kể cả khi hai thuộc tính này **giao nhau (intersectional)**, sử dụng
thư viện [`fairlearn`](https://fairlearn.org/). Ngoài ra dùng `SHAP` để tìm các đặc trưng đóng vai
trò "proxy" gián tiếp mang thông tin về sex/race.

## Tech stack

`pandas` · `numpy` · `scikit-learn` (Pipeline, ColumnTransformer, RandomizedSearchCV) ·
`fairlearn` (MetricFrame, ExponentiatedGradient) · `shap` · `matplotlib`

## Pipeline

1. **Làm sạch dữ liệu** — thay `"?"` bằng `NaN`, loại dòng thiếu (mất 7.4% dữ liệu, còn 30.162 dòng).
2. **Tiền xử lý an toàn** — `ColumnTransformer` + `OneHotEncoder` trong `Pipeline`, chỉ `fit` trên
   train để tránh rò rỉ dữ liệu (thay vì `LabelEncoder` fit trên toàn bộ dữ liệu — lỗi thường gặp).
3. **Tuning** — `RandomizedSearchCV` (3-fold Stratified CV, scoring = AUC) trên `RandomForestClassifier`.
4. **Đo fairness** với `MetricFrame`: selection rate, TPR, FPR theo từng nhóm; Demographic Parity
   difference (DP diff) và Equalized Odds difference (EO diff) — theo `sex`, `race`, và **sex × race**.
5. **Giảm thiểu bias** bằng `ExponentiatedGradient` (constraint = `DemographicParity`, ràng buộc
   theo `sex`).
6. **SHAP** — xếp hạng feature importance, đối chiếu với phân bố theo sex/race để tìm proxy feature.

## Kết quả

**Model baseline** (RandomForest, đã tuning): **Accuracy 86.69%**, **AUC 0.924**.

| | Accuracy | DP diff (sex) | EO diff (sex) | DP diff (race) | EO diff (race) | DP diff (sex×race) | EO diff (sex×race) |
|---|---|---|---|---|---|---|---|
| Baseline (tuned) | 0.8669 | 0.1544 | 0.0475 | 0.2206 | 0.6818 | 0.2787 | 1.0000 |
| Sau mitigation (ExponentiatedGradient, DP theo sex) | 0.8452 | **0.0039** | 0.2974 | 0.1059 | 0.5714 | 0.1968 | 1.0000 |

**Nhận xét chính:**
- Ràng buộc mitigation theo `sex` gần như **triệt tiêu Demographic Parity gap của sex** (0.1544 →
  0.0039), đánh đổi bằng **-2.17 điểm % accuracy** — trade-off khá hợp lý.
- Tuy nhiên **Equalized Odds diff của sex lại tăng** (0.0475 → 0.2974): tối ưu DP không đồng nghĩa
  tối ưu EO — hai định nghĩa fairness có thể mâu thuẫn nhau.
- Vì mitigation chỉ ràng buộc theo `sex`, **race** và đặc biệt **sex × race** vẫn còn bias rất lớn
  (EO diff giao nhau vẫn = 1.0000 cả trước lẫn sau) — cho thấy fairness theo từng thuộc tính riêng lẻ
  **không đảm bảo** fairness ở cấp độ giao nhau (intersectional).

**SHAP — top đặc trưng quan trọng nhất:** `marital.status_Married-civ-spouse`, `education.num`,
`capital.gain`, `age`, `relationship_Husband`, `hours.per.week`, `marital.status_Never-married`,
`occupation_Exec-managerial`, `occupation_Prof-specialty`, `capital.loss`. Hai đặc trưng
`marital.status` và `relationship` (đặc biệt giá trị `Husband`) vừa có tầm quan trọng cao với model,
vừa phân bố lệch mạnh theo `sex` → là **proxy feature khả nghi nhất** cho giới tính.

*(Hình `fairness_before_after.png` và `shap_top_features.png` do notebook xuất ra minh họa trực quan
hai phần trên.)*

## Cấu trúc repo

```
.
├── adult_income_fairness.ipynb   # Notebook chính
├── adult_income.csv              # Dữ liệu Adult Income (UCI)
└── README.md
```

## Cài đặt & chạy

```bash
pip install pandas numpy matplotlib scikit-learn fairlearn shap jupyter
jupyter notebook adult_income_fairness.ipynb
```

Chạy tuần tự các cell từ trên xuống. SHAP được tính trên mẫu con 500 dòng (thay vì toàn bộ tập test)
để notebook chạy nhanh hơn.

## Nguồn dữ liệu

[Adult Income Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)
