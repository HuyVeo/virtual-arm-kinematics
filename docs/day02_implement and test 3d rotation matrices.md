# Ngày 2 — Vector 3D, Rotation Matrix và phép quay trong Robotics

**Thời lượng:** 4 giờ  
**Tuần 1:** Toán 3D và Forward Kinematics

## Mục tiêu ngày 2

Xây nền tảng toán học để hiểu robot quay quanh các trục và đầu kẹp thay đổi vị trí như thế nào.

**Kết quả cuối ngày:**
- Viết được ma trận quay quanh trục X, Y, Z.
- Áp dụng phép quay lên vector 3D.
- Kiểm chứng kết quả bằng NumPy và pytest.
- Giải thích được sự khác nhau giữa quay một vector và đổi hệ tọa độ.

## 1. Lịch học 4 giờ

| Thời lượng | Nội dung | Việc cần làm |
|---|---|---|
| 45 phút | Ôn hình học 3D | Vector vị trí, chuẩn Euclid, hệ tọa độ Cartesian, quy tắc bàn tay phải, radian và độ |
| 45 phút | Rotation Matrix | Học ba ma trận quay cơ bản, phép nhân ma trận với vector, thứ tự phép quay |
| 120 phút | Thực hành NumPy | Viết `rotation_matrix_x/y/z()`, kiểm thử tính trực giao, định thức và phép quay liên tiếp |
| 30 phút | Tổng kết | Ghi công thức, chạy test, commit Git, tự giải thích một ví dụ không nhìn code |

### Bài tập khởi động

Biểu diễn vị trí một điểm cách gốc tọa độ 3 đơn vị theo X, 2 theo Y và 1 theo Z.

## 2. Lý thuyết cốt lõi

### A. Vector 3D

Một điểm trong không gian được biểu diễn bằng vector cột:

$$
\mathbf p =
\begin{bmatrix}
x\\
y\\
z
\end{bmatrix}
$$

Độ dài vector:

$$
\|\mathbf p\|=\sqrt{x^2+y^2+z^2}
$$

Ví dụ, với $\mathbf p=[3,2,1]^T$, khoảng cách từ điểm đến gốc tọa độ là $\sqrt{14}$.

Trong Robotics, cần phân biệt:

- **Position:** vị trí của một điểm.
- **Orientation:** hướng của một vật thể.
- **Rotation matrix:** biểu diễn phép quay hoặc hướng tương đối giữa các hệ tọa độ.

### B. Ba ma trận quay cơ bản

Với góc $\theta$, ma trận quay quanh trục Z là:

$$
R_z(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta&0\\
\sin\theta&\cos\theta&0\\
0&0&1
\end{bmatrix}
$$

Ma trận quay quanh X:

$$
R_x(\theta)=
\begin{bmatrix}
1&0&0\\
0&\cos\theta&-\sin\theta\\
0&\sin\theta&\cos\theta
\end{bmatrix}
$$

Ma trận quay quanh Y:

$$
R_y(\theta)=
\begin{bmatrix}
\cos\theta&0&\sin\theta\\
0&1&0\\
-\sin\theta&0&\cos\theta
\end{bmatrix}
$$

Trong quy ước quay chủ động, vector cột được biến đổi bởi:

$$
\mathbf p'=R\mathbf p
$$

Ví dụ quay quanh Z một góc $90^\circ$:

$$
\begin{bmatrix}
0&-1&0\\
1&0&0\\
0&0&1
\end{bmatrix}
\begin{bmatrix}
1\\0\\0
\end{bmatrix}
=
\begin{bmatrix}
0\\1\\0
\end{bmatrix}
$$

Điểm ban đầu nằm trên trục X dương; sau khi quay, nó nằm trên trục Y dương.

### C. Ba tính chất bắt buộc phải kiểm tra

Ma trận quay hợp lệ phải thỏa mãn:

$$
R^TR=I,\qquad \det(R)=1
$$

Phép quay cũng bảo toàn độ dài vector:

$$
\|R\mathbf p\|=\|\mathbf p\|
$$

Đây là các kiểm tra quan trọng hơn việc chỉ nhìn thấy kết quả số có vẻ hợp lý.

## 3. Bài thực hành — Viết code

Tạo file `python/rotations.py`:

```python
import numpy as np


def rotation_matrix_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def rotation_matrix_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rotate_vector(R: np.ndarray,
                  vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    assert R.shape == (3, 3)
    assert vector.shape == (3,)
    return R @ vector


if __name__ == "__main__":
    theta = np.deg2rad(90)
    p = np.array([1.0, 0.0, 0.0])

    R = rotation_matrix_z(theta)
    result = rotate_vector(R, p)

    print("Rotation matrix:\n", R)
    print("Original:", p)
    print("Rotated:", result)

    assert np.allclose(result, [0, 1, 0], atol=1e-10)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.isclose(np.linalg.norm(result),
                      np.linalg.norm(p))

    print("All checks passed.")
```

Chạy từ thư mục gốc repository:

```bash
python python/rotations.py
```

Kết quả mong đợi của vector sau phép quay gần bằng `[0, 1, 0]`.

## 4. Viết test bằng pytest

Tạo file `tests/test_rotations.py`:

```python
import numpy as np

from python.rotations import (
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    rotate_vector,
)


def test_rotation_z_90_degrees():
    R = rotation_matrix_z(np.pi / 2)
    result = rotate_vector(R, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(result, [0, 1, 0], atol=1e-10)


def test_rotation_matrices_are_orthogonal():
    for fn in (
        rotation_matrix_x,
        rotation_matrix_y,
        rotation_matrix_z,
    ):
        R = fn(0.73)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_rotation_preserves_vector_length():
    p = np.array([2.0, -3.0, 4.0])

    for fn in (
        rotation_matrix_x,
        rotation_matrix_y,
        rotation_matrix_z,
    ):
        result = rotate_vector(fn(1.2), p)
        assert np.isclose(np.linalg.norm(result), np.linalg.norm(p))
```

Chạy test:

```bash
python -m pytest tests/test_rotations.py -v
```

Cả ba test cần pass.

## 5. Bài tập nâng cao

### Bài 1 — Quay quanh trục X

Quay vector $[0,1,0]^T$ một góc $90^\circ$ quanh X. Hãy dự đoán kết quả trước khi chạy code.

- [ ] `[0, 0, 1]`
- [ ] `[0, 0, -1]`
- [ ] `[1, 0, 0]`

**Đáp án:** `[0, 0, 1]`. Theo quy tắc bàn tay phải, trục Y dương quay về trục Z dương.

### Bài 2 — Thứ tự phép quay

Với vector $[1,0,0]^T$, tính lần lượt:

$$
R_z(90^\circ)R_x(90^\circ)\mathbf p
$$

và

$$
R_x(90^\circ)R_z(90^\circ)\mathbf p
$$

Hai kết quả có giống nhau không?

**Đáp án:** Không nhất thiết. Phép quay 3D nói chung không giao hoán; thứ tự nhân ma trận rất quan trọng. Khi dùng vector cột, phép biến đổi ở bên phải được thực hiện trước.

### Bài 3 — Kiểm tra tính chất

Thử góc ngẫu nhiên và vector ngẫu nhiên. Viết một vòng lặp kiểm tra:

- $R^TR=I$
- $\det(R)=1$
- Độ dài vector được bảo toàn.


## 7. Checklist hoàn thành ngày 2

- [ ] Viết được ba ma trận quay X, Y, Z bằng NumPy.
- [ ] Giải thích được radian và quy tắc bàn tay phải.
- [ ] Kiểm tra tính trực giao và định thức bằng test.
- [ ] Kiểm tra phép quay bảo toàn độ dài vector.
- [ ] Hiểu thứ tự nhân các phép quay.
- [ ] Toàn bộ test pass và đã commit Git.

## Chuẩn bị cho ngày 3

Chưa cần học Quaternion hay DH hôm nay. Ngày 3 sẽ ghép rotation matrix với vector tịnh tiến để tạo homogeneous transformation $4\times4$, từ đó mô tả vị trí và hướng của từng khâu robot trong cùng một hệ tọa độ.
