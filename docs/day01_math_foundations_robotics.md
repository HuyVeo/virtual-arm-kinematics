# Day 01 — Nền tảng Toán học cho Robotics

> Mục tiêu: ôn tập vector, lượng giác, ma trận và các phép biến đổi cơ bản cần cho Kinematics, ROS 2 và điều khiển robot.
>
> Đơn vị: dùng mét (m) cho chiều dài và radian (rad) cho góc trong code Python, trừ khi ghi chú khác.

## Mục lục

1. [Vector](#1-vector)
2. [Lượng giác](#2-lượng-giác)
3. [Ma trận](#3-ma-trận)
4. [Biến đổi tọa độ trong Robotics](#4-biến-đổi-tọa-độ-trong-robotics)
5. [NumPy cho Robotics](#5-numpy-cho-robotics)
6. [Forward Kinematics của tay máy 2 khớp](#6-forward-kinematics-của-tay-máy-2-khớp)
7. [Bài tập Day 01](#7-bài-tập-day-01)
8. [Checklist hoàn thành](#8-checklist-hoàn-thành)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. Vector

### 1.1. Vector là gì?

Vector là đại lượng có độ lớn và hướng. Trong Robotics, vector thường dùng để biểu diễn vị trí, độ dời, vận tốc, gia tốc, lực và hướng.

Vector 2D:

$$
\mathbf{v} = \begin{bmatrix}v_x\\v_y\end{bmatrix}
$$

Vector 3D:

$$
\mathbf{v} = \begin{bmatrix}v_x\\v_y\\v_z\end{bmatrix}
$$

Ví dụ, `p = [0.3, 0.2, 0.4]` có thể biểu diễn tọa độ vị trí theo X, Y, Z, tính bằng mét nếu hệ tọa độ dùng đơn vị mét.

**Phân biệt:**
- **Point (điểm):** vị trí trong một hệ tọa độ.
- **Vector:** đại lượng có độ lớn và hướng.
- **Displacement (độ dời):** vector từ vị trí đầu đến vị trí cuối.
- **Coordinate frame (hệ tọa độ):** quy ước gốc và các trục dùng để biểu diễn vị trí/hướng.

### 1.2. Độ dài vector (norm)

Với vector 2D:

$$
\|\mathbf{v}\| = \sqrt{v_x^2 + v_y^2}
$$

Với vector 3D:

$$
\|\mathbf{v}\| = \sqrt{v_x^2 + v_y^2 + v_z^2}
$$

Ví dụ:

$$
\mathbf{v} = \begin{bmatrix}3\\4\end{bmatrix},
\qquad \|\mathbf{v}\| = \sqrt{3^2 + 4^2} = 5
$$

```python
import numpy as np

v = np.array([3.0, 4.0])
length = np.linalg.norm(v)
print(length)  # 5.0
```

### 1.3. Cộng, trừ và nhân vector với số

Cho:

$$
\mathbf{a}=\begin{bmatrix}1\\2\\3\end{bmatrix},\quad
\mathbf{b}=\begin{bmatrix}4\\5\\6\end{bmatrix}
$$

$$
\mathbf{a}+\mathbf{b}=\begin{bmatrix}5\\7\\9\end{bmatrix},\qquad
\mathbf{b}-\mathbf{a}=\begin{bmatrix}3\\3\\3\end{bmatrix}
$$

$$
2\mathbf{a}=\begin{bmatrix}2\\4\\6\end{bmatrix}
$$

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print(a + b)
print(b - a)
print(2 * a)
```

Hai vector cần có cùng số thành phần khi cộng/trừ theo cách thông thường.

### 1.4. Vector đơn vị (unit vector)

Vector đơn vị có norm bằng 1. Nó thể hiện hướng mà không giữ nguyên độ lớn ban đầu.

$$
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}
$$

Ví dụ:

$$
\begin{bmatrix}3\\4\end{bmatrix}
\longrightarrow
\begin{bmatrix}0.6\\0.8\end{bmatrix}
$$

Không thể chuẩn hóa vector 0 vì phép chia cho norm bằng 0 không xác định.

```python
def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return v / norm
```

### 1.5. Tích vô hướng (dot product)

$$
\mathbf{a}\cdot\mathbf{b}
= \sum_i a_i b_i
= \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta
$$

Với `a = [1, 2, 3]`, `b = [4, 5, 6]`:

$$
\mathbf{a}\cdot\mathbf{b}=1(4)+2(5)+3(6)=32
$$

Tìm góc giữa hai vector khác vector 0:

$$
\cos\theta =
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|\|\mathbf{b}\|}
$$

Ứng dụng:
- Tìm góc giữa hai hướng.
- Kiểm tra vuông góc: tích vô hướng bằng 0 (trong sai số số học).
- Tính hình chiếu của vector lên vector khác.

```python
dot = np.dot(a, b)
```

### 1.6. Tích có hướng (cross product)

Với vector 3D:

$$
\mathbf{a}\times\mathbf{b} =
\begin{bmatrix}
a_yb_z-a_zb_y\\
a_zb_x-a_xb_z\\
a_xb_y-a_yb_x
\end{bmatrix}
$$

Kết quả vuông góc với cả hai vector; hướng theo quy tắc bàn tay phải. Độ lớn là diện tích hình bình hành tạo bởi hai vector.

```python
a = np.array([1.0, 0.0, 0.0])
b = np.array([0.0, 1.0, 0.0])
print(np.cross(a, b))  # [0. 0. 1.]
```

Ứng dụng trong Robotics: mô-men lực, trục quay và vận tốc của điểm trên vật rắn. Công thức trên là cho vector 3D; với vector 2D cần diễn giải riêng.

### 1.7. Khoảng cách giữa hai điểm

Nếu hai điểm nằm trong cùng hệ tọa độ:

$$
d = \|\mathbf{p}_2-\mathbf{p}_1\|
$$

```python
def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p2 - p1))
```

Không trừ trực tiếp tọa độ được biểu diễn trong hai hệ tọa độ khác nhau nếu chưa chuyển chúng về cùng một hệ.

---

## 2. Lượng giác

### 2.1. Độ và radian

Một vòng tròn:

$$
360^\circ = 2\pi \text{ rad}
$$

Đổi độ sang radian:

$$
\theta_{\mathrm{rad}} = \theta_{\mathrm{deg}}\frac{\pi}{180}
$$

| Độ | Radian |
|---:|---:|
| 0° | 0 |
| 30° | $\frac{\pi}{6}$ |
| 45° | $\frac{\pi}{4}$ |
| 60° | $\frac{\pi}{3}$ |
| 90° | $\frac{\pi}{2}$ |
| 180° | ${\pi}$ |
| 360° | ${2\pi}$ |

```python
theta = np.deg2rad(90)
print(theta)          # 1.570796...
print(np.sin(theta))  # 1.0
```

**Quan trọng:** `np.sin()` và `np.cos()` nhận radian, không nhận độ. `np.sin(90)` là sin của 90 radian, không phải sin của 90°.

### 2.2. Sin, cos và tan

Trong tam giác vuông:

$$
\sin\theta=\frac{\text{cạnh đối}}{\text{cạnh huyền}}
$$

$$
\cos\theta=\frac{\text{cạnh kề}}{\text{cạnh huyền}}
$$

$$
\tan\theta=\frac{\text{cạnh đối}}{\text{cạnh kề}}
=\frac{\sin\theta}{\cos\theta}
$$

| Góc | $\sin\theta$ | $\cos\theta$ | $\tan\theta$ |
|---:|---:|---:|---:|
| 0° | 0 | 1 | 0 |
| 30° | $1/2$ | $\sqrt{3}/2$ | $1/\sqrt{3}$ |
| 45° | $\sqrt{2}/2$ | $\sqrt{2}/2$ | 1 |
| 60° | $\sqrt{3}/2$ | $1/2$ | $\sqrt{3}$ |
| 90° | 1 | 0 | Không xác định |

Tại 90°, tan không xác định vì mẫu số cos bằng 0.

### 2.3. Đường tròn đơn vị

Trên đường tròn bán kính 1, điểm ứng với góc $\theta$ có tọa độ:

$$
(x,y)=(\cos\theta,\sin\theta)
$$

Đây là nền tảng hình học cho ma trận quay và tính vị trí khi robot xoay quanh một khớp.

### 2.4. Định lý Pythagore và đồng nhất thức lượng giác

Định lý Pythagore:

$$
a^2+b^2=c^2
$$

Đồng nhất thức lượng giác cơ bản:

$$
\sin^2\theta+\cos^2\theta=1
$$

Đồng nhất thức này có thể dùng để kiểm tra các giá trị sin/cos.

### 2.5. Công thức góc tổng và góc hiệu

$$
\sin(\alpha+\beta)=\sin\alpha\cos\beta+\cos\alpha\sin\beta
$$

$$
\cos(\alpha+\beta)=\cos\alpha\cos\beta-\sin\alpha\sin\beta
$$

$$
\sin(\alpha-\beta)=\sin\alpha\cos\beta-\cos\alpha\sin\beta
$$

$$
\cos(\alpha-\beta)=\cos\alpha\cos\beta+\sin\alpha\sin\beta
$$

Những công thức này giải thích cách các góc khớp nối tiếp được cộng lại trong công thức Forward Kinematics.

### 2.6. Ứng dụng: tay máy phẳng hai khớp

Giả sử tay máy có hai đoạn dài $L_1,L_2$. Góc $\theta_1$ là góc đoạn thứ nhất so với trục X; $\theta_2$ là góc tương đối giữa đoạn thứ hai và đoạn thứ nhất.

$$
x=L_1\cos\theta_1+L_2\cos(\theta_1+\theta_2)
$$

$$
y=L_1\sin\theta_1+L_2\sin(\theta_1+\theta_2)
$$

Ví dụ, $L_1=0.30$ m, $L_2=0.25$ m và cả hai góc bằng 0:

$$
(x,y)=(0.55,0)\text{ m}
$$

```python
def planar_arm_fk(
    theta1: float,
    theta2: float,
    L1: float = 0.30,
    L2: float = 0.25,
) -> np.ndarray:
    # theta1, theta2 phải là radian
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return np.array([x, y])

p = planar_arm_fk(np.deg2rad(0), np.deg2rad(0))
print(p)  # [0.55 0.  ]
```

---

## 3. Ma trận

### 3.1. Kích thước ma trận

Ma trận là bảng số gồm hàng và cột. Ma trận có $m$ hàng, $n$ cột có kích thước $m\times n$.

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}
$$

Ma trận trên có kích thước $2\times3$.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.shape)  # (2, 3)
```

### 3.2. Cộng, trừ và nhân với số

Cho:

$$
A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
B=\begin{bmatrix}5&6\\7&8\end{bmatrix}
$$

$$
A+B=\begin{bmatrix}6&8\\10&12\end{bmatrix}
$$

Cộng/trừ thông thường yêu cầu hai ma trận cùng kích thước. Nhân với số thực hiện trên từng phần tử.

```python
A = np.array([[1., 2.], [3., 4.]])
B = np.array([[5., 6.], [7., 8.]])
print(A + B)
print(2 * A)
```

### 3.3. Nhân ma trận với vector và ma trận

Nếu $A$ có kích thước $m\times n$, vector đầu vào phải có $n$ phần tử. Kết quả có $m$ phần tử.

$$
\begin{bmatrix}1&2\\3&4\end{bmatrix}
\begin{bmatrix}5\\6\end{bmatrix}
=
\begin{bmatrix}17\\39\end{bmatrix}
$$

Nếu $A$ là $m\times n$ và $B$ là $n\times p$, tích $AB$ có kích thước $m\times p$.

```python
A = np.array([[1., 2.],
              [3., 4.]])
v = np.array([5., 6.])
print(A @ v)  # [17. 39.]
```

**Cảnh báo NumPy:**
- `A @ B`: nhân ma trận.
- `A * B`: nhân từng phần tử tương ứng (element-wise), nếu shape tương thích.
- Nhìn chung $AB\ne BA$; thứ tự nhân ma trận quan trọng.

### 3.4. Ma trận chuyển vị (transpose)

Chuyển vị đổi hàng thành cột:

$$
A^T=\begin{bmatrix}1&3\\2&4\end{bmatrix}
$$

```python
print(A.T)
```

### 3.5. Ma trận đơn vị (identity matrix)

$$
I_3=\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}
$$

Nhân ma trận đơn vị với vector phù hợp sẽ giữ nguyên vector:

$$
I\mathbf{v}=\mathbf{v}
$$

```python
I = np.eye(3)
v = np.array([0.3, 0.2, 0.4])
print(I @ v)
```

### 3.6. Định thức và nghịch đảo

Với ma trận vuông $2\times2$:

$$
A=\begin{bmatrix}a&b\\c&d\end{bmatrix},
\qquad \det(A)=ad-bc
$$

Nếu định thức khác 0, ma trận có nghịch đảo:

$$
A^{-1}=\frac{1}{ad-bc}
\begin{bmatrix}d&-b\\-c&a\end{bmatrix}
$$

Nghịch đảo thỏa $AA^{-1}=I$. Ma trận có định thức bằng 0 là ma trận suy biến và không có nghịch đảo.

```python
A = np.array([[1., 2.], [3., 4.]])
print(np.linalg.det(A))
print(np.linalg.inv(A))
print(A @ np.linalg.inv(A))  # gần bằng ma trận đơn vị
```

Khi giải $A\mathbf{x}=\mathbf{b}$, nếu hệ vuông và có nghiệm duy nhất, thường nên dùng `np.linalg.solve(A, b)` thay vì tính nghịch đảo rồi nhân.

### 3.7. Ma trận quay 2D

Quay vector quanh gốc tọa độ một góc $\theta$, ngược chiều kim đồng hồ:

$$
R(\theta)=\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
$$

$$
\mathbf{p}'=R(\theta)\mathbf{p}
$$

Quay $[1,0]^T$ đi 90° cho kết quả $[0,1]^T$.

```python
def rotation_matrix_2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

R = rotation_matrix_2d(np.deg2rad(90))
p = np.array([1.0, 0.0])
print(R @ p)  # xấp xỉ [0. 1.]
```

Tính chất của ma trận quay hợp lệ:

$$
R^TR=I,\qquad R^{-1}=R^T,\qquad \det(R)=1
$$

Ma trận quay giữ nguyên độ dài và góc giữa các vector; nó đổi hướng mà không kéo giãn hình học.

### 3.8. Ma trận quay 3D

Quay quanh trục X:

$$
R_x(\theta)=\begin{bmatrix}
1&0&0\\
0&\cos\theta&-\sin\theta\\
0&\sin\theta&\cos\theta
\end{bmatrix}
$$

Quay quanh trục Y:

$$
R_y(\theta)=\begin{bmatrix}
\cos\theta&0&\sin\theta\\
0&1&0\\
-\sin\theta&0&\cos\theta
\end{bmatrix}
$$

Quay quanh trục Z:

$$
R_z(\theta)=\begin{bmatrix}
\cos\theta&-\sin\theta&0\\
\sin\theta&\cos\theta&0\\
0&0&1
\end{bmatrix}
$$

Thứ tự quay quan trọng: nói chung $R_xR_y\ne R_yR_x$. Góc trong các công thức trên dùng radian khi tính bằng NumPy.

### 3.9. Biến đổi thuần nhất (homogeneous transformation)

Trong 2D, kết hợp quay và tịnh tiến bằng ma trận $3\times3$:

$$
T=\begin{bmatrix}
\cos\theta&-\sin\theta&t_x\\
\sin\theta&\cos\theta&t_y\\
0&0&1
\end{bmatrix}
$$

Biểu diễn điểm $(x,y)$ bằng tọa độ thuần nhất $[x,y,1]^T$:

$$
\begin{bmatrix}x'\\y'\\1\end{bmatrix}
=T\begin{bmatrix}x\\y\\1\end{bmatrix}
$$

Trong 3D, ma trận thuần nhất có kích thước $4\times4$:

$$
T=\begin{bmatrix}
R_{3\times3}&\mathbf{t}_{3\times1}\\
0&0&0&1
\end{bmatrix}
$$

Trong đó $R$ là ma trận quay, $\mathbf{t}=[t_x,t_y,t_z]^T$ là vector tịnh tiến.

Luôn xác định phép biến đổi đi từ hệ tọa độ nào sang hệ tọa độ nào. Không dùng tọa độ của hai hệ khác nhau trong cùng phép tính nếu chưa chuyển đổi.

### 3.10. Ghép biến đổi

Theo quy ước vector cột, nếu $T_{AB}$ biến tọa độ từ hệ A sang hệ B và $T_{BC}$ biến từ B sang C thì:

$$
T_{AC}=T_{BC}T_{AB}
$$

Với robot nhiều khâu nối tiếp:

$$
T_{0n}=T_{01}T_{12}\cdots T_{(n-1)n}
$$

Thứ tự nhân không thể tùy ý đảo.

### 3.11. Jacobian — khái niệm mở đầu

Jacobian liên hệ vận tốc khớp và vận tốc đầu công tác:

$$
\dot{\mathbf{x}}=J(\mathbf{q})\dot{\mathbf{q}}
$$

- $\mathbf{q}$: góc hoặc vị trí các khớp.
- $\dot{\mathbf{q}}$: vận tốc các khớp.
- $\dot{\mathbf{x}}$: vận tốc đầu công tác.
- $J$: ma trận Jacobian phụ thuộc cấu hình robot.

Day 01 chỉ cần hiểu ý nghĩa. Việc tính và sử dụng Jacobian sẽ học sâu hơn trong phần tiếp theo.

---

## 4. Biến đổi tọa độ trong Robotics

Robot thường có nhiều hệ tọa độ: hệ gốc robot, hệ tại từng khớp, hệ camera và hệ đầu công tác (end-effector).

Trước khi biến đổi tọa độ, hãy xác định:
1. Điểm/vector đang được biểu diễn trong hệ nào?
2. Muốn chuyển sang hệ nào?
3. Ma trận đang dùng quy ước biến đổi từ nguồn sang đích hay ngược lại?
4. Đơn vị chiều dài và góc có nhất quán không?

Một lỗi thường gặp là dùng đúng công thức nhưng sai hệ tọa độ hoặc sai thứ tự nhân ma trận. Kết quả có thể vẫn là một vector hợp lệ về mặt số học nhưng không có ý nghĩa vật lý.

---

## 5. NumPy cho Robotics

| Nhiệm vụ | NumPy |
|---|---|
| Tạo vector | `np.array([1., 2., 3.])` |
| Tạo ma trận đơn vị | `np.eye(3)` |
| Norm | `np.linalg.norm(v)` |
| Dot product | `np.dot(a, b)` |
| Cross product | `np.cross(a, b)` |
| Nhân ma trận | `A @ B` |
| Chuyển vị | `A.T` |
| Định thức | `np.linalg.det(A)` |
| Nghịch đảo | `np.linalg.inv(A)` |
| Giải hệ tuyến tính | `np.linalg.solve(A, b)` |
| Đổi độ sang radian | `np.deg2rad(angle)` |
| Đổi radian sang độ | `np.rad2deg(angle)` |
| Sin/cos | `np.sin(theta)`, `np.cos(theta)` |
| So sánh gần bằng | `np.allclose(a, b)` |

NumPy dùng mảng 1D như `np.array([1, 2])` cho vector thông thường. Nếu cần biểu diễn vector cột rõ ràng, dùng `np.array([[1], [2]])`, shape là `(2, 1)`.

Do sai số số thực, các kết quả như `6.123e-17` có thể được xem là gần bằng 0. Dùng `np.allclose()` hoặc `pytest.approx()` thay vì so sánh dấu phẩy động bằng `==` khi kiểm thử kết quả số.

---

## 6. Forward Kinematics của tay máy 2 khớp

**Forward Kinematics (FK)** tính vị trí đầu công tác từ các góc khớp đã biết.

Với tay máy phẳng 2 khớp:
- $L_1,L_2$: chiều dài hai đoạn tay (m).
- $\theta_1$: góc đoạn thứ nhất so với trục X.
- $\theta_2$: góc tương đối của đoạn thứ hai so với đoạn thứ nhất.
- Góc truyền vào hàm Python là radian.

$$
x=L_1\cos\theta_1+L_2\cos(\theta_1+\theta_2)
$$

$$
y=L_1\sin\theta_1+L_2\sin(\theta_1+\theta_2)
$$

Với $L_1=0.30$ m và $L_2=0.25$ m:

| $\theta_1$ | $\theta_2$ | Vị trí đầu tay máy |
|---:|---:|---:|
| 0° | 0° | (0.55, 0) m |
| 90° | 0° | (0, 0.55) m |
| 0° | 90° | (0.30, 0.25) m |

Các góc ở bảng là độ để dễ đọc; cần đổi sang radian trước khi gọi hàm NumPy.

```python
def planar_arm_fk(
    theta1: float,
    theta2: float,
    L1: float = 0.30,
    L2: float = 0.25,
) -> np.ndarray:
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return np.array([x, y])

print(planar_arm_fk(np.deg2rad(0), np.deg2rad(0)))
print(planar_arm_fk(np.deg2rad(90), np.deg2rad(0)))
print(planar_arm_fk(np.deg2rad(0), np.deg2rad(90)))
```

---

## 7. Bài tập Day 01

Hãy tự làm trước khi xem lời giải.

1. Tính norm của vector `[3, 4]`.
2. Chuẩn hóa vector `[3, 4]` và kiểm tra norm của kết quả bằng 1.
3. Tính dot product của `[1, 2, 3]` và `[4, 5, 6]`.
4. Tính cross product của `[1, 0, 0]` và `[0, 1, 0]`.
5. Tính `sin(30°)`, `cos(60°)` và `sin(90°)` bằng NumPy.
6. Tính `A @ v` với `A = [[1, 2], [3, 4]]` và `v = [5, 6]`.
7. Quay vector `[1, 0]` đi 90° ngược chiều kim đồng hồ.
8. Kiểm tra $R^TR=I$ với ma trận quay 2D.
9. Tính FK cho cả ba trường hợp trong bảng ở mục 6.
10. Viết test để đảm bảo hàm chuẩn hóa ném `ValueError` khi đầu vào là vector 0.

### Kết quả tham chiếu

- Norm `[3, 4]`: `5`.
- Vector chuẩn hóa: `[0.6, 0.8]`.
- Dot product: `32`.
- Cross product: `[0, 0, 1]`.
- `sin(30°)`: `0.5`; `cos(60°)`: `0.5`; `sin(90°)`: `1.0`.
- `A @ v`: `[17, 39]`.
- Quay `[1, 0]` 90°: gần bằng `[0, 1]`.
- FK: `(0.55, 0)`, `(0, 0.55)`, `(0.30, 0.25)` mét.

---

## 8. Checklist hoàn thành

-  Giải thích được point, vector, displacement và coordinate frame.
-  Tính được norm, vector đơn vị, dot product, cross product và khoảng cách.
- Đổi được độ sang radian và hiểu vì sao NumPy dùng radian.
-  Hiểu sin/cos qua tam giác vuông và đường tròn đơn vị.
-  Nhân được ma trận với vector và ma trận.
-  Hiểu ma trận đơn vị, chuyển vị, định thức và nghịch đảo.
-  Tạo được ma trận quay 2D và kiểm tra tính chất của nó.
-  Giải thích được ý nghĩa của ma trận biến đổi thuần nhất.
-  Tự viết được hàm FK cho tay máy phẳng hai khớp.
-  Chạy được unit tests và lưu công việc bằng Git.

**Tiêu chuẩn tự đánh giá:** có thể giải thích các công thức bằng lời, tự viết lại hàm FK mà không nhìn mẫu, và dùng test để phát hiện kết quả sai.

---

## 9. Tài liệu tham khảo

- NumPy — Array creation: https://numpy.org/doc/stable/user/basics.creation.html
- NumPy — Linear algebra: https://numpy.org/doc/stable/reference/routines.linalg.html
- NumPy — Trigonometric functions: https://numpy.org/doc/stable/reference/routines.math.html
- Modern Robotics (Kevin M. Lynch, Frank C. Park): https://modernrobotics.org/
- ROS 2 documentation: https://docs.ros.org/

---

## Ghi chú học tập

- Ngày học: YYYY-MM-DD
- Những phần đã hiểu:
- Những phần cần ôn lại:
- Lỗi đã gặp và cách sửa:
- Commit Git:
