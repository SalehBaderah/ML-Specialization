| Week | Description                      | Slides 📄                          | Notebooks 📓                       |
|------|----------------------------------|------------------------------------|-----------------------------------|
| 2    | Regression with Multiple Variables             | [Week 2 Slides](https://github.com/SalehBaderah/ML-Specialization/tree/main/Course1-Supervised%20ML/week2-Regression%20with%20multiple%20input%20variable/Handwritten_notes_slides) | [Week 2 Notebook](https://github.com/SalehBaderah/ML-Specialization/tree/main/Course1-Supervised%20ML/week2-Regression%20with%20multiple%20input%20variable/Notebooks) |


#  Regression with Multiple Variables

<img width="650" height="231" alt="Screenshot 2025-07-28 175423" src="https://github.com/user-attachments/assets/f9ddf817-6382-4714-a8a8-ae9de639dd73" />


---

## Model


If we have $n$ features:

$$
\mathbf{x}^{(i)} = 
\begin{bmatrix}
x_1^{(i)} \\
x_2^{(i)} \\
\vdots \\
x_n^{(i)}
\end{bmatrix}
\quad \text{and} \quad
\mathbf{w} = 
\begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_n
\end{bmatrix}
$$

Instead of a single feature $x$, we use a **vector of features**:

$$
\hat{y}^{(i)} = w_1x_1^{(i)} + w_2x_2^{(i)} + \cdots + w_nx_n^{(i)} + b
$$

Or, in vector notation:

$$
\hat{y}^{(i)} = \vec{w} . \mathbf{x}^{(i)} + b
$$





---

##  Cost Function

We use the **Mean Squared Error (MSE)**:

$$
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

- $m$: number of training examples  
- $y^{(i)}$: true (actual) target value  
- $\hat{y}^{(i)}$: predicted value

---

##  Gradient Descent

**Partial Derivatives:**

$$
\frac{\partial J(\mathbf{w}, b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) \cdot x_j^{(i)}
$$

$$
\frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)
$$

**Update Rules:**

$$
w_j := w_j - \alpha \cdot \frac{\partial J}{\partial w_j}
$$

$$
b := b - \alpha \cdot \frac{\partial J}{\partial b}
$$

---

##  Notation

- $\alpha$: learning rate  
- $j$: index of a feature  
- $m$: number of training examples  
- $x_j^{(i)}$: the $j^{th}$ feature in the $i^{th}$ example

---
# feature scaling
- Preprocessing step
-  used to normalize the range of features of data.

## 1-  Maximum Normalization 

Maximum normalization scales each feature by dividing it by its `maximum value`, so that all feature values lie in the range [0, 1].

<img width="259" height="70" alt="image" src="https://github.com/user-attachments/assets/0efc44b7-e1b7-4cd0-af32-887790b1e9ef" />


## 2- z-score normalization

 all features will have a mean of 0 and a standard deviation of 1.

formula:

<img width="271" height="90" alt="image" src="https://github.com/user-attachments/assets/ad7c6520-e69c-4048-b1f5-109ea087c6fa" />

where $j$ selects a feature or a column in the $\mathbf{X}$ matrix. $µ_j$ is the mean of all the values for feature (j) and $\sigma_j$ is the standard deviation of feature (j).




## 3-Mean normailzation
- The mean becomes 0
- The values are typically in the range [−1,1]

formula:

<img width="375" height="68" alt="image" src="https://github.com/user-attachments/assets/b2b17eaa-ff40-4675-bb14-01506816a9f1" />

