| Week | Description                      | Slides 📄                          | Notebook 📓                       |
|------|----------------------------------|------------------------------------|-----------------------------------|
| 1    | Introduction & Model             | [Week 1 Slides](slides/week1.pdf) | [Week 1 Notebook](notebooks/week1.ipynb) |


#  Simple Linear Regression

Simple Linear Regression models the relationship between **one input (feature)** and **one output (target)** using a straight line.

---

##  Model

The model predicts the output $\hat{y}$ as a linear function of the input $x^{(i)}$:

$$
\hat{y}^{(i)} = wx^{(i)} + b \tag{1}
$$

- $\hat{y}^{(i)}$: predicted output  
- $x^{(i)}$: input feature of the $i^{th}$ example  
- $w$: weight (slope of the line)  
- $b$: bias (intercept)

---

##  Cost Function

The cost function measures the difference between the model's predictions $\hat{y}^{(i)}$ and the actual values $y^{(i)}$. It tells us how well the model is performing.

We use the **Mean Squared Error (MSE)**:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2 \tag{2}
$$

- $m$: number of training examples  
- $y^{(i)}$: actual target value  
- $\hat{y}^{(i)}$: predicted value  

In the specialization, the cost function was written as:

$$
f_{w,b}(x^{(i)}) = wx^{(i)} + b
$$

This is equivalent to $\hat{y}^{(i)}$.

---

##  Gradient Descent

**Gradients (Partial Derivatives):**

$$
\frac{\partial J(w, b)}{\partial w} = \frac{1}{m} \sum\limits_{i=0}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) \cdot x^{(i)} \tag{3}
$$

$$
\frac{\partial J(w, b)}{\partial b} = \frac{1}{m} \sum\limits_{i=0}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) \tag{4}
$$

**Parameter Update Rules:**

$$
w := w - \alpha \cdot \frac{\partial J(w, b)}{\partial w} \tag{5}
$$

$$
b := b - \alpha \cdot \frac{\partial J(w, b)}{\partial b} \tag{6}
$$

---

##  Notation Summary

- $w$ = weight (slope)  
- $b$ = bias (intercept)  
- $\alpha$ = learning rate  
- $x^{(i)}$ = input of the $i^{th}$ training example  
- $y^{(i)}$ = true label of the $i^{th}$ example  
- $\hat{y}^{(i)}$ = predicted value  
- $m$ = number of training examples





