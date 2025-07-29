| Week | Description                      | Slides 📄                          | Notebook 📓                       |
|------|----------------------------------|------------------------------------|-----------------------------------|
| 1    | Introduction & Model             | [Week 1 Slides]([slides/week1.pdf](https://github.com/SalehBaderah/ML-Specialization/blob/main/Course1-Supervised%20ML/week1-Intro%20to%20ML/Slides/C1_W1.pdf) | [Week 1 Notebook](notebooks/week1.ipynb) |


#  Simple Linear Regression

Simple Linear Regression models the relationship between **one input (feature)** and **one output (target)** using a straight line.

---

##  Model

The model predicts the output $\hat{y}$ as a linear function of the input $x^{(i)}$:



<img width="123" height="26" alt="image" src="https://github.com/user-attachments/assets/7f5ad72c-2a7b-4d20-b3f2-4fa68836abe7" />





- $\hat{y}^{(i)}$: predicted output  
- $x^{(i)}$: input feature of the $i^{th}$ example  
- $w$: weight (slope of the line)  
- $b$: bias (intercept)

---

##  Cost Function

The cost function measures the difference between the model's predictions $\hat{y}^{(i)}$ and the actual values $y^{(i)}$. It tells us how well the model is performing.

We use the **Mean Squared Error (MSE)**:

<img width="410" height="94" alt="image" src="https://github.com/user-attachments/assets/d6878fe2-3d26-42bd-a0ee-3430853e303c" />



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

<img width="411" height="93" alt="image" src="https://github.com/user-attachments/assets/05b1722b-cdbc-4c90-95d7-198add74eb3c" />




<img width="361" height="90" alt="image" src="https://github.com/user-attachments/assets/b15c6115-dee5-4e67-bf50-83b439e0abd1" />


**Parameter Update Rules:**

<img width="281" height="77" alt="image" src="https://github.com/user-attachments/assets/caba0608-2777-4891-ac64-ba9c0b509ba1" />


<img width="267" height="77" alt="image" src="https://github.com/user-attachments/assets/9b3cd614-f3a4-43f8-999d-68c93252d6ad" />


---

##  Notation Summary

- $w$ = weight (slope)  
- $b$ = bias (intercept)  
- $\alpha$ = learning rate  
- $x^{(i)}$ = input of the $i^{th}$ training example  
- $y^{(i)}$ = true label of the $i^{th}$ example  
- $\hat{y}^{(i)}$ = predicted value  
- $m$ = number of training examples





