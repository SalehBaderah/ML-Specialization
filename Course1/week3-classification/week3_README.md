| Week | Description                      | Slides 📄                          | Notebooks 📓                       |
|------|----------------------------------|------------------------------------|-----------------------------------|
| 3    | classification             | [Week 3 Slides](https://github.com/SalehBaderah/ML-Specialization/blob/main/Course1-Supervised%20ML/week3-classification/Slides/C1_W3.pdf) | [Week 3 Notebooks](https://github.com/SalehBaderah/ML-Specialization/tree/main/Course1-Supervised%20ML/week3-classification/Notebooks) |


# Logistic Regression 

is a supervised learning algorithm used for binary classification,It predicts the probability that an input belongs to class 1 or 0. 

- It starts with our linear regression model 
  - $z = \mathbf{w} \cdot  \mathbf{x}^{(i)} + b$ 


- Then applies the sigmoid function to squash the output between 0 and 1
  - $σ(z) = \frac{1}{1+e^{-z}}\tag{1}$
  - 
  <img width="400" height="381" alt="Screenshot 2025-07-28 182946" src="https://github.com/user-attachments/assets/065604c8-78ff-4edf-8451-6a436814c26f" />


---

## Cost function for logistic regression

Recall that for logistic regression, the cost function is of the form 

<img width="464" height="93" alt="image" src="https://github.com/user-attachments/assets/4908e479-59f2-47cc-ba05-d1ad242a81fa" />


where
* $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ is the cost for a single data point, which is:

    $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \tag{2}$$
    


---
# Gradient descent for logistic regression
Recall the gradient descent algorithm utilizes the gradient calculation:


<img width="326" height="134" alt="image" src="https://github.com/user-attachments/assets/dc41858f-8c04-4432-9b5c-c16c8dda5107" />


Where each iteration performs simultaneous updates on $w_j$ for all $j$, where

<img width="676" height="300" alt="image" src="https://github.com/user-attachments/assets/34f9d835-6e96-486b-b4d8-203a050f0081" />


* m is the number of training examples in the data set      
* $f_{\mathbf{w},b}(x^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target
* For a logistic regression model  
    $z = \mathbf{w} \cdot \mathbf{x} + b$  
    $f_{\mathbf{w},b}(x) = g(z)$  
    where $g(z)$ is the sigmoid function:  
    $g(z) = \frac{1}{1+e^{-z}}$   
