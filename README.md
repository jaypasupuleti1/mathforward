
***

# 🌌 Project MathForward: The Math Suite
**Goal:** To take the formulas from my **Multivariable Calculus** and **Linear Algebra** textbooks and turn them into functional Python tools.

---

## 📈 Project 1: The "Smart" Stock Picker
**The Concept:** Instead of picking stocks based on hype, this uses **Optimization** to find the highest return for the lowest risk.

* **The Math:** * **Expected Value ($E[R]$):** Calculates the average "win" over 252 trading days.
    * **Volatility ($\sigma$):** Uses the **Square Root of Time** rule ($\sigma \times \sqrt{252}$) because risk grows non-linearly.
    * **Sharpe Ratio:** Finds the **Gradient (slope)** of the risk-reward line to pick the "mathematical winner."
* **Real-World Use:** This is the foundation of **Quantitative Finance** used by hedge funds to balance portfolios.



---

## 👤 Project 2: Eigenface Reconstructor
**The Concept:** This shows how computers "see" faces by breaking a photo down into its most important mathematical parts.

* **The Math:** * **Vectorization:** Turns a $64 \times 64$ photo into a single vector with **4,096 dimensions**.
    * **SVD (Singular Value Decomposition):** Decomposes 400 faces to find the "core ingredients" (the **Eigenfaces**).
    * **Projection:** Projects **your** photo onto those Eigenfaces using **Dot Products** to find the "weights" that describe your face.
* **Real-World Use:** This is how **Facial Recognition** and **Image Compression** work—turning a 1MB photo into just 50 numbers.



---

## 🛠️ Quick Setup
1.  **Install:** `pip install numpy matplotlib scikit-learn yfinance pillow`
2.  **Run Stocks:** `python stock_math.py` (Enter tickers like AAPL, TSLA)
3.  **Run Faces:** `python face_math.py` (Upload a JPG to see the reconstruction)

# --- THE "WHY" BEHIND THE DIFFERENCE ---
# If the reconstruction looks different, it's because of "Projection Error."
# 1. The computer only knows the 400 faces it was trained on. 
# 2. It's trying to build YOUR face using THEIR features (Eigenfaces).
# 3. Anything unique about you that wasn't in the dataset is treated as 
#    "Noise" and filtered out. This is a trade-off: 
#    We lose a little detail, but we save 98% more storage space!
