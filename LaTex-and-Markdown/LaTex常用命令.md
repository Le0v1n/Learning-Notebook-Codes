
### 1. 语法

以下是包含 LaTeX 代码和符号说明的数学排版表格：

<div align=center>

| 数学符号 | LaTeX 代码 | 符号说明 |
| :-: | :-: | :-: |
| $A \ B$ | `$A \ B$` | 空格 |
| $A \quad B$ | `$A \quad B$` | 四个空格 |
| $A \\ B$ | `$A \\ B$` | 换行 |
| $\{a, b \}$ | `\\{a, b \\}` | 转义字符 `\` |
| $\hat{x}$ | `$\\hat{x}$` | 帽子 |
| $\bar{x}$ | `$\\bar{x}$` | 短横线 |
| $\overline{xyz}$ | `$\\overline{xyz}$` | 长横线 |
| $\underline{xyz}$ | `$\\underline{xyz}$` | 长下划线 |
| $\dot{x}$ | `$\\dot{x}$` | 一个点 |
| $\ddot{x}$ | `$\\ddot{x}$` | 两个点 |
| $\vec{x}$ | `$\\vec{x}$` | 矢量 |
| $\overrightarrow{x}$ | `$\\overrightarrow{x}$` | 长矢量 |
| $\left( abc \right)$ | `$\\left( abc \\right)$` | 长括小括号 |
| $\left[ abc \right]$ | `$\\left[ abc \\right]$` | 长括中括号 |
| $\underset{A}{B}$ | `$\underset{A}{B}$` | 在下方写 |
| $\overset{A}{B}$ | `$\overset{A}{B}$` | 在上方写 |

</div>

### 2. 字体

<div align=center>

| 数学符号 | LaTeX 代码 | 符号说明 |
| :-: | :- | :- |
| $\rm{Hello}$ | `$\rm{Hello}$` | 非斜体罗马字体 |
| $\mathit{Hello}$ | `$\mathit{Hello}$` | 斜体字体 |
| $\mathsf{Hello}$ | `$\mathsf{Hello}$` | Sans serif 字体 |
| $\mathtt{Hello}$ | `$\mathtt{Hello}$` | Typerwriter 字体 |
| $\mathcal{Hello}$ | `$\mathcal{Hello}$` | Calligraphic 字体 |
| $\mathbb{Hello}$ | `$\mathbb{Hello}$` | Blackboard bold 字体 |
| $\boldsymbol{Hello}$ | `$\boldsymbol{Hello}$` | Boldsymbol bold 字体 |

</div>

### 3. 矩阵、对齐、分段函数

1. 【矩阵】$\left[\begin{matrix}a & b \cr c & d\end{matrix}\right]$
   ```
   \left[\begin{matrix}
      a & b \cr 
      c & d
   \end{matrix}\right]
   ```
2. 【矩阵】$\left\lgroup\begin{matrix}a & b \cr c & d\end{matrix}\right\rgroup$
   ```
   \left\lgroup\begin{matrix}
      a & b \cr
       c & d
   \end{matrix}\right\rgroup
   ```
3. 【对齐】
   $$
      \begin{aligned}
      a_1 &= b_1 + c_1 \\
      a_2 &= b_2 + c_2 + d_2 \\
      a_3 &= b_3 + c_3
      \end{aligned}
   $$

   ```
   \begin{aligned}
   a_1 &= b_1 + c_1 \\
   a_2 &= b_2 + c_2 + d_2 \\
   a_3 &= b_3 + c_3
   \end{aligned}
   ```
4. 【分段函数】语法中的 `\\` 等价于 `\cr`，表示换行。
   $$
   sign(x) = 
   \begin{cases}
      1, & x > 0 \\ 
      0, & x = 0 \cr 
      -1, & x < 0
   \end{cases}
   $$

   ```
   sign(x) = 
   \begin{cases}
      1, & x > 0 \\ 
      0, & x = 0 \cr 
      -1, & x < 0
   \end{cases}
   ```

### 4. 希腊字母

<div align=center>

| 数学符号 | LaTeX 代码 | 对应大写字母 | LaTeX 代码 |
| :-: | :- | :-: | :- |
| $\alpha$ | `$\alpha$` | $\Gamma$ | `$\Gamma$` |
| $\beta$ | `$\beta$` | $\Delta$ | `$\Delta$` |
| $\gamma$ | `$\gamma$` | $\Theta$ | `$\Theta$` |
| $\delta$ | `$\delta$` | $\Delta$ | `$\Delta$` |
| $\epsilon$ | `$\epsilon$` | | |
| $\varepsilon$ | `$\varepsilon$` | | |
| $\zeta$ | `$\zeta$` | | |
| $\eta$ | `$\eta$` | | |
| $\theta$ | `$\theta$` | $\Theta$ | `$\Theta$` |
| $\vartheta$ | `$\vartheta$` | $\varTheta$ | `$\varTheta$` |
| $\iota$ | `$\iota$` | | |
| $\kappa$ | `$\kappa$` | | |
| $\lambda$ | `$\lambda$` | $\Lambda$ | `$\Lambda$` |
| $\mu$ | `$\mu$` | | |
| $\nu$ | `$\nu$` | | |
| $\xi$ | `$\xi$` | $\Xi$ | `$\Xi$` |
| $\pi$ | `$\pi$` | $\Pi$ | `$\Pi$` |
| $\varpi$ | `$\varpi$` | $\varPi$ | `$\varPi$` |
| $\rho$ | `$\rho$` | | |
| $\varrho$ | `$\varrho$` | | |
| $\sigma$ | `$\sigma$` | $\Sigma$ | `$\Sigma$` |
| $\varsigma$ | `$\varsigma$` | $\varSigma$ | `$\varSigma$` |
| $\tau$ | `$\tau$` | | |
| $\upsilon$ | `$\upsilon$` | $\Upsilon$ | `$\Upsilon$` |
| $\phi$ | `$\phi$` | $\Phi$ | `$\Phi$` |
| $\varphi$ | `$\varphi$` | $\varPhi$ | `$\varPhi$` |
| $\chi$ | `$\chi$` | | |
| $\psi$ | `$\psi$` | $\Psi$ | `$\Psi$` |
| $\omega$ | `$\omega$` | $\Omega$ | `$\Omega$` |

</div>

### 5. 运算符

<div align=center>

| 数学符号 | LaTeX 代码 | 说明 |
| :-: | :- | :-: |
| $\ll$ | `$\ll$` | 远小于 |
| $\gg$ | `$\gg$` | 远大于 |
| $\approx$ | `$\approx$` | 约等于 |
| $\sim$ | `$\sim$` | 相似 |
| $\ne$ | `$\ne$` | 不等于 |
| $\in$ | `$\in$` | 属于 |
| $\cup$ | `$\cup$` | 交 |
| $\cap$ | `$\cap$` | 并 |
| $\pm$ | `$\pm$` | 加减 (plusminus) |
| $\div$ | `$\div$` | 除法 |
| $\cdot$ | `$\cdot$` | 点乘 |
| $\odot$ | `$\odot$` | 圈点乘 |
| $\oplus$ | `$\oplus$` | 圈加 |
| $\otimes$ | `$\otimes$` | 圈乘 |
| $\prod$ | `$\prod$` | 连乘 |
| $\int$ | `$\int$` | 积分 |
| $\partial$ | `$\partial$` | 偏导 |

</div>

### 6. 其他符号

<div align=center>

| 数学符号 | LaTeX 代码 | 说明 |
| :-: | :- | :- |
| $\dots$ | `$\dots$` | 省略号 |
| $\cdots$ | `$\cdots$` | 居中省略号 |
| $\Re$ | `$\Re$` | 实部 |
| $\nabla$ | `$\nabla$` | 梯度符号 |
| $\triangle$ | `$\triangle$` | 三角形 |
| $\angle$ | `$\angle$` | 角度符号 |
| $\infty$ | `$\infty$` | 无穷大 |
| $\dag$ | `$\dag$` | 剪影标记 |
| $\ddag$ | `$\ddag$` | 双剪影标记 |
| $\S$ | `$\S$` | 资料标记 |
| $\because$ | `$\because$` | 因为 |
| $\therefore$ | `$\therefore$` | 所以 |
| $\leftrightarrow$ | `$\leftrightarrow$` | 左右箭头 |
| $\Leftrightarrow$ | `$\Leftrightarrow$` | 左右双箭头 |
| $\nleftrightarrow$ | `$\nleftrightarrow$` | 非左右箭头 |
| $\nLeftrightarrow$ | `$\nLeftrightarrow$` | 非左右双箭头 |
| $\varnothing$ | `$\varnothing$` | 空集符号 |

</div>

# 参考

1. [如何使用jupyter编写数学公式(译)](https://www.jianshu.com/p/93ccc63e5a1b)