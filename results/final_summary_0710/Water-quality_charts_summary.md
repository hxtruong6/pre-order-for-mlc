# Partial Abstention Results - Water-quality

## Dataset: Water-quality

### Metric: aabs (Average Absolute Error)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=0, ymax=15,
        ytick={0,3,...,15},
        enlarge x limits=+0.01,
        ymajorgrids=true,   
        bar width=0.3cm,    
        xmajorgrids=true,
    ]      
        \addplot[white, thin] coordinates {
            (1,0) (2,0) (3,0) (4,0) (5,0) (6,0) (7,0) (8,0)
        };
        
        % Noisy level = 0.0
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,6.97) (2,6.89) (3,6.94) (4,6.89)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,1.55) (6,1.33) (7,1.55) (8,1.37) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,8.79) (2,8.66) (3,8.79) (4,8.67)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,1.09) (6,0.89) (7,1.12) (8,0.92) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,10.2) (2,10.04) (3,10.15) (4,10.06)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,0.73) (6,0.58) (7,0.77) (8,0.61) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,12.27) (2,12.14) (3,12.26) (4,12.19) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,0.48) (6,0.23) (7,0.49) (8,0.25) 
        };
    \end{axis}
\end{tikzpicture}
```

### Metric: abs (Absolute Error)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=0, ymax=70,
        ytick={0,10,...,70},
        enlarge x limits=+0.01,
        ymajorgrids=true,   
        bar width=0.3cm,    
        xmajorgrids=true,
    ]      
        \addplot[white, thin] coordinates {
            (1,0) (2,0) (3,0) (4,0) (5,0) (6,0) (7,0) (8,0)
        };
        
        % Noisy level = 0.0
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,42.45) (2,42.08) (3,42.36) (4,42.17)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,8.02) (6,6.04) (7,7.92) (8,6.23) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,54.72) (2,54.1) (3,54.62) (4,54.1)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,8.77) (6,6.32) (7,9.15) (8,6.51) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,60.61) (2,59.72) (3,60.38) (4,59.72)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,6.23) (6,4.29) (7,6.51) (8,4.48) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,66.04) (2,65.09) (3,65.8) (4,65.09) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,5.09) (6,2.08) (7,5.28) (8,2.26) 
        };
    \end{axis}
\end{tikzpicture}
```

### Metric: arec (Average Recall)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=50, ymax=80,
        ytick={50,60,...,80},
        enlarge x limits=+0.01,
        ymajorgrids=true,   
        bar width=0.3cm,    
        xmajorgrids=true,
    ]      
        \addplot[white, thin] coordinates {
            (1,0) (2,0) (3,0) (4,0) (5,0) (6,0) (7,0) (8,0)
        };
        
        % Noisy level = 0.0
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,74.89) (2,75.19) (3,74.87) (4,75.19)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,72.23) (6,73.61) (7,72.32) (8,73.07) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,73.27) (2,73.82) (3,73.24) (4,73.81)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,69.35) (6,72.3) (7,69.37) (8,71.91) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,70.3) (2,71.33) (3,70.22) (4,71.23)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,65.57) (6,69.77) (7,65.55) (8,69.72) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,66.1) (2,67.15) (3,66.04) (4,67.1) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,59.12) (6,64.78) (7,59.16) (8,64.83) 
        };
    \end{axis}
\end{tikzpicture}
```

### Metric: f1_pa (F1 Score for Partial Abstention)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=40, ymax=70,
        ytick={40,50,...,70},
        enlarge x limits=+0.01,
        ymajorgrids=true,   
        bar width=0.3cm,    
        xmajorgrids=true,
    ]      
        \addplot[white, thin] coordinates {
            (1,0) (2,0) (3,0) (4,0) (5,0) (6,0) (7,0) (8,0)
        };
        
        % Noisy level = 0.0
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,61.61) (2,60.95) (3,61.59) (4,60.86)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,60.94) (6,57.17) (7,61.02) (8,54.74) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,59.93) (2,59.26) (3,59.92) (4,59.23)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,59.58) (6,56.16) (7,59.57) (8,54.15) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,56.81) (2,56.57) (3,56.72) (4,56.42)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,57.43) (6,53.81) (7,57.35) (8,52.52) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,51.54) (2,51.4) (3,51.56) (4,51.3) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,53.65) (6,49.19) (7,53.72) (8,48.08) 
        };
    \end{axis}
\end{tikzpicture}
```

### Metric: hamming_accuracy_pa (Hamming Accuracy for Partial Abstention)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=50, ymax=80,
        ytick={50,60,...,80},
        enlarge x limits=+0.01,
        ymajorgrids=true,   
        bar width=0.3cm,    
        xmajorgrids=true,
    ]      
        \addplot[white, thin] coordinates {
            (1,0) (2,0) (3,0) (4,0) (5,0) (6,0) (7,0) (8,0)
        };
        
        % Noisy level = 0.0
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,73.01) (2,73.36) (3,72.99) (4,73.35)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,71.79) (6,73.26) (7,71.88) (8,72.69) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,70.7) (2,71.34) (3,70.66) (4,71.33)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,69.01) (6,72.05) (7,69.02) (8,71.65) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,66.93) (2,68.14) (3,66.85) (4,68.02)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,65.31) (6,69.6) (7,65.28) (8,69.53) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,61.36) (2,62.62) (3,61.3) (4,62.54) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,58.93) (6,64.7) (7,58.97) (8,64.75) 
        };
    \end{axis}
\end{tikzpicture}
```

### Metric: rec (Recall)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=0, ymax=10,
        ytick={0,2,...,10},
        enlarge x limits=+0.01,
        ymajorgrids=true,   
        bar width=0.3cm,    
        xmajorgrids=true,
    ]      
        \addplot[white, thin] coordinates {
            (1,0) (2,0) (3,0) (4,0) (5,0) (6,0) (7,0) (8,0)
        };
        
        % Noisy level = 0.0
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,3.11) (2,3.87) (3,3.11) (4,3.77)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,1.6) (6,2.64) (7,1.6) (8,2.36) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,1.7) (2,1.98) (3,1.65) (4,1.84)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,0.9) (6,1.32) (7,0.94) (8,1.23) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,1.42) (2,1.46) (3,1.42) (4,1.37)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,0.52) (6,1.37) (7,0.47) (8,1.27) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,0.85) (2,0.85) (3,0.85) (4,0.9) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,0.05) (6,0.28) (7,0.05) (8,0.38) 
        };
    \end{axis}
\end{tikzpicture}
```

### Metric: subset0_1_pa (Subset 0-1 Accuracy for Partial Abstention)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=0, ymax=10,
        ytick={0,2,...,10},
        enlarge x limits=+0.01,
        ymajorgrids=true,   
        bar width=0.3cm,    
        xmajorgrids=true,
    ]      
        \addplot[white, thin] coordinates {
            (1,0) (2,0) (3,0) (4,0) (5,0) (6,0) (7,0) (8,0)
        };
        
        % Noisy level = 0.0
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,2.93) (2,3.69) (3,2.93) (4,3.59)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,1.6) (6,2.64) (7,1.6) (8,2.36) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,1.7) (2,1.98) (3,1.65) (4,1.84)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,0.9) (6,1.32) (7,0.94) (8,1.23) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,1.42) (2,1.46) (3,1.42) (4,1.37)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,0.52) (6,1.37) (7,0.47) (8,1.27) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,0.8) (2,0.8) (3,0.8) (4,0.85) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,0.05) (6,0.28) (7,0.05) (8,0.38) 
        };
    \end{axis}
\end{tikzpicture}
``` 