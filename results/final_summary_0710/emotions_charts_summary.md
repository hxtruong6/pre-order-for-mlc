# Partial Abstention Results - Charts Summary

This document contains TikZ charts for all datasets and evaluation metrics from the PartialAbstention experiments.

## Dataset: emotions

### Metric: aabs (Average Absolute Error)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=0, ymax=50,
        ytick={0,10,...,50},
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
            (1,8.38) (2,8.38) (3,8.38) (4,8.38)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,4.36) (6,4.33) (7,4.5) (8,4.38) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,16.33) (2,16.34) (3,16.36) (4,16.37)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,4.62) (6,4.89) (7,4.65) (8,5.0) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,27.29) (2,27.33) (3,27.25) (4,27.35)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,5.23) (6,5.13) (7,5.3) (8,5.22) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,40.89) (2,40.85) (3,40.89) (4,40.86) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,4.4) (6,4.21) (7,4.51) (8,4.28) 
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
        ymin=0, ymax=100,
        ytick={0,20,...,100},
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
            (1,28.67) (2,28.84) (3,28.67) (4,28.84)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,17.37) (6,17.2) (7,17.87) (8,17.2) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,46.38) (2,46.46) (3,46.38) (4,46.46)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,17.12) (6,17.7) (7,17.12) (8,18.13) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,67.7) (2,67.87) (3,67.61) (4,67.87)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,20.07) (6,18.55) (7,20.07) (8,19.14) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,86.17) (2,86.09) (3,86.17) (4,86.09) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,18.29) (6,17.02) (7,19.14) (8,17.45) 
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
        ymin=60, ymax=100,
        ytick={60,70,...,100},
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
            (1,82.71) (2,83.05) (3,82.66) (4,83.08)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,81.28) (6,81.84) (7,81.36) (8,81.78) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,84.68) (2,85.01) (3,84.7) (4,84.98)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,81.04) (6,82.16) (7,81.07) (8,81.65) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,85.76) (2,86.07) (3,85.73) (4,85.96)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,78.5) (6,80.17) (7,78.6) (8,79.92) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,85.36) (2,85.41) (3,85.33) (4,85.41) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,71.71) (6,76.08) (7,71.62) (8,76.04) 
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
        ymin=50, ymax=90,
        ytick={50,60,...,90},
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
            (1,69.15) (2,69.32) (3,69.08) (4,69.35)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,68.67) (6,67.58) (7,68.73) (8,66.93) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,70.59) (2,70.85) (3,70.61) (4,70.77)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,68.52) (6,67.97) (7,68.54) (8,66.44) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,69.23) (2,69.34) (3,69.26) (4,69.2)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,66.94) (6,65.55) (7,67.16) (8,64.41) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,63.25) (2,62.92) (3,63.17) (4,62.9) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,61.04) (6,61.53) (7,60.89) (8,60.8) 
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
        ymin=60, ymax=100,
        ytick={60,70,...,100},
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
            (1,81.13) (2,81.5) (3,81.07) (4,81.52)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,80.43) (6,81.02) (7,80.49) (8,80.95) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,81.68) (2,82.07) (3,81.69) (4,82.03)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,80.12) (6,81.25) (7,80.15) (8,80.68) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,80.43) (2,80.84) (3,80.41) (4,80.68)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,77.32) (6,79.09) (7,77.41) (8,78.81) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,75.16) (2,75.27) (3,75.12) (4,75.27) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,70.41) (6,75.02) (7,70.29) (8,74.96) 
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
        ymin=0, ymax=50,
        ytick={0,10,...,50},
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
            (1,37.77) (2,38.78) (3,37.77) (4,38.95)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,34.74) (6,36.25) (7,34.74) (8,35.92) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,42.32) (2,43.33) (3,42.49) (4,43.25)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,32.8) (6,36.75) (7,33.22) (8,34.4) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,44.09) (2,44.94) (3,44.01) (4,44.6)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,25.96) (6,30.27) (7,26.13) (8,29.18) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,42.83) (2,42.99) (3,42.83) (4,43.16) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,14.75) (6,22.34) (7,14.5) (8,22.43) 
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
        ymin=0, ymax=50,
        ytick={0,10,...,50},
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
            (1,37.77) (2,38.78) (3,37.77) (4,38.95)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,34.74) (6,36.25) (7,34.74) (8,35.92) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,41.18) (2,42.22) (3,41.35) (4,42.13)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,32.74) (6,36.71) (7,33.17) (8,34.35) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,41.11) (2,41.99) (3,41.02) (4,41.63)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,25.9) (6,30.21) (7,26.01) (8,29.18) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,36.16) (2,36.34) (3,36.16) (4,36.53) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,14.75) (6,22.34) (7,14.5) (8,22.43) 
        };
    \end{axis}
\end{tikzpicture}
``` 