# GpositivePseAAC Dataset - Partial Abstention Charts

## Metric: aabs (Average Absolute Error)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=0, ymax=80,
        ytick={0,10,...,80},
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
            (1,12.87) (2,12.87) (3,13.01) (4,12.96)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,11.47) (6,11.95) (7,11.37) (8,11.71) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,27.17) (2,27.17) (3,27.19) (4,27.19)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,14.5) (6,14.99) (7,14.36) (8,15.18) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,48.92) (2,48.99) (3,48.92) (4,48.99)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,18.6) (6,19.85) (7,18.62) (8,19.82) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,70.8) (2,70.82) (3,70.8) (4,70.82) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,20.88) (6,21.79) (7,21.05) (8,21.79) 
        };
    \end{axis}
\end{tikzpicture}
```

## Metric: abs (Absolute Error)

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
            (1,35.65) (2,35.46) (3,36.04) (4,35.65)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,33.34) (6,34.31) (7,33.15) (8,33.34) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,57.03) (2,57.03) (3,57.13) (4,57.03)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,37.68) (6,38.45) (7,37.48) (8,38.54) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,80.74) (2,80.84) (3,80.74) (4,80.84)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,46.45) (6,47.99) (7,46.16) (8,47.99) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,94.8) (2,94.8) (3,94.8) (4,94.8) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,52.69) (6,53.46) (7,52.88) (8,53.37) 
        };
    \end{axis}
\end{tikzpicture}
```

## Metric: arec (Average Recall)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=80, ymax=100,
        ytick={80,85,...,100},
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
            (1,86.75) (2,86.85) (3,86.8) (4,86.95)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,86.75) (6,86.75) (7,86.66) (8,86.66) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,90.2) (2,90.27) (3,90.17) (4,90.22)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,86.7) (6,87.09) (7,86.68) (8,87.11) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,93.52) (2,93.54) (3,93.52) (4,93.5)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,86.37) (6,87.84) (7,86.22) (8,87.88) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,94.12) (2,94.15) (3,94.12) (4,94.15) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,80.76) (6,83.12) (7,80.86) (8,83.24) 
        };
    \end{axis}
\end{tikzpicture}
```

## Metric: f1_pa (F1 Score for Partial Abstention)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=60, ymax=80,
        ytick={60,65,...,80},
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
            (1,72.83) (2,72.57) (3,72.96) (4,72.77)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,72.86) (6,72.39) (7,72.78) (8,72.2) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,74.46) (2,74.38) (3,74.42) (4,74.28)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,71.99) (6,71.91) (7,72.12) (8,71.95) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,77.99) (2,77.99) (3,77.99) (4,77.87)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,71.93) (6,73.02) (7,71.81) (8,73.12) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,66.0) (2,65.98) (3,66.0) (4,65.98) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,63.24) (6,64.45) (7,63.52) (8,64.48) 
        };
    \end{axis}
\end{tikzpicture}
```

## Metric: hamming_accuracy_pa (Hamming Accuracy for Partial Abstention)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=70, ymax=90,
        ytick={70,75,...,90},
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
            (1,84.76) (2,84.87) (3,84.8) (4,84.97)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,85.01) (6,84.94) (7,84.92) (8,84.87) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,86.45) (2,86.54) (3,86.42) (4,86.47)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,84.43) (6,84.78) (7,84.43) (8,84.79) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,87.32) (2,87.34) (3,87.32) (4,87.27)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,83.3) (6,84.84) (7,83.12) (8,84.91) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,79.61) (2,79.61) (3,79.61) (4,79.61) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,75.77) (6,78.45) (7,75.83) (8,78.58) 
        };
    \end{axis}
\end{tikzpicture}
```

## Metric: rec (Recall)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=40, ymax=90,
        ytick={40,50,...,90},
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
            (1,72.26) (2,72.64) (3,72.26) (4,72.83)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,71.87) (6,72.25) (7,71.48) (8,72.06) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,76.11) (2,76.3) (3,76.01) (4,76.2)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,69.36) (6,71.67) (7,69.46) (8,71.87) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,81.98) (2,82.18) (3,81.98) (4,82.08)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,64.45) (6,70.23) (7,64.16) (8,70.52) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,81.89) (2,82.08) (3,81.89) (4,82.08) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,46.72) (6,55.69) (7,47.4) (8,56.56) 
        };
    \end{axis}
\end{tikzpicture}
```

## Metric: subset0_1_pa (Subset 0-1 Accuracy for Partial Abstention)

```latex
\begin{tikzpicture}
    \begin{axis}[
        name=ax1,
        scale=0.45,
        symbolic x coords={1,2,3,4,5,6,7,8},
        xtick=data,
        xmin=1, xmax=8,
        ymin=40, ymax=80,
        ytick={40,50,...,80},
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
            (1,71.65) (2,72.04) (3,71.65) (4,72.24)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,71.42) (6,71.87) (7,71.1) (8,71.68) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,73.33) (2,73.54) (3,73.23) (4,73.43)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,68.51) (6,70.79) (7,68.64) (8,70.96) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,75.32) (2,75.55) (3,75.32) (4,75.43)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,63.14) (6,68.91) (7,62.82) (8,69.22) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,62.03) (2,62.4) (3,62.03) (4,62.4) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,44.38) (6,53.36) (7,45.09) (8,54.33) 
        };
    \end{axis}
\end{tikzpicture}
``` 