# Scene Dataset - Partial Abstention Charts

## Metric: aabs (Average Absolute Error)

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
            (1,22.88) (2,22.91) (3,22.93) (4,23.0)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,21.42) (6,21.6) (7,21.38) (8,21.63) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,30.63) (2,30.65) (3,30.62) (4,30.69)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,21.86) (6,22.22) (7,21.91) (8,22.28) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,43.86) (2,43.89) (3,43.86) (4,43.92)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,21.2) (6,21.66) (7,21.33) (8,21.78) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,60.48) (2,60.51) (3,60.47) (4,60.51) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,18.34) (6,18.88) (7,18.39) (8,18.99) 
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
            (1,42.29) (2,42.25) (3,42.29) (4,42.29)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,40.22) (6,40.42) (7,40.09) (8,40.42) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,55.03) (2,55.05) (3,54.96) (4,55.05)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,42.77) (6,42.96) (7,43.02) (8,43.12) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,76.57) (2,76.57) (3,76.57) (4,76.59)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,47.44) (6,47.94) (7,47.82) (8,48.21) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,93.21) (2,93.23) (3,93.23) (4,93.25) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,52.31) (6,53.12) (7,52.2) (8,53.39) 
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
        ymin=85, ymax=100,
        ytick={85,90,...,100},
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
            (1,94.74) (2,94.84) (3,94.69) (4,94.78)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,94.36) (6,94.62) (7,94.28) (8,94.59) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,95.56) (2,95.72) (3,95.54) (4,95.7)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,94.24) (6,94.64) (7,94.23) (8,94.58) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,95.99) (2,96.23) (3,95.97) (4,96.22)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,92.11) (6,93.35) (7,92.09) (8,93.36) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,94.84) (2,95.11) (3,94.82) (4,95.11) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,85.44) (6,89.07) (7,85.45) (8,89.15) 
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
        ymin=60, ymax=90,
        ytick={60,70,...,90},
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
            (1,83.27) (2,83.03) (3,83.09) (4,82.81)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,82.39) (6,82.41) (7,82.14) (8,82.3) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,84.76) (2,84.85) (3,84.73) (4,84.75)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,82.62) (6,82.72) (7,82.64) (8,82.44) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,83.65) (2,84.11) (3,83.66) (4,84.09)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,78.34) (6,79.92) (7,78.27) (8,79.8) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,76.84) (2,77.45) (3,76.77) (4,77.42) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,67.95) (6,72.77) (7,67.89) (8,72.82) 
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
            (1,93.18) (2,93.3) (3,93.11) (4,93.22)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,92.82) (6,93.13) (7,92.72) (8,93.1) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,93.6) (2,93.83) (3,93.57) (4,93.79)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,92.63) (6,93.11) (7,92.62) (8,93.02) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,92.85) (2,93.27) (3,92.83) (4,93.26)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,89.99) (6,91.51) (7,89.94) (8,91.51) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,86.97) (2,87.65) (3,86.91) (4,87.64) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,82.17) (6,86.53) (7,82.17) (8,86.6) 
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
        ymin=0, ymax=90,
        ytick={0,20,...,90},
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
            (1,79.44) (2,80.89) (3,79.27) (4,80.76)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,77.73) (6,79.93) (7,77.48) (8,79.89) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,81.49) (2,83.09) (3,81.41) (4,83.09)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,75.8) (6,79.21) (7,75.7) (8,79.1) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,80.95) (2,82.26) (3,80.93) (4,82.3)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,65.41) (6,72.75) (7,65.31) (8,73.12) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,76.11) (2,77.32) (3,76.03) (4,77.23) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,38.31) (6,54.94) (7,38.22) (8,55.73) 
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
        ymin=0, ymax=90,
        ytick={0,20,...,90},
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
            (1,77.44) (2,79.03) (3,77.26) (4,78.9)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,75.98) (6,78.32) (7,75.75) (8,78.27) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,78.56) (2,80.42) (3,78.47) (4,80.42)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,73.8) (6,77.42) (7,73.7) (8,77.33) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,75.96) (2,77.61) (3,75.93) (4,77.66)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,63.48) (6,71.13) (7,63.41) (8,71.54) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,65.05) (2,66.82) (3,64.93) (4,66.69) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,36.97) (6,53.88) (7,36.84) (8,54.68) 
        };
    \end{axis}
\end{tikzpicture}
```
