# VirusPseAAC Dataset - Partial Abstention Charts

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
            (1,34.11) (2,34.27) (3,34.11) (4,34.27)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,28.0) (6,29.05) (7,28.01) (8,29.05) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,43.54) (2,43.62) (3,43.54) (4,43.62)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,28.05) (6,28.41) (7,28.41) (8,28.74) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,52.74) (2,52.78) (3,52.74) (4,52.78)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,24.88) (6,25.15) (7,24.76) (8,25.28) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,58.98) (2,59.06) (3,59.06) (4,59.06) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,19.52) (6,20.09) (7,19.57) (8,20.37) 
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
            (1,74.83) (2,74.83) (3,74.83) (4,74.83)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,63.26) (6,64.23) (7,64.72) (8,64.23) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,81.9) (2,81.9) (3,81.9) (4,81.9)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,63.26) (6,63.75) (7,63.26) (8,63.98) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,91.8) (2,91.8) (3,91.8) (4,91.8)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,65.25) (6,65.72) (7,64.76) (8,66.93) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,92.76) (2,93.0) (3,93.0) (4,93.0) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,55.53) (6,55.79) (7,55.78) (8,56.74) 
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
            (1,85.66) (2,85.82) (3,85.66) (4,85.82)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,84.06) (6,84.86) (7,84.22) (8,84.78) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,87.24) (2,87.41) (3,87.24) (4,87.45)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,83.26) (6,83.71) (7,83.38) (8,83.59) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,87.44) (2,87.56) (3,87.44) (4,87.56)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,80.06) (6,81.63) (7,80.1) (8,81.55) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,86.4) (2,86.52) (3,86.36) (4,86.52) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,74.72) (6,76.82) (7,74.68) (8,76.98) 
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
        ymin=30, ymax=60,
        ytick={30,40,...,60},
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
            (1,51.66) (2,51.47) (3,51.66) (4,51.65)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,51.76) (6,51.9) (7,52.74) (8,51.78) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,55.74) (2,56.08) (3,55.74) (4,56.16)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,53.84) (6,52.71) (7,53.63) (8,51.48) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,50.92) (2,50.83) (3,50.92) (4,50.83)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,49.83) (6,50.02) (7,50.0) (8,49.55) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,41.07) (2,41.07) (3,40.94) (4,41.07) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,47.16) (6,46.67) (7,47.03) (8,46.02) 
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
            (1,78.22) (2,78.41) (3,78.22) (4,78.41)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,77.77) (6,78.54) (7,78.01) (8,78.42) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,77.16) (2,77.41) (3,77.16) (4,77.47)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,76.56) (6,77.06) (7,76.62) (8,76.77) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,73.07) (2,73.31) (3,73.07) (4,73.31)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,73.21) (6,75.12) (7,73.34) (8,75.0) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,66.32) (2,66.52) (3,66.2) (4,66.52) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,68.46) (6,70.84) (7,68.38) (8,70.87) 
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
            (1,40.1) (2,41.57) (3,40.1) (4,42.04)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,36.23) (6,43.48) (7,36.23) (8,43.96) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,49.55) (2,50.51) (3,49.55) (4,50.75)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,36.03) (6,39.91) (7,36.28) (8,40.15) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,48.32) (2,48.56) (3,48.32) (4,48.56)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,26.56) (6,33.32) (7,26.32) (8,33.32) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,45.92) (2,45.92) (3,45.92) (4,45.92) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,13.3) (6,19.11) (7,13.06) (8,20.57) 
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
            (1,34.76) (2,36.34) (3,34.76) (4,36.85)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,31.83) (6,39.16) (7,32.23) (8,39.29) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,40.35) (2,41.47) (3,40.35) (4,41.73)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,32.81) (6,36.71) (7,32.9) (8,36.98) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,36.92) (2,37.19) (3,36.92) (4,37.19)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,23.27) (6,30.19) (7,23.26) (8,30.41) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,25.63) (2,25.63) (3,25.63) (4,25.63) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,11.29) (6,16.83) (7,10.8) (8,18.28) 
        };
    \end{axis}
\end{tikzpicture}
``` 