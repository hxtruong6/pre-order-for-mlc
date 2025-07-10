# HumanPseAAC Dataset - Partial Abstention Charts

## Metric: aabs (Average Absolute Error)

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
            (1,84.29) (2,84.29) (3,84.29) (4,84.29)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,82.56) (6,82.79) (7,82.55) (8,82.87) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,87.06) (2,87.06) (3,87.06) (4,87.06)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,79.64) (6,80.08) (7,79.7) (8,80.24) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,88.75) (2,88.75) (3,88.74) (4,88.75)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,69.97) (6,70.87) (7,69.97) (8,71.0) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,87.7) (2,87.7) (3,87.71) (4,87.71) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,45.42) (6,47.15) (7,45.67) (8,47.29) 
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
        ymin=90, ymax=100,
        ytick={90,92,...,100},
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
            (1,95.65) (2,95.65) (3,95.65) (4,95.65)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,95.27) (6,95.33) (7,95.3) (8,95.43) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,98.68) (2,98.68) (3,98.68) (4,98.68)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,96.67) (6,96.83) (7,96.68) (8,96.81) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,99.73) (2,99.73) (3,99.73) (4,99.73)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,98.0) (6,98.08) (7,98.04) (8,98.12) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,99.86) (2,99.86) (3,99.86) (4,99.86) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,98.02) (6,98.04) (7,98.2) (8,98.12) 
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
            (1,98.23) (2,98.23) (3,98.23) (4,98.23)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,97.82) (6,97.82) (7,97.79) (8,97.83) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,98.19) (2,98.19) (3,98.19) (4,98.19)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,96.86) (6,96.95) (7,96.86) (8,96.98) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,97.73) (2,97.74) (3,97.73) (4,97.74)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,94.17) (6,94.5) (7,94.16) (8,94.51) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,96.12) (2,96.14) (3,96.12) (4,96.14) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,84.68) (6,86.13) (7,84.73) (8,86.17) 
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
            (1,55.64) (2,55.6) (3,55.64) (4,55.6)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,54.12) (6,53.82) (7,53.87) (8,53.82) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,54.52) (2,54.47) (3,54.55) (4,54.47)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,50.91) (6,50.88) (7,50.87) (8,50.97) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,48.34) (2,48.34) (3,48.37) (4,48.36)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,45.29) (6,45.86) (7,45.41) (8,45.76) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,38.88) (2,38.84) (3,38.91) (4,38.86) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,36.51) (6,37.48) (7,36.5) (8,37.39) 
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
            (1,88.68) (2,88.68) (3,88.68) (4,88.68)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,87.45) (6,87.31) (7,87.33) (8,87.29) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,85.97) (2,85.96) (3,85.97) (4,85.96)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,84.52) (6,84.61) (7,84.45) (8,84.63) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,79.75) (2,79.8) (3,79.74) (4,79.8)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,80.52) (6,81.08) (7,80.52) (8,81.02) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,68.41) (2,68.56) (3,68.41) (4,68.56) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,71.91) (6,73.73) (7,71.88) (8,73.74) 
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
            (1,82.39) (2,82.42) (3,82.39) (4,82.42)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,79.14) (6,79.3) (7,78.98) (8,79.43) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,80.49) (2,80.5) (3,80.49) (4,80.5)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,69.22) (6,70.09) (7,69.16) (8,70.46) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,74.92) (2,74.98) (3,74.89) (4,74.95)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,45.72) (6,48.6) (7,45.72) (8,48.66) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,64.58) (2,64.6) (3,64.62) (4,64.65) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,9.03) (6,12.17) (7,9.16) (8,12.38) 
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
        ymin=0, ymax=60,
        ytick={0,10,...,60},
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
            (1,52.43) (2,52.52) (3,52.43) (4,52.52)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,49.75) (6,49.65) (7,49.51) (8,49.57) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,49.89) (2,49.93) (3,49.89) (4,49.93)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,43.31) (6,44.08) (7,43.43) (8,44.34) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,40.39) (2,40.52) (3,40.31) (4,40.44)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,29.73) (6,32.35) (7,29.9) (8,32.38) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,27.63) (2,27.66) (3,27.7) (4,27.76) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,7.53) (6,10.43) (7,7.7) (8,10.61) 
        };
    \end{axis}
\end{tikzpicture}
``` 