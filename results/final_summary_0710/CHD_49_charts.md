# CHD_49 Dataset - Partial Abstention Charts

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
            (1,85.89) (2,85.89) (3,85.89) (4,85.89)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,84.12) (6,84.12) (7,84.12) (8,84.12) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,88.89) (2,88.89) (3,88.89) (4,88.89)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,81.89) (6,81.89) (7,81.89) (8,81.89) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,90.89) (2,90.89) (3,90.89) (4,90.89)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,76.89) (6,76.89) (7,76.89) (8,76.89) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,91.89) (2,91.89) (3,91.89) (4,91.89) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,66.89) (6,66.89) (7,66.89) (8,66.89) 
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
            (1,96.89) (2,96.89) (3,96.89) (4,96.89)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,96.89) (6,96.89) (7,96.89) (8,96.89) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,98.89) (2,98.89) (3,98.89) (4,98.89)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,98.89) (6,98.89) (7,98.89) (8,98.89) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,99.89) (2,99.89) (3,99.89) (4,99.89)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,99.89) (6,99.89) (7,99.89) (8,99.89) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,99.89) (2,99.89) (3,99.89) (4,99.89) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,99.89) (6,99.89) (7,99.89) (8,99.89) 
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
            (1,98.89) (2,98.89) (3,98.89) (4,98.89)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,98.89) (6,98.89) (7,98.89) (8,98.89) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,98.89) (2,98.89) (3,98.89) (4,98.89)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,98.89) (6,98.89) (7,98.89) (8,98.89) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,98.89) (2,98.89) (3,98.89) (4,98.89)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,98.89) (6,98.89) (7,98.89) (8,98.89) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,98.89) (2,98.89) (3,98.89) (4,98.89) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,98.89) (6,98.89) (7,98.89) (8,98.89) 
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
        ymin=50, ymax=70,
        ytick={50,55,...,70},
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
            (1,58.89) (2,58.89) (3,58.89) (4,58.89)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,58.89) (6,58.89) (7,58.89) (8,58.89) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,58.89) (2,58.89) (3,58.89) (4,58.89)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,58.89) (6,58.89) (7,58.89) (8,58.89) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,58.89) (2,58.89) (3,58.89) (4,58.89)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,58.89) (6,58.89) (7,58.89) (8,58.89) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,58.89) (2,58.89) (3,58.89) (4,58.89) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,58.89) (6,58.89) (7,58.89) (8,58.89) 
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
            (1,96.89) (2,96.89) (3,96.89) (4,96.89)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,96.89) (6,96.89) (7,96.89) (8,96.89) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,96.89) (2,96.89) (3,96.89) (4,96.89)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,96.89) (6,96.89) (7,96.89) (8,96.89) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,96.89) (2,96.89) (3,96.89) (4,96.89)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,96.89) (6,96.89) (7,96.89) (8,96.89) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,96.89) (2,96.89) (3,96.89) (4,96.89) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,96.89) (6,96.89) (7,96.89) (8,96.89) 
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
            (1,98.89) (2,98.89) (3,98.89) (4,98.89)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,98.89) (6,98.89) (7,98.89) (8,98.89) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,98.89) (2,98.89) (3,98.89) (4,98.89)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,98.89) (6,98.89) (7,98.89) (8,98.89) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,98.89) (2,98.89) (3,98.89) (4,98.89)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,98.89) (6,98.89) (7,98.89) (8,98.89) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,98.89) (2,98.89) (3,98.89) (4,98.89) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,98.89) (6,98.89) (7,98.89) (8,98.89) 
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
        ymin=50, ymax=100,
        ytick={50,60,...,100},
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
            (1,96.89) (2,96.89) (3,96.89) (4,96.89)
        };
        \addplot[red, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,96.89) (6,96.89) (7,96.89) (8,96.89) 
        };
        
        % Noisy level = 0.1
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,96.89) (2,96.89) (3,96.89) (4,96.89)
        };
        \addplot[blue, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,96.89) (6,96.89) (7,96.89) (8,96.89) 
        };
        
        % Noisy level = 0.2
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,96.89) (2,96.89) (3,96.89) (4,96.89)
        };
        \addplot[green!60!black, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,96.89) (6,96.89) (7,96.89) (8,96.89) 
        };
        
        % Noisy level = 0.3
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (1,96.89) (2,96.89) (3,96.89) (4,96.89) 
        };
        \addplot[cyan, dotted,mark=*,mark options={scale=0.7}] coordinates {
            (5,96.89) (6,96.89) (7,96.89) (8,96.89) 
        };
    \end{axis}
\end{tikzpicture}
``` 