\NeedsTeXFormat{LaTeX2e}
\RequirePackage{fmtcount}
\ProvidesClass{srsdocument}[2023/03/01 SRS Document]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions

\LoadClass{report}

\RequirePackage{setspace}
\RequirePackage{titlesec}
\RequirePackage{parskip}
\RequirePackage{titling}
\RequirePackage{tabularx}
\RequirePackage{longtable}
\RequirePackage{tcolorbox}
\tcbuselibrary{skins, breakable}
\RequirePackage{keyval}
\RequirePackage{needspace}
\RequirePackage{amsthm}

% --- Definición de entornos matemáticos ---
\theoremstyle{definition}
\newtheorem{definition}{Definition}[chapter]
\newtheorem{example}{Example}[chapter]

\theoremstyle{remark}
\newtheorem*{remark}{Remark}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{proposition}{Proposition}[chapter]
% --- Definición del entorno 'requirement' ---
% auto counter: Numera los requisitos automáticamente
% number within=section: Reinicia el contador en cada sección (ej. REQ-1.1, REQ-1.2)
\newtcolorbox[auto counter, number within=part]{requirement}[2][]{%
    enhanced,
    breakable, % Permite que la caja se divida si ocupa más de una página
    colback=white, % Fondo blanco
    colframe=black!75!white, % Borde gris oscuro/negro
    colbacktitle=gray!15!white, % Fondo del título en gris muy claro
    coltitle=black, % Color del texto del título
    fonttitle=\bfseries, % Título en negrita
    title={ \space \space  REQ-\thetcbcounter:~#2}, % Formato del título: REQ-1.1: Nombre
    arc=2mm, % Bordes ligeramente redondeados
    boxrule=0.5mm, % Grosor del borde
    #1 % Permite añadir opciones extra si es necesario
}
% --- Entorno para Iteraciones de IA ---
\newtcolorbox[auto counter]{aiiteration}[1][]{%
    enhanced,
    breakable,
    colback=blue!2!white, % Fondo ligeramente azulado para diferenciarlo de los requisitos
    colframe=blue!40!black, % Borde azul oscuro
    colbacktitle=blue!10!white,
    coltitle=black,
    fonttitle=\bfseries\large,
    title={Iteración IA \#\thetcbcounter},
    arc=2mm,
    boxrule=0.5mm,
    #1
}
% --- Entorno para Casos de Uso (Estilo Grupo 7) ---
% Contador global: no se reinicia por sección, numeración continua CU-01, CU-02...
\newtcolorbox[auto counter]{usecase}[2][]{%
    enhanced,
    breakable,
    parbox=false,
    before={\par\needspace{8\baselineskip}},
    colback=white,
    colframe=black!80!white,
    colbacktitle=gray!20!white,
    coltitle=black,
    fonttitle=\bfseries,
    title={CU-\padzeroes[2]{\thetcbcounter}:~#2},
    arc=1mm,
    boxrule=0.5mm,
    #1
}
% --- Comando para estructurar el interior de un Requisito ---
\newcommand{\reqbody}[5]{
    \vspace{-2mm} % Ajuste de margen superior
    \begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash\bfseries}p{4.5cm} X}
        
        Descripción & #1 \\
        \hline
        Entradas & #2 \\
        \hline
        Procesamiento & #3 \\
        \hline
        Salidas & #4 \\
        \hline
        Excepciones y Errores & #5 \\
        
    \end{tabularx}
}

% --- Fila individual de Caso de Uso (permite corte de página entre filas, usando minipage para alinear listas) ---
\newcommand{\curow}[2]{%
    \par\noindent
    \begin{minipage}[t]{4.6cm}
        \vspace{0pt}\raggedright\bfseries #1
    \end{minipage}%
    \hfill
    \begin{minipage}[t]{\dimexpr\linewidth-4.8cm\relax}
        \vspace{0pt}#2
    \end{minipage}
    \par\nopagebreak\vspace{1.5mm}\noindent\rule{\linewidth}{0.4pt}\vspace{1.5mm}%
    \par
}
\newcommand{\curowlast}[2]{%
    \par\noindent
    \begin{minipage}[t]{4.6cm}
        \vspace{0pt}\raggedright\bfseries #1
    \end{minipage}%
    \hfill
    \begin{minipage}[t]{\dimexpr\linewidth-4.8cm\relax}
        \vspace{0pt}#2
    \end{minipage}
    \par
}

% --- Comando para los campos del Caso de Uso ---
\makeatletter
\define@key{cubody}{nombre}{\def\cu@nombre{#1}}
\define@key{cubody}{id}{\def\cu@id{#1}}
\define@key{cubody}{actores}{\def\cu@actores{#1}}
\define@key{cubody}{objetivo}{\def\cu@objetivo{#1}}
\define@key{cubody}{descripcion}{\def\cu@descripcion{#1}}
\define@key{cubody}{tipo}{\def\cu@tipo{#1}}
\define@key{cubody}{precondiciones}{\def\cu@precondiciones{#1}}
\define@key{cubody}{dependencias}{\def\cu@dependencias{#1}}
\define@key{cubody}{normal}{\def\cu@normal{#1}}
\define@key{cubody}{alternativas}{\def\cu@alternativas{#1}}
\define@key{cubody}{postcondiciones}{\def\cu@postcondiciones{#1}}
\define@key{cubody}{importancia}{\def\cu@importancia{#1}}
\define@key{cubody}{comentarios}{\def\cu@comentarios{#1}}

\newcommand{\cubody}[1]{%
    \setkeys{cubody}{nombre=N/A, id=N/A, actores=N/A, objetivo=N/A, descripcion=N/A, tipo=N/A, precondiciones=N/A, dependencias=N/A, normal=N/A, alternativas=N/A, postcondiciones=N/A, importancia=N/A, comentarios=N/A}%
    \setkeys{cubody}{#1}%
    \vspace{-2mm}%
    {\setlength{\parskip}{0pt}%
    \curow{Nombre}{\cu@nombre}%
    \curow{ID}{\cu@id}%
    \curow{Actores}{\cu@actores}%
    \curow{Objetivo}{\cu@objetivo}%
    \curow{Descripción}{\cu@descripcion}%
    \curow{Tipo}{\cu@tipo}%
    \curow{Precondiciones}{\cu@precondiciones}%
    \curow{Dependencias}{\cu@dependencias}%
    \curow{Secuencia Normal}{\cu@normal}%
    \curow{Secuencias Alternativas}{\cu@alternativas}%
    \curow{Postcondiciones}{\cu@postcondiciones}%
    \curow{Importancia}{\cu@importancia}%
    \curowlast{Comentarios}{\cu@comentarios}}%
}
\makeatother

% --- Entorno para Historias de Usuario ---
\newtcolorbox{userstory}[3][]{%
    enhanced,
    breakable,
    parbox=false,
    before={\par\needspace{8\baselineskip}},
    colback=white,
    colframe=black!80!white,
    colbacktitle=gray!20!white,
    coltitle=black,
    fonttitle=\bfseries,
    title={#2:~#3},
    arc=1mm,
    boxrule=0.5mm,
    #1
}

\makeatletter
\define@key{usbody}{idnombre}{\def\us@idnombre{#1}}
\define@key{usbody}{tarjeta}{\def\us@tarjeta{#1}}
\define@key{usbody}{prioridad}{\def\us@prioridad{#1}}
\define@key{usbody}{estimacion}{\def\us@estimacion{#1}}
\define@key{usbody}{conversacion}{\def\us@conversacion{#1}}
\define@key{usbody}{criterios}{\def\us@criterios{#1}}

\newcommand{\usbody}[1]{%
\vspace{2mm}
    \setkeys{usbody}{tarjeta=N/A, prioridad=N/A, estimacion=N/A, conversacion=N/A, criterios=N/A}%
    \setkeys{usbody}{#1}%
    \vspace{-2mm}%
    {\setlength{\parskip}{0pt}%
    \curow{Tarjeta}{\us@tarjeta}%
    \curow{Prioridad}{\us@prioridad}%
    \curow{Estimación}{\us@estimacion}%
    \curow{Conversación}{\us@conversacion}%
    \curowlast{Criterios de aceptación}{\us@criterios}}%
}
\makeatother
% Comandos obligatorios dentro de cada iteración
\newcommand{\aiprompt}[1]{\textbf{\textcolor{blue!60!black}{Prompt Empleado:}} \\ \textit{``#1''} \vspace{3mm}\\}
\newcommand{\aicontext}[1]{\textbf{\textcolor{blue!60!black}{Contexto Proporcionado:}} \\ #1 \vspace{3mm}\\}
\newcommand{\aiartifacts}[1]{\textbf{\textcolor{blue!60!black}{Artefactos Generados:}} \\ #1 \vspace{3mm}\\}
\newcommand{\aianalysis}[3]{
    \textbf{\textcolor{blue!60!black}{Análisis de Resultados:}}
    \begin{itemize}
        \item \textbf{Correctas:} #1
        \item \textbf{Parciales/Intentadas:} #2
        \item \textbf{Fallidas:} #3
    \end{itemize}
    \vspace{1mm}
}

\newcommand{\aicorrections}[1]{\textbf{\textcolor{blue!60!black}{Correcciones Aplicadas:}} \\ #1 \vspace{2mm}}
\newcommand{\version}[1]{\def\theversion{#1}}
\newenvironment{figuresrs}[1][red!10!white]{
  \begin{figure}[h!]
    \centering
    \begin{tcolorbox}[enhanced, colback=#1, colframe=red!50!black, arc=2mm, boxrule=0.5mm, halign=center, width=0.95\textwidth]
        
}{%
    \end{tcolorbox}
  \end{figure}
}

\newenvironment{figuresrswide}[1][red!10!white]{
  \begin{figure}[h!]
    \centering
    \begin{tcolorbox}[enhanced, colback=#1, colframe=red!50!black, arc=2mm, boxrule=0.5mm, halign=center, width=23.5cm, center, left=0mm, right=0mm, top=0mm, bottom=0mm]
        
}{%
    \end{tcolorbox}
  \end{figure}
}
\newcommand{\revision}[4]{%
	\hline
	#1 & #2 & #3 & #4\\}

\renewcommand{\maketitle}{%
\hypersetup{pageanchor=false}
\begin{titlepage}
	\begin{flushright}
		\rule{\textwidth}{2pt}
		\vspace{0.5in}

		\begin{bfseries}
			\Huge
			Casper\\
			\vspace{0.5in}
			\thetitle
			\vspace{0.5in}

			\LARGE Version \theversion\\
			\vspace{1in}
		\end{bfseries}

		\Large
		Author: \theauthor
		\vspace{0.25in}

		Date: \today

		\vfill
		\rule{\textwidth}{2pt}
	\end{flushright}
\end{titlepage}
\hypersetup{pageanchor=true}
}

\setlength{\parindent}{0.25in}

\titleformat{\chapter}[hang]{\LARGE\bfseries}{\thechapter}{0.5em}{}
\titlespacing*{\chapter}{0pt}{-24pt}{0pt}

\onehalfspacing
