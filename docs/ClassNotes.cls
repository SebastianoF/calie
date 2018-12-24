% ClasseArticolo.cls Versione personale basata su quella di...
%
% LaTeX class for articles published in the Journal of Graph
% Algorithms and Applications, http://jgaa.info
%
% Impunemente implementata da JL per le sue esigenze! La maggio parte della roba risulta quindi essere inutile...!
%
% Based on previous package jga-article.sty file


\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ClassNotes}[2012/02/28 v1.0 LaTeX class for my article]

\DeclareOption*{%
\PassOptionsToClass{\CurrentOption}{article}%
}

\ProcessOptions\relax

\LoadClass[twoside]{article}

\RequirePackage{hyperref}
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}


%%%
\usepackage[english]{babel}
\usepackage{euscript}
\usepackage[utf8]{inputenc}
%%%


\setlength{\oddsidemargin}{20pt}
\setlength{\evensidemargin}{20pt}
\setlength{\marginparwidth}{-10pt}
\setlength{\textwidth}{425pt}


%%Utile da modificare
\newcommand{\ShortIssueNo}{UCL, GIFT-Surg project}
\newcommand{\IssueNo}{GIFT-Surg project}

%%Utile da modificare
\newcommand{\IssueDue}{
  \vspace{0.2in}
  %\hrule
  \begin{center}
    %\parbox{3.25in}{%
      {\normalsize\sf \today } \\
      {\large\sf Version Notes 0.0} 
      
      %{\normalsize\sf  }

    %}
  \end{center}
  %\hrule
  \vspace{0.15in}

  }


\newcommand{\HeadingAuthorInfo}{}
\newcommand{\HeadingAuthor}[1]{\renewcommand{\HeadingAuthorInfo}{#1}}
\newcommand{\HeadingTitleInfo}{}
\newcommand{\HeadingTitle}[1]{\renewcommand{\HeadingTitleInfo}{#1}}

%Ridefinizione per il quoziente!
\def\quotient#1#2{%
    \raise1ex\hbox{$#1$}\Big/\lower1ex\hbox{$#2$}%
}



%%%
\fancyhead{} % clear all fields
\fancyhead[LE]{\thepage~~~~\sf \HeadingTitleInfo\/}  %\textit{}
\fancyhead[RO]{ ~~ \textrm{\HeadingAuthorInfo}~~~~\thepage}  % {\sf Statistical Study Proposal},
\fancyfoot[C]{}
\renewcommand{\headrulewidth}{0.0pt}
\renewcommand{\footrulewidth}{0.0pt}
%%%

\newtoks\auth@toks
\newtoks\authpdf@toks
\newtoks\mail@toks
\newtoks\affil@toks
\newtoks\title@toks
\newtoks\titlepdf@toks
\newtoks\history@toks
\newtoks\editor@toks
\newtoks\ack@toks


\newcounter{affil@count}
\renewcommand{\theaffil@count}{\textsuperscript{\arabic{affil@count}}}

\newcounter{author@count}
\setcounter{author@count}{0}

\renewcommand{\author}[3][]{%
  \if!#1!%
    \auth@toks=\expandafter{\the\auth@toks\@author{#2}}%
    \mail@toks=\expandafter{\the\mail@toks\@email{#2}{#3}}%
  \else%
    \auth@toks=\expandafter{\the\auth@toks\@authoraff{#1}{#2}}%
    \mail@toks=\expandafter{\the\mail@toks\@email{#2}{#3}}%
  \fi%
  \ifnum \value{author@count} > 0%
        \authpdf@toks=\expandafter{\the\authpdf@toks{, #2}}%
    \else%
        \authpdf@toks=\expandafter{\the\authpdf@toks{#2}}%
    \fi%
    \stepcounter{author@count}%
}


\renewcommand{\title}[1]{
  \title@toks=\expandafter{\the\title@toks\@title{#1}}
  \titlepdf@toks={#1}
  }

\renewcommand{\@title}[1]{
    \begin{center}
        \Large\bf \par#1
    \end{center}
    }    

\renewcommand{\@author}[1]{
    {\large{\emph{#1}}}
}
  
\newcommand{\@authoraff}[2]{
    {\large{\emph{#2$\,$}}\formataffilmark{#1}}
}

\newcommand\@marksep{,}
\newcommand\displaysep{$^,$}
\newcommand\refentry[1]{\ref{#1}}


\newcommand\formataffilmark[1]{%
  \let\@marksep\@refsepinit
  \@for\@ii:=#1\do{%
    \@marksep
    \refentry{\@ii}%
  }%
}
\def\@refsepinit{\let\@marksep\displaysep}

\newcommand{\@email}[2]{
    \footnotesize{\href{mailto:#2}{\textsf{#2}} }
}

\newcommand{\@oneaffil}[1]{
\centering%
  {\par#1}%
}%

\newcommand{\@affil}[2]{%
\centering%
  {\par\ref{#1}#2}%
}%

\newcounter{history@count}
\setcounter{history@count}{0}

\newcommand{\submitted}[1]{
        \stepcounter{history@count}%
        \history@toks=\expandafter{\the\history@toks\@history{Submitted}{#1}}%
}

\newcommand{\reviewed}[1]{
        \stepcounter{history@count}%
        \history@toks=\expandafter{\the\history@toks\@history{Reviewed}{#1}}%
}

\newcommand{\revised}[1]{
        \stepcounter{history@count}%
        \history@toks=\expandafter{\the\history@toks\@history{Revised}{#1}}%
}

\newcommand{\accepted}[1]{
        \stepcounter{history@count}%
        \history@toks=\expandafter{\the\history@toks\@history{Accepted}{#1}}%
}

\newcommand{\final}[1]{
        \stepcounter{history@count}%
        \history@toks=\expandafter{\the\history@toks\@history{Final}{#1}}%
}

\newcommand{\published}[1]{
        \stepcounter{history@count}%
        \history@toks=\expandafter{\the\history@toks\@history{Published}{#1}}%
}

\newcommand{\type}[1]{
        \stepcounter{history@count}%
        \editor@toks=\expandafter{\the\editor@toks\@editor{Article type}{#1}}%
}

\newcommand{\editor}[1]{
        \stepcounter{history@count}%
        \editor@toks=\expandafter{\the\editor@toks\@editor{Communicated by}{#1}}%
}

\newcommand{\historysize}{\fontsize{8}{9.5}\selectfont}

\newcommand{\@history}[2]{
\parbox[c][0.7cm]{2.2cm}{\begin{center}\historysize{\textsf{#1: }\par #2}\end{center}}
}

\newcommand{\@editor}[2]{
\parbox[c][0.7cm]{5cm}{\begin{center}\historysize{\textsf{#1: }\par #2}\end{center}}
}

\newcommand{\Ack}[1]{%
    \ack@toks=\expandafter{\the\ack@toks\@ack{#1}}%
}%

\newcommand{\@ack}[1]{%
    {\footnotesize{#1}}%
}%

\def\printauthors{\the\auth@toks}
\def\printaffiliations{\the\affil@toks}
\def\printtitle{\the\title@toks}
\def\printhistory{\the\history@toks}
\def\printeditor{\the\editor@toks}
\def\printack{\the\ack@toks}
\def\printmail{%
    \ifnum \value{author@count} > 1%
        {\footnotesize{\emph{E-mail addresses:}}\the\mail@toks}%
    \else%
        {\footnotesize{\emph{E-mail address:}}\the\mail@toks}%
    \fi%
}%

\newcommand{\history}{
\ifnum \value{history@count} > 0%
    \vfill%
    \begin{center}%
    \fbox{\parbox{0.98\textwidth}{\centering \printhistory\\%
    \printeditor}}%
    \end{center}%
\fi%
\vfill%
\hrule%
\vspace{1mm}%
\printack%

\noindent\printmail%
}%


\renewcommand{\maketitle}{%
\hypersetup{pdftitle=\the\titlepdf@toks,pdfauthor=\the\authpdf@toks}
\printtitle%
\begin{center}%
\printauthors%
\vspace{5mm}%
\printaffiliations%
\vspace{5mm}%
\end{center}%
}%
  
\newcommand{\Body}{%
    \history%
    \thispagestyle{empty} %
    \clearpage\pagestyle{fancy}%
}%
  
\newtheorem{lemma}{Lemma}
\newtheorem{theorem}{Theorem}
\newenvironment{proof}{\par\addvspace\topsep\noindent
{\bf Proof:} \ignorespaces }{\qed}
\newcommand{\qed}{\hspace*{\fill}$\Box$\ifmmode\else\par\addvspace\topsep\fi}





















%%%%%%%%%%%%%%%%%%%%%%%%%%% Definizione teoremi: $%%%%%

\newtheorem{definition}{Definition}[section]
\newtheorem{example}{Example}[section]
%\newtheorem{lemma}{Lemma}[section]
\newtheorem{observation}{Observation}[section]
\newtheorem{quest}{Question}[section]
\newtheorem{theo}{Theorem}[section]
\newtheorem{prop}{Property}[section]
\newtheorem{coroll}[theo]{Corollary}
\newtheorem{algo}{Algorithm}


