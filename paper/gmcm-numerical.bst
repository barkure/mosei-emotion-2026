%% SPDX-License-Identifier: MIT
%% Independently written, small BibTeX style for the reference categories
%% in the project: book, journal article, conference paper, web resource.
%% No SORT: numbering follows first citation.
%% Project subset of GB/T 7714-2025: journal/conference/preprint entries.
%% Not a general-purpose implementation for all document types.
ENTRY
  { author editor title edition address publisher year pages
    journal booktitle volume number url urldate eprint language date }
  {} {}
INTEGERS { name.index name.count entry.count }
STRINGS { names result part field.name }

FUNCTION {format.names}
{ 'names :=
  "" 'result :=
  #1 'name.index :=
  names num.names$ 'name.count :=
  { name.index name.count > { #0 } { name.index #3 > { #0 } { #1 } if$ } if$ }
  { names name.index "{vv~}{ll}{~f{~}}{, jj}" format.name$ 'part :=
    part "others" = { "et~al." 'part := } { } if$
    result empty$
      { part 'result := }
      { result ", " * part * 'result := }
    if$
    name.index #1 + 'name.index :=
  } while$
  name.count #3 >
    { result language empty$ { ", et~al" } { language "zh" = { ", 等" } { ", et~al" } if$ } if$ * 'result := }
    { } if$
  result
}

FUNCTION {required}
{ 'field.name :=
  duplicate$ empty$
    { pop$ "Missing " field.name * " in " * cite$ * warning$ "" }
    { }
  if$
}

FUNCTION {author.or.editor}
{ author empty$
    { editor "author/editor" required }
    { author }
  if$
  duplicate$ empty$ { } { format.names } if$
}

FUNCTION {comma.field}
{ duplicate$ empty$ { pop$ } { ", " write$ write$ } if$ }

FUNCTION {start.entry}
{ newline$
  "\bibitem{" write$ cite$ write$ "}" write$ newline$
  author.or.editor write$
  ". " write$ title "title" required write$
}

FUNCTION {end.entry}
{ "." write$ newline$ }

FUNCTION {book}
{ start.entry
  "[M]. " write$
  edition empty$ { } { edition write$ ". " write$ } if$
  address "address" required write$
  ": " write$ publisher "publisher" required write$
  year "year" required comma.field
  pages empty$ { } { ": " write$ pages write$ } if$
  end.entry
}

FUNCTION {article}
{ start.entry
  "[J]. " write$ journal "journal" required write$
  year "year" required comma.field
  volume "volume" required comma.field
  number empty$ { } { "(" write$ number write$ ")" write$ } if$
  ": " write$ pages "pages" required write$
  end.entry
}

FUNCTION {inproceedings}
{ start.entry
  "[C]//" write$ booktitle "booktitle" required write$
  volume empty$ { } { ": " write$ volume write$ } if$
  publisher empty$ { } {
    ". " write$
    address empty$ { } { address write$ ": " write$ } if$
    publisher write$
  } if$
  year "year" required comma.field
  pages empty$ { } { ": " write$ pages write$ } if$
  end.entry
}

FUNCTION {misc}
{ start.entry
  eprint empty$
    { "[EB/OL]. " write$ }
    { "[PP/OL]. " write$
      edition empty$ { } { edition write$ ". " write$ } if$
      "arXiv" write$
    } if$
  date empty$ { year write$ } { "(" write$ date write$ ")" write$ } if$
  "[" write$ urldate "urldate" required write$ "]. " write$
  "\url{" write$ url "url" required write$ "}" write$
  end.entry
}

%% AI tool disclosure: name, version, developer, release date.
FUNCTION {aitool}
{ newline$
  "\bibitem{" write$ cite$ write$ "}" write$ newline$
  author.or.editor write$
  ", " write$ title "title" required write$
  ", " write$ publisher "publisher" required write$
  ", " write$ date "date" required write$
  "." write$ newline$
}

FUNCTION {online} { misc }
FUNCTION {webpage} { misc }
FUNCTION {default.type}
{ "Unsupported type for " cite$ *
  "; use book, article, inproceedings, or misc/online, or select another style." * warning$
  start.entry
  year comma.field
  end.entry
}

READ
FUNCTION {count.entry} { entry.count #1 + 'entry.count := }
FUNCTION {initialize} { #0 'entry.count := }
EXECUTE {initialize}
ITERATE {count.entry}
FUNCTION {begin.bib}
{ preamble$ empty$ { } { preamble$ write$ newline$ } if$
  "\begin{thebibliography}{" write$
  entry.count int.to.str$ write$ "}" write$ newline$
}
EXECUTE {begin.bib}
ITERATE {call.type$}
FUNCTION {end.bib}
{ newline$ "\end{thebibliography}" write$ newline$ }
EXECUTE {end.bib}
