<a name="top"></a>
## Table of Contents
<ul>
	<li><a href="#vimscript">Vim Script</a></li>

</ul>

<a name="vimscript"></a>
##Vim Script

```vim
g:var "global
a:var "arguments
l:var "function local
b:var "buffer local
w:var "window local
t:var "tab local
v:var	"vim defined
let
unlet
unlet!

types
string start with '0' is false
[]
{}

if
elseif
endif

for var in list
endfor

for [a, b] in [[1,2],[3,4]]
endfor

while asdf
endwhile

try
catch ?
finally
endtry

"Functions
function
function! "use this if duplicated
call ~()
delete ~()
function A()
function B(arg1, ...)

command! -nargs=1  CycleCommand call CycleThatSyntax<f-args>
```


