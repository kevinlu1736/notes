
<style>
</style>

# Differential Equations

## Table of Contents
<ul>
	<li><a href="#solve">Solve Differential Equations</a>
		<ul>
				<li><a href="#separable">Separable</a></li>
		</ul>
  </li>
	<li><a href="#miscellaneous">Miscellaneous</a></li>
</ul>

<a name="solve"></a>
## Solve Differential Equations

<a name="separable"></a>
### Separable
Example 1: <br/><br/>
$x\sqrt{1-y}\ dx - \sqrt{1-x^2}\ dy = 0$

$x\sqrt{1-y}\ dx = \sqrt{1-x^2}\ dy$

$\dfrac{x}{\sqrt{1-x^2}}dx = \dfrac{1}{\sqrt{1-y}}dy$ &nbsp;&nbsp;&nbsp;&nbsp; Lost $x=\pm 1$ and $y=1$

$\displaystyle\int \dfrac{x}{\sqrt{1-x^2}}dx = \displaystyle\int \dfrac{1}{\sqrt{1-y}}dy$

$u = 1 - x^2 \implies du = -2x dx$

$v = 1 - y \implies dv = -1 dy$

$\displaystyle\int -\dfrac{1}{2}\dfrac{1}{\sqrt{u}}du = \displaystyle\int \dfrac{-1}{\sqrt{v}}dv$

$\displaystyle\int -\dfrac{1}{2}u^{-\frac{1}{2}}du = \displaystyle\int -v^{-\frac{1}{2}}dv$

$ u^{\frac{1}{2}} = 2v^{\frac{1}{2}} + C$

$ (1-x^2)^{\frac{1}{2}} = 2(1-y)^{\frac{1}{2}} + C$

Substitute $x=\pm 1$ and $y=1$ to original ODE $\implies 0 - 0 = 0$

<a name="miscellaneous"></a>
## Miscellaneous

### Long division to solve recursive integral
??

<a name="bottom"></a>
