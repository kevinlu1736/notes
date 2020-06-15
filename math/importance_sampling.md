<style>
	.color-important {
		color: red;
	}
</style>
## Importance Sampling

If:<br/>
$ \mathop{\mathbb{E}}_f[x] = \sum\limits_x x\ f(x) $ 

$ \mathop{\mathbb{E}}_g[x] = \sum\limits_x x\ g(x) $ 

Then:
$ \mathop{\mathbb{E}}_g[x] = \sum\limits_x x\ \dfrac{g(x)f(x)}{f(x)} 
													 = \dfrac{g(x)}{f(x)} \sum\limits_x x\ f(x)
													 = \dfrac{g(x)}{f(x)} \mathop{\mathbb{E}}_f[x]$  


<div class="color-important">
This derivation tells that we can estimate the expectation for g from what we know about f.<br/>
This is applicable to discrete distributions.
</div>


For any unnormalized distribution:

$\mathop{\mathbb{E}}_g[x] = \sum\limits_x x\ \dfrac{h(x)}{z} 
                          = \sum\limits_x x\ \dfrac{h(x)f(x)}{zf(x)}
													= \dfrac{1}{z} \sum\limits_x x\ f(x) \dfrac{h(x)}{f(x)}$

If all x is just 1, 
$\mathop{\mathbb{E}}_g[1] = 1$

$\implies 1 = \sum\limits_x \dfrac{h(x)}{z} = 1$

$\implies z = \sum\limits_x h(x) = \sum\limits_x f(x)\dfrac{h(x)}{f(x)}$

$\implies z = \dfrac{1}{n}\sum\limits_{i=1}^n\ \dfrac{h(x)}{f(x)} = \overline w$ (normalization factor)

$\implies \mathop{\mathbb{E}}_g[x] = \dfrac{\overline{xw}}{\overline w}$
	   
	

