<a name="top"></a>
## Table of Contents
<ul>
	<li><a href="#oop">OOP</a></li>
	<li><a href="#functions">Functions</a></li>
	<li><a href="#types">Types</a></li>
	<li><a href="#control">Control Structures</a></li>
	<li><a href="#multithreading">Multithreading</a></li>
	<li><a href="#yield>Yield</a></li>

</ul>

<a name="oop"></a>
## OOP
### Data Class

- Automatically implement hash_code, equals, toString

```kotlin
data class Animal(val name: String, val gender: String)
```
### Sealed Class

- All the subclasses are known, can't declare subclasses outside this file

```kotlin
sealed class Element
```

### Properties
#### Setter

```kotlin
private var prop: String = "..."
set (v) {
	println("New value: $v")
	field = v
}

```

#### Getter

```kotlin
private var _os: String? = null
var os: String
	get() {
		return 3
	}
```

#### Delegated Property(Lazy)

```kotlin
val os: String by lazy {
	System.getProperty("os.name")
}
```

#### Generic getter and setter
```kotlin
class Prop(var field: String) {
	operator fun getValue(thisRef: Any?, p: KProperty<*>): String {
		println("You read me")
		return field
	}
	
	operator fun setValue(thisRef: Any?, p: KProperty<*>, v: String) {
		println("You write me")
		field = v
	}
}

var p1 by Prop("initial")
```

#### Easy access to Properties

```kotlin
class Example(val a: Int, val b: String?, val c: Boolean)

fun main()	{
	val ex = Example(1, null, true)
	
	with(ex) {
		println("a = $a, b = $b, c = $c")
	}
}
```

<a href="#top">Back to Top</a>
<a name="functions"></a>

## Functions

### Default Params

```kotlin
fun getFirstWord(s: String, separator: String = " "): String {
	val index = s.indexOf(separator)
	return if (index < 0) s else s.substring(0, index)
}
```
### Extension Function and Properties
```kotlin
fun String.getFirstWord(separator: String = " "): String {
	val index = indexOf(separator)
	return if (index < 0) s else substring(0, index)
}

//OR

val String.firstWord: String
get() {
	val index = indexOf(separator)
	return if (index < 0) s else substring(0, index)
}

//use
fun main(args: Array<String>) {
	println(
		"Jane Doe".getFirstWord()
	)
}
```

### Function returned when entounter null
```kotlin
fun test2(str: String?): String? {
	str ?: return null
}
```

### Lambda
#### Optimize Performance
```kotlin
//generate code here instead of lambda
inline fun repeat(times: Int, body: (Int) -> Unit) {
	for (index in 0 until times) {
		body(index)
	}
}

fun main(args: Array<String>) {
	repeat (6) {
		println('fuck')
	}
}
```

<a name="types"></a>

## Types

### Auto Cast
```kotlin
if (e is Text) {
	sb.append(e.text)
}
```

### Map
```kotlin
val map = mapOf(
	"k1" to 1,
	"k2" to 2,
	"k3" to 3
)

for ((key, value) in map.entries) {
	println("$key -> $value")
}
```

### Conditional assignment
```kotlin
val s = if (System.currentTimeMills() % 2L == 0L) {
	"Get up!"
} else {
	"Sleep"
}
```

### Filter and Map
```kotlin
fun main(args: Array<String>) {
	val numbers = (1..100).toList()
	val list = numbers
				.filter { it % 16 == 0 }
				.also { print(it) }
				.map { "0x" + it.toString(16) } //tostring(16) convert to hex
}
```

<a href="#top">Back to Top</a>
<a name="control"></a>

## Control Structures

### Foreach and Cases
```kotlin
fun Element.extractText(): String {
	val sb = StringBuilder()
	fun extractText(e: Element) {
		when (e) {
			is Text -> sb.append(e.text)
			is Container -> e.children.forEach(::extractText)
			else -> error("Unrecognized element: $e")
		}
	}
}

fun test(e: Example) = when (e.a) {
		1, 3, 5 -> "Odd"
		2, 4, 6 -> "Even"
		else -> "Too big"
}
```

<a href="#top">Back to Top</a>
<a name="multithreading"></a>

##Multithreading

### Coroutines
```kotlin
//Multithreading implement (out of memory)
fun threads(n: Int) {
	val threads = List(n) {
		thread {
			sleep(1000L)
			println(it)
		}
	}
	
	threads.forEach {it.join()}
}

//coroutines (no out of memory)
fun coroutines(n: Int) = runBlocking {
	val jobs = List(100_000) {
		async { 
			delay(1000L)
			println(it)
		}
	}
}

fun main(args: Array<String>) = coroutines(100_000)
```

#### Suspend Function with Coroutine
```kotlin
//Eliminate the cascaded callbacks
suspend fun CallbackService.request(from: String) =
	suspendCoroutine<CallbackService.Response> {cont ->
		try{
			request(from) {r -> 
							cont.resume(r)
			}
		} catch (e: Throwable) {
			cont.resumeWithException(e)
		}
}

var r1 = s1.request(s2.name)
var r2 = r2.request(s1.name)

from (from in listOf("a", "b", "c")) {
	println(s1.request(from).message)
}
```

<a name="yield"></a>
##Yield

```kotlin
fun main(args: Array<String>) {
	val seq = buildSequence {
		var a = 1
		var b = 1
		
		while (true) {
			yield(a)
			val tmp = a
			a = b
			b += tmp
		}
	}
	
	println(
		seq.take(20).toList()
	)
}
```