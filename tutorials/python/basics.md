---
description: Here is some basic information about the Python programming language.
---

# Basics

In this chapter code blocks used often. `>>>` symbol is used to denote the commands entered to the interpreter. The following line is the output of the Python interpreter.

### Print

The first command everybody should learn is `print()`. It simply prints the provided string to the console.

{% hint style="info" %}
**String:** Anything written between 2 apostrophe sign or quotation marks is called a string: Example: `'string1'`   `"string2"`
{% endhint %}

{% hint style="danger" %}
Combining and apostrophe or quotation marks will give an error. 

**Incorrect:** `'error1" "error2'`
{% endhint %}

```python
>>> print('Hello World! This is Salih!')
Hello World! This is Salih.

>>> print("I am an Applied Mathematician!")
I am an Applied Mathematician!
```

### Variables

To create a variable use  `=`   sign followed by a variable name and then assign it to any value you want. You can print the value of the variable using `print()`  function.

```python
>>> a = 5
>>> b = 'This is a string.'
>>> print(a)
5
>>> print(b)
This is a string.
```

### Data Types

Use `type()` command to check the data type of a variable.

* **Text type:** String \(`str`\)
* **Numeric Types:** Integer \(`int`\), Float \(`float`\), Complex \(`complex`\)
* **Sequence Types:** List \(`list`\), Tuple \(`tuple`\), Range \(`range`\)
* **Mapping Type \(Hash Table\):** Dictionary \(`dict`\)
* **Set Types:** Set **\(**`set`, or `frozenset`\)
* **Boolean Types:** Boolean \(`bool`\)

The list above is taken from the [w3schools.com](https://www.w3schools.com/python/python_datatypes.asp). Examples belonging to each category are given below. Putting a pound `#` symbol before typing anything converts the code into a comment line in Python.

```python
# String Example
>>> s = 'This is a string'
>>> print(type(s))
<class 'str'>

# Integer Example          # Other commands giving the same results
>>> i = 5                  # i = int(5)
>>> print(type(i))
<class 'int'>

# Float Example
>>> f = 3.0                # f = float(3)
>>> print(type(s))
<class 'float'>

# Complex Example          # j is used to denote complex i in Python
>>> c = 1 + 3j             # c = complex(1, 3)
>>> print(type(s))
<class 'complex'>

# String Example
>>> s = 'This is a string'
>>> print(type(s))
<class 'str'>

# String Example
>>> s = 'This is a string'
>>> print(type(s))
<class 'str'>

# String Example
>>> s = 'This is a string'
>>> print(type(s))
<class 'str'>

# String Example
>>> s = 'This is a string'
>>> print(type(s))
<class 'str'>

# String Example
>>> s = 'This is a string'
>>> print(type(s))
<class 'str'>

# String Example
>>> s = 'This is a string'
>>> print(type(s))
<class 'str'>
```





