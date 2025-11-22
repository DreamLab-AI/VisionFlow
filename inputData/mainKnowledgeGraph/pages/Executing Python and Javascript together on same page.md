- Code
	- {{runpage}}
	- ```javascript
	  logseq.can_run = "java?";
	  ```
	- ```python
	  import js
	
	  def pyfun(str):
	    return js.logseq.can_run[:-1] + str + " & py"
	
	  js.logseq.kits.pyfun = pyfun
	  ```
	- ```javascript
	  logseq.can_run = logseq.kits.pyfun("script");
	  ```
	- ```python
	  js.logseq.can_run += "thon"
	  ```
	- ```javascript
	  alert("Logseq can edit & run: clojure & " + logseq.can_run + " & r-language");
	  ```
	-