GIT_ROOT := $(shell git rev-parse --show-toplevel)

ex1-opencl.json: main.fut
	futhark bench --backend=opencl \
		--json opencl.json \
		$<

ex1-c.json: main.fut
	futhark bench --backend=c \
		--json c.json \
		$<

clean: 
	rm -rf ./main ./main.c ./data ./*.json

plot:
	python $(GIT_ROOT)/plot.py prefix 10 10
