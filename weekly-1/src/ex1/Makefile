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
	
