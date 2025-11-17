.PHONY: data clean plot all

all: plot

plot: test_process.pdf test_process_idx.pdf

test_process.pdf: benchmark
	python plot.py exercise1 test_process $(SIZES)

test_process_idx.pdf: benchmark
	python plot.py exercise1 test_process_idx $(SIZES)

benchmark: exercise1-opencl.json exercise1-c.json

SIZES = 100 1000 10000 100000 1000000 10000000

exercise1-opencl.json: exercise1.fut
	futhark bench --backend=opencl \
	              --json exercise1-opencl.json \
	              exercise1.fut

exercise1-c.json: exercise1.fut
	futhark bench --backend=c \
	              --json exercise1-c.json \
	              exercise1.fut

clean:
	rm -fr data exercise1 *.pdf *.bin *.c *.pyc *.json
