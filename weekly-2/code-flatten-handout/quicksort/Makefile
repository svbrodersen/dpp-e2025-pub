clean:
	rm -rf ./quicksort-flat.c quicksort-flat

bench-json:
	futhark bench --backend=cuda --json=quicksort-flat-cuda.json quicksort-flat.fut
	futhark bench --backend=c --json=quicksort-flat-c.json quicksort-flat.fut
