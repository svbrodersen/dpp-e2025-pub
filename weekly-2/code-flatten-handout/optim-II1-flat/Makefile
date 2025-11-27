run: optimize-flat-II1.fut mk-datasets.fut util.fut
	futhark bench --backend=cuda optimize-flat-II1.fut

datasets: mk-datasets.fut
	mkdir -p data
	futhark cuda mk-datasets.fut
	echo "10i64 10000000i64 100000i64" | ./mk-datasets -e mkResults -b > data/res10x10000000x100000.out
	echo "100i64 1000000i64  10000i64" | ./mk-datasets -e mkResults -b > data/res100x1000000x10000.out
	echo "1000i64 100000i64   1000i64" | ./mk-datasets -e mkResults -b > data/res1000x100000x1000.out
	echo "10000i64 10000i64    100i64" | ./mk-datasets -e mkResults -b > data/res10000x10000x100.out
	echo "100000i64 1000i64     10i64" | ./mk-datasets -e mkResults -b > data/res100000x1000x10.out
	
clean:
	rm -rf *.c optimize-flat-II1 mk-datasets util
	rm -rf *.actual *.expected
