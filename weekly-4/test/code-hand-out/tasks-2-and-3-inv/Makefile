validate: ops.fut test-red.fut test-hist.fut gen-vjp2-comminv.fut red-adj-ppad.fut
	futhark bench --backend=cuda test-red.fut -e validate_sop
	futhark bench --backend=cuda test-red.fut -e validate_mul
	futhark bench --backend=cuda test-hist.fut -e validate_sop
	futhark bench --backend=cuda test-hist.fut -e validate_mul

run: red-adj-ppad.fut ops.fut test-red.fut test-hist.fut gen-vjp2-comminv.fut
	futhark bench --backend=cuda test-red.fut
	futhark bench --backend=cuda test-hist.fut

datasets:
	mkdir -p data
	futhark dataset --f64-bounds=0.795:1.22 -g [100000]f64 -g f64 -b > data/red-100K.in
	futhark dataset --f64-bounds=1.0:2.0 -g [32]f64 --i64-bounds=0:31 -g [1000000]i64 --f64-bounds=0.795:1.22 -g [1000000]f64 --f64-bounds=0.8:1.19 -g [32]f64 -b > data/histo-31-1M.in
	futhark dataset --f64-bounds=1.0:2.0 -g [401]f64 --i64-bounds=0:400 -g [1000000]i64 --f64-bounds=0.795:1.22 -g [1000000]f64 --f64-bounds=0.8:1.19 -g [401]f64 -b > data/histo-401-1M.in
	futhark dataset --f64-bounds=1.0:2.0 -g [50000]f64 --i64-bounds=0:49999 -g [1000000]i64 --f64-bounds=0.795:1.22 -g [1000000]f64 --f64-bounds=0.8:1.19 -g [50000]f64 -b > data/histo-50K-1M.in
	futhark dataset --f64-bounds=0.5:2.0 -g [32]f64 --i64-bounds=0:31 -g [10000000]i64 --f64-bounds=0.795:1.22 -g [10000000]f64 --f64-bounds=0.5:2.0 -g [32]f64 -b > data/histo-31-10M.in
	futhark dataset -g [401]f64 --i64-bounds=0:400 -g [10000000]i64 --f64-bounds=0.795:1.22 -g [10000000]f64 -g [401]f64 -b > data/histo-401-10M.in
	futhark dataset -g [50000]f64 --i64-bounds=0:49999 -g [10000000]i64 --f64-bounds=0.795:1.22 -g [10000000]f64 -g [50000]f64 -b > data/histo-50K-10M.in
	futhark dataset -g [32]f64 -g [32]f64 --i64-bounds=0:31 -g [10000000]i64 -g [10000000]f64 -g [10000000]f64 -g [32]f64 -g [32]f64 -b > data/histo-31-10M-tup.in
	futhark dataset -g [401]f64 -g [401]f64 --i64-bounds=0:400 -g [10000000]i64 -g [10000000]f64 -g [10000000]f64 -g [401]f64 -g [401]f64 -b > data/histo-401-10M-tup.in
	futhark dataset -g [50000]f64 -g [50000]f64 --i64-bounds=0:49999 -g [10000000]i64 -g [10000000]f64 -g [10000000]f64 -g [50000]f64 -g [50000]f64 -b > data/histo-50K-10M-tup.in

clean:
	rm -rf *.c ops gen-vjp2-comminv test-red red-adj-ppad test-hist
	rm -rf *.actual *.expected
