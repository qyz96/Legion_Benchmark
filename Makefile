all: myblas.so

.PHONY: clean

myblas.so: myblas.c
	gcc myblas.c -o myblas.so -shared -fPIC -m64 -I${MKLROOT}/include -shared -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

clean:
	rm -f myblas.so
