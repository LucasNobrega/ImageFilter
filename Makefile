prepare:
	rm -rf build
	mkdir build

all:
	cd build; cmake .. ; make

run:
	cd build/app; ./ImageFilter_CUDA