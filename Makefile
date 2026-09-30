all:
	nvcc -O2 grayscale.cu -o grayscale

run:
	./grayscale

clean:
	rm -f grayscale input.ppm output.pgm
