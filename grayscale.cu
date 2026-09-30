#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void rgbToGray(
    unsigned char *r,
    unsigned char *g,
    unsigned char *b,
    unsigned char *gray,
    int pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pixels)
    {
        gray[idx] =
            (unsigned char)(
                0.299f * r[idx] +
                0.587f * g[idx] +
                0.114f * b[idx]
            );
    }
}

int main()
{
    const int width = 1000;
    const int height = 1000;
    const int pixels = width * height;

    size_t size = pixels * sizeof(unsigned char);

    unsigned char *r = (unsigned char*)malloc(size);
    unsigned char *g = (unsigned char*)malloc(size);
    unsigned char *b = (unsigned char*)malloc(size);
    unsigned char *gray = (unsigned char*)malloc(size);

    // Generate RGB image data
    for (int i = 0; i < pixels; i++)
    {
        r[i] = rand() % 256;
        g[i] = rand() % 256;
        b[i] = rand() % 256;
    }

    // Save input RGB image as PPM
    FILE *input = fopen("input.ppm", "wb");

    fprintf(input, "P6\n%d %d\n255\n", width, height);

    for (int i = 0; i < pixels; i++)
    {
        fputc(r[i], input);
        fputc(g[i], input);
        fputc(b[i], input);
    }

    fclose(input);

    unsigned char *d_r;
    unsigned char *d_g;
    unsigned char *d_b;
    unsigned char *d_gray;

    cudaMalloc(&d_r, size);
    cudaMalloc(&d_g, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_gray, size);

    cudaMemcpy(d_r, r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, g, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (pixels + threads - 1) / threads;

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    rgbToGray<<<blocks, threads>>>(
        d_r,
        d_g,
        d_b,
        d_gray,
        pixels
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    cudaMemcpy(
        gray,
        d_gray,
        size,
        cudaMemcpyDeviceToHost
    );

    // Save grayscale image as PGM
    FILE *output = fopen("output.pgm", "wb");

    fprintf(output, "P5\n%d %d\n255\n", width, height);
    fwrite(gray, sizeof(unsigned char), pixels, output);

    fclose(output);

    printf("CUDA GPU Image Grayscale Processing\n");
    printf("Image size: %d x %d\n", width, height);
    printf("Pixels processed: %d\n", pixels);
    printf("CUDA blocks: %d\n", blocks);
    printf("Threads per block: %d\n", threads);
    printf("GPU kernel execution time: %.4f ms\n", milliseconds);

    printf("Sample grayscale values: ");

    for (int i = 0; i < 10; i++)
    {
        printf("%d ", gray[i]);
    }

    printf("\n");

    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_gray);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(r);
    free(g);
    free(b);
    free(gray);

    return 0;
}
