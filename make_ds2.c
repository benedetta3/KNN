#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char** argv){
    if(argc < 4){
        printf("Uso: %s N D output.ds2\n", argv[0]);
        return 1;
    }

    int32_t N = atoi(argv[1]);
    int32_t D = atoi(argv[2]);
    const char* outfile = argv[3];

    FILE* f = fopen(outfile, "wb");
    if(!f){
        perror("open");
        return 1;
    }

    // Scriviamo header N, D
    fwrite(&N, sizeof(int32_t), 1, f);
    fwrite(&D, sizeof(int32_t), 1, f);

    // Generazione valori
    srand(12345);
    float v;

    long total = (long)N * (long)D;
    for(long i = 0; i < total; i++){
        v = ((float)rand()/RAND_MAX) * 20.0f - 10.0f;
        fwrite(&v, sizeof(float), 1, f);
    }

    fclose(f);
    return 0;
}
