#include <stdio.h>
#include <ctype.h>

#define BUFFER_SIZE 256

int main(int argc, char** argv)
{
    if (argc < 4) {
        printf("usage : %s scale input output\n", argv[0]);
        return 1;
    }

    double scale;
    sscanf(argv[1], "%lg", &scale);

    FILE* fp = fopen(argv[2], "r");
    FILE* out = fopen(argv[3], "w");

    if (!fp) {
        printf("error : cannot open input file\n");
        return 1;
    }

    if (!out) {
        fclose(fp);
        printf("error : cannot open output file\n");
        return 1;
    }

    char buffer[BUFFER_SIZE];

    while (fgets(buffer, BUFFER_SIZE, fp))
    {
        if (buffer[0] == 'v' && isspace(buffer[1])) {
            double vertex[3];
            sscanf(buffer, "v %lg %lg %lg", &vertex[0], &vertex[1], &vertex[2]);
            vertex[0] *= scale;
            vertex[1] *= scale;
            vertex[2] *= scale;
            fprintf(out, "v %g %g %g\n", vertex[0], vertex[1], vertex[2]);
        } else {
            fputs(buffer, out);
        }
    }

    fclose(fp);
    fclose(out);
}
