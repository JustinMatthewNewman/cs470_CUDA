//gcc compare_image.c -o compare
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  int cflag = 0;
  int aflag = 0;
  int opt;
  while ((opt = getopt(argc, argv, "ca")) != -1) {
    switch (opt) {
      case 'c':
        cflag = 1;
        break;
      case 'a':
        aflag = 1;
        break;
      default:
        fprintf(stderr, "Usage for Image Comparison Percent: %s [-c] file1 file2\n", argv[0]);
        fprintf(stderr, "Usage for Image Combination: %s [-a] file1 file2 outputFile\n", argv[0]);
        return 1;
    }
  }

  char command[256];
  if (cflag) {
    sprintf(command, "compare -metric AE -fuzz 10%% %s %s null: 2>&1", argv[optind], argv[optind+1]);
    int result = system(command);
    float diff;
    if (sscanf(command, "%f", &diff) == 1) {
      printf("Percentage Difference: %d %%\n", (int)diff);
    } else {
      printf("Failed to calculate percentage difference.\n");
    }
  } else if (aflag) {
    sprintf(command, "composite -gravity center %s %s %s", argv[optind], argv[optind+1], argv[optind+2]);
    int result = system(command);
  }
  return 0;
}

