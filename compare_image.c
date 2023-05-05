//gcc compare_image.c -o compare
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  int cflag = 0;
  int aflag = 0;
  int bflag = 0;
  int opt;
  while ((opt = getopt(argc, argv, "cba")) != -1) {
    switch (opt) {
      case 'c':
        cflag = 1;
        break;
      case 'a':
        aflag = 1;
        break;
      case 'b':
        bflag = 1;
        break;
      default:
        fprintf(stderr, "Usage for Image Comparison: %s [-c] file1 file2 file3\n", argv[0]);
        fprintf(stderr, "Usage for Image Combination: %s [-a] file1 file2 outputFile\n", argv[0]);
        return 1;
    }
  }

  char command[256];
  if (cflag) {
    sprintf(command, "compare %s %s %s: 2>&1", argv[optind], argv[optind+1], argv[optind+1]);
    printf("%s", command);
    system(command);
  } else if (aflag) {
    sprintf(command, "composite -gravity center %s %s -blend 50x50 %s", argv[optind], argv[optind+1], argv[optind+2]);
    int result = system(command);
  } else if (bflag) {
    sprintf(command, "composite -gravity center %s %s %s", argv[optind], argv[optind+1], argv[optind+2]);
    int result = system(command);

  }
  return 0;
}

