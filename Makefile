default: serial

serial: serial.c
	gcc serial.c -I$HOME/local/include -L$HOME/local/lib -Wl,-Bstatic -lpng -Wl,-Bdynamic -lz -lm -o serial

clean:
	rm -f serial output_serial.png