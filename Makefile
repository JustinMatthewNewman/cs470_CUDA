default: serial

serial: serial.c
	gcc -o serial serial.c -lm

clean:
	rm -f serial output_serial.png