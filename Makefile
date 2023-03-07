default: serial

serial: serial.c
	gcc -o serial serial.c

clean:
	rm -f serial