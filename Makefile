default: serial

serial: serial.c
	gcc serial.c -lpng -lm -o serial
	gcc serial.c -lpng -lm -o encode_video/serial

clean:
	rm -f serial output_serial.png encode_video/serial
	rm -rf test