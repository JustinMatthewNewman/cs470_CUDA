default: serial

serial: serial.c
	gcc serial.c -lpng -lm -o serial
	gcc serial.c -lpng -lm -o encode_video/serial
	nvcc par.cu -lpng -lm -o par

clean:
	rm -f serial output_serial.png encode_video/serial par
	rm -rf test
