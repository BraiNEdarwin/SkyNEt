This experiment is made to comfigure the switches on the dipstick so that all connections from 1 device are on hte output.
It makes a switch matrix in utils. Then it makes an integer from each row and sends that to the arduino. The arduino then recieves a string and makes 8 bytes out of it.
then it sends it via SPI.
make sure you program the arduino first.
