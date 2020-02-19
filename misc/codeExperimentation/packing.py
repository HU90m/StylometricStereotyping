import numpy as np
from struct import pack, unpack

a = np.float64(23427)
b = np.float64(231)

c = a/b

vector = [c, c/b, c, c/b]

# standard double which is 4 bytes
x = pack('>d', c)
# machine double which is 8 bytes
y = pack('d', c)


packed_vector = pack(f'{len(vector)}d', *vector)

with open("output.dat", "wb") as output_file:
    output_file.write(packed_vector)

with open("output.dat", "rb") as input_file:
    z = input_file.read(8 * len(vector))

print("Output:", vector)
print("Input:", unpack(f'{len(vector)}d', z))
