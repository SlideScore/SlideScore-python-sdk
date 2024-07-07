import sys
import math
from typing import List

from bitarray import bitarray
from bitarray.util import ba2int, int2ba


class OmegaEncoder():
    encoding_options = [
        'naturalOnly', # As is, ints must be > 1
        'naturalZero', # Stored as int + one
        'integers' # Stored as mapping like [0, 1, -1, 2, -2] -> [1, 2, 3, 4, 5]
    ]

    def check_type(self, encoding_type: str):
        """Checks if the specified type is supported"""
        if encoding_type not in self.encoding_options:
            raise Exception('Invalid type, select from: ' + ', '.join(self.encoding_options))

    def prepare_encode_num_using_type(self, num: int, selected_type: str) -> int:
        """Maps the 'naturalZero' and 'integers' types to the Omega natural type, i.e. all > 1 """
        if selected_type == 'naturalOnly':
            return num
        elif selected_type == 'naturalZero':
            return num + 1
        elif selected_type == 'integers':
            # [0, 1, -1, 2, -2] -> [1, 2, 3, 4, 5]
            return abs(2 * num) + (1 if num <= 0 else 0)
        else:
            raise Exception("Don't know how to encode using: " + selected_type)

    def finalize_decode_num_using_type(self, num: int, selected_type: str) -> int:
        """Performs the inverse of the prepare function, by mapping the natural type of the omega encoding to all
        integers, or zero-based integers."""
        if selected_type == 'naturalOnly':
            return num
        elif selected_type == 'naturalZero':
            return num - 1
        elif selected_type == 'integers':
            # [1, 2, 3, 4, 5] -> [0, 1, -1, 2, -2]
            return math.ceil(num / 2 * (1 if num % 2 == 0 else -1))
        else:
            raise Exception("Don't know how to decode using: " + selected_type)


    def encode(self, raw_nums: List[int], encoding_type = str) -> bitarray:
        """Encode a list of integers into a Elias Omega encoded bitarray. 
        
        Encoding types:
            'naturalOnly': Integers must be > 1.
            'naturalZero': Integers must be > 0.
            'integers': Can use all integers, also < 0. 
                Less efficient than other encoding types
        """
        self.check_type(encoding_type)
        bits = bitarray()
        # Perform mapping if needed
        nums = map(lambda num : self.prepare_encode_num_using_type(num, encoding_type), raw_nums)

        for num in nums:
            new_bits = self.encode_raw(num, bitarray([0]))
            bits += new_bits
        return bits

    def encode_raw(self, num: int, encoded: bitarray):
        """Internal method to do one pass of Elias Omega Encoding"""
        if num < 1: raise Exception('Number lower than 1 got passed to encode_raw')
        if num == 1: return encoded # Step 2
        bit_encoded_num = int2ba(num)
    
        encoded = bit_encoded_num + encoded # Step 3
        new_num = len(bit_encoded_num) - 1 # step 4
        return self.encode_raw(new_num, encoded) # step 5

    def decode(self, encoded: bitarray, selected_type: str) -> List[int]:
        """Decode an Elias Omega encoded bitarray into a list of integers. 
        Must use the same type as used when encoding, to correctly map the integer results.
        """
        self.check_type(selected_type)

        raw_nums = []
        offset = 0
        while len(encoded) > offset:
            new_num, offset = self.decode_raw(1, encoded, offset)

            raw_nums.append(new_num)
                
        nums = [self.finalize_decode_num_using_type(num, selected_type) for num in raw_nums]
        return nums

    def decode_raw(self, num: int, encoded: bitarray, offset: int):
        """Internal method to do one pass of Elias Omega decoding"""
        encoded_slice = encoded[offset : offset + 1 + num]
        # print('offset', offset, 'end', offset + 1 + num, 'slice', encoded_slice, '[0]', encoded_slice[0])

        if encoded_slice[0] == 0: return num, offset + 1
        # If the next bit is a "1" then read it plus N more bits, and use that binary number as the new value of N.

        offset += num + 1

        new_num = ba2int(encoded_slice)
        return self.decode_raw(new_num, encoded, offset)


def stress_test_omega_encoding():
    """Testing method to see how fast the encoding and decoding is for the OmegaEncoder class above"""
    import time
    import random
    import brotli
    num_points = 100 * 1000
    random_ints = [(i % 256) + 1 for i in range(num_points)]
    t0 = time.time()
    omega_encoder = OmegaEncoder()
    omega_encoded_bits = omega_encoder.encode(random_ints, "naturalOnly")
    t1 = time.time()
    print(f'Encoding took: {(t1 - t0):.2f} seconds for {len(random_ints)} numbers in {omega_encoded_bits.nbytes} bytes')
    decoded_nums = omega_encoder.decode(omega_encoded_bits, "naturalOnly")
    print(f'Decoding took: {(time.time() - t1):.2f} seconds')
    if decoded_nums == random_ints:
        print('And the input is the same as the output')
    else:
        print('ERROR: Input and output not the same')

    t2 = time.time()
    compressed_bytes = brotli.compress(omega_encoded_bits.tobytes())
    print(f'Compressing took: {(time.time() - t2):.2f} seconds, to {len(compressed_bytes)} bytes')
    open('./omega_encoded_nums.bin', 'wb').write(omega_encoded_bits.tobytes())

if __name__ == '__main__':
    # Usage python omega_encode.py [...list of ints to encode]
    if len(sys.argv[1:]) > 0:
        passed_ints = list(map(int, sys.argv[1:]))
        omega_encoder = OmegaEncoder()
        res = omega_encoder.encode(passed_ints, 'naturalOnly')

        print(f'{res.nbytes} bytes needed to represent {len(passed_ints)} ints')
        if len(res) < 50:
            print(res)
        print(omega_encoder.decode(res, 'naturalOnly'))
    else:
        stress_test_omega_encoding()