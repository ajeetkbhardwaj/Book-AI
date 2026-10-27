import struct
import collections
import os

# Struct layout mapping:
# int32_t id        (4 bytes)
# char name[32]     (32 bytes)
# char dept[16]     (16 bytes)
# 4 bytes padding inserted by C++ to align the next 8-byte boundary
# double salary     (8 bytes)
# Total = 64 bytes
RECORD_FORMAT = '<i 32s 16s 4x d'
RECORD_SIZE = 64
FILE_PATH = 'build/row_store.bin' # Replace 'build/' depending on where you run this

if os.path.exists(FILE_PATH):
    with open(FILE_PATH, 'rb') as f:
        data = f.read()

    num_records = len(data) // RECORD_SIZE
    print(f"Total Records: {num_records}")

    print("\n--- First 5 Records ---")
    for i in range(min(5, num_records)):
        offset = i * RECORD_SIZE
        record_data = data[offset:offset + RECORD_SIZE]
        emp_id, name_b, dept_b, salary = struct.unpack(RECORD_FORMAT, record_data)
        
        name = name_b.decode('utf-8').rstrip('\x00')
        dept = dept_b.decode('utf-8').rstrip('\x00')
        print(f"ID: {emp_id} | Name: {name:<10} | Dept: {dept:<12} | Salary: ${salary:.2f}")
else:
    print(f"Database file {FILE_PATH} not found.")