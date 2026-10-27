import struct
import os

BASE_PATH = 'build/col_store' # Adjust if running from a different working directory

salary_file = f'{BASE_PATH}_salary.bin'
id_file = f'{BASE_PATH}_id.bin'
name_file = f'{BASE_PATH}_name.bin'
dept_file = f'{BASE_PATH}_department.bin'

if os.path.exists(salary_file) and os.path.exists(id_file) and os.path.exists(name_file) and os.path.exists(dept_file):
    with open(id_file, 'rb') as f: id_data = f.read()
    with open(name_file, 'rb') as f: name_data = f.read()
    with open(dept_file, 'rb') as f: dept_data = f.read()
    with open(salary_file, 'rb') as f: salary_data = f.read()

    num_records = len(id_data) // 4
    print(f"Total Records: {num_records}")

    print("\n--- First 5 Records (Reconstructed from Columns) ---")
    for i in range(min(5, num_records)):
        emp_id = struct.unpack('<i', id_data[i*4:(i+1)*4])[0]
        name = name_data[i*32:(i+1)*32].decode('utf-8').rstrip('\x00')
        dept = dept_data[i*16:(i+1)*16].decode('utf-8').rstrip('\x00')
        salary = struct.unpack('<d', salary_data[i*8:(i+1)*8])[0]
        print(f"ID: {emp_id} | Name: {name:<10} | Dept: {dept:<12} | Salary: ${salary:.2f}")
else:
    print("Database column files not found. Run the C++ program first.")
