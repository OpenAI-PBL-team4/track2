Python 3.10.2 (tags/v3.10.2:a58ebcc, Jan 17 2022, 14:12:15) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import os

folder_path = 'D:\\wet_files'
file_names = os.listdir(folder_path)

for file_name in file_names:
    if file_name.endswith('_cleaned.txt'):
        input_file_path = os.path.join(folder_path, file_name)
        output_file_path = os.path.join(folder_path, file_name.replace('_cleaned.txt', '_tokenized.txt'))
        tokenize_file(input_file_path, output_file_path, tokenizer)
        print(f'Tokenized {file_name}')
