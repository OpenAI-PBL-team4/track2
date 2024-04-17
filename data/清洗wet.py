import re

def clean_text(text):
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 将多个空格替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_wet_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                cleaned_line = clean_text(line)
                if cleaned_line:  # Ignore empty lines
                    output_file.write(cleaned_line + '\n')

# 示例：清洗第一个 WET 文件
clean_wet_file('D/data/file_0.warc.wet', '/mnt/data/file_0_cleaned.txt')
