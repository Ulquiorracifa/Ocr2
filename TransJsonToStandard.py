import json

path = "D:\Development\pycharm_workspace\Ocr2\\resultNormal30.json"
outPath = "D:\Development\pycharm_workspace\Ocr2\\result.standard.json"

with open(path, 'r', encoding='UTF-8') as load_f:
    load_dict = json.load(load_f)
    print(load_dict)
    print(type(load_dict))

result = []

# spec

# normal----------------
for c in load_dict:
    result.append({
        'id': c['id'],
        'fileName': c['filename'],
        'Code': c['invoice']['invoiceCode'],
        'No': c['invoice']['invoiceNo'],
        'Date': c['invoice']['invoiceDate'],
        'Amount': c['invoice']['invoiceAmount'],
        'Verify': c['invoice']['verifyCode']
    })

# result = json.dumps(result).encode().decode("unicode-escape")

with open(outPath, "w", encoding='UTF-8') as f:
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    # json_str = json.dumps(json_str, indent=4)
    f.write(json_str + '\n')
    print("加载入文件完成...")
