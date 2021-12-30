import re

# 读取内容
def load_data(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read()
    return data

# 正则获取每个人的信息
def get_info(data):
    obj_lx = re.compile('\(1738455198\)(.*?)2021-', re.S)
    obj_ak = re.compile('\(365314956\)(.*?)2021-', re.S)
    obj_bb = re.compile('\(379763326\)(.*?)2021-', re.S)
    obj_cr = re.compile('\(1224673159\)(.*?)2021-', re.S)
    obj_bob = re.compile('\(1341770220\)(.*?)2021-', re.S)
    obj_holy = re.compile('\(1015796089\)(.*?)2021-', re.S)

    holy_content = obj_holy.findall(data)
    lx_content = obj_lx.findall(data)
    ak_content = obj_ak.findall(data)
    bb_content = obj_bb.findall(data)
    cr_content = obj_cr.findall(data)
    bob_content = obj_bob.findall(data)

    all_content = holy_content+lx_content+ak_content+bb_content+cr_content+bob_content

    return holy_content, lx_content, ak_content, bb_content, cr_content, bob_content, all_content

# 对数据集进行处理，手动过滤一些不要的信息
def process_data(content):
    target = content
    word_list = []

    for words in target:
        # 先手动对信息进行处理，去除不要的信息
        filtered_word = words.replace('\n', '').replace('[图片]', '')
        if len(filtered_word) < 100 and len(filtered_word) >=1:
            word_list.append(filtered_word)

    return word_list

data = load_data('./data/聊天记录.txt')
holy_content, lx_content, ak_content, bb_content, cr_content, bob_content, all_content = get_info(data)
word_list = process_data(lx_content)

