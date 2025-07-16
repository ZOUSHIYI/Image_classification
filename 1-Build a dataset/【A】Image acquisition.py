'''
    "本代码仅供编程科普教学、科研学术等非盈利用途。

    "请遵守国家相关法律法规和互联网数据使用规定。
  
    "请勿用于商业用途，请勿高频长时间访问服务器，请勿用于网络攻击，请勿恶意窃取信息，请勿用于作恶。
  
    "任何后果与作者无关。
'''
import os
import time
import random
import re
import requests
import urllib3
from tqdm import tqdm

urllib3.disable_warnings()

# 随机 User-Agent 列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/114.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/97.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/111.0.0.0 Safari/537.36',
]

# 百度图片请求参数中部分 header
headers = {
    'Connection': 'keep-alive',
    'Accept': 'text/plain, */*; q=0.01',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    'Referer': 'https://image.baidu.com/',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

# 创建根文件夹
if not os.path.exists('dataset'):
    os.makedirs('dataset')
    print('新建 dataset 文件夹')
else:
    print('文件夹 dataset 已存在')

# 主函数：爬取指定关键词的图片
def craw_single_class(keyword, DOWNLOAD_NUM=100):
    save_dir = os.path.join('dataset', keyword)
    os.makedirs(save_dir, exist_ok=True)
    print(f'保存路径: {save_dir}')

    count = 1  # 控制翻页
    num = 0  # 计数已下载图片
    FLAG = True  # 下载开关

    with tqdm(total=DOWNLOAD_NUM, desc=f"下载 {keyword}") as pbar:
        while FLAG:
            page = 30 * count

            # 构造请求参数
            params = {
                'tn': 'resultjson_com',
                'ipn': 'rj',
                'ct': '201326592',
                'is': '',
                'fp': 'result',
                'queryWord': keyword,
                'cl': '2',
                'lm': '-1',
                'ie': 'utf-8',
                'oe': 'utf-8',
                'adpicid': '',
                'st': '-1',
                'z': '',
                'ic': '',
                'word': keyword,
                's': '',
                'se': '',
                'tab': '',
                'width': '',
                'height': '',
                'face': '0',
                'istype': '2',
                'qc': '',
                'nc': '1',
                'pn': str(page),
                'rn': '30',
            }

            headers['User-Agent'] = random.choice(USER_AGENTS)
            try:
                response = requests.get('https://image.baidu.com/search/acjson',
                                        headers=headers,
                                        params=params,
                                        timeout=10,
                                        verify=False)
                response.raise_for_status()
                json_data = response.json().get('data')
            except Exception as e:
                print(f"[错误] 请求失败：{e}")
                break

            if not json_data:
                print("无有效数据，跳过该页")
                break

            for x in json_data:
                img_url = x.get("thumbURL")
                if not img_url:
                    continue
                img_type = x.get("type") or "jpg"
                title = x.get("fromPageTitleEnc") or "unknown"
                title_clean = re.sub(r'[\\/:*?"<>|]', "_", title)

                try:
                    headers['User-Agent'] = random.choice(USER_AGENTS)
                    resp = requests.get(img_url, timeout=10, verify=False)
                    resp.raise_for_status()
                    time.sleep(random.uniform(0.5, 1.5))  # 防封

                    file_path = os.path.join(save_dir, f"{num}_{title_clean}.{img_type}")
                    with open(file_path, 'wb') as f:
                        f.write(resp.content)

                    num += 1
                    pbar.update(1)
                    if num >= DOWNLOAD_NUM:
                        FLAG = False
                        print(f"[完成] {keyword} 图片共下载 {num} 张")
                        break

                except Exception as e:
                    print(f"[跳过] 下载失败：{e}")
                    continue

            count += 1


# 类别列表（可自定义）
class_list = [
    "兔子", "刺猬", "啄木鸟", "大猩猩", "天鹅", "小浣熊", "斑马", "昆虫", "松鼠", "树袋熊", "梅花鹿", "毛毛虫",
    "水母", "河马", "海豚", "海豹", "海马", "火烈鸟", "火鸡", "灰狼", "熊", "熊猫", "蝙蝠", "金鱼", "霍加狓",
    "马", "驴", "鬓狗", "鲨鱼", "鲸鱼", "鸽子", "鹅", "鹦鹉", "鹬", "麋鹿"
]

# 执行多类爬取
for each in class_list:
    craw_single_class(each, DOWNLOAD_NUM=100)
