import glob
import json
import os
import re
import time
import urllib.parse
from hashlib import sha1
from urllib.parse import urlparse, unquote

import frontmatter
import markdown
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import GetPosts, NewPost, EditPost

from logger import Logger

log = Logger()
BASE_PATH = os.getcwd()
POST_PATH = os.path.join(BASE_PATH, "_posts")
SHA1_PATH = os.path.join(BASE_PATH, ".md_sha1")
README_PATH = os.path.join(BASE_PATH, "README.md")
config_file_txt = ""

if os.path.exists(os.path.join(os.getcwd(), "diy_config.txt")):
    config_file_txt = os.path.join(os.getcwd(), "diy_config.txt")
else:
    config_file_txt = os.path.join(os.getcwd(), "config.txt")

config_info = {}

with open(config_file_txt, 'rb') as f:
    config_info = json.loads(f.read())

username = config_info["USERNAME"]
password = config_info["PASSWORD"]
xmlrpc_php = config_info["XMLRPC_PHP"]

try:
    if os.environ["USERNAME"]:
        username = os.environ["USERNAME"]

    if os.environ["PASSWORD"]:
        password = os.environ["PASSWORD"]

    if os.environ["XMLRPC_PHP"]:
        xmlrpc_php = os.environ["XMLRPC_PHP"]
except:
    log.error("无法获取github的secrets配置信息,开始使用本地变量")

url_info = urlparse(xmlrpc_php)
domain_name = url_info.netloc
wp = Client(xmlrpc_php, username, password)


def post_url(link):
    return f'https://{domain_name}/p/{link}/'


# 获取已发布文章id列表
def get_posts():
    log.info('正在获取服务器文章列表...')
    posts = wp.call(GetPosts({'post_type': 'post', 'number': 1000000000}))
    post_link_id_list = []
    for post in posts:
        post_link_id_list.append({
            "id": post.id,
            "link": unquote(post.link)
        })
    log.info(f'获取服务器文章完成，共：{len(post_link_id_list)}')
    log.info(posts[0])
    return post_link_id_list


# 创建post对象
def create_post_obj(title, content, link, post_status, terms_names_post_tag, terms_names_category):
    post_obj = WordPressPost()
    post_obj.title = title
    post_obj.content = content
    post_obj.link = link
    post_obj.post_status = post_status
    post_obj.comment_status = "open"
    post_obj.terms_names = {
        # 文章所属标签，没有则自动创建
        'post_tag': terms_names_post_tag,
        # 文章所属分类，没有则自动创建
        'category': terms_names_category
    }
    return post_obj


# 新建文章
def new_post(title, content, link, post_status, terms_names_post_tag, terms_names_category):
    post_obj = create_post_obj(title=title,
                               content=content,
                               link=link,
                               post_status=post_status,
                               terms_names_post_tag=terms_names_post_tag,
                               terms_names_category=terms_names_category)
    log.info(f'正在发布文章：{title}, 链接为：{link} ...')
    try:
        # 先获取id
        id = wp.call(NewPost(post_obj))
        log.info(f'发布文章完成：{title}')
    except:
        id = None
        log.info(f'发布文章失败：{title}')
    return id


# 更新文章
def edit_post(id, title, content, link, post_status, terms_names_post_tag, terms_names_category):
    post_obj = create_post_obj(title=title,
                               content=content,
                               link=link,
                               post_status=post_status,
                               terms_names_post_tag=terms_names_post_tag,
                               terms_names_category=terms_names_category)
    try:
        res = wp.call(EditPost(id, post_obj))
        log.info(f'更新文章成功：{title}，结果：{res}')
    except:
        id = None
        log.info(f'更新文章失败：{title}')
    return id


# 获取markdown文件中的内容
def read_md(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        post = frontmatter.load(f)
        content = post.content
        metadata = post.metadata
        log.info(f"post.metadata ===>> {post.metadata}")
    return content, metadata


def get_md_list(dir_path):
    log.info(f'正在获取本地MD文件列表：{dir_path} ...')
    md_l1 = glob.glob(f'{dir_path}/*/*.md')
    md_l2 = glob.glob(f'{dir_path}/*/*/*.md')
    md_list = md_l1 + md_l2
    log.info(f'MD文章个数：{len(md_list)}')
    return md_list


# 计算sha1
def get_sha1(filename):
    sha1_obj = sha1()
    with open(filename, 'rb') as f:
        sha1_obj.update(f.read())
    result = sha1_obj.hexdigest()
    return result


# 将字典写入文件
def write_dic_info_to_file(dic_info, file):
    dic_info["update_time"] = time.strftime('%Y-%m-%d-%H-%M-%S')
    log.info(f'记录的文章列表为：{dic_info.keys()}')
    dic_info_str = json.dumps(dic_info)
    file = open(file, 'wt', encoding='utf8')
    file.write(dic_info_str)
    file.close()
    return True


# 将文件读取为字典格式
def read_dic_from_file(file):
    file_byte = open(file, 'rt', encoding='utf8')
    file_info = file_byte.read()
    if not file_info:
        return {}
    dic = json.loads(file_info)
    file_byte.close()
    return dic


# 获取md_sha1_dic
def get_md_sha1_dic(file):
    result = {}
    if os.path.exists(file):
        result = read_dic_from_file(file)
    else:
        write_dic_info_to_file({}, file)
    return result


def update_md_sha1_dict(sha1_dict, link, sha1_value):
    sha1_dict[link] = {
        "hash_value": sha1_value,
        "link": link,
        "encode_link": urllib.parse.quote(link, safe='').lower()
    }


def post_link_id_list_2_link_id_dic(post_link_id_list):
    link_id_dic = {}
    for post in post_link_id_list:
        link_id_dic[post["link"]] = post["id"]
    return link_id_dic


def href_info(link):
    return f'<br/><br/><br/>\n\n\n\n本文永久更新地址: [{link}]({link})'


def parse_dir(path, base_path, name):
    path = path.replace(base_path, '').split(name)[0]
    path = path.strip('/')
    dirs = path.split('/')
    if len(dirs) == 1:
        return dirs[0], ''
    return dirs[0], dirs[1]


# 在README.md中插入信息文章索引信息，更容易获取google的收录
def insert_index_info_in_readme(link_id_dic):
    # 获取_posts下所有markdown文件
    md_list = get_md_list(POST_PATH)
    # 生成插入列表
    insert_info = ""
    md_maps = {}
    for md in md_list:
        d0, d1 = parse_dir(md, POST_PATH, os.path.basename(md).split(".")[0])
        if d0 not in md_maps:
            md_maps[d0] = {}
        if d1 not in md_maps[d0]:
            md_maps[d0][d1] = []
        md_maps[d0][d1].append(md)

    for d0, d1_maps in md_maps.items():
        insert_info = f'{insert_info}## {d0}{os.linesep * 2}'
        for d1, mds in d1_maps.items():
            if d1:
                insert_info = f'{insert_info}### {d1}{os.linesep * 2}'
            mds.sort(reverse=True)
            for md in mds:
                (content, metadata) = read_md(md)
                title = metadata.get("title", "")
                link = title
                url = post_url(link_id_dic[link])
                insert_info = f'{insert_info}[{title}]({url}){os.linesep * 2}'
        insert_info = f'{insert_info}{os.linesep * 2}'

    # 替换 ---start--- 到 ---end--- 之间的内容
    insert_info = "---start---\n## 目录(" + time.strftime('%Y年%m月%d日') + "更新)" + "\n" + insert_info + "---end---"

    # 获取README.md内容
    with open(README_PATH, 'r', encoding='utf-8') as f:
        readme_md_content = f.read()

    if readme_md_content:
        new_readme_md_content = re.sub(r'---start---(.|\n)*---end---', insert_info, readme_md_content)
    else:
        new_readme_md_content = insert_info

    with open(README_PATH, 'w', encoding='utf-8') as f:
        f.write(new_readme_md_content)
    return True


def main():
    # 1. 获取网站数据库中已有的文章列表
    post_link_id_list = get_posts()
    link_id_dic = post_link_id_list_2_link_id_dic(post_link_id_list)
    # 2. 获取md_sha1_dic
    # 查看目录下是否存在md_sha1.txt,如果存在则读取内容；
    # 如果不存在则创建md_sha1.txt,内容初始化为{}，并读取其中的内容；
    # 将读取的字典内容变量名，设置为 md_sha1_dic
    md_sha1_dic = get_md_sha1_dic(SHA1_PATH)

    # 3. 开始同步
    # 读取_posts目录中的md文件列表
    md_list = get_md_list(POST_PATH)

    for md in md_list:
        # 计算md文件的sha1值，并与md_sha1_dic做对比
        sha1_value = get_sha1(md)
        # 读取md文件信息
        (content, metadata) = read_md(md)
        # 获取title
        title = metadata.get("title", "")
        # 如果sha1与md_sha1_dic中记录的相同，则打印：XX文件无需同步;
        if ((title in md_sha1_dic.keys()) and
                ("hash_value" in md_sha1_dic[title]) and
                (sha1_value == md_sha1_dic[title]["hash_value"])):
            log.info(md + "无需同步")
        # 如果sha1与md_sha1_dic中记录的不同，则开始同步
        else:
            # 读取md文件信息
            (content, metadata) = read_md(md)
            # 获取title
            title = metadata.get("title", "")
            terms_names_post_tag = metadata.get("tags", domain_name)
            terms_names_category = metadata.get("categories", domain_name)
            post_status = "publish"
            link = title

            content = markdown.markdown(content + href_info(post_url(link)), extensions=['tables', 'fenced_code'])
            # 如果文章无id,则直接新建
            if not (post_url(link) in link_id_dic.keys()):
                post_id = new_post(title, content, link, post_status, terms_names_post_tag, terms_names_category)
                log.info("new_post==>>", {
                    "title": title,
                    "content": content,
                    "link": link,
                    "post_status": post_status,
                    "terms_names_post_tag": terms_names_post_tag,
                    "terms_names_category": terms_names_category
                })
            # 如果文章有id, 则更新文章
            else:
                # 获取id
                id = link_id_dic[post_url(link)]
                post_id = edit_post(id, title, content, link, post_status, terms_names_post_tag, terms_names_category)

                log.info("edit_post==>>", {
                    "id": id,
                    "title": title,
                    "content": content,
                    "link": link,
                    "post_status": post_status,
                    "terms_names_post_tag": terms_names_post_tag,
                    "terms_names_category": terms_names_category
                })

            if post_id:
                update_md_sha1_dict(md_sha1_dic, link, sha1_value)
                link_id_dic[link] = post_id

    # 4. 重建md_sha1_dic
    write_dic_info_to_file(md_sha1_dic, SHA1_PATH)
    # 5. 将链接信息写入insert_index_info_in_readme
    insert_index_info_in_readme(link_id_dic)


main()
