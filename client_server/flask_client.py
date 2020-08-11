# -*- coding:utf-8 -*-
 ######################################################
#        > File Name: flask_client.py
#      > Author: Bruce
 #     > Mail: wayne_s@126.com
 #     > Created Time: Sat Nov  2 10:50:57 CST 2019
 ######################################################

from flask import Flask, send_file

app = Flask(__name__)

@app.route('/login')
def login():
    #登录
    return send_file('templates/login.html')

@app.route('/register')
def register():
    #注册
    return send_file('templates/register.html')

@app.route('/baseinfo')
def baseinfo():
    #基本信息
    return send_file('templates/base_info.html')

@app.route('/diagnosis')
def diagnosis():
    #辅助诊疗
    return send_file('templates/Assisted_diagnosis.html')

@app.route('/reference')
def reference():
    #指标重要程度参考表
    return send_file('templates/index_reference_table.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

