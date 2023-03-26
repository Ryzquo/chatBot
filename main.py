#!/user/bin/env python
# -*- coding:utf-8 -*-

import pyaudio
import requests
import pyttsx3

from ppasr.predict import PPASRPredictor


class chatBot:
    def __init__(self, 
                 botName="chatBot", 
                 address="http://127.0.0.1",
                 port="填代理端口", 
                 apiKey="填openai api key", 
                 interval_time=0.5
        ):
        self.botName = botName
        
        # 设置代理服务器的地址和端口
        self.proxies = {
            f"http": f"{address}:{port}",
            f"https": f"{address}:{port}"
        }
        # ChatGPT API的URL
        self.url = "https://api.openai.com/v1/chat/completions"
        # 请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {apiKey}"
        }
        
        # 语音播报
        self.botSay = pyttsx3.init()
        # self.botSay.setProperty('rate', 200)
        # self.botSay.setProperty('volume', 1)
        voices = self.botSay.getProperty('voices')
        self.botSay.setProperty('voice',voices[0].id)
        
        # 录音
        self.interval_time = interval_time
        self.CHUNK = int(16000 * self.interval_time)
        # self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.FORMAT, 
            channels=self.CHANNELS, rate=self.RATE, 
            input=True
        )
        
        # 加载语音识别
        self.predictor = PPASRPredictor(
            model_tag='conformer_streaming_fbank_wenetspeech'
        )
        
    def send_request(self, messages):
        # 请求参数
        parameters = {
                      "model": "gpt-3.5-turbo-0301", #gpt-3.5-turbo-0301
                      "messages":messages# [{"role": "user", "content": context}]
                    }
        # 发送请求
        response = requests.post(self.url, headers=self.headers, 
                                 json=parameters, 
                                 proxies=self.proxies)

        # 解析响应*
        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]
            return text
        else:
            print(response)
            return "抱歉，出现了一些错误。"

    def start_conversation(self, messages, botName="chatBot"):
        # 初始对话
        messages.append({"role": "user", "content": "你好!"})
        response = self.send_request(messages)
        print(f"{botName}: ", response["content"], end=" ")
        # 说
        self.say(response["content"])
        print(">")
        
        count_useGpt = 0
        last_text = ""
        # 进入对话循环
        while True:
            buf = self.stream.read(self.CHUNK)
            result = self.predictor.predict_stream(
                audio_data=buf, use_pun=False, is_end=False
            )
            if result is None: continue
            score, text = result['score'], result['text']
            
            if text and (last_text == text):
                print(f"{last_text} > {text}")
                count_useGpt += 1
            else:
                count_useGpt = 0
            
            if count_useGpt>2:
                print(f">> {text}")
                
                user_message={"role": "user", "content": text}
                # 将用户输入添加到messages中
                messages.append(user_message)
                response = self.send_request(messages)
                print(f"{botName}: ", response["content"], end=" ")
                self.say(response["content"])
                print(">")
                
                #将API接口返回的内容添加至messages，以用作多轮对话
                messages.append(response)
                
                # 如果API返回的内容包含"再见"，则结束对话循环
                if "再见" in text:
                    break
            
                # 重置录音
                self.stream.stop_stream()
                self.stream.close()
                self.stream = self.audio.open(
                    format=self.FORMAT, 
                    channels=self.CHANNELS, rate=self.RATE, 
                    input=True
                )
                # 重置流式识别
                self.predictor.reset_stream()
                
                last_text = ""
                text = ""
                count_useGpt = 0
                
            last_text = text
            
    def say(self, content):
        self.botSay.say(content)
        self.botSay.runAndWait()
        self.botSay.stop()

if __name__ == "__main__":
    # 初始化
    messages=[{"role": "system", "content": "你是一个聊天机器人"}] 
    sEar = chatBot(botName="chatBot", interval_time=1)
    sEar.start_conversation(messages)
