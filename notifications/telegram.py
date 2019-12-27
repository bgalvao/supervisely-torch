# test bot
import requests
import json


with open('./notifications/telegram.json', 'r') as f:
    d = json.load(f)
    token = d['token']
    chat_id = d['chat_id']

base_url = 'https://api.telegram.org/bot'
method = '/sendMessage'

url = base_url + token + method


def send_message(text):
    # https://core.telegram.org/bots/api#sendmessage
    params = {
        'chat_id': chat_id,
        'text': text,
    }
    response = requests.post(url=url, params=params)
    print('telegram text status code:', response.status_code)
    # print(response.json())
    # print(response.headers)


if __name__ == '__main__':
    print(token, '\n', chat_id)
    # send_message("trying reading from file")