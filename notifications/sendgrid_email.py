# using SendGrid's Python Library
# https://github.com/sendgrid/sendgrid-python
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

import json

with open('./notifications/sendgrid.json', 'r') as file:
    d = json.load(file)
    from_email = d['from_email']
    to_emails = d['to_emails']
    api_key = d['api_key']


def send_email(
    html_content, from_email=from_email, to_emails=to_emails,
    subject='Model training update'):

    message = Mail(
        from_email=from_email, to_emails=to_emails, subject=subject,
        html_content=html_content
    )

    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        print("sendgrid email status code:", response.status_code)
        # print(response.body)
        # print(response.headers)
    except Exception as e:
        print(e.message)


if __name__ == '__main__':
    send_email("test message")