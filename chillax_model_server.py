import socket
import os
import struct
import json
from chillax_models import make_depression_prediction, make_hs_prediction, make_offensive_prediction

SOCK_PATH = "/tmp/CoreFxPipe_ChillaxSocket"

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
    try:
        os.remove(SOCK_PATH)
    except OSError:
        pass
    sock.bind(SOCK_PATH)
    sock.listen()
    print("listening for connection..")
    conn, addr = sock.accept()
    while True:
        with conn:
            try:
                while True:
                    amount_expected = struct.unpack('I', conn.recv(4))[0]
                    print("amount_expected :", amount_expected)

                    message = conn.recv(amount_expected)
                    try:
                        json_message = json.loads(message)
                    except json.JSONDecodeError as er:
                        print(er)

                    # Send data
                    off_prediction = make_offensive_prediction(json_message["Message"])
                    hs_prediction = make_hs_prediction(json_message["Message"])
                    depression_prediction = make_depression_prediction(json_message["Message"])
                    is_off = False
                    is_hs = False
                    is_depression = False
                    if off_prediction[0] == 1:
                        is_off = True
                    if hs_prediction[0] == 1:
                        is_hs = True
                    if depression_prediction[0] == 1:
                        is_depression = True
                    response = json.dumps({'IsOffensive':is_off, 'IsHatespeech': is_hs, 'IsDepression' : is_depression,'Message':json_message['Message']})
                    conn.sendall(
                        struct.pack(
                            'I',
                            len(response.encode())
                        )
                        + response.encode('utf-8')
                    )
                    print("Sent message: ", response)

            except (struct.error, KeyboardInterrupt) as e:
                print(e)

            finally:
                print("listening for connection..")
                conn, addr = sock.accept()