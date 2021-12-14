from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

#IP = '192.168.0.24'
#大学
IP = '172.20.61.72'

def show_color_osc(pos_list, reid_list):
    
    try:
        for i in range(0, len(reid_list)):
            PORT_COLOR = 10000 + reid_list[i]

            # UDPのクライアントを作る
            client = udp_client.UDPClient(IP, PORT_COLOR)

            # メッセージを作って送信する
            msg = OscMessageBuilder(address='/pos')
            msg.add_arg(pos_list[i][0])
            msg.add_arg(pos_list[i][2])
            m = msg.build()

            client.send(m)
    except Exception:
        pass
    
def change_particles(distance_list, reid_list):
    for i in range(0, len(reid_list)):
            PORT_DIS = 1100 + reid_list[i]
            # UDPのクライアントを作る
            client = udp_client.UDPClient(IP, PORT_DIS)

            # メッセージを作って送信する
            msg = OscMessageBuilder(address='/dis')
            
            msg.add_arg(distance_list[i])
            print(distance_list[i])
            m = msg.build()
             
            client.send(m)