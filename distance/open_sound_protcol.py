def measure_distance_osc(distance_list):
    
    try:
        # case = comb(n, r)
        case = comb(Human_Number, 2, exact=True)
    
        diff_distance = [SOCIAL_DISTANCE for j in range(case)]

        if distance_list[2][2] == 0:
            Detected_Human_Number = 2
        elif distance_list[1][2] == 0:
            Detected_Human_Number = 1
        else:
            Detected_Human_Number = 3
            
        for i in range(0, Detected_Human_Number-1):
            
            if distance_list[i][2] == 0:
                    continue
            
            real_distance_realsense_width = distance_list[i][2]
            x1 = distance_list[i][0]

            
            d1 = distance_list[i][2]
            
            for j in range(i+1, Detected_Human_Number):
                
                if distance_list[j][2] == 0:
                    continue
                
#                real_distance_realsense_width = 0.24 * distance_list[j][2] + 452
                x2 = real_distance_realsense_width/1280*(distance_list[j][0]-640)
                #x2 = distance_list[j][0]-640
                
                d2 = distance_list[j][2]
                
                distance = ((x1 - x2)**2 + (d1 - d2)**2)**0.5
                
                if(type(distance) is complex):
                    print("complex")
                else:
                    #diff_distance[(i+j+2)%3] = distance
            
                    if i==0:
                        diff_distance[0] = min(diff_distance[0] , distance)
                  
                    if i == 1 or j == 1:
                        diff_distance[1] = min(diff_distance[1], distance)
                    
                    if j == 2:
                        diff_distance[2] = min(diff_distance[2], distance)
        
        if distance_list[2][2] == 0:
            diff_distance[2] = 0
        if distance_list[1][2] == 0:
            diff_distance[1] = 0
        
        
        #print(diff_distance)
        
        #print(distance_list[0][3])
        
        #TouchDesignerへ
        
        for i in range(0, 3):
            PORT = 1100 + distance_list[i][3]
            # UDPのクライアントを作る
            client = udp_client.UDPClient(IP, PORT)

            # メッセージを作って送信する
            msg = OscMessageBuilder(address='/dis')
            
            msg.add_arg(diff_distance[i])
                
            m = msg.build()
             
            client.send(m)
            
    
    except (TypeError, NameError):
        print(TypeError)
        print(NameError)