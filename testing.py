            # if frame_count % FPS == 0:
            #     still_pos = []
            #     last_obj = temp_obj
            #     temp_obj = still_object_detection(temp_obj, box_pos, wiggle_percent=0.5)
            #     for old in last_obj:
            #         for new in temp_obj:
            #             iou = get_iou(new, old)
            #             if iou > 0.9:
            #                 still_pos.append(new)
            #     if len(still_obj.keys()) == 0:
            #         for i, pos in enumerate(still_pos):
            #             still_obj.update({i:(frame_count-FPS, frame_count, pos)})
            #             # cv2.imwrite(f"./output/still_obj/still_{frame_count}_{i}.jpg", frame[pos[1]: pos[3], pos[0]: pos[2]])
            #             cv2.imwrite(f"C:\\Users\\black\\Documents\\VNPT\\Object-Differences-Video\\output\\still_obj\\still_{frame_count}_{i}.jpg", frame[pos[1]: pos[3], pos[0]: pos[2]])
            #     else:
            #         for pos in still_pos:
            #             for key, value in still_obj.items():
            #                 past_still = value[2]
            #                 iou = get_iou(past_still, pos)
            #                 if iou > 0.50:
            #                     new_value = (value[0], frame_count, pos)
            #                     still_obj.update({key:new_value})
            #                     if len(still_pos) != 0:
            #                         still_pos.remove(pos)
            #         if len(still_pos) > 0:
            #             for pos in still_pos:
            #                 still_obj.update({len(still_obj.keys()):(frame_count-FPS, frame_count, pos)})  
            #                 # cv2.imwrite(f"./output/still_obj/still_{frame_count}_{i}.jpg", frame[pos[1]: pos[3], pos[0]: pos[2]])
            #                 cv2.imwrite(f"C:\\Users\\black\\Documents\\VNPT\\Object-Differences-Video\\output\\still_obj\\still_{frame_count}_{i}.jpg", frame[pos[1]: pos[3], pos[0]: pos[2]])          
            #     print(temp_obj)
            #     print(still_obj)