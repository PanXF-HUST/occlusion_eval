import torch

def occlud_eval(im_result):
    '''
    im_result: image_result from pPose_nms/pose_nms(function),
    im_reslut{
            'keypoints': merge_pose - 0.3,
            'kp_score': merge_score,
            'proposal_score': torch.mean(merge_score) + bbox_scores_pick[j] + 1.25 * max(merge_score),
            'human_boxes': bboxes_pick[j],
            'bboxes_scores': bbox_scores_pick[j],
            'eval_score': 0,
            'coded_top':[],
            'coded_bottom':[],
            'coded_left':[],
            'coded_right':[]
            }

    return eval_score
    '''
    # final_result = im_result
    for human in im_result['result']:
        # kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        hm_bboxes_score = human['bboxes_scores']
        evalscore = human['eval_score']


        # for n in range(kp_scores.shape[0]):
        head = max(kp_scores[0:5])
        shoulder = (kp_scores[5]+kp_scores[6])/2
        # left_arm = (0.7*kp_scores[7]+0.3*kp_scores[9])
        # right_arm = (0.7*kp_scores[8]+0.3*kp_scores[10])
        left_arm = (0.1 * kp_scores[5] +0.6 * kp_scores[7] + 0.3 * kp_scores[9])
        right_arm = (0.1 * kp_scores[6] + 0.6 * kp_scores[8] + 0.3 * kp_scores[10])
        hip = (kp_scores[11]+kp_scores[12])/2
        body = (shoulder + hip)/2
        left_leg =(0.15*kp_scores[11]+0.6*kp_scores[13]+0.25*kp_scores[15])
        right_leg = (0.15*kp_scores[12]+0.6*kp_scores[14]+0.25*kp_scores[16])

        eval_part_score = (head,shoulder,left_arm,right_arm,hip,body,left_leg,right_leg)

        occlued_part = occlued_part_list(eval_part_score)
        human['occlude_part'] = occlued_part

        # define human = 0.15*head + 0.25*shoulder + 0.075*(left_arm + right_arm) + 0.1*(left_leg + right_leg)
        body_score = 0.15*head + 0.25*shoulder + 0.25*hip + 0.075*(left_arm + right_arm) + 0.1*(left_leg + right_leg)

        evalscore = 0.4*hm_bboxes_score + 0.6*body_score
        human['eval_score'] = evalscore
        # im_result.append({
        #     'eval_score': evalscore
        # })


    # print('with occlusion eval score the result is:',im_result)
    # print('______________________________________________________')
    return im_result

def occlued_part_list(part_scores):
    part_name = ['head','shoulder','left_arm','right_arm','hip','body','left_leg','right_leg']
    occlued_part = []
    for i in range(8):
        if part_scores[i] < 0.2:
            occlued_part.append(part_name[i])

    return occlued_part

