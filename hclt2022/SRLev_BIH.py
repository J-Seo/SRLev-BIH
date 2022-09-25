import numpy as np
import math
import json
from hclt2022.BERT_imitate_human_inference import BERT_inference
from pororo import Pororo
from soynlp.hangle import jamo_levenshtein
srl = Pororo(task="srl", lang="ko")
col = Pororo(task="col", lang="ko")
pos = Pororo(task="pos", lang="ko")
# !pip install python-mecab-ko를 반드시 시행해야 함
# !pip install kollocate

BIH_predictions = BERT_inference()

def bilinear_interpolation(total_srl_score, bih_prediction):
    x_1 = 0
    y_1 = 0
    x_2 = 1.001
    y_2 = 4.501

    q_11 = 0

    interpolated_x = np.float(total_srl_score)
    interpolated_y = np.float(bih_prediction)

    q_21 = interpolated_x
    q_12 = interpolated_y

    q_22 = interpolated_y - ((q_21 / 1.0) - (q_12 / 4.5)) + (4.5 - q_12)

    denominator = (x_2 - x_1) * (y_2 - y_1)
    first_nu = (x_2 - interpolated_x) * (y_2 - interpolated_y) * q_11
    second_nu = (interpolated_x - x_1) * (y_2 - interpolated_y) * q_21
    third_nu = (x_2 - interpolated_x) * (interpolated_y - y_1) * q_12
    forth_nu = (interpolated_x - x_1) * (interpolated_y - y_1) * q_22

    point = (first_nu / denominator) + (second_nu / denominator) + (third_nu / denominator) + (
            forth_nu / denominator)

    return point

def adjust_zero_division(score):
    if score == 0:
        adjust_score = 0.00000001
        return adjust_score
    else:
        return score


def srlev_bih(prediction_file, reference_file, is_mean=False, total_srl_scores = list(), total_bih_scores = list(), total_srlbih_scores = list()):
    with open(reference_file, 'r', encoding='utf-8') as f, open(prediction_file, 'r', encoding='utf-8') as f1:
        refs_list = f.readlines()
        preds_list = list(f1)

        for refs, prd_jsonl, bih_prediction in zip(refs_list, preds_list, BIH_predictions):
            ref = refs.split(' = ')
            prd = json.loads(prd_jsonl)['hypothesis']
            srl_ref1, srl_ref2, srl_ref3 = srl(ref[1].replace(' [EOS]', '').strip()), \
                               srl(ref[2].replace(' [EOS]', '').strip()), \
                               srl(ref[3].replace(' [EOS]', '').strip())

            ref1_list, ref2_list, ref3_list, prd_list = list(), list(), list(), list()

            # 1번 정답 문장에 대한 SRL 튜플을 리스트에 저장
            for srl_ref1_sentence in srl_ref1:
                for srl_ref1_token in srl_ref1_sentence:
                    if type(srl_ref1_token) is not tuple:
                        continue
                    elif srl_ref1_token[1] != '-':
                        ref1_list.append(srl_ref1_token)

            # 2번 정답 문장에 대한 SRL 튜플을 리스트에 저장
            for srl_ref2_sentence in srl_ref2:
                for srl_ref2_token in srl_ref2_sentence:
                    if type(srl_ref2_token) is not tuple:
                        continue
                    elif srl_ref2_token[1] != '-':
                        ref2_list.append(srl_ref2_token)


            # 3번 정답 문장에 대한 SRL 튜플을 리스트에 저장
            for srl_ref3_sentence in srl_ref3:
                for srl_ref3_token in srl_ref3_sentence:
                    if type(srl_ref3_token) is not tuple:
                        continue
                    elif srl_ref3_token[1] != '-':
                        ref3_list.append(srl_ref3_token)


            # 생성 문장에 대한 SRL 튜플을 리스트에 저장
            for srl_sentence in srl(prd.strip()):
                for srl_token in srl_sentence:
                    if type(srl_token) is not tuple:
                        continue
                    elif srl_token[1] != '-':
                        prd_list.append(srl_token)

            ref1_srl_score, ref2_srl_score, ref3_srl_score = [], [], []
            for single_prd in prd_list:
                concate_pred = ''
                pos_single_prd = pos(single_prd[0])
                for pos_single_prd_element in pos_single_prd:
                    if pos_single_prd_element[1].startswith('N') or pos_single_prd_element[1].startswith('V') \
                            or pos_single_prd_element[1].startswith('X'):
                        concate_pred += pos_single_prd_element[0]
                    elif concate_pred != '':
                        break

                for single_ref1 in ref1_list:
                    if single_prd[1] == single_ref1[1]:

                        concate_ref1 = ''
                        pos_single_ref1 = pos(single_ref1[0])

                        for pos_single_ref1_element in pos_single_ref1:
                            if pos_single_ref1_element[1].startswith('N') or pos_single_ref1_element[1].startswith('V') \
                                    or pos_single_ref1_element[1].startswith('X'):
                                concate_ref1 += pos_single_ref1_element[0]
                            elif concate_ref1 != '':
                                break

                        score1 = jamo_levenshtein(concate_pred, concate_ref1)
                        if score1 < 0.5:
                            ref1_srl_score.append(1)
                            break

                for single_ref2 in ref2_list:
                    if single_prd[1] == single_ref2[1]:
                        pos_single_ref2 = pos(single_ref2[0])
                        concate_ref2 = ''

                        for pos_single_ref2_element in pos_single_ref2:
                            if pos_single_ref2_element[1].startswith('N') or pos_single_ref2_element[1].startswith('V') \
                                    or pos_single_ref2_element[1].startswith('X'):
                                concate_ref2 += pos_single_ref2_element[0]
                            elif concate_ref2 != '':
                                break

                        score2 = jamo_levenshtein(concate_pred, concate_ref2)
                        if score2 < 0.5:
                            ref2_srl_score.append(1)
                            break

                for single_ref3 in ref3_list:
                    if single_prd[1] == single_ref3[1]:
                        pos_single_ref3 = pos(single_ref3[0])
                        concate_ref3 = ''

                        for pos_single_ref3_element in pos_single_ref3:
                            if pos_single_ref3_element[1].startswith('N') or pos_single_ref3_element[1].startswith('V') \
                                    or pos_single_ref3_element[1].startswith('X'):
                                concate_ref3 += pos_single_ref3_element[0]
                            elif concate_ref3 != '':
                                break

                        score3 = jamo_levenshtein(concate_pred, concate_ref3)
                        if score3 < 0.5:
                            ref3_srl_score.append(1)
                            break

            # 가장 높은 점수를 보이는 케이스가 존재하면 바로 패스!
            # 생성 문장 중심으로.. 일치 정도에 대해서 평균

            num_of_prd_list = adjust_zero_division(len(prd_list))

            mean_ref1_srl_score = np.sum(ref1_srl_score) / num_of_prd_list
            mean_ref2_srl_score = np.sum(ref2_srl_score) / num_of_prd_list
            mean_ref3_srl_score = np.sum(ref3_srl_score) / num_of_prd_list

            if is_mean == True:
                total_srl_score = np.mean([mean_ref1_srl_score, mean_ref2_srl_score, mean_ref3_srl_score])
            else:
                total_srl_score = np.max([mean_ref1_srl_score, mean_ref2_srl_score, mean_ref3_srl_score])
            if math.isnan(total_srl_score) == True:
                total_srl_score = 0.0
            total_srl_scores.append(total_srl_score)
            total_bih_scores.append(bih_prediction)
            total_srlbih_scores.append(bilinear_interpolation(total_srl_score,bih_prediction))

            #print("정답1:", ref1_list)
            #print("정답2:", ref2_list)
            #print("정답3:", ref3_list)
            #print("생성:", prd_list)
            #print("각 점수:", mean_ref1_srl_score, mean_ref2_srl_score, mean_ref3_srl_score)
            #print("평균 점수:", total_srl_scores)

        total_mean_score = np.mean(total_srl_scores)
        total_bih_score = np.mean(total_bih_scores)
        total_srlbih_score = np.mean(total_srlbih_scores)
    return total_mean_score, total_bih_score, total_srlbih_score
