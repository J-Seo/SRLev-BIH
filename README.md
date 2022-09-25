# SRLev-BIH 2022
SRLev-BIH: 한국어 일반 상식 추론 및 생성 능력 평가 지표

생성 결과 및 평가 `*.json` 파일은 모두 korean_commogen 폴더에 있습니다.:
HCLT2022 논문의 평가용 `*.json` 파일은 모두 human_evaluation 폴더에 있습니다.:
```bash

- korean_commongen/
  - korean_commongen_official_test.json: 평가 데이터
  
  - quantitative_eval/: 모델 생성 데이터
    - KoGPT2_quantitative.json
    - KoBART_quantitative.json
    - mBART_quantitative.json
    - mBART_50_quantitative.json
    - mT5_small_quantitative.json
    - mT5_base_quantitative.json
    - mT5_large_quantitative.json
  
- human_evaluations/
  - integrated_42_human_labeled_test.json: SRLev-BIH 성능 측정을 위한 평가 데이터
 
- results/
  - predict_results_BERT_imitate_human.txt: BIH 모델의 추론 결과 

- checkpoint/ : 학습된 BIH 모델에 대한 체크포인트 저장소

- commonsense_score.py: SRLev-BIH 실행 

```
## 1. 가상환경 conda를 활용한 설치 방법 (python 3.7)

#### conda (예시: cuda 11.1)

```bash
$ conda create -n evaluation python=3.7
```

#### KoNLPy (우분투 기준)
```bash
$ pip install konlpy    # Python 3.x
$ sudo apt-get install curl git
$ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

#### 패키지 설치 (torch 1.6 기준)
```bash
$ pip install -r requirements.txt
```

#### :sunglasses: 각 평가 지표에 대한 설치 패키지 정보

#### SRLev
```bash
$ pip install proro  # 형태소 분절 / SRL 파싱 
$ pip install soynlp # 자모 르벤스타인 
$ pip install python-mecab-ko # mecab기반의 분절
```

#### BERT_imitates_human (BIH)

```bash
$ pip install transformers=4.21.0 # 형태소 분절 / SRL 파싱 
$ pip install datasets # 자모 르벤스타인 
$ pip install python-mecab-ko # mecab기반의 분절
$ pip install sklearn # 평가용.. 다만 이 패키지에서는 사용하지 않음.
```

#### :disappointed_relieved: 만약 cuda version 문제로 상위 버전의 torch를 사용한다면..?

아래와 같은 방법으로 특정 torch를 설치하기.

(예시)

```bash
$ pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## 2. 실행 방법

#### SRLev-BIH 평가용 데이터에 사용하기 

```bash
$ python commonsense_score.py --is_mean True --reference_file hclt2022/korean_commongen/korean_commongen_official_test.txt --hypothesis_file hclt2022/human_evaluations/integrated_42_human_labeled_test.json --model_name_or_path 'hclt2022/checkpoint' --own_task_name 'BERT_imitate_human' --do_predict True --output_dir 'hclt2022/results'

# is_mean False로 max 기반의 점수화가 기본 세팅, True로 하는 경우에는 평균 기반으로 점수화
```

#### SRLev-BIH KoGPT2에 사용하기 

```bash
$ python commonsense_score.py --is_mean True --reference_file hclt2022/korean_commongen/korean_commongen_official_test.txt --hypothesis_file hclt2022/korean_commongen/quantitative_eval/KoGPT2_quantitative.json --model_name_or_path 'hclt2022/checkpoint' --own_task_name 'BERT_imitate_human' --do_predict True --output_dir 'hclt2022/results'

# is_mean False로 max 기반의 점수화가 기본 세팅, True로 하는 경우에는 평균 기반으로 점수화
```

## Paper
Please cite the following work when using this data:
> SRL Score 2022: 한국어 일반 상식 추론을 위한 평가지표
> HCLT 2022
