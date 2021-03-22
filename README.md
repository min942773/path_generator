# 소프트웨어종합설계팀 안내사항

- [pretrain된 PG모델](https://drive.google.com/file/d/1dQNxyiP4g4pdFQD6EPMQdzNow9sQevqD/view?usp=sharing)을 다운받아 '../commonsense-qa/saved_models/pretrain_generator'로 옮겨주세요.
- path들을 확인하기 편하게 하기 위하여 small_csqa라는 데이터셋을 만들어두었습니다.
- 해당 데이터는 google drive에 추가하도록 하겠습니다. (오늘내로)
- 아래의 코드를 그대로 실행해주세요 (small_csqa.config에 ablation으로 PG를 생성안하도록 바꾸어놓음)
- 제 [노션](https://www.notion.so/Path-Generator-90508348ed7a4123874ad41925c3206b)에 오류났던 부분 올려놓았는데 혹시 오류나면 참고해주세요.
- 그 외 궁금한 부분은 이슈로 질문주세요!

### 1. Download Data

First, you need to download all the necessary data in order to train the model:

```bash
cd commonsense-qa
bash scripts/download.sh
```

### 2. Preprocess

To preprocess the data, run:

```bash
python preprocess.py
```

### 3. Using the path generator to connect question-answer entities 
(Modify ./config/path_generate.config to specify the dataset and gpu device)

```bash
./scripts/run_generate.sh
```

### 4. Commonsense QA system training
```bash
bash scripts/run_main.sh ./config/small_csqa.config
```
Training process and final evaluation results would be stored in './saved_models/' 




