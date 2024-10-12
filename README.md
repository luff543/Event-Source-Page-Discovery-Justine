# Event Source Page Discovery
這個專案主要是要訓練一個焦點式爬蟲模型，希望可以找到目標網站的活動來源頁面。我們使用強化學習中的DQN、Deep SARSA、A2C當作是爬蟲演算法，也比較的先前工作的Policy Gradent方法。希望能找到最適合這個任務的方法。
This project aims to train a focused web crawler model to discover event source pages of target websites. We employ reinforcement learning algorithms such as DQN, Deep SARSA, and A2C as our web crawling strategies and compare them with the previous Policy Gradient method. The goal is to find the most suitable approach for this specific task.
## Environment and Requirement
Python version: 3.8
We offer two ways to set up the environment
- pip install -r requirements.txt
- docker `ubuntu:20.04 `
```
## first build docker
docker pull tensorflow/tensorflow:2.6.0-gpu
docker run --gpus all -itd -v [your path]: --network host --name [Create your Container ID] tensorflow/tensorflow:2.6.0-
cd [your path]
pip install -r requirements.txt

## install google chrome
apt-get install -y wget
wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt-get install ./google-chrome-stable_current_amd64.deb
google-chrome --version ## check chrome exist

## run docker 
docker restart [your Container ID]
docker exec -it [your Container ID] bash
cd [your path]
python deep_q_network_step_notfix_bert_bilstm_coor_top3.py > log/dqn_log.log
ctrl + d  (leave docker)
```
## Folder Structure
```
|--- dataset_google_search
|	|--- fintunetraindata.csv
|	|--- google_search_rank_train - contain news.csv
|	|--- google_search_rank_vali - contain_news.csv
|	|--- newdata_test - remove error url.csv
|	|--- generate_dataset
|	    |--- ...
|
|--- ESPS_dataset
|   |--- test92_down.csv
|   |--- test_no_down_pre.csv
|   |--- test_url.txt
|   |--- train_down.csv
|   |--- train_url.txt
|   |--- vali_down.csv
|   |--- vali_no_down.csv
|   |--- vali_url.txt
|
|--- ESPS_model
|   |--- maxrepeat_secondrepeat_pagetitle_middlesegment__google_search_data_down_shape_change_batch12_epoch8.h5
|
|--- Preprocess
|	|--- tagtokenizemore.txt
|	|--- Tokenize.ipynb
|	|--- Tokenize.py
|	|--- vocab.txt
|
|--- html_rank_train
	|--- ...
|
|--- html_rank_vali
	|--- ...
|
|--- src
|	|--- Args.py
|	|--- PolicyDeepthEnvFixCoordinate_singlefile.py
|	|--- StringTool.py
|	|--- Tool.py
|	|--- URLTool.py
|	|--- ESPS_API.py
|	|--- ESPS_reward.py
|
|---  official
	|--- ...
|--- ActorCritic_usetagpath_step_nofix.py
|--- collect_url.py
|--- config.ini
|--- deep_q_network_step_notfix_bert_bilstm_coor_top3.py
|--- deep_q_network_without_bilstm.py
|--- deep_q_network_without_coor.py
|--- deep_q_network_without_top3.py
|--- ESPS_data.ipynb
|--- evaluate.ipynb
|--- EvaluateMultitaskModelNoFixStep.py
|--- EvaluateMultitaskModelNoFixStep_sarsa.py
|--- EvaluateMultitaskModelNoStep_policy.py
|--- EvaluateMultitaskModelWithFixStep_policy.py
|--- EvaluateMultitaskModelWithFixStepTop3.py
|--- finetune_ActorCritic_step_nofix.py
|--- finetune_DQN_step_notfix.py
|--- finetune_policy gradent_step_nofix.py
|--- finetune_SARSA_step_nofix.py
|--- negativapage.csv
|--- policy gradent_step_nofix.py
|--- positivepage.csv
|--- README.md
|--- requirements.txt
|--- run.sh
|--- SARSA_step_nofix_bert_bilstm_coor_top3.py
```
## Other need to download
Other necessary data: https://1drv.ms/f/s!AqlIgICzxHbeh40aujkkpQKoDBDdsQ?e=A8fsao

## Train the model (Scenario A)
```
python deep_q_network_step_notfix_bert_bilstm_coor_top3.py --epoch 51 --train 40 --cost 0.4 > log/dqn_log_2023_xx_xx.log
python SARSA_step_nofix_bert_bilstm_coor_top3.py > log/sarsa_log_2023_xx_xx.log
python policy gradent_step_nofix.py > log/polcy_log_2023_xx_xx.log
python ActorCritic_usetagpath_step_nofix.py > log/polcy_log_2023_xx_xx.log
```
## Scenario B
```
python deep_q_network_step_notfix_bert_bilstm_coor_top3.py --epoch 51 --train 40 --cost 0.4 --esps > log/dqn_log_2023_xx_xx.log
```
## Scenario C, D
```
python finetune_DQN_step_nofix.py > log/finetune_dqn_log_2023_xx_xx.log
python finetune_SARSA_step_nofix.py > log/finetune_sarsa_log_2023_xx_xx.log
python finetune_ActorCritic_step_nofix.py > log/finetune_a2c_log_2023_xx_xx.log
python finetune_policy gradient_step_nofix.py > log/finetune_policy_log_2023_xx_xx.log
```
+ need to change load_pretrain_model
+ Scenario C: load_pretrain_model = False 
+ Scenario D: load_pretrain_model = True

## model
+ https://1drv.ms/f/s!AqlIgICzxHbeh5Uo2YQyimjcntjbMw?e=wiOOBO
## How to test the trained agent:
```
## asset method
python EvaluateMultitaskModelNoFixStep.py --modelname [model path/model.h5] --numofthread [float] --multitask --bilstm --top3 # dqn
python EvaluateMultitaskModelNoFixStep_sarsa.py --modelname [model path/model.h5] --numofthread [float] --multitask --bilstm --top3 
python EvaluateMultitaskModelNoFixStep_policy.py --modelname [model path/model.h5] --numofthread [float] --multitask --bilstm --top3 

## fix step
python EvaluateMultitaskModelWithFixStepTop3.py --modelname [model path/model.h5] --steps 3 --numofthread [float] #dqn,a2c,sarsa
python EvaluateMultitaskModelWithFixStepTop3_policy.py --modelname [model path/model.h5] --steps 3 --numofthread [float]

## threshold
python EvaluateMultitaskModel_threshold.py --modelname [model path/model.h5] -wbl 0 -u --numofthread [float] #dqn,a2c,sarsa
python EvaluateMultitaskModel_threshold_policy.py --modelname [model path/model.h5] -wbl 0 -u --numofthread [float]
Default: numofthread = 3
```
testing 後都會生成.txt需要再將這些檔案路徑放 evalucte.ipynb 進行評估
threshold method 會生成兩個.txt檔
1. xxx_traindata_base_xxx.txt (計算train data所有點擊的效能並產生其threshold)
2. xxx_evaluationdata_base_xxx.txt(最終放入 evalucte.ipynb的檔案)

fix step and asset method會生成xxx_evaluationdata.txt檔
## ESPS_dataset generate
```
python collect_url.py
```
+ 運行結束會產生每個domain url上所有可點擊的連結的.txt檔
+ 需要再運行 ESPS_data.ipynb 將這先連結標記成1 or 0 (是活動來源或不是活動來源)
+ 最後會產生
  + test92_down.csv
  + test_no_down_pre.csv
  + train_down.csv
  + vali_down.csv
  + vali_no_down.csv
