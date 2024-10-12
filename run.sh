#python EvaluateMultitaskModelNoStep_policy.py --modelname finetunemodel2/finetune_policy_gradient_nofixstep_eplision_0.7_epoch0.h5 --numofthread 3
#python EvaluateMultitaskModelNoStep_policy.py --modelname finetunemodel2/finetune_policy_gradient_nofixstep_eplision_0.7_epoch1.h5 --numofthread 3
#python EvaluateMultitaskModelNoStep_policy.py --modelname finetunemodel2/finetune_policy_gradient_nofixstep_eplision_0.7_epoch2.h5 --numofthread 3
#python EvaluateMultitaskModelNoStep_policy.py --modelname finetunemodel2/finetune_policy_gradient_nofixstep_eplision_0.7_epoch3.h5 --numofthread 3
#python EvaluateMultitaskModelNoStep_policy.py --modelname finetunemodel2/finetune_policy_gradient_nofixstep_eplision_0.7_epoch4.h5 --numofthread 3
#python EvaluateMultitaskModelNoStep_policy.py --modelname finetunemodel2/finetune_policy_gradient_nofixstep_eplision_0.7_epoch5.h5 --numofthread 3

#python EvaluateMultitaskModelNoFixStep.py --modelname stepnofixmodel/pretrain_dqn_without_top3_fixreward_choose.h5 --numofthread 3 --multitask --bilstm
#python EvaluateMultitaskModelNoFixStep.py --modelname stepnofixmodel/pretrain_dqn_train40_epoch100_choose.h5 --numofthread 3 --multitask --bilstm --top3

#python EvaluateMultitaskModelNoFixStep.py --modelname stepnofixESPS/pretrain_dqn_new_nofixstep_bertbisltm_coor_top3_cost0.4_ESPSreward_choose.h5 --numofthread 3 --multitask --bilstm --top3
#python EvaluateMultitaskModelNoStep_policy.py --modelname stepnofixESPS/pretrain_policy_gradient_ESPS_step_nofix_choose.h5 --numofthread 3

#python EvaluateMultitaskModelNoFixStep.py --modelname stepnofixmodel/pretrain_dqn_new_nofixstep_bertbisltm_coor_top3_cost0.4_fixreward_choose.h5 --numofthread 3 --multitask --bilstm --top3

#python EvaluateMultitaskModelNoFixStep.py --modelname finetunemodel2/Actor_finetune_A2C_nofixstep_eplision_0.7_epoch0.h5 --numofthread 3 --bilstm --top3
#python EvaluateMultitaskModelNoFixStep.py --modelname finetunemodel2/Actor_finetune_A2C_nofixstep_eplision_0.7_epoch1.h5 --numofthread 3 --bilstm --top3
#python EvaluateMultitaskModelNoFixStep.py --modelname finetunemodel2/Actor_finetune_A2C_nofixstep_eplision_0.7_epoch2.h5 --numofthread 3 --bilstm --top3
#python EvaluateMultitaskModelNoFixStep.py --modelname finetunemodel2/Actor_finetune_A2C_nofixstep_eplision_0.7_epoch3.h5 --numofthread 3 --bilstm --top3
#python EvaluateMultitaskModelNoFixStep.py --modelname finetunemodel2/Actor_finetune_A2C_nofixstep_eplision_0.7_epoch4.h5 --numofthread 3 --bilstm --top3
#python EvaluateMultitaskModelNoFixStep.py --modelname finetunemodel2/Actor_finetune_A2C_nofixstep_eplision_0.7_epoch5.h5 --numofthread 3 --bilstm --top3
python deep_q_network_step_notfix_bert_bilstm_coor_top3.py --epoch 51 --train 40 --cost 0.4