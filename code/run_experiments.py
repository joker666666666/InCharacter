import argparse
import os
import pdb
import sys

import global_variable as gv
from characters import character_labels
from personality_tests import personality_assessment
from utils import logger_main as logger


def count_files_in_directory(directory_path):
    try:
        # 获取目录中的所有文件和子目录
        entries = os.listdir(directory_path)
        # 过滤出文件
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
        return len(files)
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return 0


parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--agent_llm', default='gpt-3.5', choices=["chatglm", "qwen", 'gpt-3.5', 'gpt-4'], help='Agent LLM')
parser.add_argument('--folder_name', default='memory1', choices=["memory1", "emotion1", "emotion2", "emotion3", "emotion4"])

# 解析参数
args = parser.parse_args()

questionnaire = '16Personalities'
dataset = "characterllm"
agent_llm = args.agent_llm
folder_name = args.folder_name
logger.info(f"{agent_llm}, {folder_name}")
interview_folder_path = f'/home/huanglebj/InCharacter2/results/{dataset}/{folder_name}/{agent_llm}'
file_count = count_files_in_directory(interview_folder_path)
if file_count != 9:
    logger.info("The number of files in the interview folder is not 9.")
    sys.exit(0)

eval_method = 'interview_assess_batch_anonymous'
eval_llm = 'gpt-3.5'
repeat_times = 1
characters = list(character_labels['pdb'].keys())

logger.info('Start testing eval methods')
print(folder_name)

results = {}
cnt = 0
for character in characters:
    cnt += 1
    print(cnt)

    result = personality_assessment(
        character, "charactereval", agent_llm, 
        questionnaire, eval_method, eval_llm, repeat_times=repeat_times, folder_name=args.folder_name, dataset=dataset)
    results[(character, "charactereval")] = result 
    
logger.info(f"Completion_token:{gv.completion_tokens}, Prompt_token:{gv.prompt_tokens}")
logger.info('Questionnaire: {}, Eval Method: {}, Repeat Times: {}, Agent LLM: {}, Eval LLM: {}'.format(questionnaire, eval_method, repeat_times, agent_llm, eval_llm))   

from utils import avg

personality_consistency = {} 

for analysis_key in result['analysis'].keys():
    analysis_values = [ v['analysis'][analysis_key] for v in results.values()]
    analysis_value = avg(analysis_values)
    
    logger.info('Analyzing {}: {:.4f}'.format(analysis_key, analysis_value))
    personality_consistency[analysis_key] = analysis_value

preds = { rpa: {dim: result['dims'][dim]['all_scores'] for dim in result['dims']} for rpa, result in results.items()}

if questionnaire in ['BFI', '16Personalities']:
    label_settings = ['pdb']
    labels_pdb = { rpa: {dim: character_labels['pdb'][rpa[0]][questionnaire][dim] for dim in result['dims']} for rpa, result in results.items()} 
else:
    label_settings = ['annotation']
    labels_pdb = { rpa: {dim: character_labels['annotation'][rpa[0]][questionnaire][dim] for dim in result['dims']} for rpa, result in results.items()} 

for label_setting in label_settings:
    labels = { rpa: {dim: character_labels[label_setting][rpa[0]][questionnaire][dim] for dim in result['dims']} for rpa, result in results.items()} #e.g. { "score": 65.88130032806441, "type": "H"}


    from personality_tests import calculate_measured_alignment

    measured_alignment = calculate_measured_alignment(preds, labels, questionnaire, labels_pdb=labels_pdb)                        
    
    single_acc = measured_alignment['all']['single_acc']['all']
    single_mse = measured_alignment['all']['single_mse']['all']
    single_mae = measured_alignment['all']['single_mae']['all']
    full_acc = measured_alignment['all']['full_acc']
    
    
    logger.info('Single Acc, Full Acc, Single MSE, Single MAE')
    logger.info('Alignment {}: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(label_setting.upper()[:3], single_acc, full_acc, single_mse, single_mae))

          
                                         
                        
                        

                    
                    
                        
                        
                            
            
                
            
            

    
        
    
