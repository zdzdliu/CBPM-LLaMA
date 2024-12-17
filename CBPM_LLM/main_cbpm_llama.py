import os
import re
import json
import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_for_llm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_dir', help='path to the training dataset (json file)',
                        default=os.path.join(os.path.dirname(__file__), 'dataset','trainlist.json'))
    parser.add_argument('--test_dataset_dir', help='path to the test dataset (json file)',
                        default=os.path.join(os.path.dirname(__file__), 'dataset','testlist.json'))
    parser.add_argument('--save_result_dir', help='path to save the prediction resutls',
                        default=os.path.join(os.path.dirname(__file__), 'results'))
    parser.add_argument('--llms_dir', help='path to pre-train LLM',
                        default=os.path.join(os.path.dirname(__file__), 'models', 'llama-3-8b-bnb-4bit/'))
    parser.add_argument('--device_map', help='GPU to use',
                        default={'gpu':[0,1]})
    args = parser.parse_args()
    return args


class Predict_BP_By_LLM:
    def __init__(self, train_dataset_dir: str, test_dataset_dir: str, save_result_dir: str, llms_dir: str, device_map: dict) -> None:
        self.train_dataset_dir = train_dataset_dir
        self.test_dataset_dir = test_dataset_dir
        self.save_result_dir = save_result_dir
        self.llms_dir = llms_dir
        self.device_map = device_map
        self.max_seq_length = 2048
    

    def __call__(self, *args: any, **kwds: any) -> any:
        self.train_on_single_gpu()
    

    def train_on_single_gpu(self, *args: any, **kwds: any) -> any:

        # load model
        model, tokenizer = self.load_model(self.llms_dir)
        
        #load training json file
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ### Instruction:
        {}
        ### Input:
        {}
        ### Response:
        {}
        """
        EOS_TOKEN = tokenizer.eos_token
        dataset = self.load_dataset_from_json(self.train_dataset_dir, EOS_TOKEN, alpaca_prompt)

        # train model
        self.train_model(model, tokenizer, dataset, self.max_seq_length, self.save_result_dir)
        
        # Enable native 2x faster inference
        FastLanguageModel.for_inference(model)
        
        # result analyse
        save_model_result_file_name = os.path.join(self.save_result_dir, 'bpestimation.csv')
        result_analyse = Result_Analyse(model=model, file_name=self.test_dataset_dir, tokenizer=tokenizer, alpaca_prompt=alpaca_prompt, savefilename=save_model_result_file_name)
        result_analyse()

    def formatting_prompts_func(self, examples, eos_token: str, alpaca_prompt: str) -> any:
        instructions = examples['instruction']
        inputs = examples['input']
        outputs = examples['output']
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + eos_token
            texts.append(text)

        return {'text': texts}

    def load_dataset_from_json(self, train_data: str, eos_token: str, alpaca_prompt: str) -> any:
        my_dataset = load_dataset('json', data_dir='', data_files=train_data, split='train')
        my_dataset = my_dataset.map(lambda x: self.formatting_prompts_func(x, eos_token, alpaca_prompt), batched=True)
        return my_dataset


    def load_model(self, model_path: str) -> any:
        max_seq_length_in_model = self.max_seq_length   # Choose any! We auto support RoPE Scaling internally!
        dtype_in_model =  None                          # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit_in_model = True                    # Use 4bit quantization to reduce memory usage. Can be False.
        # load llama model
        bnb_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            # quantization_config = bnb_config,
            max_seq_length = max_seq_length_in_model,
            dtype = dtype_in_model,
            load_in_4bit = load_in_4bit_in_model,
            # llm_int8_enable_fp32_cpu_offload = True,
            device_map = 'auto'
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = 64,     # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,       # Supports any, but = 0 is optimized
            bias = 'none',          # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = 'unsloth',     # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,     # We support rank stabilized LoRA
            loftq_config = None     # And LoftQ
        )
        return model, tokenizer
    

    def train_model(self, model: str, tokenizer: any, dataset: any, max_seq_length: int, output_dir: str) -> any:
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = 'text',
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False,
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                max_steps = 250, 
                learning_rate = 2e-4,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = 'adamw_8bit',
                weight_decay = 0.01,
                lr_scheduler_type = 'linear',
                seed = 3407,
                output_dir = output_dir,
                num_train_epochs = 1
            ) 
        )
        trainer_stats = trainer.train()
        return trainer_stats


class Result_Analyse:
    def __init__(self, model, file_name, tokenizer, alpaca_prompt, savefilename) -> None:
        self.model = model
        self.test_name = file_name
        self.tokenizer = tokenizer
        self.alpaca_prompt = alpaca_prompt
        self.save_file_name = savefilename

    def __call__(self, *args: any, **kwds: any) -> any:
        with open(self.test_name, 'r', encoding='utf-8') as file:
            testData = json.load(file)

        estSBPAll = []   #estimated sbp
        refSBPAll = []   #reference sbp
        BaseSBPAll = []  #calibrated sbp
        estDBPAll = []   #estimated dbp
        refDBPAll = []   #reference dbp
        BaseDBPAll = []  #calibrated dbp
        for i in range(len(testData)):
            if i % 100 == 0:
                print('Completed: ', i, 'Remaining: ', len(testData) - i)
            refSBPAll.append(float(testData[i]['refsbp']))
            refDBPAll.append(float(testData[i]['refdbp']))
            BaseSBPAll.append(float(testData[i]['basesbp']))
            BaseDBPAll.append(float(testData[i]['basedbp']))

            inputs = self.tokenizer([self.alpaca_prompt.format(testData[i]['instruction'], testData[i]['input'],"",)], return_tensors = "pt").to("cuda")
            outputs = self.model.generate(**inputs,  pad_token_id=self.tokenizer.eos_token_id, max_new_tokens = 64, use_cache = True)
            outputs = self.tokenizer.batch_decode(outputs)
            outputs = outputs[0]
            outputs = self.postprocess(outputs)

            
            map, pp = float(self.extract_numbers(self.map_str(outputs))), float(self.extract_numbers(self.pp_str(outputs)))
            sbp, dbp = map+2*pp/3, map-pp/3
            estSBPAll.append(round(sbp,1))
            estDBPAll.append(round(dbp,1))
        
        estSBPAll = np.array(estSBPAll)
        refSBPAll = np.array(refSBPAll)
        estDBPAll = np.array(estDBPAll)
        refDBPAll = np.array(refDBPAll)
        BaseSBPAll = np.array(BaseSBPAll)
        BaseDBPAll = np.array(BaseDBPAll)

        # save results
        # [CalFreeEst_SBP, Reference_SBP, CalFreeEst_DBP, Reference_DBP, Baseline_SBP, Baseline_DBP]
        Results = np.array([estSBPAll, refSBPAll,estDBPAll, refDBPAll, BaseSBPAll, BaseDBPAll])
        Results = np.transpose(Results,(1,0))
        df = pd.DataFrame(Results)
        df.to_csv(self.save_file_name)

        print('########calibration-free results########')
        err = estSBPAll-refSBPAll
        MAE = np.average(np.abs(err))
        print('SBP MAE: %.3f ME: %.3f STD: %.3f' % (MAE, np.average(err),np.std(err))) 
        err = estDBPAll-refDBPAll
        MAE = np.average(np.abs(err))
        print('SBP MAE: %.3f ME: %.3f STD: %.3f' % (MAE, np.average(err),np.std(err))) 

        print('########calibration-based results########')
        rario = 0.3
        err = estSBPAll*rario+BaseSBPAll*(1-rario)-refSBPAll
        MAE = np.average(np.abs(err))
        print('SBP MAE: %.3f ME: %.3f STD: %.3f' % (MAE, np.average(err),np.std(err)))
        err = estDBPAll*rario+BaseDBPAll*(1-rario)-refDBPAll
        MAE = np.average(np.abs(err))
        print('DBP MAE: %.3f ME: %.3f STD: %.3f' % (MAE, np.average(err),np.std(err)))

    
    def sbp_str(self, source_str: str) -> any:
        start_index = source_str.find("predicted SBP is")
        end_index = source_str.find(" mmHg", start_index) + len(" mmHg")
        return source_str[start_index:end_index]
    
    def dbp_str(self, source_str: str) -> any:
        start_index = source_str.find("predicted DBP is") 
        end_index = source_str.find(" mmHg", start_index) + len(" mmHg")
        return source_str[start_index:end_index]

    def map_str(self, source_str: str) -> any:
        start_index = source_str.find("Predicted_map:")
        end_index = source_str.find(" mmHg", start_index) + len(" mmHg")
        return source_str[start_index:end_index]

    def pp_str(self, source_str: str) -> any:
        start_index = source_str.find("Predicted_pp:")
        end_index = source_str.find(" mmHg", start_index) + len(" mmHg")
        return source_str[start_index:end_index]

    def extract_numbers(self, des_str: str) -> any:
        numbers = re.findall(r'\d+', des_str)
        if len(numbers)==0:
            return 0
        else:
            return float(numbers[0])+float(numbers[1])/10

    def postprocess(self, response: str) -> any:
        messages = response.split("Response:\n")
        if not messages:
            raise ValueError("Invalid template for prompt. The template should include the term 'Response:'")
        return "".join(messages[1:])
    
    


if __name__ == '__main__':
    parse = parse_for_llm()
    pre_bp = Predict_BP_By_LLM(train_dataset_dir=parse.train_dataset_dir, test_dataset_dir=parse.test_dataset_dir, save_result_dir=parse.save_result_dir,
                                llms_dir_list=parse.llms_dir, device_map=parse.device_map)
    pre_bp()
