from langchain_google_genai import GoogleGenerativeAI
import transformers
import torch
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain_openai import OpenAI
from openai import OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import LlamaCpp
from llama_cpp import Llama
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, \
    PreTrainedTokenizerFast, PreTrainedModel, BitsAndBytesConfig
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import config

os.environ["OPENAI_API_KEY"] = config.open_ai_api_key
os.environ["GOOGLE_API_KEY"] = config.google_api_key


def get_bnb_config() -> BitsAndBytesConfig:
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bnb_config


def get_llama_variant_tokenizer(model_name) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_llama_variant_model(model_name) -> PreTrainedModel:
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    return model


def get_llama_variant_pipeline(model_name: str, tokenizer: PreTrainedTokenizerFast = None, adapters_name: str = None) -> transformers.pipelines:
    tokenizer = tokenizer if tokenizer else get_llama_variant_tokenizer(model_name)
    model = get_llama_variant_model(model_name)

    if adapters_name:
        model = PeftModel.from_pretrained(model, adapters_name)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    return pipeline


# AiMavenAi/AiMaven-PrometheusModels
def get_aimaven_prompt(final_prompt_str: str, tokenizer: PreTrainedTokenizerFast) -> list:
    messages = [
        {"role": "user", "content": final_prompt_str}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def parse_aimaven_models_completion(raw_completion: str, size: int) -> str:
    # print(raw_completion)
    final_completion = raw_completion[size:]
    return final_completion


def get_aimaven_prometheus_7b_completion(prompts_list: list[str],
                                         max_new_tokens: int = 4096,
                                         temperature: float = 0.7, top_k: int = 50, top_p: float = 0.95
                                         ) -> list[str]:
    torch.cuda.empty_cache()
    model_name = "AiMavenAi/AiMaven-Prometheus"

    tokenizer = get_llama_variant_tokenizer(model_name)
    pipeline = get_llama_variant_pipeline(model_name, tokenizer)

    completions = []
    for prompt in prompts_list:
        # prompt = get_aimaven_prompt(prompt, tokenizer)
        output = pipeline(prompt,
                          max_length=max_new_tokens,
                          do_sample=True,
                          truncation=True,
                          num_return_sequences=1,
                          temperature=temperature,
                          top_k=top_k,
                          top_p=top_p
                          )
        # print(output[0]["generated_text"])
        size = len(prompt)
        completion = parse_aimaven_models_completion(output[0]["generated_text"], size)
        # print(completion)
        completions.append(completion)
    return completions


def get_fine_tuned_aimaven_prometheus_7b_completion(prompts_list: list[str],max_tokens: int = 4096):
    torch.cuda.empty_cache()
    model_name = "AiMavenAi/AiMaven-Prometheus"
    adapters_name = "shaswatamitra/aimaven-prometheus-finetuned2"

    pipeline = get_llama_variant_pipeline(model_name, adapters_name=adapters_name)

    completions = []

    for prompt in prompts_list:
        sequences = pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            max_length=max_tokens,
            truncation=True
        )
        size = len(prompt)
        completions.append(sequences[0]['generated_text'][size:])

    return completions


def get_llama_31_8b_completion(prompts_list: list[str], max_tokens: int = 4096) -> list:
    model_path = "/media/shaswata/4TB_2/Desktop_Backup/Context_CTI/models/Meta-Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cuda",
    )
    completions = []

    for prompt in prompts_list:
        messages = [
            # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": prompt},
        ]
        outputs = pipeline(
            messages,
            max_new_tokens=max_tokens,
        )
        out = outputs[0]["generated_text"][-1]['content']
        # print(outputs[0]["generated_text"][-1])
        completions.append(out)

    return completions


def get_mistral_nemo_12b_instruct_completion(prompts_list: list[str], max_tokens: int = 4096) -> list:
    model_path = "/media/shaswata/4TB_2/Desktop_Backup/Context_CTI/models/Mistral-Nemo-Instruct-2407"

    tokenizer = MistralTokenizer.from_file(f"{model_path}/tekken.json")
    model = Transformer.from_folder(model_path)

    completions = []
    for prompt in prompts_list:
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        out_tokens, _ = generate([tokens], model, max_tokens=max_tokens, temperature=0.35,
                                 eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
        result = tokenizer.decode(out_tokens[0])
        completions.append(result)
        torch.cuda.empty_cache()

    return completions


def get_mistral_nemo_minitron_8b_base_completion(prompts_list: list[str], max_tokens: int = 2048) -> list:
    model_path = "/media/shaswata/4TB_2/Desktop_Backup/Context_CTI/models/Mistral-NeMo-Minitron-8B-Base"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = 'cuda'
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)

    completions = []
    for prompt in prompts_list:
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        outputs = model.generate(inputs, max_length=max_tokens)
        output_text = tokenizer.decode(outputs[0])
        output_text = output_text.split("In this task, you are given a sentence with a missing word that can be an object")[0]
        output_text = output_text.split("Answer:")[-1]
        completions.append(output_text)

    return completions


# OpenAI GPT-4o
def get_gpt_4o_completion(prompts_list: list[str]) -> list:
    client = OpenAI()

    completions = []
    for prompt in prompts_list:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ])
        message = response.choices[0].message.content
        completions.append(message)

    return completions


# OpenAI GPT-3.5 Turbo
def get_gpt3_5_completion(prompts_list: list[str],
                          temperature: float = 0.7
                          ) -> list[str]:
    llm = OpenAI(
        # temperature=temperature
    )

    completions = []
    for prompt in prompts_list:
        prompt = PromptTemplate.from_template(prompt)

        llm_chain = LLMChain(prompt=prompt, llm=llm)
        try:
            out = llm_chain.invoke({'question': prompt})
            completions.append(out['text'])
        except Exception as e:
            print(f"Error occurred: {e} for Prompt: {prompt}")
            completions.append("Error Occurred!")

    return completions


# Llama-2-7B
def get_llama2_7b_completion(prompts_list: list[str],
                             temperature: float = 0.6,
                             top_p: float = 0.9,
                             top_k: int = 50,
                             max_tokens: int = 4096
                             ) -> list[str]:
    torch.cuda.empty_cache()
    model_path = "/media/shaswata/4TB_2/Desktop_Backup/Context_CTI/models/Llama-2-7b-chat-hf"

    pipeline = get_llama_variant_pipeline(model_path)

    completions = []

    for prompt in prompts_list:
        sequences = pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            max_length=max_tokens,
            truncation=True
        )
        size = len(prompt)
        completions.append(sequences[0]['generated_text'][size:])

    return completions


def get_fine_tuned_llama2_7b_completion(prompts_list: list[str],max_tokens: int = 4096):
    torch.cuda.empty_cache()
    model_path = "/media/shaswata/4TB_2/Desktop_Backup/Context_CTI/models/Llama-2-7b-chat-hf"
    adapters_name = "shaswatamitra/llama2-7b-chat-hf-finetuned2"

    pipeline = get_llama_variant_pipeline(model_path, adapters_name=adapters_name)

    completions = []

    for prompt in prompts_list:
        sequences = pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            max_length=max_tokens,
            truncation=True
        )
        size = len(prompt)
        completions.append(sequences[0]['generated_text'][size:])

    return completions



# Google Palm2-text-bison@001
def get_palm2_text_bison001_completion(prompts_list: list[str],
                                       max_tokens: int = 4096,
                                       temperature: float = 0.0,
                                       top_p: float = 0.95,
                                       top_k: int = 40
                                       ) -> list[str]:
    llm = GoogleGenerativeAI(model="models/text-bison-001",
                             maxOutputTokens=max_tokens,
                             # temperature=temperature,
                             # topP=top_p,
                             # topK=top_k
                             )

    completions = []
    for prompt in prompts_list:
        completions.append(llm.invoke(prompt))

    return completions


# QWEN1.5-7B-Chat
def get_qwen_7b_completion(prompts_list: list[str],
                           max_tokens: int = 4096,
                           temperature: float = 0.01,
                           top_p: float = 0.95,
                           top_k: int = 40
                           ) -> list[str]:
    torch.cuda.empty_cache()
    model_id = "Qwen/Qwen1.5-7B-Chat"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    completions = []
    for prompt in prompts_list:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(model_inputs.input_ids,
                                       max_new_tokens=max_tokens,
                                       # temperature=temperature,    # Commented to allow default temperature
                                       # top_p=top_p,    # Commented to allow default value
                                       # top_k=top_k     # Commented to allow default value
                                       )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        completions.append(response)

    return completions


# Mistral-7B
def get_mistral_7b_completion(prompts_list: list[str],
                              max_tokens: int = 2048,
                              temperature: float = 1,
                              top_p: float = 1.0,
                              top_k: int = 40) -> list[str]:
    torch.cuda.empty_cache()
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    pipeline = get_llama_variant_pipeline(model_name)
    completions = []
    for prompt in prompts_list:
        sequences = pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            max_length=max_tokens,
            truncation=True
        )
        size = len(prompt)
        completions.append(sequences[0]['generated_text'][size:])

    return completions


def get_fine_tuned_mistral_7b_completion(prompts_list: list[str], max_tokens: int = 4096):
    torch.cuda.empty_cache()
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    adapters_name = "shaswatamitra/mistral-7b-v2-finetuned2"

    pipeline = get_llama_variant_pipeline(model_name, adapters_name=adapters_name)

    completions = []

    for prompt in prompts_list:
        sequences = pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            max_length=max_tokens,
            truncation=True
        )
        size = len(prompt)
        completions.append(sequences[0]['generated_text'][size:])

    return completions


# westlake-7B
def get_westlake_7b_completion(prompts_list: list[str],
                               max_tokens: int = 2048,
                               temperature: float = 0.75,
                               top_p: float = 1,
                               top_k: int = 50) -> list[str]:
    torch.cuda.empty_cache()
    model_name = "senseable/WestLake-7B-v2"

    pipeline = get_llama_variant_pipeline(model_name)
    completions = []
    for prompt in prompts_list:
        sequences = pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            max_length=max_tokens,
            truncation=True
        )
        size = len(prompt)
        completions.append(sequences[0]['generated_text'][size:])

    return completions


def get_fine_tuned_westlake_7b_completion(prompts_list: list[str], max_tokens: int = 4096):
    torch.cuda.empty_cache()
    model_name = "senseable/WestLake-7B-v2"
    adapters_name = "shaswatamitra/westlake-finetuned2"

    pipeline = get_llama_variant_pipeline(model_name, adapters_name=adapters_name)

    completions = []

    for prompt in prompts_list:
        sequences = pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            max_length=max_tokens,
            truncation=True
        )
        size = len(prompt)
        completions.append(sequences[0]['generated_text'][size:])

    return completions


# westseverus-7B
def get_westseverus_7b_completion(prompts_list: list[str],
                                  max_tokens: int = 2048,
                                  temperature: float = 0.75,
                                  top_p: float = 1,
                                  top_k: int = 50
                                  ) -> list[str]:
    torch.cuda.empty_cache()
    model_name = "FelixChao/WestSeverus-7B-DPO-v2"

    pipeline = get_llama_variant_pipeline(model_name)
    completions = []
    for prompt in prompts_list:
        sequences = pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            max_length=max_tokens,
            truncation=True
        )
        size = len(prompt)
        completions.append(sequences[0]['generated_text'][size:])

    return completions


def get_fine_tuned_westseverus_7b_completion(prompts_list: list[str], max_tokens: int = 4096):
    torch.cuda.empty_cache()
    model_name = "FelixChao/WestSeverus-7B-DPO-v2"
    adapters_name = "shaswatamitra/westseverus-finetuned2"

    pipeline = get_llama_variant_pipeline(model_name, adapters_name=adapters_name)

    completions = []

    for prompt in prompts_list:
        sequences = pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            max_length=max_tokens,
            truncation=True
        )
        size = len(prompt)
        completions.append(sequences[0]['generated_text'][size:])

    return completions


if __name__ == "__main__":

    prompt1 = """You are an honest assistant and you answer questions based only on the information that is given to you
    in the prompt. If you cannot answer you simply say you do not know the answer. Based on this information, generate a
    complete response filled with complete information for the query while considering global knowledge retrieved from
    internet and local knowledge retrieved from local wikis. Also, while answering, only provide the answer without
    mentioning its source such as global knowledge or local knowledge.
    
    global_knowledge: The features of mobile that Pegasus
    targeted include reading text messages, call snooping, collecting passwords, location tracking, accessing the
    device's microphone and camera, and harvesting information from apps.
    
    local_knowledge: Document 1: remotely accessing text messages, iMessages, calls, emails, logs, and more from apps
    including Gmail, Facebook, Skype, WhatsApp, Viber, Facetime, Calendar, Line, Mail.Ru, WeChat, Surespot, Tango,
    Telegram, and others.
    
    question: What features of mobile did Pegasus targeted?
    
    answer:"""

    prompt2 = """You are an honest assistant and you answer questions based only on the information that is given to you
    in the prompt. If you cannot answer you simply say you do not know the answer. Based on this information, generate a
    complete response filled with complete information for the query while considering global knowledge retrieved from
    internet and local knowledge retrieved from local wikis. Also, while answering, only provide the answer without
    mentioning its source such as global knowledge or local knowledge.
    
    global_knowledge: The features of mobile that Pegasus
    targeted include reading text messages, call snooping, collecting passwords, location tracking, accessing the
    device's microphone and camera, and harvesting information from apps.

    local_knowledge: Document 1: remotely accessing text messages, iMessages, calls, emails, logs, and more from apps
    including Gmail, Facebook, Skype, WhatsApp, Viber, Facetime, Calendar, Line, Mail.Ru, WeChat, Surespot, Tango,
    Telegram, and others.

    question: What is Pegasus?
    
    answer:"""

    prompt3 = "Is there any vulnerability in Weston Embedded uC-HTTP Server?"
    prompt4 = "What is Netgear RAX30 JSON Parsing getblockschedule() stack-based buffer overflow vulnerability?"

    models_list = [
        # get_aimaven_prometheus_7b_completion,
        # get_gpt3_5_completion,
        # get_llama2_7b_completion,
        # get_palm2_text_bison001_completion,
        # get_qwen_7b_completion,
        # get_mistral_7b_completion,
        # get_westlake_7b_completion,
        # get_westseverus_7b_completion,
    ]

    prompts = [
        # prompt1,
        # prompt2,
        prompt3,
        prompt4
    ]

    completions = get_westlake_7b_completion(prompts)
    for completion in completions:
        print(f"Output:\n{completion}")

    completions = get_fine_tuned_westlake_7b_completion(prompts)
    for completion in completions:
        print(f"Output:\n{completion}")
