# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
import json
import time
import re
import requests
from codebleu import calc_codebleu

try:
    from .compute_hx import compute_hx as compute_step_hx
except ImportError:
    from compute_hx import compute_hx as compute_step_hx

DEFAULT_BETA_H = 0.01
DEFAULT_DELTA_H_EPS = 1e-6


def extract_python_code(prediction):
    """使用正则表达式提取三反引号包裹的Python代码"""
    pattern = r"```python(.*?)```"
    code_blocks = re.findall(pattern, prediction, re.DOTALL)
    code = "\n".join(code_blocks)
    return code


def extract_step_idx(step_text: str, prefix_pattern: str) -> int | None:
    """从 Step 标题中提取步骤编号，用于对齐思维文本和代码片段。"""
    match = re.search(prefix_pattern, step_text)
    if not match:
        return None
    return int(match.group(1))


def align_text_and_code_spans(
    text_spans: list[dict],
    code_spans: list[dict],
) -> tuple[list[dict], list[dict]]:
    """按 Step 编号对齐自然语言 reasoning span 和代码 span。"""
    code_by_step = {}
    for code_span in code_spans:
        # 代码里通常是 “# Step N: ...”，先建立 step_idx -> code_span 的索引。
        step_idx = extract_step_idx(
            str(code_span.get("step_code", "")),
            r"(?im)^\s*#?\s*Step\s+(\d+)\s*[:：].*$",
        )
        if step_idx is not None:
            code_by_step[step_idx] = code_span

    aligned_text_spans = []
    aligned_code_spans = []
    for text_span in text_spans:
        # 文本 reasoning 里通常是 “Step N: ...”，用相同编号匹配代码片段。
        step_idx = extract_step_idx(
            str(text_span.get("step_text", "")),
            r"(?im)^\s*Step\s+(\d+)\s*[:：].*$",
        )
        if step_idx is None:
            continue
        aligned_text_spans.append(text_span)
        aligned_code_spans.append(
            code_by_step.get(
                step_idx,
                # 某些输出可能缺少对应代码注释，补一个空 code_span 保持长度一致。
                {
                    "char_start": text_span["char_start"],
                    "char_end": text_span["char_end"],
                    "token_start": text_span["token_start"],
                    "token_end": text_span["token_end"],
                    "step_code": None,
                },
            )
        )
    return aligned_text_spans, aligned_code_spans


def extract_gold_region(extra_info=None, sample_info=None):
    """从 extra_info 或样本同名字段中读取 gold support region。"""
    for source in (extra_info, sample_info):
        if not isinstance(source, dict):
            continue
        for key in ("gold_regions", "gold_region", "X_G", "x_g"):
            value = source.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple, str)) and len(value) == 0:
                continue
            if key == "gold_regions" and isinstance(value, list):
                # compute_hx.py 支持 {"gold_regions": [...]}，这里做统一包装。
                return {"gold_regions": value}
            return value
    return None


def compute_step_hx_values(text_spans, code_spans, gold_region):
    """调用 compute_hx.py 计算每个 Step 的 H*(X_k)，失败时回退为 0。"""
    aligned_text_spans, aligned_code_spans = align_text_and_code_spans(text_spans or [], code_spans or [])
    if not aligned_text_spans or not gold_region:
        # 没有结构化 step 或没有 gold region 时，不能可靠计算 H(X)，返回零信号。
        return aligned_text_spans, aligned_code_spans, [0.0] * len(aligned_text_spans), None

    try:
        step_hx_values = compute_step_hx(aligned_text_spans, aligned_code_spans, gold_region)
    except Exception as exc:
        print(f"Warning! compute_hx failed, fallback to zero hx: {type(exc).__name__}: {exc}")
        return aligned_text_spans, aligned_code_spans, [0.0] * len(aligned_text_spans), exc

    if len(step_hx_values) != len(aligned_text_spans):
        print(
            "Warning! compute_hx returned unexpected length: "
            f"{len(step_hx_values)} vs {len(aligned_text_spans)}"
        )
    normalized_values = []
    for idx in range(len(aligned_text_spans)):
        try:
            # 防御性转换，避免 compute_hx 返回 numpy 标量、字符串或长度不一致。
            normalized_values.append(float(step_hx_values[idx]))
        except (IndexError, TypeError, ValueError):
            normalized_values.append(0.0)
    return aligned_text_spans, aligned_code_spans, normalized_values, None


def _fill_span_values(target_tensor, span, value, valid_response_length):
    """把某个 step 的标量值铺到对应 token span 上。"""
    token_start = max(0, min(int(span.get("token_start", 0)), int(valid_response_length)))
    token_end = max(token_start, min(int(span.get("token_end", 0)), int(valid_response_length)))
    if token_end > token_start:
        target_tensor[token_start:token_end] = float(value)


def compute_hx(text_spans, code_spans, response_ids=None, valid_response_length=None, gold_region=None):
    """
    根据 text/code step spans 计算 token 级 H(X)。

    真实的 step 级 H(X) 由 compute_hx.py 计算；本函数只负责把每个 step
    的 H(X) 展开到对应的 reasoning/code token 区间。
    """
    if response_ids is not None:
        hx_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
    else:
        max_token_end = 0
        for span in list(text_spans or []) + list(code_spans or []):
            max_token_end = max(max_token_end, int(span.get("token_end", 0)))
        if valid_response_length is not None:
            max_token_end = max(max_token_end, int(valid_response_length))
        hx_tensor = torch.zeros(max_token_end, dtype=torch.float32)

    if valid_response_length is not None:
        valid_response_length = int(valid_response_length)
    else:
        valid_response_length = hx_tensor.shape[-1]

    aligned_text_spans, aligned_code_spans, step_hx_values, _ = compute_step_hx_values(
        text_spans=text_spans,
        code_spans=code_spans,
        gold_region=gold_region,
    )
    for step_hx, text_span, code_span in zip(step_hx_values, aligned_text_spans, aligned_code_spans):
        # reasoning 文本 token 使用该 step 的 H(X)。
        _fill_span_values(hx_tensor, text_span, step_hx, valid_response_length)
        if code_span.get("step_code") is not None:
            # 代码 token 也使用对应 step 的 H(X)，实现论文中的 step-level token 信号。
            _fill_span_values(hx_tensor, code_span, step_hx, valid_response_length)
    return hx_tensor


def compute_delta_h(prev_h, curr_h, eps=DEFAULT_DELTA_H_EPS):
    """计算相邻两个推理步骤之间的相对熵下降 δH。"""
    prev_h = float(prev_h or 0.0)
    curr_h = float(curr_h or 0.0)
    return (prev_h - curr_h) / (prev_h + eps)


def get_candidate_space_size(step_text, code_text, extra_info, prev_candidate_state):
    """
    X_k 候选空间大小的旧占位接口。

    当前真实计算已经迁移到 compute_hx.py，这个函数暂时保留是为了兼容早期代码。
    """
    return None


def _tokenize_without_special_tokens(tokenizer, text):
    """不添加特殊 token 的轻量 tokenizer 封装，用于字符位置到 token 位置映射。"""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        try:
            encoded = tokenizer(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer(text)
        if isinstance(encoded, dict):
            return encoded.get("input_ids", [])
        return encoded


def _encoded_length(tokenizer, text):
    return len(_tokenize_without_special_tokens(tokenizer, text))


def extract_step_spans(response_str, tokenizer, response_ids, valid_response_length):
    """
    从模型输出中提取 reasoning step span 和 code step span。

    - text_spans：来自 <think>...</think> 中的 “Step N: ...”
    - code_spans：来自 ```python 代码块中的 “# Step N: ...”
    同时把字符区间映射到 response token 区间，后续用于铺 H(X)/δH。
    """
    valid_response_length = int(valid_response_length)
    reasoning_end = response_str.find("</think>")
    text_search = response_str if reasoning_end < 0 else response_str[:reasoning_end]
    text_matches = list(re.finditer(r"(?im)^\s*Step\s+\d+\s*[:：].*$", text_search))
    text_spans = []

    for idx, match in enumerate(text_matches):
        # 当前 Step 的文本范围：从当前 Step 标题到下一个 Step 标题之前。
        char_start = match.start()
        char_end = text_matches[idx + 1].start() if idx + 1 < len(text_matches) else len(text_search)
        token_start = _encoded_length(tokenizer, response_str[:char_start])
        token_end = _encoded_length(tokenizer, response_str[:char_end])

        token_start = max(0, min(token_start, valid_response_length))
        token_end = max(token_start, min(token_end, valid_response_length))
        if token_end > token_start:
            text_spans.append(
                {
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_start": token_start,
                    "token_end": token_end,
                    "step_text": response_str[char_start:char_end].strip(),
                }
            )

    code_spans = []
    code_block_pattern = r"```(?:python)?\s*(.*?)```"
    for code_block_match in re.finditer(code_block_pattern, response_str, re.DOTALL | re.IGNORECASE):
        # 只在代码块内部识别 “# Step N: ...”，避免误匹配普通文本。
        code_block_start = code_block_match.start(1)
        code_block_text = code_block_match.group(1)
        code_matches = list(re.finditer(r"(?im)^\s*#?\s*Step\s+\d+\s*[:：].*$", code_block_text))

        for idx, match in enumerate(code_matches):
            # 当前 Step 的代码范围：从当前代码 Step 注释到下一个代码 Step 注释之前。
            char_start = code_block_start + match.start()
            char_end = code_block_start + (
                code_matches[idx + 1].start() if idx + 1 < len(code_matches) else len(code_block_text)
            )
            token_start = _encoded_length(tokenizer, response_str[:char_start])
            token_end = _encoded_length(tokenizer, response_str[:char_end])

            token_start = max(0, min(token_start, valid_response_length))
            token_end = max(token_start, min(token_end, valid_response_length))
            if token_end > token_start:
                code_spans.append(
                    {
                        "char_start": char_start,
                        "char_end": char_end,
                        "token_start": token_start,
                        "token_end": token_end,
                        "step_code": response_str[char_start:char_end].strip(),
                    }
                )

    return {"text_spans": text_spans, "code_spans": code_spans}


def build_entropy_reward(
    response_str,
    tokenizer,
    response_ids,
    valid_response_length,
    extra_info=None,
    gold_region=None,
    code_text=None,
    beta_h=DEFAULT_BETA_H,
):
    """
    为单条 response 构造 token 级 entropy reward。

    compute_hx.py 返回 step 级 H*(X_k)；本函数按论文公式计算相邻 step 的 δH，
    并把 β_h * δH 铺到对应 reasoning/code token 上。
    """
    span_info = extract_step_spans(response_str, tokenizer, response_ids, valid_response_length)
    text_spans = span_info["text_spans"]
    code_spans = span_info["code_spans"]
    aligned_text_spans, aligned_code_spans, step_hx_values, hx_error = compute_step_hx_values(
        text_spans=text_spans,
        code_spans=code_spans,
        gold_region=gold_region,
    )

    hx_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
    delta_h_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
    entropy_reward = torch.zeros_like(response_ids, dtype=torch.float32)

    for step_idx, (step_hx, text_span, code_span) in enumerate(
        zip(step_hx_values, aligned_text_spans, aligned_code_spans)
    ):
        # 第一个 step 没有前序 step，δH 置 0；后续 step 使用相邻 H 值计算相对下降。
        delta_h = 0.0 if step_idx == 0 else compute_delta_h(step_hx_values[step_idx - 1], step_hx)
        _fill_span_values(hx_tensor, text_span, step_hx, valid_response_length)
        _fill_span_values(delta_h_tensor, text_span, delta_h, valid_response_length)
        if delta_h != 0.0:
            _fill_span_values(entropy_reward, text_span, float(beta_h) * delta_h, valid_response_length)

        if code_span.get("step_code") is not None:
            # 同一 step 的代码 token 共享该 step 的 H(X)/δH 信号。
            _fill_span_values(hx_tensor, code_span, step_hx, valid_response_length)
            _fill_span_values(delta_h_tensor, code_span, delta_h, valid_response_length)
            if delta_h != 0.0:
                _fill_span_values(entropy_reward, code_span, float(beta_h) * delta_h, valid_response_length)

    entropy_reward_sum = float(entropy_reward[: int(valid_response_length)].sum().item())
    hx_values = hx_tensor[: int(valid_response_length)].detach().cpu().tolist()
    delta_h_values = delta_h_tensor[: int(valid_response_length)].detach().cpu().tolist()
    return {
        "reward": entropy_reward,
        "hx_values": hx_values,
        "delta_h_values": delta_h_values,
        "hx_mean": sum(hx_values) / len(hx_values) if hx_values else 0.0,
        "delta_h_mean": sum(delta_h_values) / len(delta_h_values) if delta_h_values else 0.0,
        "entropy_reward_sum": entropy_reward_sum,
        "num_steps": len(aligned_text_spans),
        "step_hx_values": step_hx_values,
        "text_spans": aligned_text_spans,
        "code_spans": aligned_code_spans,
        "hx_error": str(hx_error) if hx_error else "",
    }

@register("dapo")
class DAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False, step: int=0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
            
        print("\n\n===== Starting Reward Calculation =====")
        print(f"Processing batch with {len(data)} samples")
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # 第一步：收集同一个prompt的所有response信息
        prompt_groups = defaultdict(list)
        print("\n=== Step 1: Grouping responses by prompt ===")
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            
            # 存储这个response的信息，稍后处理
            prompt_groups[prompt_str].append(i)
            #print(f"  Sample {i} -> Prompt: '{prompt_str[:50]}...'")
        
        print(f"\nFound {len(prompt_groups)} unique prompts in batch")
        # for prompt, indices in prompt_groups.items():
        #     print(f"  Prompt: '{prompt[:50]}...' has {len(indices)} responses")
        
        # 第二步：处理每个response并收集正确代码
        print("\n=== Step 2: Calculating initial scores and collecting correct codes ===")
        all_scores = []  # 存储每个response的分数信息
        correct_codes_by_prompt = defaultdict(list)  # 按prompt存储正确代码
        
        # 第一次遍历：计算分数并收集正确代码
        for prompt_str, indices in prompt_groups.items():
            print(f"\nProcessing prompt: '{prompt_str[:50]}...'")
            for i in indices:
                data_item = data[i]
                
                # 提取response
                response_ids = data_item.batch['responses']
                prompt_length = data_item.batch['prompts'].shape[-1]
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                response_str = self.tokenizer.decode(valid_response_ids)
                
                # 提取其他必要信息
                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            
                data_source = data_item.non_tensor_batch['data_source']
                extra_info = data_item.non_tensor_batch.get('extra_info', None)
                # gold_regions 来自数据字段；用于 compute_hx.py 判断 X_G 是否仍被当前候选空间覆盖。
                gold_region = extract_gold_region(extra_info=extra_info, sample_info=data_item.non_tensor_batch)
                
                print(f"\n  Calculating score for sample {i}...")
                print(f"  Response: '{response_str[:50]}...'")
                print(f"  Ground truth: '{ground_truth[:50]}...'")
                
                # 使用compute_score计算分数
                dict_scores = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                
                
                # 保存分数信息
                all_scores.append({
                    "index": i,
                    "total_score": dict_scores['score'],
                    "dict_scores": dict_scores,
                    "valid_response_length": valid_response_length,
                    "response_ids": response_ids,
                    "extra_info": extra_info,
                    "gold_region": gold_region,
                    "prompt_str": prompt_str,
                    "response_str": response_str
                })
                
                print(f"  Initial scores for sample {i}:")
                print(f"    Score: {dict_scores['score']:.2f}")
                print(f"    Format: {dict_scores['format']:.2f}")
                print(f"    Execute: {dict_scores['execute']:.2f}")
                print(f"    Answer: {dict_scores['answer']:.2f}")
                print(f"    CodeBLEU: {dict_scores['codebleu']:.4f}")
                print(f"    Filepath: {dict_scores.get('filepath', 0.0):.2f}")
                
                # 如果答案正确，保存代码
                # 修改：使用 acc (Total>=3.0) 作为判断标准，确保答案正确才算 Correct
                if dict_scores.get('acc', False):
                    try:
                        # 提取代码
                        code_str = response_str.split("</think>")[-1].replace("<|im_end|>", "").replace("", "").replace("<_end>", "").replace("<|endoftext|>", "").strip()
                        cleaned_code = extract_python_code(code_str)
                        # print("\ncode_str code:{}".format(code_str))
                        # print("\nCleaned code:{}".format(cleaned_code))
                        correct_codes_by_prompt[prompt_str].append(cleaned_code)
                        print(f"  ✅ Sample {i} is CORRECT, saving code for future comparison")
                    except Exception as e:
                        print(f"  ❌ Error extracting code for correct answer: {e}")
                else:
                    print(f"  ❌ Sample {i} is INCORRECT")
        
        # 第三步：修正错误答案的codebleu分数
        print("\n=== Step 3: Adjusting CodeBLEU scores for incorrect answers ===")
        for score_info in all_scores:
            if not score_info["dict_scores"].get('acc', False):  # 只要不是完全正确 (acc=True)，都尝试修正
                prompt_str = score_info["prompt_str"]
                correct_codes = correct_codes_by_prompt.get(prompt_str, [])
                old_bleu = score_info["dict_scores"].get('codebleu', 0)
                
                print(f"\n  Adjusting sample {score_info['index']} (incorrect answer)")
                print(f"  Prompt has {len(correct_codes)} correct solutions")
                
                if correct_codes:
                    # 提取当前response的代码
                    try:
                        code_str = score_info["response_str"].split("</think>")[-1].replace("<|im_end|>", "").replace("", "").replace("<_end>", "").replace("<|endoftext|>", "").strip()
                        # _, _, cleaned_code = code_exec_result(code_str)
                        cleaned_code=extract_python_code(code_str)
                        # 计算与所有正确代码的平均codebleu
                        total_bleu = 0.0
                        print(f"  Calculating CodeBLEU against {len(correct_codes)} correct solutions")
                        
                        for idx, correct_code in enumerate(correct_codes):
                            try:
                                codebleu_result = calc_codebleu(
                                    [cleaned_code],
                                    [correct_code],
                                    lang="python",
                                    weights=(0.1, 0.1, 0.4, 0.4)
                                )
                                bleu_score = codebleu_result["codebleu"]
                                total_bleu += bleu_score
                                print(f"    Against correct solution {idx+1}: CodeBLEU = {bleu_score:.4f}")
                            except Exception as e:
                                print(f"    ❌ CodeBLEU calculation error: {e}")
                                total_bleu += 0.0
                        
                        avg_bleu = total_bleu / len(correct_codes)
                        
                    #     # 修正分数
                        score_info["dict_scores"]['codebleu'] = avg_bleu
                   
                        score_info['total_score'] =score_info["dict_scores"]['score']- old_bleu + avg_bleu
                        
                        print(f"score_info: {score_info}")
                        print(f"  CodeBLEU adjustment for sample {score_info['index']}:")
                        print(f"    Prompt: '{prompt_str[:50]}...'")
                        print(f"    Old CodeBLEU: {old_bleu:.4f} -> New: {avg_bleu:.4f}")
                        print(f"    Old total: {score_info['dict_scores']['score']:.2f} -> New: {score_info['total_score']:.2f}")
                    except Exception as e:
                        print(f"  ❌ Error processing incorrect answer: {e}")

                        # 修正分数 - 修复点在这里
                    #     old_total_score = score_info['total_score']
                    #     new_total_score = old_total_score - old_bleu + avg_bleu
                        
                    #     score_info["dict_scores"]['codebleu'] = avg_bleu
                    #     score_info['total_score'] = new_total_score
                        
                    #     print(f"  CodeBLEU adjustment for sample {score_info['index']}:")
                    #     print(f"    Old CodeBLEU: {old_bleu:.4f} -> New: {avg_bleu:.4f}")
                    #     print(f"    Old total: {old_total_score:.2f} -> New: {new_total_score:.2f}")
                    # except Exception as e:
                    #     print(f"  ❌ Error processing incorrect answer: {e}")
                else:
                    print(f"  ⚠️ No correct solutions for this prompt, keeping original CodeBLEU: {old_bleu:.4f}")
        
        # 第四步：填充结果张量
        print("\n=== Step 4: Filling result tensors ===")
        # 初始化分项张量
        reward_tensor_items = {
            'score':[],
            'format': [],
            'execute': [],
            'answer': [],
            'codebleu': [],
            'filepath': [],
            'acc': [],
            'hx_mean': [],
            'delta_h_mean': [],
            'entropy_reward_sum': []
        }
        # print('----------------naive.py----reward_tensor_items-------')
        # print(all_scores)
        
        print("\nFinal scores:")
        for score_info in all_scores:
            i = score_info["index"]
            valid_response_length = score_info["valid_response_length"]
            total_score = score_info["total_score"]
            dict_scores = score_info["dict_scores"]
            # 提取最终代码块，传给 entropy 计算逻辑备用；当前主要依赖 text/code spans。
            code_text = extract_python_code(
                score_info["response_str"].split("</think>")[-1]
                .replace("<|im_end|>", "")
                .replace("", "")
                .replace("<_end>", "")
                .replace("<|endoftext|>", "")
                .strip()
            )
            # 构造 token 级 entropy reward；如果没有 gold_regions 或 compute_hx 失败，会安全回退为 0。
            entropy_info = build_entropy_reward(
                response_str=score_info["response_str"],
                tokenizer=self.tokenizer,
                response_ids=score_info["response_ids"],
                valid_response_length=valid_response_length,
                extra_info=score_info["extra_info"],
                gold_region=score_info["gold_region"],
                code_text=code_text,
                beta_h=DEFAULT_BETA_H,
            )
            
            # 设置总分
            reward_tensor[i, valid_response_length - 1] = total_score
            # 在原有 PSR 末 token 奖励基础上，叠加 step-level entropy token 奖励。
            reward_tensor[i] += entropy_info["reward"].to(reward_tensor.device)
            
            # 设置分项分数
            for key in ['format', 'execute', 'answer', 'codebleu', 'filepath', 'acc']:
                if key in dict_scores:
                    reward_tensor_items[key].append(dict_scores[key])
            reward_tensor_items['score'].append(total_score)
            reward_tensor_items['hx_mean'].append(entropy_info["hx_mean"])
            reward_tensor_items['delta_h_mean'].append(entropy_info["delta_h_mean"])
            reward_tensor_items['entropy_reward_sum'].append(entropy_info["entropy_reward_sum"])
            # 设置总分到 'score' 键
            # reward_tensor_items['score'][i, valid_response_length - 1] = total_score
            # print(f"  Sample {i}:")
            # print(f"    Total reward: {total_score:.2f}")
            # print(f"    Format: {dict_scores['format']:.2f}")
            # print(f"    Execute: {dict_scores['execute']:.2f}")
            # print(f"    Answer: {dict_scores['answer']:.2f}")
            # print(f"    CodeBLEU: {dict_scores['codebleu']:.4f}")
        
        print("\n===== Reward Calculation Completed =====")
        # print('----------------naive.py----reward_tensor_items-------')
        # print(reward_tensor_items)
    
        if return_dict:
            return {
                "reward_tensor": reward_tensor, # Tensor[batch_size, response_length]
                "reward_extra_info": reward_tensor_items,  # Dict[List[batch_size]]   Dict[Tensor[batch_size response_length]] {'format': [-1, -1, -0.5, ...], 'execute': [...], 'answer': [...]}
            }
        else:
            return reward_tensor
